"""
Feature Store Materialization
==============================
Materialises user_historical_features from the offline store (Parquet / BigQuery)
into the online store (Redis / GCP Memorystore) so the inference API can serve
them at sub-5 ms latency.

Two modes:
  full        -- backfill ALL data (first run only)
  incremental -- only materialise new data since the last successful run
                 (run this hourly in production via Cloud Scheduler)

GCP production: replace the FileSource with a BigQuerySource and schedule this
script as a Vertex AI Pipeline component or Cloud Run Job triggered by
Cloud Scheduler.

Usage:
    python feature_store/materialize.py          # incremental (default)
    python feature_store/materialize.py --full   # full backfill
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import structlog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logger = structlog.get_logger()


def run_materialization(full: bool = False) -> None:
    import feast

    repo_path = ROOT / "feature_store"
    store = feast.FeatureStore(repo_path=str(repo_path))

    # Determine time bounds
    end_date   = datetime.now(tz=timezone.utc)
    if full:
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        logger.info(
            "full_materialization",
            start=str(start_date),
            end=str(end_date),
        )
        store.materialize(
            start_date=start_date,
            end_date=end_date,
            feature_views=["user_historical_features"],
        )
    else:
        logger.info("incremental_materialization", end=str(end_date))
        store.materialize_incremental(
            end_date=end_date,
            feature_views=["user_historical_features"],
        )

    logger.info("materialization_complete")
    print("\n✅ Materialisation complete — Redis now reflects the offline store.")


def verify_online_store(n_users: int = 5) -> None:
    """Spot-check that a handful of users have features in the online store."""
    import feast
    import pandas as pd

    repo_path = ROOT / "feature_store"
    store = feast.FeatureStore(repo_path=str(repo_path))

    # Pull a few user IDs from the offline store
    try:
        df = pd.read_parquet(ROOT / "data/processed/transactions.parquet")
        sample_users = df["user_id"].drop_duplicates().head(n_users).tolist()
    except FileNotFoundError:
        print("⚠️  No offline data found. Run `python data/generate_data.py` first.")
        return

    entity_rows = [{"user_id": uid} for uid in sample_users]
    feature_vector = store.get_online_features(
        features=[
            "user_historical_features:user_mean_amount",
            "user_historical_features:user_txn_count",
            "user_historical_features:user_fraud_rate",
        ],
        entity_rows=entity_rows,
    ).to_df()

    print("\n── Online store spot-check ───────────────────────────────────────")
    print(feature_vector.to_string(index=False))
    missing = feature_vector["user_mean_amount"].isna().sum()
    print(f"\n  Missing values: {missing}/{n_users}")
    if missing == 0:
        print("  ✅ All sampled users have online features")
    else:
        print("  ⚠️  Some users are missing; re-run with --full if needed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialise Feast features to Redis")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full backfill instead of incremental update",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Spot-check the online store after materialisation (default: True)",
    )
    args = parser.parse_args()

    run_materialization(full=args.full)
    if args.verify:
        verify_online_store()


if __name__ == "__main__":
    main()
