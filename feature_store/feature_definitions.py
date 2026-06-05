"""
Feast Feature Store Definitions
=================================
Defines the feature registry for the real-time fraud detection pipeline.

Feature architecture
--------------------
┌──────────────────────────────────────────────────────────────────────────┐
│  OFFLINE STORE  (GCS Parquet / BigQuery)                                 │
│  - user_historical_features  ← nightly batch job materialises these      │
│    into the online store every 24 h                                      │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │  feast materialize / feast materialize-incremental
┌───────────────────────────────▼──────────────────────────────────────────┐
│  ONLINE STORE  (Redis / GCP Memorystore)                                 │
│  - user_historical_features  ← low-latency reads for inference           │
│  - user_streaming_features   ← Bytewax / aggregator writes directly      │
│    (10-min and 1-h sliding window aggregates)                            │
└──────────────────────────────────────────────────────────────────────────┘

Usage:
    cd feature_store && feast apply
    python feature_store/materialize.py
"""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, PushSource, Project
from feast.types import Float32, Float64, Int64

# ── Project ────────────────────────────────────────────────────────────────────

project = Project(
    name="fraud_detection",
    description="Real-time fraud detection feature store",
    tags={"team": "ml-platform", "domain": "fraud", "env": "prod"},
)

# ── Entities ───────────────────────────────────────────────────────────────────

user = Entity(
    name="user_id",
    description=(
        "Unique identifier for a platform user. "
        "Used as the join key for all user-scoped feature views."
    ),
    join_keys=["user_id"],
    tags={"pii": "pseudonymised"},
)

# ── Data sources ───────────────────────────────────────────────────────────────

# LOCAL DEV ── Parquet file on disk (works out-of-the-box, no cloud credentials)
_PARQUET_PATH = str(
    Path(__file__).resolve().parents[1] / "data/processed/transactions.parquet"
)
offline_parquet_source = FileSource(
    path=_PARQUET_PATH,
    timestamp_field="timestamp",
    created_timestamp_column="timestamp",
    name="transactions_parquet",
    description="Historical transaction Parquet file (local dev)",
)

# ─────────────────────────────────────────────────────────────────────────────
# GCP PRODUCTION ── uncomment to switch to BigQuery offline store
# from feast.infra.offline_stores.bigquery_source import BigQuerySource
#
# offline_bq_source = BigQuerySource(
#     name="transactions_bigquery",
#     table="<your-project>.fraud_features.user_transaction_features",
#     timestamp_field="timestamp",
#     description="User transaction features in BigQuery",
# )
# ─────────────────────────────────────────────────────────────────────────────

# Streaming push source: the Bytewax aggregator pushes real-time windows here.
# Feast routes the push to the online store (Redis) immediately.
transaction_stream_push = PushSource(
    name="transaction_stream_push",
    batch_source=offline_parquet_source,
    description=(
        "Streaming push source for real-time 10-min / 1-h window aggregates. "
        "Written by the Bytewax aggregator service."
    ),
)

# ── Feature views ──────────────────────────────────────────────────────────────

user_historical_fv = FeatureView(
    name="user_historical_features",
    entities=[user],
    ttl=timedelta(days=30),
    schema=[
        Field(name="user_mean_amount",  dtype=Float32,
              description="User's mean transaction amount (lifetime)"),
        Field(name="user_std_amount",   dtype=Float32,
              description="User's transaction amount std dev (lifetime)"),
        Field(name="user_txn_count",    dtype=Int64,
              description="User's total transaction count (lifetime)"),
        Field(name="user_fraud_rate",   dtype=Float32,
              description="User's historical fraud rate (lifetime)"),
        Field(name="amount_vs_user_avg",dtype=Float32,
              description="Current amount / user mean amount ratio"),
        Field(name="amount_z_score",    dtype=Float32,
              description="Z-score of current amount vs. user distribution"),
        Field(name="high_amount_flag",  dtype=Int64,
              description="1 if amount is >3 std devs above user mean"),
    ],
    source=offline_parquet_source,
    online=True,
    description=(
        "Historical per-user spending patterns. Materialised to Redis daily "
        "via `feast materialize-incremental`. Used by the inference API to "
        "enrich real-time transactions."
    ),
    tags={"tier": "batch-materialized", "latency": "daily"},
)

user_streaming_fv = FeatureView(
    name="user_streaming_features",
    entities=[user],
    ttl=timedelta(hours=2),          # Entries expire after 2 h of inactivity
    schema=[
        Field(name="user_txn_count_10m",   dtype=Int64,
              description="Number of transactions by this user in last 10 min"),
        Field(name="user_avg_amount_10m",  dtype=Float32,
              description="Average transaction amount in last 10 min"),
        Field(name="user_txn_count_1h",    dtype=Int64,
              description="Number of transactions by this user in last 1 h"),
        Field(name="user_avg_amount_1h",   dtype=Float32,
              description="Average transaction amount in last 1 h"),
        Field(name="last_txn_amount",      dtype=Float32,
              description="Amount of the most recent transaction"),
    ],
    source=transaction_stream_push,
    online=True,
    description=(
        "Real-time sliding-window aggregates written by the Bytewax streaming "
        "aggregator. Sub-second freshness. Redis TTL: 2 h."
    ),
    tags={"tier": "streaming", "latency": "sub-second"},
)
