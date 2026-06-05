"""
Data & Concept Drift Monitoring  (Evidently AI)
================================================
Compares a rolling window of production predictions against the training
baseline to detect:

  - Data drift      : shift in input feature distributions
                      (e.g. sudden influx of high-risk country transactions)
  - Target drift    : shift in model output distribution
                      (e.g. fraud probability mean jumping from 3 % → 18 %)
  - Prediction drift: change in model confidence / decision boundary

Designed to run on a schedule (e.g. every hour via Cloud Scheduler).
Results are written to data/monitoring/ as HTML reports + JSON summaries.

In GCP production, the JSON summaries can be pushed to:
  - Cloud Monitoring custom metrics
  - BigQuery for trend analysis
  - Pub/Sub to trigger automated retraining pipelines

Usage:
    python monitoring/drift_detector.py
    python monitoring/drift_detector.py --window-hours 1
    python monitoring/drift_detector.py --report-html
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import structlog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logger = structlog.get_logger()

MONITOR_DIR     = ROOT / "data/monitoring"
PREDICTION_DIR  = ROOT / "data/predictions"
TRAINING_DATA   = ROOT / "data/processed/transactions.parquet"
MONITOR_DIR.mkdir(parents=True, exist_ok=True)

# Alert thresholds
DRIFT_SHARE_THRESHOLD   = 0.20    # >20 % of features drifted → alert
FRAUD_RATE_THRESHOLD    = 0.05    # >5 % fraud rate in window → alert
LATENCY_P99_THRESHOLD_MS = 100.0  # >100 ms p99 → alert


# ── Data loaders ───────────────────────────────────────────────────────────────

def load_training_reference() -> pd.DataFrame:
    """Load training data as the reference distribution for drift analysis."""
    if not TRAINING_DATA.exists():
        raise FileNotFoundError(
            f"Training data not found at {TRAINING_DATA}. "
            "Run `python data/generate_data.py` first."
        )
    df = pd.read_parquet(TRAINING_DATA)
    logger.info("reference_loaded", rows=len(df))
    return df


def load_recent_predictions(window_hours: int = 24) -> pd.DataFrame:
    """Load prediction logs from the last N hours."""
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=window_hours)
    records = []

    for log_file in sorted(PREDICTION_DIR.glob("predictions_*.jsonl")):
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    logged_at = pd.Timestamp(rec.get("logged_at", ""))
                    if logged_at.tzinfo is None:
                        logged_at = logged_at.tz_localize("UTC")
                    if logged_at >= cutoff:
                        records.append(rec)
                except (json.JSONDecodeError, Exception):
                    continue

    if not records:
        logger.warning("no_predictions_found", window_hours=window_hours)
        return pd.DataFrame()

    df = pd.DataFrame(records)
    logger.info("predictions_loaded", rows=len(df), window_hours=window_hours)
    return df


# ── Drift analysis ─────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "amount",
    "feat_user_txn_count_10m",
    "feat_user_avg_amount_10m",
    "feat_user_txn_count_1h",
    "feat_user_mean_amount",
    "feat_amount_vs_user_avg",
]

REFERENCE_COLUMN_MAP = {
    "feat_user_txn_count_10m":  "user_txn_count_10m",
    "feat_user_avg_amount_10m": "user_avg_amount_10m",
    "feat_user_txn_count_1h":   "user_txn_count_1h",
    "feat_user_mean_amount":    "user_mean_amount",
    "feat_amount_vs_user_avg":  "amount_vs_user_avg",
}


def run_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    report_html: bool = False,
) -> dict:
    """
    Run Evidently AI drift tests and return a JSON-serialisable summary.
    """
    from evidently import ColumnMapping
    from evidently.metrics import (
        DataDriftTable,
        DatasetDriftMetric,
        ColumnDriftMetric,
    )
    from evidently.report import Report
    from evidently.test_suite import TestSuite
    from evidently.tests import (
        TestShareOfDriftedColumns,
        TestColumnDrift,
        TestMeanInNSigmas,
    )

    # Rename current df columns to match reference
    current_renamed = current.rename(columns=REFERENCE_COLUMN_MAP)

    # Align columns
    shared_cols = [c for c in NUMERIC_FEATURES if c in reference.columns
                   and c in current_renamed.columns]

    ref_slice = reference[shared_cols].dropna()
    cur_slice = current_renamed[shared_cols].dropna()

    if cur_slice.empty:
        return {"error": "No valid current data for drift analysis"}

    column_mapping = ColumnMapping(numerical_features=shared_cols)

    # ── Data drift report ──────────────────────────────────────────────────────
    report = Report(metrics=[DatasetDriftMetric(), DataDriftTable()])
    report.run(
        reference_data=ref_slice,
        current_data=cur_slice,
        column_mapping=column_mapping,
    )

    report_dict = report.as_dict()
    dataset_drift = report_dict["metrics"][0]["result"]
    drift_share   = dataset_drift.get("share_of_drifted_columns", 0.0)
    n_drifted     = dataset_drift.get("number_of_drifted_columns", 0)
    n_features    = dataset_drift.get("number_of_columns", len(shared_cols))

    # ── Statistical tests per feature ─────────────────────────────────────────
    suite = TestSuite(tests=[
        TestShareOfDriftedColumns(lt=DRIFT_SHARE_THRESHOLD + 0.01),
        TestMeanInNSigmas(column_name="amount", n=3),
    ])
    suite.run(
        reference_data=ref_slice,
        current_data=cur_slice,
        column_mapping=column_mapping,
    )
    suite_dict    = suite.as_dict()
    tests_passed  = all(t["status"] == "SUCCESS" for t in suite_dict["tests"])

    summary = {
        "run_at":               datetime.utcnow().isoformat(),
        "reference_rows":       len(ref_slice),
        "current_rows":         len(cur_slice),
        "n_features_tested":    n_features,
        "n_features_drifted":   n_drifted,
        "drift_share":          round(drift_share, 4),
        "dataset_drift_detected": drift_share > DRIFT_SHARE_THRESHOLD,
        "tests_passed":         tests_passed,
        "feature_details":      {},
    }

    # Per-column drift stats
    for metric in report_dict["metrics"][1:]:
        col_name = metric.get("result", {}).get("column_name")
        if col_name:
            summary["feature_details"][col_name] = {
                "drift_detected": metric["result"].get("drift_detected", False),
                "stattest":       metric["result"].get("stattest_name", ""),
                "p_value":        metric["result"].get("p_value"),
            }

    # ── Save HTML report ───────────────────────────────────────────────────────
    if report_html:
        ts  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out = MONITOR_DIR / f"drift_report_{ts}.html"
        report.save_html(str(out))
        logger.info("html_report_saved", path=str(out))
        summary["html_report"] = str(out)

    return summary


# ── Target drift (fraud rate) ──────────────────────────────────────────────────

def check_target_drift(current: pd.DataFrame) -> dict:
    """Check if the model's predicted fraud rate has drifted significantly."""
    if current.empty or "fraud_probability" not in current.columns:
        return {}

    fraud_rate    = float(current["is_fraud"].mean()) if "is_fraud" in current.columns else None
    pred_mean     = float(current["fraud_probability"].mean())
    pred_p99      = float(current["fraud_probability"].quantile(0.99))
    latency_p99   = float(current["latency_ms"].quantile(0.99)) if "latency_ms" in current.columns else None

    alerts = []
    if fraud_rate is not None and fraud_rate > FRAUD_RATE_THRESHOLD:
        alerts.append(f"ALERT: Predicted fraud rate {fraud_rate:.1%} exceeds threshold {FRAUD_RATE_THRESHOLD:.1%}")
    if latency_p99 is not None and latency_p99 > LATENCY_P99_THRESHOLD_MS:
        alerts.append(f"ALERT: p99 latency {latency_p99:.0f} ms exceeds {LATENCY_P99_THRESHOLD_MS:.0f} ms SLA")

    return {
        "predicted_fraud_rate":   round(fraud_rate, 4) if fraud_rate is not None else None,
        "fraud_prob_mean":         round(pred_mean, 4),
        "fraud_prob_p99":          round(pred_p99, 4),
        "latency_p99_ms":          round(latency_p99, 2) if latency_p99 else None,
        "alerts":                  alerts,
    }


# ── GCP alert sink ─────────────────────────────────────────────────────────────

def send_gcp_alert(summary: dict) -> None:
    """
    Write a custom metric to Cloud Monitoring when drift is detected.
    Requires: google-cloud-monitoring, GOOGLE_APPLICATION_CREDENTIALS.

    In production, a Cloud Monitoring alert policy picks up these metrics
    and pages on-call via PagerDuty / OpsGenie.
    """
    try:
        from google.cloud import monitoring_v3
        from google.protobuf.timestamp_pb2 import Timestamp
        import time

        client     = monitoring_v3.MetricServiceClient()
        project_id = os.getenv("GCP_PROJECT_ID", "your-gcp-project")
        project_name = f"projects/{project_id}"

        series = monitoring_v3.TimeSeries()
        series.metric.type = "custom.googleapis.com/fraud_detection/drift_share"
        series.resource.type = "global"

        now = time.time()
        ts  = Timestamp()
        ts.seconds = int(now)
        ts.nanos   = int((now % 1) * 1e9)

        point = monitoring_v3.Point()
        point.interval.end_time   = ts
        point.interval.start_time = ts
        point.value.double_value  = summary.get("drift_share", 0.0)
        series.points = [point]

        client.create_time_series(
            name=project_name, time_series=[series]
        )
        logger.info("gcp_metric_written", drift_share=summary.get("drift_share"))

    except ImportError:
        logger.debug("google-cloud-monitoring not installed; skipping GCP metric push")
    except Exception as e:
        logger.warning("gcp_metric_failed", error=str(e))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run drift monitoring checks")
    parser.add_argument("--window-hours", type=int, default=24,
                        help="Hours of prediction history to analyse (default: 24)")
    parser.add_argument("--report-html", action="store_true",
                        help="Save a full Evidently HTML report to data/monitoring/")
    args = parser.parse_args()

    print(f"🔍  Running drift analysis (window={args.window_hours}h) …")

    # 1. Load data
    try:
        reference = load_training_reference()
    except FileNotFoundError as e:
        print(f"❌  {e}")
        sys.exit(1)

    current = load_recent_predictions(window_hours=args.window_hours)

    if current.empty:
        print("⚠️   No recent predictions found. Start the inference API and send some traffic first.")
        print("     Example: curl -X POST http://localhost:8080/predict -d '{...}'")
        return

    # 2. Feature drift
    try:
        drift_summary = run_evidently_report(
            reference=reference,
            current=current,
            report_html=args.report_html,
        )
    except ImportError:
        print("⚠️   evidently not installed. Run: pip install evidently")
        drift_summary = {"error": "evidently not installed"}

    # 3. Target / latency drift
    target_summary = check_target_drift(current)

    # 4. Merge + save
    full_summary = {**drift_summary, "target_metrics": target_summary}
    ts           = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path     = MONITOR_DIR / f"drift_summary_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(full_summary, f, indent=2, default=str)

    # 5. Console output
    print(f"\n── Drift summary ({'⚠️  DRIFT DETECTED' if drift_summary.get('dataset_drift_detected') else '✅  No drift'}) ──")
    print(f"  Features tested    : {drift_summary.get('n_features_tested', 'N/A')}")
    print(f"  Features drifted   : {drift_summary.get('n_features_drifted', 'N/A')}")
    print(f"  Drift share        : {drift_summary.get('drift_share', 0):.1%}")
    if target_summary:
        print(f"  Pred fraud rate    : {target_summary.get('predicted_fraud_rate', 0):.2%}")
        print(f"  Latency p99        : {target_summary.get('latency_p99_ms', 'N/A')} ms")
        for alert in target_summary.get("alerts", []):
            print(f"  🚨  {alert}")
    print(f"\n  Report saved to: {out_path}")

    # 6. Push to GCP Monitoring (no-op if credentials not configured)
    if drift_summary.get("dataset_drift_detected") or target_summary.get("alerts"):
        send_gcp_alert(full_summary)

    # 7. Exit with non-zero code if drift detected (useful in CI/CD pipelines)
    if drift_summary.get("dataset_drift_detected"):
        sys.exit(1)


if __name__ == "__main__":
    main()
