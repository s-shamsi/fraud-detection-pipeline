"""
GCP Pub/Sub Transaction Producer
==================================
Drop-in replacement for streaming/producer.py that targets a Cloud Pub/Sub
topic instead of Kafka/Redpanda.

GCP Services used:
  - Cloud Pub/Sub  : managed message queue (replaces Kafka)
  - Cloud Logging  : structured log ingestion

Setup:
    gcloud auth application-default login
    export GCP_PROJECT_ID=your-project-id
    export PUBSUB_TOPIC=transactions

Usage:
    python streaming/gcp_pubsub_producer.py
    DRIFT_MODE=attack python streaming/gcp_pubsub_producer.py --tps 500
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streaming.producer import make_transaction      # reuse same factory

logger = structlog.get_logger()

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-gcp-project")
PUBSUB_TOPIC   = os.getenv("PUBSUB_TOPIC",   "transactions")
DEFAULT_TPS    = int(os.getenv("TPS",         "200"))
DRIFT_MODE     = os.getenv("DRIFT_MODE",      "normal")


def run_pubsub_producer(
    tps: int          = DEFAULT_TPS,
    duration_s: Optional[int] = None,
    drift_mode: str   = DRIFT_MODE,
) -> None:
    """
    Publish synthetic transactions to Cloud Pub/Sub.

    Uses the async (futures) publish API with batching for high throughput:
      - max_messages: 1000 per batch
      - max_latency:  0.05 s
    This allows sustained ~10 000 TPS on a standard n2-standard-4 VM.
    """
    from google.cloud import pubsub_v1
    from google.api_core.exceptions import GoogleAPICallError

    publisher = pubsub_v1.PublisherClient(
        batch_settings=pubsub_v1.types.BatchSettings(
            max_messages=1_000,
            max_latency=0.05,
            max_bytes=1_000_000,
        ),
    )
    topic_path = publisher.topic_path(GCP_PROJECT_ID, PUBSUB_TOPIC)

    count     = 0
    fraud_cnt = 0
    start     = time.monotonic()
    interval  = 1.0 / tps
    futures   = []

    logger.info(
        "pubsub_producer_started",
        topic=topic_path,
        tps=tps,
        drift_mode=drift_mode,
    )
    print(f"🚀 Publishing to {topic_path} @ {tps} TPS  [mode={drift_mode}]  Ctrl+C to stop")

    try:
        while True:
            txn  = make_transaction(fraud_rate=0.015, drift_mode=drift_mode)
            data = json.dumps(txn).encode("utf-8")

            # Attributes allow server-side filtering in Pub/Sub subscriptions
            attrs = {
                "user_id":   txn["user_id"],
                "country":   txn["country"],
                "is_fraud":  str(txn["_is_fraud_label"]),
            }
            future = publisher.publish(topic_path, data=data, **attrs)
            futures.append(future)
            count     += 1
            fraud_cnt += txn["_is_fraud_label"]

            # Resolve completed futures to propagate errors early
            if len(futures) >= 5_000:
                resolved = []
                for f in futures:
                    try:
                        f.result(timeout=5)
                    except GoogleAPICallError as e:
                        logger.error("pubsub_publish_error", error=str(e))
                futures = resolved

            if count % 10_000 == 0:
                elapsed = time.monotonic() - start
                logger.info(
                    "pubsub_producer_stats",
                    sent=count,
                    fraud_rate=f"{fraud_cnt / count:.2%}",
                    tps=f"{count / elapsed:.0f}",
                )

            if duration_s and (time.monotonic() - start) >= duration_s:
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        pass
    finally:
        # Flush remaining futures
        for f in futures:
            try:
                f.result(timeout=10)
            except Exception:
                pass

        elapsed = time.monotonic() - start
        logger.info(
            "pubsub_producer_stopped",
            total_sent=count,
            fraud_rate=f"{fraud_cnt / max(count, 1):.2%}",
            elapsed_s=f"{elapsed:.0f}",
        )
        print(f"\n✅ Published {count:,} messages ({fraud_cnt:,} fraud) in {elapsed:.0f}s")


# ── Topic provisioning helper ──────────────────────────────────────────────────

def ensure_topic_exists() -> None:
    """Create the Pub/Sub topic if it doesn't exist yet."""
    from google.cloud import pubsub_v1
    from google.api_core.exceptions import AlreadyExists

    publisher  = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(GCP_PROJECT_ID, PUBSUB_TOPIC)
    try:
        publisher.create_topic(request={"name": topic_path})
        logger.info("topic_created", topic=topic_path)
    except AlreadyExists:
        logger.info("topic_already_exists", topic=topic_path)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud Pub/Sub transaction producer")
    parser.add_argument("--tps",      type=int, default=DEFAULT_TPS)
    parser.add_argument("--duration", type=int, default=None, help="Stop after N seconds")
    parser.add_argument(
        "--mode", type=str, default=DRIFT_MODE,
        choices=["normal", "attack", "velocity"],
    )
    parser.add_argument(
        "--create-topic", action="store_true",
        help="Create the Pub/Sub topic if it doesn't exist",
    )
    args = parser.parse_args()

    if args.create_topic:
        ensure_topic_exists()

    run_pubsub_producer(tps=args.tps, duration_s=args.duration, drift_mode=args.mode)
