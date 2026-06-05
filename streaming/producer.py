"""
Transaction Stream Producer
=============================
Simulates a live feed of financial transactions at configurable TPS.

LOCAL DEV : streams to Redpanda (Docker) on localhost:9092
GCP PROD  : see streaming/gcp_pubsub_producer.py — identical interface,
            streams to a Cloud Pub/Sub topic instead.

Three drift modes (activate via env var DRIFT_MODE):
  normal   — baseline fraud rate ~1.5 %
  attack   — coordinated fraud spike (rate → 25 %, high amounts)
  velocity — velocity fraud burst (same user, many small txns in seconds)

Usage:
    python streaming/producer.py
    DRIFT_MODE=attack TPS=500 python streaming/producer.py
    python streaming/producer.py --tps 200 --duration 60
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logger = structlog.get_logger()

# ── Configuration ──────────────────────────────────────────────────────────────

BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC             = os.getenv("KAFKA_TOPIC",              "transactions")
DEFAULT_TPS       = int(os.getenv("TPS", "100"))
DRIFT_MODE        = os.getenv("DRIFT_MODE", "normal")   # normal | attack | velocity

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "clothing", "restaurant", "gas_station",
    "online_retail", "travel", "entertainment", "pharmacy", "utilities",
]
LEGIT_COUNTRIES   = ["US", "GB", "DE", "FR", "JP", "CA", "AU", "NL", "SE"]
HIGH_RISK_REGIONS = ["XX1", "XX2", "XX3", "XX4"]

# ── Transaction factory ────────────────────────────────────────────────────────

def make_transaction(
    fraud_rate: float = 0.015,
    drift_mode: str = "normal",
    burst_user: Optional[str] = None,
) -> dict:
    """
    Generate a single synthetic transaction payload.

    In 'attack' mode the fraud rate jumps to ~25 % and amounts are inflated —
    this is the scenario that Evidently AI drift monitoring is designed to catch.
    """
    user_id = burst_user or f"user_{random.randint(0, 999):05d}"

    if drift_mode == "attack":
        is_fraud  = random.random() < 0.25          # coordinated spike
        amount    = random.lognormvariate(5.8, 1.1) if is_fraud else random.lognormvariate(3.8, 0.5)
        country   = random.choice(HIGH_RISK_REGIONS) if is_fraud else random.choice(LEGIT_COUNTRIES)
        category  = random.choice(["electronics", "online_retail", "travel"]) if is_fraud else random.choice(MERCHANT_CATEGORIES)

    elif drift_mode == "velocity":
        is_fraud  = True
        amount    = round(random.uniform(0.50, 4.99), 2)   # card-testing micro-amounts
        country   = random.choice(LEGIT_COUNTRIES)
        category  = random.choice(["online_retail", "electronics"])

    else:  # normal
        is_fraud  = random.random() < fraud_rate
        amount    = random.lognormvariate(5.5, 0.9) if is_fraud else random.lognormvariate(3.8, 0.5)
        country   = random.choice(HIGH_RISK_REGIONS) if is_fraud else random.choice(LEGIT_COUNTRIES)
        category  = random.choice(MERCHANT_CATEGORIES)

    now = datetime.now(tz=timezone.utc)
    return {
        "transaction_id":   f"txn_{int(now.timestamp() * 1e6)}_{random.randint(100, 999)}",
        "user_id":          user_id,
        "amount":           round(amount, 2),
        "merchant_category": category,
        "country":          country,
        "timestamp":        now.isoformat(),
        "hour_of_day":      now.hour,
        "day_of_week":      now.weekday(),
        # Ground-truth label — available for monitoring/evaluation but
        # NOT sent to the inference API in production
        "_is_fraud_label":  int(is_fraud),
    }


# ── Producer ───────────────────────────────────────────────────────────────────

def build_producer() -> KafkaProducer:
    for attempt in range(1, 6):
        try:
            producer = KafkaProducer(
                bootstrap_servers=BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=str.encode,
                compression_type="snappy",
                linger_ms=5,
                batch_size=32768,
                acks="all",
                retries=3,
            )
            logger.info("kafka_connected", brokers=BOOTSTRAP_SERVERS)
            return producer
        except NoBrokersAvailable:
            logger.warning(f"kafka_unavailable_retrying (attempt {attempt}/5)")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Cannot connect to Kafka at {BOOTSTRAP_SERVERS}.")


def run_producer(
    tps: int          = DEFAULT_TPS,
    duration_s: Optional[int] = None,
    fraud_rate: float = 0.015,
    drift_mode: str   = DRIFT_MODE,
) -> None:
    """
    Stream transactions to Kafka.

    In 'velocity' mode, we periodically pick a single user and send 10 rapid
    transactions from them to test velocity-based fraud detection.
    """
    producer = build_producer()

    interval  = 1.0 / tps
    count     = 0
    fraud_cnt = 0
    start     = time.monotonic()
    burst_user = None
    burst_remaining = 0

    logger.info("producer_started", tps=tps, topic=TOPIC, drift_mode=drift_mode)
    print(f"🚀 Streaming to {TOPIC} @ {tps} TPS  [mode={drift_mode}]  Ctrl+C to stop")

    try:
        while True:
            # Velocity burst logic
            if drift_mode == "velocity":
                if burst_remaining <= 0:
                    burst_user = f"user_{random.randint(0, 999):05d}"
                    burst_remaining = random.randint(8, 20)
            else:
                burst_user = None

            txn = make_transaction(
                fraud_rate=fraud_rate,
                drift_mode=drift_mode,
                burst_user=burst_user,
            )
            if burst_remaining > 0:
                burst_remaining -= 1

            producer.send(TOPIC, value=txn, key=txn["user_id"])
            count += 1
            fraud_cnt += txn["_is_fraud_label"]

            if count % 5_000 == 0:
                elapsed = time.monotonic() - start
                actual_tps = count / elapsed
                logger.info(
                    "producer_stats",
                    sent=count,
                    fraud_rate=f"{fraud_cnt / count:.2%}",
                    actual_tps=f"{actual_tps:.0f}",
                    elapsed_s=f"{elapsed:.0f}",
                )

            if duration_s and (time.monotonic() - start) >= duration_s:
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        pass
    finally:
        producer.flush()
        producer.close()
        elapsed = time.monotonic() - start
        logger.info(
            "producer_stopped",
            total_sent=count,
            fraud_rate=f"{fraud_cnt / max(count, 1):.2%}",
            elapsed_s=f"{elapsed:.0f}",
        )
        print(f"\n✅ Sent {count:,} transactions ({fraud_cnt:,} fraud) in {elapsed:.0f}s")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transaction stream producer")
    parser.add_argument("--tps",      type=int,   default=DEFAULT_TPS, help="Transactions per second")
    parser.add_argument("--duration", type=int,   default=None,        help="Stop after N seconds")
    parser.add_argument("--mode",     type=str,   default=DRIFT_MODE,
                        choices=["normal", "attack", "velocity"],
                        help="Drift injection mode")
    args = parser.parse_args()
    run_producer(tps=args.tps, duration_s=args.duration, drift_mode=args.mode)
