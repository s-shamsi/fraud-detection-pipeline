"""
Streaming Aggregation Engine
==============================
Consumes the transaction Kafka topic, maintains per-user sliding windows
(10 min and 1 h), and writes the aggregated features to Redis so the
inference API can read them at sub-5 ms latency.

Architecture
------------
Kafka topic ──► this process (stateful, in-memory deques) ──► Redis

Streaming features written per user_id:
  user_txn_count_10m    — transaction count in last 10 min
  user_avg_amount_10m   — average amount in last 10 min
  user_txn_count_1h     — transaction count in last 1 h
  user_avg_amount_1h    — average amount in last 1 h
  last_txn_amount       — amount of most recent transaction

GCP Production note:
  For managed streaming at scale, replace this process with a
  Cloud Dataflow (Apache Beam) pipeline using the same logic:
    - Input : PubSubIO.readStrings() from the transactions topic
    - Window: SlidingWindows.of(Duration.standardMinutes(10))
    - Sink  : RedisIO.write() to GCP Memorystore
  See gcp/dataflow_aggregator.py for the Beam template stub.

Usage:
    python streaming/aggregator.py
    REDIS_HOST=my-memorystore-ip python streaming/aggregator.py
"""

import json
import os
import signal
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Tuple

import redis
import structlog
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

logger = structlog.get_logger()
ROOT = Path(__file__).resolve().parents[1]

# ── Config ─────────────────────────────────────────────────────────────────────

BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC             = os.getenv("KAFKA_TOPIC",              "transactions")
GROUP_ID          = os.getenv("KAFKA_GROUP_ID",           "fraud-aggregator-v1")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB",   "0"))

WINDOW_10M_SECS = 10 * 60     # 600 seconds
WINDOW_1H_SECS  = 60 * 60     # 3600 seconds
REDIS_KEY_TTL   = 2 * 3600    # 2 h — auto-expire inactive users
FLUSH_INTERVAL  = 100          # Write to Redis every N messages (batch)

# ── Type alias ─────────────────────────────────────────────────────────────────

Event = Tuple[float, float]     # (unix_timestamp, amount)

# ── Redis helpers ──────────────────────────────────────────────────────────────

def build_redis_client() -> redis.Redis:
    for attempt in range(1, 6):
        try:
            r = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=2,
                retry_on_timeout=True,
            )
            r.ping()
            logger.info("redis_connected", host=REDIS_HOST, port=REDIS_PORT)
            return r
        except redis.ConnectionError:
            logger.warning(f"redis_unavailable_retrying (attempt {attempt}/5)")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Cannot connect to Redis at {REDIS_HOST}:{REDIS_PORT}")


def write_user_features(
    r: redis.Redis,
    user_id: str,
    features: dict,
) -> None:
    """
    Write streaming features to Redis using a hash.
    Key format: fraud_detection:streaming:<user_id>

    This key format is intentionally compatible with the Feast online store
    layout so that `feast get_online_features()` can also resolve these values.
    """
    key = f"fraud_detection:streaming:{user_id}"
    pipeline = r.pipeline(transaction=False)
    pipeline.hset(key, mapping={k: str(round(v, 6)) for k, v in features.items()})
    pipeline.expire(key, REDIS_KEY_TTL)
    pipeline.execute()


# ── State management ────────────────────────────────────────────────────────────

class UserWindowState:
    """Maintains per-user sliding-window event deques in memory."""

    __slots__ = ("events",)

    def __init__(self) -> None:
        self.events: Deque[Event] = deque()

    def add(self, ts: float, amount: float) -> None:
        self.events.append((ts, amount))

    def trim_and_aggregate(self, now: float) -> dict:
        """Prune expired events and compute window statistics."""
        cutoff_1h = now - WINDOW_1H_SECS

        # Prune events older than the longest window
        while self.events and self.events[0][0] < cutoff_1h:
            self.events.popleft()

        if not self.events:
            return {
                "user_txn_count_10m":  0,
                "user_avg_amount_10m": 0.0,
                "user_txn_count_1h":   0,
                "user_avg_amount_1h":  0.0,
                "last_txn_amount":     0.0,
            }

        cutoff_10m = now - WINDOW_10M_SECS
        events_10m = [(t, a) for t, a in self.events if t >= cutoff_10m]
        events_1h  = list(self.events)

        cnt_10m = len(events_10m)
        avg_10m = sum(a for _, a in events_10m) / cnt_10m if cnt_10m else 0.0
        cnt_1h  = len(events_1h)
        avg_1h  = sum(a for _, a in events_1h)  / cnt_1h  if cnt_1h  else 0.0

        return {
            "user_txn_count_10m":  cnt_10m,
            "user_avg_amount_10m": avg_10m,
            "user_txn_count_1h":   cnt_1h,
            "user_avg_amount_1h":  avg_1h,
            "last_txn_amount":     events_1h[-1][1] if events_1h else 0.0,
        }


# ── Aggregator main loop ───────────────────────────────────────────────────────

def run_aggregator() -> None:
    r = build_redis_client()

    # Connect to Kafka
    for attempt in range(1, 6):
        try:
            consumer = KafkaConsumer(
                TOPIC,
                bootstrap_servers=BOOTSTRAP_SERVERS,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                group_id=GROUP_ID,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                max_poll_records=500,
                session_timeout_ms=30_000,
            )
            break
        except NoBrokersAvailable:
            logger.warning(f"kafka_unavailable_retrying (attempt {attempt}/5)")
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError("Cannot connect to Kafka")

    logger.info("aggregator_started", topic=TOPIC, group=GROUP_ID)

    # Per-user in-memory state
    user_states: dict[str, UserWindowState] = defaultdict(UserWindowState)

    # Dirty set — track which users need a Redis write
    dirty: set[str] = set()

    processed   = 0
    redis_writes = 0
    start        = time.monotonic()

    # Graceful shutdown
    shutdown = False
    def _shutdown(sig, frame):
        nonlocal shutdown
        shutdown = True
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        for message in consumer:
            if shutdown:
                break

            txn = message.value
            try:
                user_id = txn["user_id"]
                amount  = float(txn["amount"])
                ts_str  = txn.get("timestamp", "")
                ts      = (
                    datetime.fromisoformat(ts_str).timestamp()
                    if ts_str else time.time()
                )
            except (KeyError, ValueError) as e:
                logger.warning("bad_message", error=str(e))
                continue

            user_states[user_id].add(ts, amount)
            dirty.add(user_id)
            processed += 1

            # Flush to Redis in batches
            if len(dirty) >= FLUSH_INTERVAL:
                now = time.time()
                pipeline = r.pipeline(transaction=False)
                for uid in dirty:
                    feats = user_states[uid].trim_and_aggregate(now)
                    key   = f"fraud_detection:streaming:{uid}"
                    pipeline.hset(key, mapping={k: str(round(v, 6)) for k, v in feats.items()})
                    pipeline.expire(key, REDIS_KEY_TTL)
                pipeline.execute()
                redis_writes += len(dirty)
                dirty.clear()

            if processed % 10_000 == 0:
                elapsed = time.monotonic() - start
                logger.info(
                    "aggregator_stats",
                    processed=processed,
                    redis_writes=redis_writes,
                    users_tracked=len(user_states),
                    throughput_tps=f"{processed / elapsed:.0f}",
                )

    finally:
        # Final flush
        if dirty:
            now = time.time()
            for uid in dirty:
                feats = user_states[uid].trim_and_aggregate(now)
                write_user_features(r, uid, feats)

        consumer.close()
        logger.info(
            "aggregator_stopped",
            total_processed=processed,
            users_tracked=len(user_states),
        )
        print(f"\n✅ Aggregator stopped. Processed {processed:,} events.")


# ── Bytewax integration note ───────────────────────────────────────────────────
#
# For a fully stateful stream-processing graph, the logic above maps cleanly
# to Bytewax 0.20+ operators:
#
#   flow = Dataflow("fraud_aggregator")
#   inp  = flow.input("kafka", KafkaSource([BOOTSTRAP_SERVERS], [TOPIC]))
#   keyed = inp.map("parse", parse_txn).key_on("uid", lambda x: x[0])
#   windowed = keyed.fold_window(
#       "agg_10m",
#       SlidingWindow(length=timedelta(minutes=10), slide=timedelta(minutes=1), ...),
#       builder=lambda: {"amounts": [], "count": 0},
#       folder=lambda acc, x: {"amounts": acc["amounts"] + [x[1]], "count": acc["count"] + 1},
#   )
#   windowed.output("redis", RedisSink(host=REDIS_HOST))
#
# See gcp/dataflow_aggregator.py for the Apache Beam / Cloud Dataflow equivalent.
# ──────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    run_aggregator()
