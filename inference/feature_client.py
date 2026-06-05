"""
Feature Client
===============
Async Redis client that reads streaming and historical features for the
inference service.  Uses direct Redis hash access (bypassing the Feast SDK
in the hot path) to achieve sub-5 ms feature fetch latency.

Key format: fraud_detection:streaming:<user_id>
             fraud_detection:historical:<user_id>

The Feast online store also uses Redis; it materialises historical features
under its own key scheme.  For the inference hot path, we read both key
schemes here for maximum flexibility.
"""

import os
from typing import Optional

import structlog

logger = structlog.get_logger()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))


class FeatureClient:
    """
    Async Redis client for real-time feature retrieval.

    Maintains a single connection pool; safe to share across FastAPI workers.
    """

    def __init__(self) -> None:
        self._redis = None

    async def connect(self) -> None:
        import redis.asyncio as aioredis

        self._redis = await aioredis.from_url(
            f"redis://{REDIS_HOST}:{REDIS_PORT}",
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
            socket_connect_timeout=2,
            socket_timeout=2,
            retry_on_timeout=True,
        )
        await self._redis.ping()
        logger.info("redis_connected", host=REDIS_HOST, port=REDIS_PORT)

    async def get_online_features(self, user_id: str) -> dict:
        """
        Fetch the complete online feature set for a user.

        Merges streaming features (written by the aggregator) and historical
        features (materialised from the offline store by Feast).
        Returns an empty dict if the user is brand new (cold start).
        """
        pipeline = self._redis.pipeline(transaction=False)
        pipeline.hgetall(f"fraud_detection:streaming:{user_id}")
        pipeline.hgetall(f"fraud_detection:historical:{user_id}")
        results = await pipeline.execute()

        streaming  = results[0] or {}
        historical = results[1] or {}

        # Merge: streaming features take priority (fresher)
        merged = {**historical, **streaming}

        # Cast all values to float for safe np.array construction
        typed: dict[str, float] = {}
        for k, v in merged.items():
            try:
                typed[k] = float(v)
            except (ValueError, TypeError):
                pass

        if not typed:
            logger.debug("cold_start_user", user_id=user_id)

        return typed

    async def set_historical_features(
        self,
        user_id: str,
        features: dict,
        ttl_seconds: int = 86_400,   # 24 h
    ) -> None:
        """
        Write historical features for a user (called by the materialization job).
        """
        key = f"fraud_detection:historical:{user_id}"
        await self._redis.hset(
            key,
            mapping={k: str(round(float(v), 6)) for k, v in features.items()},
        )
        await self._redis.expire(key, ttl_seconds)

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            logger.info("redis_disconnected")

    async def health_check(self) -> bool:
        """Returns True if Redis is reachable."""
        try:
            return await self._redis.ping()
        except Exception:
            return False


# ── Merchant category encoding (must match data/generate_data.py) ──────────────

_MERCHANT_CATEGORY_INDEX: dict[str, int] = {
    "clothing":     0,
    "electronics":  1,
    "entertainment":2,
    "gas_station":  3,
    "grocery":      4,
    "online_retail":5,
    "pharmacy":     6,
    "restaurant":   7,
    "travel":       8,
    "utilities":    9,
}

_HIGH_RISK_REGIONS = {"XX1", "XX2", "XX3", "XX4"}


def encode_merchant_category(cat: str) -> int:
    return _MERCHANT_CATEGORY_INDEX.get(cat.lower().strip(), 5)   # default: online_retail


def encode_country(country: str) -> float:
    """Returns 1.0 if country is a high-risk region, 0.0 otherwise."""
    return 1.0 if country.upper() in _HIGH_RISK_REGIONS else 0.0
