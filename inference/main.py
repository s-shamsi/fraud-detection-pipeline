"""
Fraud Detection Inference API
===============================
FastAPI application serving real-time fraud scores with sub-50 ms latency.

Request flow
────────────
POST /predict
  1. Validate request schema (Pydantic, ~0.5 ms)
  2. Async fetch streaming + historical features from Redis  (~3–5 ms)
  3. Assemble 16-dimensional feature vector                  (~0.1 ms)
  4. ONNX Runtime inference                                  (~1–2 ms)
  5. Log prediction to file for drift monitoring             (async, non-blocking)
  6. Return FraudPredictionResponse                          (~0.5 ms)
  ─────────────────────────────────────────────────────────────────────
  Total (p99 target)                                         < 50 ms

Deployment targets:
  Local dev   — uvicorn inference.main:app --reload
  GCP Prod    — Cloud Run (--min-instances=1 to avoid cold starts)
  Alternative — Vertex AI Prediction endpoint (replace main.py with handler.py)

Usage:
    uvicorn inference.main:app --host 0.0.0.0 --port 8080 --workers 2
    # or via Docker:
    docker build -t fraud-inference ./inference && docker run -p 8080:8080 fraud-inference
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import structlog
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest

from inference.feature_client import (
    FeatureClient,
    encode_country,
    encode_merchant_category,
)
from inference.schemas import (
    FraudPredictionRequest,
    FraudPredictionResponse,
    HealthResponse,
    PredictionLog,
)

logger = structlog.get_logger()

# ── Prometheus metrics ─────────────────────────────────────────────────────────

PRED_COUNT = Counter(
    "fraud_predictions_total",
    "Total predictions",
    ["result", "risk_tier"],
)
E2E_LATENCY = Histogram(
    "fraud_e2e_latency_seconds",
    "End-to-end prediction latency",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)
FEAT_LATENCY = Histogram(
    "fraud_feature_fetch_latency_seconds",
    "Feature store fetch latency",
    buckets=[0.001, 0.003, 0.005, 0.01, 0.025, 0.05],
)

# ── Global state ───────────────────────────────────────────────────────────────

_onnx_session:   Optional[ort.InferenceSession] = None
_feature_client: Optional[FeatureClient]        = None
_model_metadata: dict                           = {}

# ── Lifespan ───────────────────────────────────────────────────────────────────

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise and cleanly shut down resources."""
    global _onnx_session, _feature_client, _model_metadata

    # 1. Load ONNX model
    onnx_path = ARTIFACT_DIR / "fraud_detector.onnx"
    if not onnx_path.exists():
        raise RuntimeError(
            f"ONNX model not found at {onnx_path}. "
            "Run `python model/train.py && python model/convert_to_onnx.py` first."
        )

    session_opts = ort.SessionOptions()
    session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "2"))
    session_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    _onnx_session = ort.InferenceSession(
        str(onnx_path),
        sess_options=session_opts,
        providers=["CPUExecutionProvider"],
    )
    logger.info(
        "onnx_model_loaded",
        path=str(onnx_path),
        inputs=[i.name for i in _onnx_session.get_inputs()],
    )

    # 2. Load metadata (feature list, threshold)
    meta_path = ARTIFACT_DIR / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            _model_metadata = json.load(f)

    # 3. Connect to feature store (Redis)
    _feature_client = FeatureClient()
    await _feature_client.connect()

    logger.info("inference_service_ready", version=MODEL_VERSION)
    yield

    # Cleanup
    if _feature_client:
        await _feature_client.close()
    logger.info("inference_service_stopped")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description=(
        "Sub-50 ms fraud scoring powered by XGBoost/ONNX Runtime and "
        "a Feast feature store backed by GCP Memorystore Redis."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness + readiness check used by Cloud Run and Kubernetes."""
    fs_ok = (
        await _feature_client.health_check()
        if _feature_client else False
    )
    return HealthResponse(
        status="ok" if (_onnx_session and fs_ok) else "degraded",
        model_loaded=_onnx_session is not None,
        feature_store_connected=fs_ok,
    )


@app.get("/metrics", tags=["ops"])
async def metrics() -> Response:
    """Prometheus scrape endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


@app.post("/predict", response_model=FraudPredictionResponse, tags=["inference"])
async def predict(req: FraudPredictionRequest) -> FraudPredictionResponse:
    """
    Score a financial transaction for fraud.

    The response includes the fraud probability, a binary decision, and a
    risk tier (LOW / MEDIUM / HIGH / CRITICAL) for downstream routing.
    """
    t0 = time.monotonic()

    if _onnx_session is None:
        raise HTTPException(503, detail="Model not loaded")

    # ── 1. Fetch features ──────────────────────────────────────────────────────
    feat_t0 = time.monotonic()
    online_feats = await _feature_client.get_online_features(user_id=req.user_id)
    feat_ms = (time.monotonic() - feat_t0) * 1_000
    FEAT_LATENCY.observe((time.monotonic() - feat_t0))

    # ── 2. Build feature vector ────────────────────────────────────────────────
    fv = _build_feature_vector(req, online_feats)

    # ── 3. ONNX inference ──────────────────────────────────────────────────────
    input_name = _onnx_session.get_inputs()[0].name
    onnx_out   = _onnx_session.run(
        None,
        {input_name: fv.reshape(1, -1).astype(np.float32)},
    )

    # onnxmltools output: [predicted_labels, probabilities]
    # probabilities is a list of {0: p_legit, 1: p_fraud} dicts
    if isinstance(onnx_out[1], list):
        fraud_prob = float(onnx_out[1][0].get(1, 0.0))
    else:
        fraud_prob = float(onnx_out[1][0, 1])

    threshold = _model_metadata.get("threshold", 0.40)
    is_fraud  = fraud_prob >= threshold
    risk_tier = _risk_tier(fraud_prob)
    total_ms  = (time.monotonic() - t0) * 1_000

    # ── 4. Metrics ─────────────────────────────────────────────────────────────
    E2E_LATENCY.observe(time.monotonic() - t0)
    PRED_COUNT.labels(
        result="fraud" if is_fraud else "legit",
        risk_tier=risk_tier,
    ).inc()

    # ── 5. Async prediction log (non-blocking) ─────────────────────────────────
    asyncio.create_task(
        _log_prediction(req, fraud_prob, is_fraud, risk_tier, total_ms, online_feats)
    )

    logger.info(
        "prediction",
        txn_id=req.transaction_id,
        user_id=req.user_id,
        fraud_prob=round(fraud_prob, 4),
        is_fraud=is_fraud,
        risk_tier=risk_tier,
        latency_ms=round(total_ms, 2),
        feature_fetch_ms=round(feat_ms, 2),
    )

    return FraudPredictionResponse(
        transaction_id=req.transaction_id,
        user_id=req.user_id,
        fraud_probability=round(fraud_prob, 5),
        is_fraud=is_fraud,
        risk_tier=risk_tier,
        latency_ms=round(total_ms, 2),
        model_version=MODEL_VERSION,
    )


@app.post("/batch-predict", tags=["inference"])
async def batch_predict(requests: list[FraudPredictionRequest]) -> list[FraudPredictionResponse]:
    """
    Score up to 100 transactions in a single call.
    Uses asyncio.gather for concurrent feature fetches.
    """
    if len(requests) > 100:
        raise HTTPException(400, "Maximum 100 transactions per batch")
    return await asyncio.gather(*[predict(r) for r in requests])


# ── Feature assembly ───────────────────────────────────────────────────────────

#  Feature order MUST match data/generate_data.py::get_feature_columns()
#  Index  Feature
#  ─────  ──────────────────────────────
#   0     amount
#   1     hour_of_day
#   2     day_of_week
#   3     is_weekend
#   4     merchant_category_enc
#   5     is_high_risk_country
#   6     user_mean_amount
#   7     user_std_amount
#   8     user_txn_count
#   9     user_fraud_rate
#  10     amount_vs_user_avg
#  11     amount_z_score
#  12     high_amount_flag
#  13     user_txn_count_10m      ← streaming
#  14     user_avg_amount_10m     ← streaming
#  15     user_txn_count_1h       ← streaming

def _build_feature_vector(
    req: FraudPredictionRequest,
    feats: dict,
) -> np.ndarray:
    ts  = req.timestamp or datetime.now(tz=timezone.utc)
    amt = req.amount

    user_mean   = feats.get("user_mean_amount", amt)
    user_std    = feats.get("user_std_amount",  0.0)
    user_count  = feats.get("user_txn_count",   0.0)
    user_fr     = feats.get("user_fraud_rate",  0.0)

    amt_vs_avg  = amt / (user_mean + 1e-6)
    amt_z       = (amt - user_mean) / (user_std + 1.0)
    high_flag   = float(amt_z > 3.0)

    return np.array([
        amt,
        float(ts.hour),
        float(ts.weekday()),
        float(ts.weekday() >= 5),
        float(encode_merchant_category(req.merchant_category)),
        float(encode_country(req.country)),
        float(user_mean),
        float(user_std),
        float(user_count),
        float(user_fr),
        float(amt_vs_avg),
        float(amt_z),
        high_flag,
        float(feats.get("user_txn_count_10m",  0)),
        float(feats.get("user_avg_amount_10m", amt)),
        float(feats.get("user_txn_count_1h",   0)),
    ], dtype=np.float32)


def _risk_tier(prob: float) -> str:
    if prob >= 0.80:
        return "CRITICAL"
    if prob >= 0.50:
        return "HIGH"
    if prob >= 0.30:
        return "MEDIUM"
    return "LOW"


# ── Prediction logging ─────────────────────────────────────────────────────────

_LOG_DIR = Path(os.getenv("PREDICTION_LOG_DIR", "data/predictions"))
_LOG_DIR.mkdir(parents=True, exist_ok=True)


async def _log_prediction(
    req: FraudPredictionRequest,
    fraud_prob: float,
    is_fraud: bool,
    risk_tier: str,
    latency_ms: float,
    feats: dict,
) -> None:
    """
    Append prediction record to a JSONL file for drift monitoring.

    In GCP production, replace file writes with:
      - BigQuery streaming inserts  (google-cloud-bigquery)
      - Cloud Logging structured logs (google-cloud-logging)
      - Pub/Sub for downstream consumers
    """
    record = PredictionLog(
        transaction_id=req.transaction_id,
        user_id=req.user_id,
        amount=req.amount,
        merchant_category=req.merchant_category,
        country=req.country,
        timestamp=req.timestamp or datetime.now(tz=timezone.utc),
        fraud_probability=fraud_prob,
        is_fraud=is_fraud,
        risk_tier=risk_tier,
        latency_ms=latency_ms,
        model_version=MODEL_VERSION,
        logged_at=datetime.now(tz=timezone.utc),
        feat_user_txn_count_10m=feats.get("user_txn_count_10m"),
        feat_user_avg_amount_10m=feats.get("user_avg_amount_10m"),
        feat_user_txn_count_1h=feats.get("user_txn_count_1h"),
        feat_user_mean_amount=feats.get("user_mean_amount"),
        feat_amount_vs_user_avg=feats.get("amount_vs_user_avg"),
    ).model_dump(mode="json")

    log_file = _LOG_DIR / f"predictions_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
