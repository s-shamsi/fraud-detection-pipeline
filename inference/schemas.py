"""
Inference API – Pydantic Schemas
==================================
Request and response models for the FastAPI fraud detection endpoint.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class FraudPredictionRequest(BaseModel):
    """
    Incoming transaction payload sent to POST /predict.
    Only raw transaction fields are required — enrichment features
    (user history, streaming windows) are fetched from the feature store.
    """

    transaction_id:    str   = Field(...,  description="Unique transaction identifier")
    user_id:           str   = Field(...,  description="User identifier (used as feature store key)")
    amount:            float = Field(...,  gt=0, description="Transaction amount in USD")
    merchant_category: str   = Field(...,  description="Merchant category code")
    country:           str   = Field(...,  description="ISO country code of the merchant")
    timestamp:         Optional[datetime] = Field(
        default=None,
        description="Transaction timestamp (UTC); defaults to request time if omitted",
    )

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Transaction amount must be positive")
        return round(v, 2)

    @field_validator("merchant_category")
    @classmethod
    def normalise_category(cls, v: str) -> str:
        return v.lower().strip()

    @field_validator("country")
    @classmethod
    def normalise_country(cls, v: str) -> str:
        return v.upper().strip()

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id":    "txn_001234567",
                "user_id":           "user_00042",
                "amount":            349.99,
                "merchant_category": "electronics",
                "country":           "US",
                "timestamp":         "2024-06-01T14:32:00Z",
            }
        }
    }


class FraudPredictionResponse(BaseModel):
    """Fraud scoring result returned by POST /predict."""

    transaction_id:   str   = Field(...,  description="Echo of the input transaction_id")
    user_id:          str   = Field(...,  description="Echo of the input user_id")
    fraud_probability: float = Field(..., ge=0.0, le=1.0,
                                     description="P(fraud) ∈ [0, 1]")
    is_fraud:         bool  = Field(...,  description="Binary decision at the configured threshold")
    risk_tier:        str   = Field(...,  description="LOW | MEDIUM | HIGH | CRITICAL")
    latency_ms:       float = Field(...,  description="End-to-end latency in milliseconds")
    model_version:    str   = Field(...,  description="Deployed ONNX model version tag")

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id":    "txn_001234567",
                "user_id":           "user_00042",
                "fraud_probability": 0.031,
                "is_fraud":          False,
                "risk_tier":         "LOW",
                "latency_ms":        12.4,
                "model_version":     "1.0.0",
            }
        }
    }


class HealthResponse(BaseModel):
    """Response returned by GET /health."""

    status:                  str  = "ok"
    model_loaded:            bool = False
    feature_store_connected: bool = False


class PredictionLog(BaseModel):
    """
    Internal schema for logging predictions to the offline store.
    Saved as Parquet for drift monitoring and model retraining.
    """

    transaction_id:    str
    user_id:           str
    amount:            float
    merchant_category: str
    country:           str
    timestamp:         datetime
    fraud_probability: float
    is_fraud:          bool
    risk_tier:         str
    latency_ms:        float
    model_version:     str
    logged_at:         datetime
    # Streaming features at decision time (for drift analysis)
    feat_user_txn_count_10m:  Optional[float] = None
    feat_user_avg_amount_10m: Optional[float] = None
    feat_user_txn_count_1h:   Optional[float] = None
    feat_user_mean_amount:    Optional[float] = None
    feat_amount_vs_user_avg:  Optional[float] = None
