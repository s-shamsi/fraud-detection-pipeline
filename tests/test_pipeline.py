"""
Tests – data generation, model training, and ONNX conversion.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Data generation ────────────────────────────────────────────────────────────

class TestDataGeneration:

    def test_output_shape(self, sample_transactions):
        assert len(sample_transactions) == 500

    def test_fraud_rate_in_range(self, sample_transactions):
        rate = sample_transactions["is_fraud"].mean()
        assert 0.005 <= rate <= 0.10, f"Unexpected fraud rate: {rate:.3f}"

    def test_required_columns_present(self, sample_transactions):
        required = [
            "transaction_id", "user_id", "amount",
            "merchant_category", "country", "timestamp", "is_fraud",
        ]
        for col in required:
            assert col in sample_transactions.columns, f"Missing column: {col}"

    def test_amounts_are_positive(self, sample_transactions):
        assert (sample_transactions["amount"] > 0).all()

    def test_feature_columns_match(self, sample_transactions, feature_columns):
        from data.generate_data import get_feature_columns
        # Streaming feature placeholders added in __main__, skip here
        non_streaming = [c for c in feature_columns if "10m" not in c and "1h" not in c]
        for col in non_streaming:
            assert col in sample_transactions.columns, f"Feature column missing: {col}"

    def test_no_null_user_ids(self, sample_transactions):
        assert sample_transactions["user_id"].notna().all()

    def test_encoding_integrity(self, sample_transactions):
        """Encoded merchant categories should be integers 0-9."""
        assert sample_transactions["merchant_category_enc"].between(0, 9).all()


# ── Feature vector assembly ────────────────────────────────────────────────────

class TestFeatureVector:

    def test_feature_count(self, feature_columns):
        assert len(feature_columns) == 16, (
            f"Expected 16 features, got {len(feature_columns)}"
        )

    def test_feature_vector_shape(self, sample_transactions, feature_columns):
        X = sample_transactions[
            [c for c in feature_columns if c in sample_transactions.columns]
        ].fillna(0).values
        assert X.shape[1] <= len(feature_columns)

    def test_no_inf_values(self, sample_transactions, feature_columns):
        import pandas as pd
        cols = [c for c in feature_columns if c in sample_transactions.columns]
        X = sample_transactions[cols].fillna(0)
        assert not np.isinf(X.values).any(), "Inf values found in feature matrix"


# ── Inference schemas ──────────────────────────────────────────────────────────

class TestSchemas:

    def test_valid_request(self, sample_request_payload):
        from inference.schemas import FraudPredictionRequest
        req = FraudPredictionRequest(**sample_request_payload)
        assert req.amount == 199.99
        assert req.merchant_category == "electronics"
        assert req.country == "US"

    def test_amount_normalisation(self):
        from inference.schemas import FraudPredictionRequest
        req = FraudPredictionRequest(
            transaction_id="t1", user_id="u1",
            amount=1.2345678, merchant_category="grocery", country="de"
        )
        assert req.amount == 1.23          # rounded to 2dp
        assert req.country == "DE"         # uppercased
        assert req.merchant_category == "grocery"  # lowercased

    def test_negative_amount_rejected(self):
        from inference.schemas import FraudPredictionRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            FraudPredictionRequest(
                transaction_id="t1", user_id="u1",
                amount=-10.0, merchant_category="grocery", country="US"
            )

    def test_risk_tier_labels():
        from inference.main import _risk_tier
        assert _risk_tier(0.85) == "CRITICAL"
        assert _risk_tier(0.55) == "HIGH"
        assert _risk_tier(0.35) == "MEDIUM"
        assert _risk_tier(0.10) == "LOW"


# ── Feature client encoding ────────────────────────────────────────────────────

class TestFeatureClientEncoding:

    def test_known_category(self):
        from inference.feature_client import encode_merchant_category
        assert encode_merchant_category("grocery")      == 4
        assert encode_merchant_category("electronics")  == 1
        assert encode_merchant_category("GROCERY")      == 4   # case-insensitive

    def test_unknown_category_defaults(self):
        from inference.feature_client import encode_merchant_category
        assert encode_merchant_category("casino") == 5    # default: online_retail

    def test_high_risk_country(self):
        from inference.feature_client import encode_country
        assert encode_country("XX1") == 1.0
        assert encode_country("XX3") == 1.0
        assert encode_country("US")  == 0.0
        assert encode_country("GB")  == 0.0

    def test_country_case_insensitive(self):
        from inference.feature_client import encode_country
        assert encode_country("xx1") == 1.0    # should normalise
