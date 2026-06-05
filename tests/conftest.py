"""
pytest conftest – shared fixtures for the test suite.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def sample_transactions():
    """Generate a small synthetic dataset once for all tests."""
    from data.generate_data import generate_synthetic_transactions, engineer_offline_features

    df = generate_synthetic_transactions(
        n_users=50, n_transactions=500, seed=99
    )
    return engineer_offline_features(df)


@pytest.fixture(scope="session")
def feature_columns():
    from data.generate_data import get_feature_columns
    return get_feature_columns()


@pytest.fixture
def sample_request_payload():
    return {
        "transaction_id":    "test_txn_001",
        "user_id":           "user_00001",
        "amount":            199.99,
        "merchant_category": "electronics",
        "country":           "US",
    }


@pytest.fixture
def high_risk_request_payload():
    return {
        "transaction_id":    "test_txn_999",
        "user_id":           "user_00001",
        "amount":            3499.99,
        "merchant_category": "electronics",
        "country":           "XX1",   # High-risk region
    }
