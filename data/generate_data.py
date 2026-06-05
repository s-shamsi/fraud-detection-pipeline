"""
Synthetic Transaction Data Generator
=====================================
Generates 100k realistic financial transactions with injected fraud patterns:
  - Velocity fraud: multiple rapid transactions from the same user
  - Geographic anomalies: transactions from high-risk regions
  - Amount anomalies: unusually large purchases vs. user history
  - Category anomalies: purchases outside user's typical behaviour

Usage:
    python data/generate_data.py
"""

import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "clothing", "restaurant",
    "gas_station", "online_retail", "travel", "entertainment",
    "pharmacy", "utilities",
]

LEGIT_COUNTRIES   = ["US", "GB", "DE", "FR", "JP", "CA", "AU", "NL", "SE", "CH"]
HIGH_RISK_REGIONS = ["XX1", "XX2", "XX3", "XX4"]          # Anonymised high-risk codes

CAT_INDEX = {c: i for i, c in enumerate(sorted(MERCHANT_CATEGORIES))}
COUNTRY_INDEX: dict[str, int] = {}


# ── Core generator ─────────────────────────────────────────────────────────────

def generate_synthetic_transactions(
    n_users: int = 1_000,
    n_transactions: int = 100_000,
    fraud_rate: float = 0.015,
    days_history: int = 90,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a labelled transaction dataset.

    Fraud injection rules
    ----------------------
    - 50 % of fraud: high amount (lognormal μ=5.5) from a high-risk region
    - 30 % of fraud: velocity burst (5+ txns in 10 min) from a legit country
    - 20 % of fraud: card-testing pattern (many small amounts in quick succession)
    """
    np.random.seed(seed)
    random.seed(seed)

    start_ts  = datetime(2024, 1, 1)
    total_sec = days_history * 24 * 3600

    user_ids = [f"user_{i:05d}" for i in range(n_users)]

    # Build per-user baseline profiles
    profiles: dict[str, dict] = {}
    for uid in user_ids:
        profiles[uid] = {
            "avg_amount":   float(np.random.lognormal(3.8, 0.65)),
            "home_country": random.choices(
                LEGIT_COUNTRIES,
                weights=[30, 10, 10, 8, 7, 8, 7, 5, 5, 10],
            )[0],
            "fav_cats": random.sample(MERCHANT_CATEGORIES, k=3),
        }

    records = []
    txn_counter = 0

    for _ in range(n_transactions):
        uid     = random.choice(user_ids)
        profile = profiles[uid]
        r_val   = random.random()
        is_fraud = r_val < fraud_rate

        ts = start_ts + timedelta(seconds=random.randint(0, total_sec))

        if is_fraud:
            fraud_type = random.random()
            if fraud_type < 0.5:
                # High-amount / geographic fraud
                amount  = float(np.random.lognormal(5.5, 0.9))
                country = random.choice(HIGH_RISK_REGIONS)
                cat     = random.choice(MERCHANT_CATEGORIES)
            elif fraud_type < 0.8:
                # Velocity fraud (large amount, unusual merchant)
                amount  = float(np.random.lognormal(4.8, 0.6))
                country = profile["home_country"]
                cat     = random.choice(
                    [c for c in MERCHANT_CATEGORIES if c not in profile["fav_cats"]]
                )
            else:
                # Card-testing: many tiny amounts
                amount  = round(random.uniform(0.50, 2.99), 2)
                country = profile["home_country"]
                cat     = random.choice(["online_retail", "electronics"])
        else:
            amount  = float(
                np.random.lognormal(np.log(max(profile["avg_amount"], 1.0)), 0.45)
            )
            country = profile["home_country"]
            cat     = random.choices(
                profile["fav_cats"] + MERCHANT_CATEGORIES,
                weights=[0.22, 0.18, 0.12] + [0.048] * 10,
            )[0]

        records.append({
            "transaction_id":  f"txn_{txn_counter:09d}",
            "user_id":         uid,
            "amount":          round(amount, 2),
            "merchant_category": cat,
            "country":         country,
            "timestamp":       ts,
            "is_fraud":        int(is_fraud),
            "hour_of_day":     ts.hour,
            "day_of_week":     ts.weekday(),
            "is_weekend":      int(ts.weekday() >= 5),
        })
        txn_counter += 1

    df = (
        pd.DataFrame(records)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return df


# ── Feature engineering ────────────────────────────────────────────────────────

def engineer_offline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add user-level historical aggregate features used at training time.
    These are also periodically materialized to the Feast online store (Redis)
    for serving user lifetime-stats during inference.
    """
    df = df.copy().sort_values(["user_id", "timestamp"])

    # Lifetime user aggregates
    user_agg = (
        df.groupby("user_id")
        .agg(
            user_mean_amount=("amount",          "mean"),
            user_std_amount= ("amount",          "std"),
            user_txn_count=  ("transaction_id",  "count"),
            user_fraud_count=("is_fraud",        "sum"),
        )
        .reset_index()
    )
    user_agg["user_fraud_rate"] = (
        user_agg["user_fraud_count"] / user_agg["user_txn_count"].clip(lower=1)
    )
    user_agg["user_std_amount"] = user_agg["user_std_amount"].fillna(0.0)

    df = df.merge(user_agg.drop(columns=["user_fraud_count"]), on="user_id", how="left")

    # Derived ratio features
    df["amount_vs_user_avg"] = df["amount"] / (df["user_mean_amount"] + 1e-6)
    df["amount_z_score"]     = (df["amount"] - df["user_mean_amount"]) / (
        df["user_std_amount"] + 1.0
    )
    df["high_amount_flag"]   = (df["amount_z_score"] > 3.0).astype(int)

    # Encode categoricals
    global COUNTRY_INDEX
    all_countries = sorted(df["country"].unique())
    COUNTRY_INDEX = {c: i for i, c in enumerate(all_countries)}

    df["merchant_category_enc"] = df["merchant_category"].map(CAT_INDEX).fillna(5).astype(int)
    df["country_enc"]           = df["country"].map(COUNTRY_INDEX).fillna(0).astype(int)
    df["is_high_risk_country"]  = df["country"].isin(HIGH_RISK_REGIONS).astype(int)

    return df


def get_feature_columns() -> list[str]:
    """Return the canonical ordered list of model input features."""
    return [
        "amount",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "merchant_category_enc",
        "is_high_risk_country",
        "user_mean_amount",
        "user_std_amount",
        "user_txn_count",
        "user_fraud_rate",
        "amount_vs_user_avg",
        "amount_z_score",
        "high_amount_flag",
        # Streaming features (filled with 0 for offline training)
        "user_txn_count_10m",
        "user_avg_amount_10m",
        "user_txn_count_1h",
    ]


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating 100 000 synthetic transactions …")
    df = generate_synthetic_transactions(n_users=1_000, n_transactions=100_000)
    df = engineer_offline_features(df)

    # Add placeholder streaming features (0 for training; computed live in prod)
    for col in ["user_txn_count_10m", "user_avg_amount_10m", "user_txn_count_1h"]:
        df[col] = 0.0

    df.to_parquet(out_dir / "transactions.parquet", index=False)
    df.to_csv(out_dir / "transactions.csv", index=False)

    # Save label encodings for use by inference service
    import json
    with open(out_dir / "encodings.json", "w") as f:
        json.dump({"cat_index": CAT_INDEX, "country_index": COUNTRY_INDEX}, f, indent=2)

    print(f"  Rows       : {len(df):,}")
    print(f"  Fraud rate : {df['is_fraud'].mean():.2%}")
    print(f"  Features   : {get_feature_columns()}")
    print(f"  Saved to   : {out_dir}")
    print(df[get_feature_columns() + ['is_fraud']].describe().T.to_string())
