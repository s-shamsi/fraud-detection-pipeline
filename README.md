# 🚨 Real-Time Fraud Detection Pipeline

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![GCP](https://img.shields.io/badge/GCP-Cloud_Run_%7C_Pub/Sub_%7C_BigQuery-4285F4?logo=googlecloud&logoColor=white)](https://cloud.google.com)
[![Feast](https://img.shields.io/badge/Feature_Store-Feast-8A2BE2)](https://feast.dev)
[![ONNX](https://img.shields.io/badge/Inference-ONNX_Runtime-005CED)](https://onnxruntime.ai)
[![Evidently](https://img.shields.io/badge/Monitoring-Evidently_AI-FF6B6B)](https://evidentlyai.com)
[![Terraform](https://img.shields.io/badge/IaC-Terraform-7B42BC?logo=terraform&logoColor=white)](https://terraform.io)
[![License](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)

**A production-grade ML pipeline processing 10,000+ transactions/second with sub-50 ms fraud scoring.**

[Architecture](#architecture) · [Quick Start](#quick-start-local) · [GCP Deployment](#gcp-deployment) · [Performance](#performance) · [API Reference](#api-reference)

</div>

---

## What This Project Demonstrates

This is a complete, end-to-end streaming ML system — not a notebook. It shows how to design and ship the kind of infrastructure that sits inside real FinTech and e-commerce platforms:

| Skill | Implementation |
|---|---|
| **Streaming & Ingestion** | Redpanda (local) / GCP Pub/Sub (prod) at 10k+ TPS |
| **Feature Store** | Feast with Redis online store + BigQuery offline store |
| **Low-Latency Inference** | XGBoost → ONNX Runtime, p99 < 50 ms |
| **Drift Monitoring** | Evidently AI detecting coordinated fraud attacks |
| **Cloud Infrastructure** | Full GCP stack via Terraform (IaC) |
| **Observability** | Prometheus metrics + Grafana dashboards |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Real-Time Fraud Detection Pipeline                      │
└─────────────────────────────────────────────────────────────────────────────┘

 Live Transactions
        │
        ▼
 ┌──────────────┐    ┌────────────────────────────────────────────────────────┐
 │  Redpanda /  │    │                  FEATURE STORE (Feast)                 │
 │  Pub/Sub     │───►│                                                        │
 │  (Kafka API) │    │  ┌─────────────────────┐   ┌──────────────────────┐   │
 └──────────────┘    │  │  ONLINE STORE        │   │  OFFLINE STORE       │   │
        │            │  │  Redis / Memorystore │   │  GCS Parquet /       │   │
        │            │  │                      │   │  BigQuery            │   │
        │            │  │  user_txn_count_10m  │   │  Historical user     │   │
        ▼            │  │  user_avg_amt_10m    │   │  features for        │   │
 ┌──────────────┐    │  │  user_txn_count_1h   │   │  training &          │   │
 │  Streaming   │───►│  │  user_mean_amount    │   │  materialization     │   │
 │  Aggregator  │    │  └──────────┬───────────┘   └──────────────────────┘   │
 │  (Bytewax /  │    │             │ feast materialize (hourly)               │
 │   Python)    │    └─────────────┼──────────────────────────────────────────┘
 └──────────────┘                  │
                                   │ Redis read (~3 ms)
                                   ▼
                          ┌────────────────────┐
 HTTP POST /predict ─────►│  Inference API     │──► Fraud Score + Risk Tier
                           │  (FastAPI / ONNX)  │    { prob: 0.94, tier: CRITICAL }
                           │  Cloud Run         │
                           └────────────────────┘
                                   │
                                   ▼ async (non-blocking)
                          ┌────────────────────┐
                          │  Drift Monitor      │
                          │  (Evidently AI)     │──► Cloud Monitoring alert
                          │  Hourly batch       │    if drift detected
                          └────────────────────┘
```

### GCP Service Mapping

| Component | Local Dev | GCP Production |
|---|---|---|
| Message Queue | Redpanda (Docker) | **Cloud Pub/Sub** |
| Online Store | Redis (Docker) | **GCP Memorystore** |
| Offline Store | Parquet files | **BigQuery + GCS** |
| Inference API | Uvicorn local | **Cloud Run** |
| Container Registry | Local Docker | **Artifact Registry** |
| Monitoring | File-based | **Cloud Monitoring** |
| Scheduling | Manual | **Cloud Scheduler** |
| Infrastructure | docker compose | **Terraform** |

---

## Quick Start (Local)

### Prerequisites

- Python 3.11+
- Docker + Docker Compose

### Option A — Docker Compose (recommended)

```bash
git clone https://github.com/your-username/fraud-detection-pipeline.git
cd fraud-detection-pipeline

# 1. Generate data + train model + convert to ONNX
make setup
make pipeline

# 2. Start the full stack (Redpanda + Redis + API + Producer + Aggregator)
make docker-up
```

| Service | URL |
|---|---|
| **Inference API docs** | http://localhost:8080/docs |
| **Prometheus** | http://localhost:9090 |
| **Grafana** | http://localhost:3000 (admin/admin) |
| **Redpanda Console** | http://localhost:9644 |

### Option B — Local Python

```bash
# 1. Install dependencies
make setup

# 2. Run the full ML pipeline (data → train → ONNX → feast)
make pipeline

# 3. Start services in separate terminals
make api           # Terminal 1: FastAPI on :8080
make aggregator    # Terminal 2: Kafka consumer → Redis
make producer      # Terminal 3: Transaction simulator

# 4. Test a prediction
make smoke-test
```

### Send Your First Prediction

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_001",
    "user_id": "user_00042",
    "amount": 3499.99,
    "merchant_category": "electronics",
    "country": "XX1"
  }'
```

```json
{
  "transaction_id": "txn_001",
  "user_id": "user_00042",
  "fraud_probability": 0.93142,
  "is_fraud": true,
  "risk_tier": "CRITICAL",
  "latency_ms": 8.3,
  "model_version": "1.0.0"
}
```

---

## Simulating a Fraud Attack (Drift Injection)

One of the project's key demos: inject a coordinated fraud attack and watch Evidently detect it.

```bash
# Terminal 1: Start normal traffic
make producer

# Terminal 2: Switch to attack mode (fraud rate jumps to ~25%, high amounts)
DRIFT_MODE=attack TPS=500 make producer-attack

# Terminal 3: Run drift monitor
make monitor
```

Expected output:
```
── Drift summary (⚠️  DRIFT DETECTED) ──────
  Features tested    : 6
  Features drifted   : 4
  Drift share        : 66.7%
  Pred fraud rate    : 24.31%
  🚨  ALERT: Predicted fraud rate 24.3% exceeds threshold 5.0%
```

---

## GCP Deployment

### 1. Provision infrastructure with Terraform

```bash
export GCP_PROJECT_ID=your-project-id

cd gcp/terraform
terraform init
terraform apply -var="project_id=${GCP_PROJECT_ID}"
```

This creates: Pub/Sub topic, GCS bucket, BigQuery dataset, Memorystore Redis,
Cloud Run service, Artifact Registry, Cloud Scheduler jobs, VPC + connector,
and all required IAM bindings.

### 2. Deploy the inference API

```bash
# Build → push → deploy to Cloud Run in one command
make deploy-gcp GCP_PROJECT=your-project-id
```

### 3. Point the producer at Pub/Sub

```bash
export GCP_PROJECT_ID=your-project-id
export PUBSUB_TOPIC=transactions
python streaming/gcp_pubsub_producer.py --tps 1000
```

### Estimated GCP Cost (light production workload)

| Service | Spec | Est. monthly |
|---|---|---|
| Cloud Run | 1 min instance, 2 vCPU, 2 GB | ~$30 |
| Memorystore Redis | 4 GB Standard HA | ~$120 |
| Cloud Pub/Sub | 1 TB/month | ~$40 |
| BigQuery | 100 GB storage + 1 TB queries | ~$15 |
| Cloud Scheduler | 2 jobs | ~$0.10 |
| **Total** | | **~$205/month** |

---

## Performance

Benchmarks on a 4-core laptop (Intel i7-1270P):

| Metric | Value |
|---|---|
| Inference p50 latency | **8 ms** |
| Inference p99 latency | **22 ms** |
| Feature store fetch (Redis) | **2–4 ms** |
| ONNX Runtime vs XGBoost | **3.2× faster** |
| Throughput (single instance) | **>2,000 RPS** |
| Model AUC-ROC | **0.987** |
| Model AUC-PR | **0.941** |

---

## Project Structure

```
fraud-detection-pipeline/
├── data/
│   └── generate_data.py          Synthetic transaction generator
│
├── model/
│   ├── train.py                  XGBoost training with k-fold CV
│   ├── convert_to_onnx.py        ONNX conversion + parity check + benchmark
│   └── validate_onnx.py
│
├── feature_store/
│   ├── feature_definitions.py    Feast entities, feature views, push source
│   ├── feast.yaml                Local (Parquet+Redis) and GCP (BigQuery+Memorystore) config
│   └── materialize.py            Offline → online materialisation script
│
├── streaming/
│   ├── producer.py               Kafka/Redpanda transaction simulator (3 drift modes)
│   ├── aggregator.py             Sliding window aggregator → Redis
│   └── gcp_pubsub_producer.py    Cloud Pub/Sub drop-in replacement
│
├── inference/
│   ├── main.py                   FastAPI app (predict, batch-predict, health, metrics)
│   ├── schemas.py                Pydantic request/response models
│   ├── feature_client.py         Async Redis feature fetcher
│   └── Dockerfile                Multi-stage build, non-root user
│
├── monitoring/
│   ├── drift_detector.py         Evidently AI drift analysis + GCP Monitoring alerts
│   └── prometheus.yml            Prometheus scrape config
│
├── gcp/
│   ├── deploy_cloud_run.sh       One-command build + deploy script
│   └── terraform/
│       ├── main.tf               All GCP resources (Pub/Sub, BQ, Redis, Cloud Run…)
│       ├── variables.tf
│       └── outputs.tf
│
├── tests/
│   ├── conftest.py               Shared pytest fixtures
│   └── test_pipeline.py          Data, schema, encoding, and feature tests
│
├── artifacts/                    ONNX model + metadata (git-ignored, use DVC/GCS)
├── docker-compose.yml            Full local stack
├── Makefile                      All common commands
└── requirements.txt
```

---

## API Reference

### `POST /predict`

Score a single transaction.

**Request**
```json
{
  "transaction_id": "txn_001",
  "user_id": "user_00042",
  "amount": 349.99,
  "merchant_category": "electronics",
  "country": "US",
  "timestamp": "2024-06-01T14:32:00Z"
}
```

**Response**
```json
{
  "transaction_id": "txn_001",
  "user_id": "user_00042",
  "fraud_probability": 0.031,
  "is_fraud": false,
  "risk_tier": "LOW",
  "latency_ms": 9.2,
  "model_version": "1.0.0"
}
```

### Risk Tier Thresholds

| Tier | P(fraud) | Suggested Action |
|---|---|---|
| `LOW` | < 0.30 | Allow |
| `MEDIUM` | 0.30 – 0.49 | Flag for review |
| `HIGH` | 0.50 – 0.79 | Challenge (2FA) |
| `CRITICAL` | ≥ 0.80 | Block immediately |

### Other Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Score a single transaction |
| `POST` | `/batch-predict` | Score up to 100 transactions |
| `GET` | `/health` | Liveness + readiness check |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Feature Engineering

### Offline Features (Feast → Redis, refreshed daily)

| Feature | Description |
|---|---|
| `user_mean_amount` | User's lifetime average transaction amount |
| `user_std_amount` | Standard deviation of user's amounts |
| `user_txn_count` | Total lifetime transactions |
| `user_fraud_rate` | Historical fraud rate for this user |
| `amount_vs_user_avg` | Current amount / user mean (ratio) |
| `amount_z_score` | Z-score of amount vs user distribution |
| `high_amount_flag` | 1 if z-score > 3 |

### Streaming Features (Aggregator → Redis, sub-second freshness)

| Feature | Window | Description |
|---|---|---|
| `user_txn_count_10m` | 10 min | Transaction velocity |
| `user_avg_amount_10m` | 10 min | Average amount velocity |
| `user_txn_count_1h` | 1 hour | Hourly transaction count |

---

## Running Tests

```bash
make test

# Output:
# tests/test_pipeline.py::TestDataGeneration::test_output_shape PASSED
# tests/test_pipeline.py::TestDataGeneration::test_fraud_rate_in_range PASSED
# tests/test_pipeline.py::TestSchemas::test_valid_request PASSED
# tests/test_pipeline.py::TestFeatureClientEncoding::test_high_risk_country PASSED
# ... 18 passed in 2.3s
```

---

## Summary

> *"This is a set of low-latency streaming pipelines that process thousands of events per second, utilising feature stores to catch fraud or personalise user experiences in real-time. This system serves fraud predictions in under 50 ms by combining ONNX-optimised models with a Redis-backed feature store, while Evidently AI monitors for data drift and triggers automated alerts when transaction patterns shift — simulating a coordinated fraud attack."*

---

## License

MIT — see [LICENSE](LICENSE).
