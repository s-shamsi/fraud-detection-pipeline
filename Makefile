# ────────────────────────────────────────────────────────────────────────────
# Makefile – Real-Time Fraud Detection Pipeline
# ────────────────────────────────────────────────────────────────────────────
# Usage:
#   make setup            install Python dependencies
#   make pipeline         run the full local pipeline end-to-end
#   make docker-up        start all services via Docker Compose
#   make deploy-gcp       build + deploy to Cloud Run
# ────────────────────────────────────────────────────────────────────────────

.PHONY: all setup pipeline \
        generate-data train convert-onnx \
        feast-apply feast-materialize \
        api stream-start monitor \
        docker-up docker-down docker-logs \
        test lint clean deploy-gcp

PYTHON      ?= python3
VENV_DIR    ?= .venv
PIP         := $(VENV_DIR)/bin/pip
PY          := $(VENV_DIR)/bin/python
UVICORN     := $(VENV_DIR)/bin/uvicorn
GCP_PROJECT ?= $(shell echo $$GCP_PROJECT_ID)
REGION      ?= us-central1

# ── Virtualenv setup ───────────────────────────────────────────────────────────

$(VENV_DIR)/bin/activate:
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip

setup: $(VENV_DIR)/bin/activate
	$(PIP) install -r requirements.txt
	@echo "✅  Dependencies installed into $(VENV_DIR)"

# ── Full local pipeline (runs all steps in order) ─────────────────────────────

pipeline: generate-data train convert-onnx feast-apply feast-materialize
	@echo ""
	@echo "✅  Full pipeline complete. Run 'make api' to start the inference server."

# ── Step 1: Synthetic data ─────────────────────────────────────────────────────

generate-data:
	@echo "──  Generating synthetic transaction data …"
	$(PY) data/generate_data.py

# ── Step 2: Train XGBoost model ───────────────────────────────────────────────

train:
	@echo "──  Training XGBoost fraud detection model …"
	$(PY) model/train.py

# ── Step 3: Convert to ONNX ───────────────────────────────────────────────────

convert-onnx:
	@echo "──  Converting XGBoost → ONNX and running parity check …"
	$(PY) model/convert_to_onnx.py

# ── Step 4: Feast feature store ───────────────────────────────────────────────

feast-apply:
	@echo "──  Applying Feast feature store definitions …"
	cd feature_store && $(VENV_DIR)/../$(VENV_DIR)/bin/feast apply

feast-materialize:
	@echo "──  Materialising features from offline → online store (Redis) …"
	$(PY) feature_store/materialize.py --full

feast-materialize-incremental:
	$(PY) feature_store/materialize.py

# ── Step 5: Inference API ─────────────────────────────────────────────────────

api:
	@echo "──  Starting FastAPI inference service on http://localhost:8080 …"
	@echo "    API docs: http://localhost:8080/docs"
	$(UVICORN) inference.main:app \
		--host 0.0.0.0 \
		--port 8080 \
		--reload \
		--log-level info

# ── Step 6: Stream producer + aggregator ──────────────────────────────────────

producer:
	@echo "──  Starting transaction stream producer (100 TPS, normal mode) …"
	$(PY) streaming/producer.py

producer-attack:
	@echo "──  Starting transaction stream producer (ATTACK mode – drift injection) …"
	DRIFT_MODE=attack TPS=500 $(PY) streaming/producer.py

aggregator:
	@echo "──  Starting streaming aggregator (Kafka → Redis sliding windows) …"
	$(PY) streaming/aggregator.py

stream-start:
	@echo "──  Starting producer + aggregator in background …"
	$(PY) streaming/producer.py &
	$(PY) streaming/aggregator.py &
	@echo "   PIDs written to .pids"
	@echo "   Run 'make stream-stop' to stop them"

stream-stop:
	-pkill -f "streaming/producer.py"
	-pkill -f "streaming/aggregator.py"

# ── Step 7: Drift monitoring ───────────────────────────────────────────────────

monitor:
	$(PY) monitoring/drift_detector.py --report-html

monitor-continuous:
	@while true; do \
		echo "[$(date -u +%H:%M:%S)] Running drift check …"; \
		$(PY) monitoring/drift_detector.py --window-hours 1; \
		sleep 3600; \
	done

# ── Docker Compose ─────────────────────────────────────────────────────────────

docker-up:
	@echo "──  Starting full stack (Redpanda + Redis + API + Producer + Aggregator) …"
	docker compose up -d
	@echo ""
	@echo "  Inference API : http://localhost:8080/docs"
	@echo "  Redpanda admin: http://localhost:9644"
	@echo "  Prometheus    : http://localhost:9090"
	@echo "  Grafana       : http://localhost:3000  (admin/admin)"

docker-down:
	docker compose down -v

docker-logs:
	docker compose logs -f inference-api

docker-build:
	docker compose build --no-cache

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	$(VENV_DIR)/bin/pytest tests/ -v --tb=short --cov=inference --cov=model --cov=streaming

test-fast:
	$(VENV_DIR)/bin/pytest tests/ -v --tb=short -m "not slow"

# ── Linting ───────────────────────────────────────────────────────────────────

lint:
	$(VENV_DIR)/bin/ruff check . --fix
	$(VENV_DIR)/bin/mypy inference/ model/ streaming/ --ignore-missing-imports

# ── GCP Deployment ────────────────────────────────────────────────────────────

terraform-init:
	cd gcp/terraform && terraform init

terraform-plan:
	cd gcp/terraform && terraform plan -var="project_id=$(GCP_PROJECT)" -var="region=$(REGION)"

terraform-apply:
	cd gcp/terraform && terraform apply -var="project_id=$(GCP_PROJECT)" -var="region=$(REGION)" -auto-approve

terraform-destroy:
	cd gcp/terraform && terraform destroy -var="project_id=$(GCP_PROJECT)" -var="region=$(REGION)"

deploy-gcp: convert-onnx
	@echo "──  Deploying inference API to Cloud Run …"
	GCP_PROJECT_ID=$(GCP_PROJECT) REGION=$(REGION) bash gcp/deploy_cloud_run.sh

# ── Quick API test ────────────────────────────────────────────────────────────

smoke-test:
	@echo "──  Sending a test transaction to the local API …"
	curl -s -X POST http://localhost:8080/predict \
	  -H "Content-Type: application/json" \
	  -d '{"transaction_id":"test_001","user_id":"user_00042","amount":349.99,"merchant_category":"electronics","country":"US"}' \
	  | python3 -m json.tool

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache 2>/dev/null || true
	@echo "✅  Cleaned"

clean-data:
	rm -rf data/processed data/predictions data/monitoring
	@echo "✅  Data directories cleaned (run 'make generate-data' to regenerate)"

clean-artifacts:
	rm -f artifacts/*.onnx artifacts/*.json artifacts/*.pkl artifacts/*.csv
	@echo "✅  Model artifacts cleaned (run 'make pipeline' to rebuild)"
