#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# deploy_cloud_run.sh
# Builds the inference Docker image, pushes it to Artifact Registry,
# and deploys it to Cloud Run in a single command.
#
# Prerequisites:
#   gcloud auth login
#   gcloud auth configure-docker ${REGION}-docker.pkg.dev
#
# Usage:
#   ./gcp/deploy_cloud_run.sh
#   GCP_PROJECT_ID=my-project REGION=europe-west1 ./gcp/deploy_cloud_run.sh
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration (override via env vars) ──────────────────────────────────────
GCP_PROJECT_ID="${GCP_PROJECT_ID:-your-gcp-project}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-fraud-inference-api}"
REPO="${REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/fraud-detection"
MODEL_VERSION="${MODEL_VERSION:-1.0.0}"
IMAGE_TAG="${REPO}/inference:${MODEL_VERSION}"
MEMORY="${MEMORY:-2Gi}"
CPU="${CPU:-2}"
MIN_INSTANCES="${MIN_INSTANCES:-1}"
MAX_INSTANCES="${MAX_INSTANCES:-20}"

# ── Helpers ────────────────────────────────────────────────────────────────────
info()  { echo -e "\033[0;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[0;32m[ OK ]\033[0m  $*"; }
err()   { echo -e "\033[0;31m[ERR ]\033[0m  $*" >&2; exit 1; }

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ── Preflight checks ───────────────────────────────────────────────────────────
info "Checking prerequisites …"
command -v gcloud  >/dev/null 2>&1 || err "gcloud CLI not found. Install: https://cloud.google.com/sdk/install"
command -v docker  >/dev/null 2>&1 || err "docker not found."

[[ -f "${ROOT_DIR}/artifacts/fraud_detector.onnx" ]] \
  || err "ONNX model not found. Run: python model/train.py && python model/convert_to_onnx.py"

# ── Ensure Artifact Registry repo exists ──────────────────────────────────────
info "Ensuring Artifact Registry repository exists …"
gcloud artifacts repositories describe fraud-detection \
  --location="${REGION}" \
  --project="${GCP_PROJECT_ID}" >/dev/null 2>&1 \
|| gcloud artifacts repositories create fraud-detection \
     --repository-format=docker \
     --location="${REGION}" \
     --project="${GCP_PROJECT_ID}" \
     --description="Fraud detection Docker images"

# ── Copy ONNX model into inference build context ───────────────────────────────
info "Copying ONNX artefacts into build context …"
mkdir -p "${ROOT_DIR}/inference_build/artifacts"
cp "${ROOT_DIR}/artifacts/fraud_detector.onnx"   "${ROOT_DIR}/inference_build/artifacts/"
cp "${ROOT_DIR}/artifacts/model_metadata.json"   "${ROOT_DIR}/inference_build/artifacts/"

# ── Docker build ───────────────────────────────────────────────────────────────
info "Building Docker image ${IMAGE_TAG} …"
docker build \
  --file "${ROOT_DIR}/inference/Dockerfile" \
  --tag  "${IMAGE_TAG}" \
  --build-arg "MODEL_VERSION=${MODEL_VERSION}" \
  --platform linux/amd64 \
  "${ROOT_DIR}"

ok "Docker build complete"

# ── Push to Artifact Registry ──────────────────────────────────────────────────
info "Pushing image to Artifact Registry …"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
docker push "${IMAGE_TAG}"
ok "Image pushed: ${IMAGE_TAG}"

# ── Retrieve Terraform outputs ─────────────────────────────────────────────────
TF_DIR="${ROOT_DIR}/gcp/terraform"
if command -v terraform >/dev/null 2>&1 && [[ -f "${TF_DIR}/terraform.tfstate" ]]; then
  REDIS_HOST=$(terraform -chdir="${TF_DIR}" output -raw redis_host 2>/dev/null || echo "")
  VPC_CONNECTOR=$(terraform -chdir="${TF_DIR}" output -raw vpc_connector 2>/dev/null || echo "")
else
  REDIS_HOST="${REDIS_HOST:-}"
  VPC_CONNECTOR="${VPC_CONNECTOR:-}"
fi

# ── Deploy to Cloud Run ────────────────────────────────────────────────────────
info "Deploying to Cloud Run (${REGION}) …"

DEPLOY_ARGS=(
  run deploy "${SERVICE_NAME}"
  --image         "${IMAGE_TAG}"
  --region        "${REGION}"
  --project       "${GCP_PROJECT_ID}"
  --platform      managed
  --memory        "${MEMORY}"
  --cpu           "${CPU}"
  --min-instances "${MIN_INSTANCES}"
  --max-instances "${MAX_INSTANCES}"
  --port          8080
  --set-env-vars  "MODEL_VERSION=${MODEL_VERSION},GCP_PROJECT_ID=${GCP_PROJECT_ID}"
  --service-account "fraud-inference-sa@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
  --allow-unauthenticated
  --cpu-always-allocated
  --execution-environment gen2
)

[[ -n "${REDIS_HOST}" ]] && DEPLOY_ARGS+=(--set-env-vars "REDIS_HOST=${REDIS_HOST}")
[[ -n "${VPC_CONNECTOR}" ]] && DEPLOY_ARGS+=(
  --vpc-connector "${VPC_CONNECTOR}"
  --vpc-egress    private-ranges-only
)

gcloud "${DEPLOY_ARGS[@]}"

# ── Print service URL ──────────────────────────────────────────────────────────
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --region="${REGION}" \
  --project="${GCP_PROJECT_ID}" \
  --format="value(status.url)")

ok "Deployment complete!"
echo ""
echo "  Service URL  : ${SERVICE_URL}"
echo "  Health check : ${SERVICE_URL}/health"
echo "  API docs     : ${SERVICE_URL}/docs"
echo "  Metrics      : ${SERVICE_URL}/metrics"
echo ""
echo "  Test with:"
echo "    curl -X POST ${SERVICE_URL}/predict \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"transaction_id\":\"txn_001\",\"user_id\":\"user_00042\",\"amount\":349.99,\"merchant_category\":\"electronics\",\"country\":\"US\"}'"

# ── Cleanup ────────────────────────────────────────────────────────────────────
rm -rf "${ROOT_DIR}/inference_build"
