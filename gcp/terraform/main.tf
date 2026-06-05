# ────────────────────────────────────────────────────────────────────────────
# Terraform – GCP Infrastructure for Real-Time Fraud Detection Pipeline
# ────────────────────────────────────────────────────────────────────────────
# Resources provisioned:
#   - Pub/Sub topic + subscription (replaces Kafka)
#   - GCS bucket (Feast offline store + artefacts)
#   - BigQuery dataset + table (offline feature store)
#   - Memorystore Redis (Feast online store)
#   - Cloud Run service (inference API)
#   - Artifact Registry (Docker images)
#   - Cloud Scheduler job (hourly materialization + drift monitoring)
#   - VPC + Serverless VPC Connector (Cloud Run → Memorystore)
#   - IAM service accounts + bindings
#
# Usage:
#   cd gcp/terraform
#   terraform init
#   terraform plan -var="project_id=your-project"
#   terraform apply -var="project_id=your-project"
# ────────────────────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Recommended: store state in GCS
  # backend "gcs" {
  #   bucket = "your-tfstate-bucket"
  #   prefix = "fraud-detection/state"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ── APIs to enable ─────────────────────────────────────────────────────────────

locals {
  required_apis = [
    "pubsub.googleapis.com",
    "redis.googleapis.com",
    "run.googleapis.com",
    "bigquery.googleapis.com",
    "storage.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudscheduler.googleapis.com",
    "vpcaccess.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
  ]
}

resource "google_project_service" "apis" {
  for_each                   = toset(local.required_apis)
  service                    = each.value
  disable_on_destroy         = false
  disable_dependent_services = false
}

# ── Service accounts ───────────────────────────────────────────────────────────

resource "google_service_account" "inference_sa" {
  account_id   = "fraud-inference-sa"
  display_name = "Fraud Inference API Service Account"
  description  = "Used by the Cloud Run inference service"
}

resource "google_service_account" "pipeline_sa" {
  account_id   = "fraud-pipeline-sa"
  display_name = "Fraud Pipeline Service Account"
  description  = "Used by Dataflow, materialization, and monitoring jobs"
}

# Inference SA: read from GCS (artefacts), write to BigQuery (prediction logs)
resource "google_project_iam_member" "inference_gcs_reader" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.inference_sa.email}"
}

resource "google_project_iam_member" "inference_bq_writer" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.inference_sa.email}"
}

resource "google_project_iam_member" "inference_monitoring_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.inference_sa.email}"
}

# Pipeline SA: full access to GCS, BigQuery, Pub/Sub, Redis
resource "google_project_iam_member" "pipeline_gcs_admin" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.pipeline_sa.email}"
}

resource "google_project_iam_member" "pipeline_bq_admin" {
  project = var.project_id
  role    = "roles/bigquery.admin"
  member  = "serviceAccount:${google_service_account.pipeline_sa.email}"
}

resource "google_project_iam_member" "pipeline_pubsub_admin" {
  project = var.project_id
  role    = "roles/pubsub.admin"
  member  = "serviceAccount:${google_service_account.pipeline_sa.email}"
}

# ── VPC for Memorystore ────────────────────────────────────────────────────────

resource "google_compute_network" "fraud_vpc" {
  name                    = "fraud-detection-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "fraud_subnet" {
  name          = "fraud-detection-subnet"
  ip_cidr_range = "10.10.0.0/24"
  region        = var.region
  network       = google_compute_network.fraud_vpc.id
}

resource "google_vpc_access_connector" "connector" {
  name          = "fraud-vpc-connector"
  region        = var.region
  network       = google_compute_network.fraud_vpc.name
  ip_cidr_range = "10.10.1.0/28"
  min_instances = 2
  max_instances = 10
  depends_on    = [google_project_service.apis]
}

# ── Pub/Sub (replaces Kafka) ───────────────────────────────────────────────────

resource "google_pubsub_topic" "transactions" {
  name = "transactions"
  labels = {
    env     = var.environment
    service = "fraud-detection"
  }
  message_retention_duration = "86600s"   # 24 h
}

resource "google_pubsub_subscription" "transactions_aggregator" {
  name  = "transactions-aggregator-sub"
  topic = google_pubsub_topic.transactions.name

  ack_deadline_seconds       = 60
  retain_acked_messages      = true
  message_retention_duration = "86600s"

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }

  labels = {
    env     = var.environment
    service = "fraud-detection"
  }
}

resource "google_pubsub_subscription" "transactions_dead_letter" {
  name  = "transactions-dlq-sub"
  topic = google_pubsub_topic.transactions.name

  ack_deadline_seconds = 600
}

# ── GCS bucket (Feast offline store + model artefacts) ─────────────────────────

resource "google_storage_bucket" "feast_store" {
  name                        = "${var.project_id}-fraud-feast-store"
  location                    = var.region
  force_destroy               = false
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action { type = "Delete" }
    condition { age = 90 }        # Purge raw data after 90 days
  }

  labels = {
    env     = var.environment
    service = "feast-offline-store"
  }
}

resource "google_storage_bucket_object" "feast_registry_prefix" {
  name    = "feast-registry/"
  bucket  = google_storage_bucket.feast_store.name
  content = " "
}

resource "google_storage_bucket_object" "artifacts_prefix" {
  name    = "artifacts/"
  bucket  = google_storage_bucket.feast_store.name
  content = " "
}

# ── BigQuery (prediction logs + offline feature store) ─────────────────────────

resource "google_bigquery_dataset" "fraud_features" {
  dataset_id                 = "fraud_features"
  friendly_name              = "Fraud Detection Features"
  description                = "Feature store offline data and prediction logs"
  location                   = var.region
  delete_contents_on_destroy = false

  labels = {
    env     = var.environment
    service = "fraud-detection"
  }
}

resource "google_bigquery_table" "prediction_logs" {
  dataset_id          = google_bigquery_dataset.fraud_features.dataset_id
  table_id            = "prediction_logs"
  deletion_protection = false

  time_partitioning {
    type  = "DAY"
    field = "logged_at"
  }

  clustering = ["user_id", "is_fraud"]

  schema = jsonencode([
    { name = "transaction_id",    type = "STRING",    mode = "REQUIRED" },
    { name = "user_id",           type = "STRING",    mode = "REQUIRED" },
    { name = "amount",            type = "FLOAT64",   mode = "REQUIRED" },
    { name = "merchant_category", type = "STRING",    mode = "NULLABLE" },
    { name = "country",           type = "STRING",    mode = "NULLABLE" },
    { name = "timestamp",         type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "fraud_probability", type = "FLOAT64",   mode = "REQUIRED" },
    { name = "is_fraud",          type = "BOOL",      mode = "REQUIRED" },
    { name = "risk_tier",         type = "STRING",    mode = "NULLABLE" },
    { name = "latency_ms",        type = "FLOAT64",   mode = "NULLABLE" },
    { name = "model_version",     type = "STRING",    mode = "NULLABLE" },
    { name = "logged_at",         type = "TIMESTAMP", mode = "REQUIRED" },
  ])
}

resource "google_bigquery_table" "user_transaction_features" {
  dataset_id          = google_bigquery_dataset.fraud_features.dataset_id
  table_id            = "user_transaction_features"
  deletion_protection = false

  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }

  clustering = ["user_id"]

  schema = jsonencode([
    { name = "user_id",            type = "STRING",  mode = "REQUIRED" },
    { name = "timestamp",          type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "user_mean_amount",   type = "FLOAT64", mode = "NULLABLE" },
    { name = "user_std_amount",    type = "FLOAT64", mode = "NULLABLE" },
    { name = "user_txn_count",     type = "INT64",   mode = "NULLABLE" },
    { name = "user_fraud_rate",    type = "FLOAT64", mode = "NULLABLE" },
    { name = "amount_vs_user_avg", type = "FLOAT64", mode = "NULLABLE" },
    { name = "amount_z_score",     type = "FLOAT64", mode = "NULLABLE" },
    { name = "high_amount_flag",   type = "INT64",   mode = "NULLABLE" },
  ])
}

# ── Memorystore Redis (Feast online store) ─────────────────────────────────────

resource "google_redis_instance" "feast_online_store" {
  name               = "fraud-feast-online-store"
  tier               = "STANDARD_HA"      # HA with automatic failover
  memory_size_gb     = 4
  region             = var.region
  authorized_network = google_compute_network.fraud_vpc.id
  redis_version      = "REDIS_7_0"
  display_name       = "Feast Online Store – Fraud Detection"

  redis_configs = {
    maxmemory-policy = "allkeys-lru"
    maxmemory        = "3500mb"
  }

  labels = {
    env     = var.environment
    service = "feast-online-store"
  }
}

# ── Artifact Registry (Docker images) ─────────────────────────────────────────

resource "google_artifact_registry_repository" "fraud_images" {
  location      = var.region
  repository_id = "fraud-detection"
  description   = "Docker images for fraud detection services"
  format        = "DOCKER"

  labels = {
    env = var.environment
  }
}

# ── Cloud Run (inference service) ─────────────────────────────────────────────

resource "google_cloud_run_v2_service" "inference_api" {
  name     = "fraud-inference-api"
  location = var.region

  template {
    service_account = google_service_account.inference_sa.email

    scaling {
      min_instance_count = 1    # Keep warm to avoid cold starts
      max_instance_count = 20
    }

    vpc_access {
      connector = google_vpc_access_connector.connector.id
      egress    = "PRIVATE_RANGES_ONLY"
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/fraud-detection/inference:latest"

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
        cpu_idle          = false   # Always allocate CPU (needed for ONNX)
        startup_cpu_boost = true
      }

      env {
        name  = "REDIS_HOST"
        value = google_redis_instance.feast_online_store.host
      }
      env {
        name  = "REDIS_PORT"
        value = "6379"
      }
      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }
      env {
        name  = "MODEL_VERSION"
        value = var.model_version
      }
      env {
        name  = "ARTIFACT_DIR"
        value = "/app/artifacts"
      }
      env {
        name  = "ORT_THREADS"
        value = "2"
      }

      # Cloud Run health check
      liveness_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 5
        period_seconds        = 30
        timeout_seconds       = 5
        failure_threshold     = 3
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 5
        period_seconds        = 5
        timeout_seconds       = 3
        failure_threshold     = 10
      }

      ports {
        container_port = 8080
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [
    google_redis_instance.feast_online_store,
    google_vpc_access_connector.connector,
  ]
}

# Make inference API publicly accessible (remove for internal-only deployments)
resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.inference_api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ── Cloud Scheduler (hourly materialization + monitoring) ──────────────────────

resource "google_cloud_scheduler_job" "materialize_features" {
  name        = "feast-materialize-incremental"
  description = "Materialise Feast features from BigQuery to Redis hourly"
  schedule    = "0 * * * *"    # Every hour
  time_zone   = "UTC"
  region      = var.region

  http_target {
    uri         = "${google_cloud_run_v2_service.inference_api.uri}/admin/materialize"
    http_method = "POST"
    oidc_token {
      service_account_email = google_service_account.pipeline_sa.email
    }
  }
}

resource "google_cloud_scheduler_job" "drift_monitor" {
  name        = "drift-monitoring-hourly"
  description = "Run Evidently drift checks on recent predictions"
  schedule    = "30 * * * *"   # 30 min past every hour
  time_zone   = "UTC"
  region      = var.region

  http_target {
    uri         = "${google_cloud_run_v2_service.inference_api.uri}/admin/drift-check"
    http_method = "POST"
    oidc_token {
      service_account_email = google_service_account.pipeline_sa.email
    }
  }
}
