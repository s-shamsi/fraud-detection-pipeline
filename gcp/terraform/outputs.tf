output "inference_api_url" {
  description = "Cloud Run inference API public URL"
  value       = google_cloud_run_v2_service.inference_api.uri
}

output "redis_host" {
  description = "GCP Memorystore Redis host (private IP)"
  value       = google_redis_instance.feast_online_store.host
  sensitive   = true
}

output "gcs_bucket" {
  description = "GCS bucket name for Feast offline store and artefacts"
  value       = google_storage_bucket.feast_store.name
}

output "artifact_registry_repo" {
  description = "Full Artifact Registry path for Docker image pushes"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/fraud-detection"
}

output "pubsub_topic" {
  description = "Pub/Sub transactions topic name"
  value       = google_pubsub_topic.transactions.name
}

output "bigquery_dataset" {
  description = "BigQuery dataset for features and prediction logs"
  value       = google_bigquery_dataset.fraud_features.dataset_id
}

output "service_account_inference" {
  description = "Inference API service account email"
  value       = google_service_account.inference_sa.email
}
