variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "region" {
  type        = string
  description = "GCP region for all resources"
  default     = "us-central1"
}

variable "environment" {
  type        = string
  description = "Deployment environment tag"
  default     = "prod"
}

variable "model_version" {
  type        = string
  description = "Model version tag injected into the Cloud Run container"
  default     = "1.0.0"
}
