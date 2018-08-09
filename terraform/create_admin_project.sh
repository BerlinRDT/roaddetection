#!/bin/bash

# As per - https://github.com/steinim/gcp-terraform-workshop

gcloud projects create ${TF_PROJECT} \
  --set-as-default

gcloud alpha billing projects link ${TF_PROJECT} \
  --billing-account ${TF_VAR_billing_account}

gcloud iam service-accounts create terraform \
  --display-name "Terraform admin account"

gcloud iam service-accounts keys create ${TF_CREDS} \
  --iam-account terraform@${TF_PROJECT}.iam.gserviceaccount.com

gcloud projects add-iam-policy-binding ${TF_PROJECT} \
  --member serviceAccount:terraform@${TF_PROJECT}.iam.gserviceaccount.com \
  --role roles/viewer

gcloud projects add-iam-policy-binding ${TF_PROJECT} \
  --member serviceAccount:terraform@${TF_PROJECT}.iam.gserviceaccount.com \
  --role roles/storage.admin

gcloud projects add-iam-policy-binding ${TF_PROJECT} \
  --member serviceAccount:terraform@${TF_PROJECT}.iam.gserviceaccount.com \
  --role roles/compute.admin

gcloud services enable cloudresourcemanager.googleapis.com
gcloud services enable cloudbilling.googleapis.com
gcloud services enable iam.googleapis.com
gcloud services enable compute.googleapis.com


gsutil mb -p ${TF_PROJECT} gs://${TF_PROJECT}

cat > backend.tf <<EOF
terraform {
 backend "gcs" {
   bucket  = "${TF_PROJECT}"
   prefix    = "/terraform.tfstate"
   project = "${TF_PROJECT}"
 }
}
EOF