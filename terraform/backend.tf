terraform {
 backend "gcs" {
   bucket  = "berlin-rdt"
   prefix    = "/terraform.tfstate"
   project = "berlin-rdt"
 }
}
