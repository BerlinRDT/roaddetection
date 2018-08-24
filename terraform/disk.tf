resource "google_compute_disk" "default" {
  name  = "disk1"
  type  = "pd-ssd"
  project = "${var.project_name}"
  size=64
  image = "ubuntu-gpu-1535055336"
  zone  = "${data.google_compute_zones.available.names[0]}"
  labels {
    environment = "dev"
  }

  lifecycle {
    prevent_destroy = true
  }
}