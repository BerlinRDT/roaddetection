resource "google_compute_address" "default" {
  name = "gpu-server-ip",
  project = "${var.project_name}"
  region = "${var.region}"
  address_type = "EXTERNAL"

  lifecycle {
    prevent_destroy = false
  }
}


resource "google_compute_firewall" "default" {
  name = "allow-http"
  network = "default"

  allow {
    protocol = "icmp"
  }

  allow {
    protocol = "tcp"
    ports = ["80","8080","8888"]
  }

  source_ranges = ["0.0.0.0/0"]

  source_tags = ["web"]
}