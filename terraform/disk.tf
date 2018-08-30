data "google_compute_image" "my_image" {
  name    = "ubuntu-gpu-1535059965"
  project = "berlin-rdt"
}


resource "google_compute_disk" "default" {
  name  = "disk1"
  type  = "pd-ssd"
  project = "${var.project_name}"
  size=64
  image = "${data.google_compute_image.my_image.self_link}"
  zone  = "${data.google_compute_zones.available.names[0]}"
  labels {
    environment = "dev"
  }

  lifecycle {
    prevent_destroy = true
  }
}