variable "project_name" {}
variable "region" {}

provider "google" {
 region = "${var.region}"
}

data "google_compute_zones" "available" {}

resource "google_compute_instance" "default" {
 zone = "${data.google_compute_zones.available.names[0]}"
 project = "${var.project_name}"
 name = "road-detection"
 machine_type = "n1-standard-1"
 boot_disk {
   auto_delete = false
   source = "${google_compute_disk.default.name}"
 }

 network_interface {
   network = "default"
   access_config {
   }
 }

 guest_accelerator {
  type = "nvidia-tesla-k80"
  count = 1
}
 
 scheduling {
  on_host_maintenance="TERMINATE"
 } 
}

output "instance_id" {
 value = "${google_compute_instance.default.self_link}"
}