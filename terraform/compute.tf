variable "project_name" {}
variable "region" {}
variable "public_key_path" {}
variable "private_key_path" {}

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
      nat_ip = "${google_compute_address.default.address}"
    }
  }

  guest_accelerator {
    type = "nvidia-tesla-k80"
    count = 1
  }

  metadata {
    ssh-keys = "root:${file("${var.public_key_path}")}"
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
  }

  provisioner "remote-exec" {
    inline = [
      "sudo mkdir -p /etc/jupyter",
      "mkdir -p /home/ubuntu/.ssh",
      "chown -R ubuntu /home/ubuntu/.ssh"
    ]

    connection {
      type = "ssh"
      user = "root",
      private_key = "${file("${var.private_key_path}")}"
      agent = false
    }
  }

  provisioner "file" {
    source = "keys/"
    destination = "/home/ubuntu/.ssh"
    connection {
      type = "ssh"
      user = "root",
      private_key = "${file("${var.private_key_path}")}"
      agent = false
    }
  }

  provisioner "file" {
    source = "jupyter/jupyter_notebook_config.py"
    destination = "/etc/jupyter/jupyter_notebook_config.py"
    connection {
      type = "ssh"
      user = "root",
      private_key = "${file("${var.private_key_path}")}"
      agent = false
    }
  }

  provisioner "remote-exec" {
    inline = [
      "sudo su - ubuntu -c 'nohup jupyter-notebook --no-browser > /dev/null 2> /dev/null &'"
    ]
    connection {
      type = "ssh"
      user = "root",
      private_key = "${file("${var.private_key_path}")}"
      agent = false
    }
  }
}

output "instance_id" {
  value = "${google_compute_instance.default.self_link}"
}