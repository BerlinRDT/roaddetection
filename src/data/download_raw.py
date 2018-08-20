# -*- coding: utf-8 -*-
from google.cloud import storage
from google.cloud.exceptions import NotFound
import os.path

local_images_dir = 'data/raw/images/'


def download(blob):
    source_blob_name = blob.name
    file_name = source_blob_name.rsplit('/', 1)[1]
    destination_file_name = local_images_dir + file_name
    if not os.path.isfile(destination_file_name):
        blob.download_to_filename(destination_file_name)
        print('Image {} downloaded to {}.'.format(
            source_blob_name,
            destination_file_name))
    else:
        print('Image already {} exists. Skipping download'.
              format(destination_file_name))


def main():
    client = storage.Client()

    if not os.path.exists(local_images_dir):
        os.makedirs(local_images_dir)

    try:
        bucket = client.get_bucket('satellite_images')
        blobs = bucket.list_blobs()
        [download(blob) for blob in blobs if "Visual.tif" in blob.name]
    except NotFound:
        print('Sorry, that bucket does not exist!')


if __name__ == '__main__':
    main()
