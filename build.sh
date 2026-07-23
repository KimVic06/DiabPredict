#!/usr/bin/env bash

set -o errexit

pip install -r requirements.txt

apt-get update
apt-get install -y tesseract-ocr
