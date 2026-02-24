#!/usr/bin/env bash
# exit on error
set -o errexit

# Upgrade pip
pip install --upgrade pip

# Install packages with pre-built wheels only (no compilation)
pip install --only-binary=:all: pandas numpy || pip install pandas numpy

# Install rest of requirements
pip install -r requirements.txt
