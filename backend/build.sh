#!/usr/bin/env bash
# exit on error
set -o errexit

# Generate enviornment.py from env vars (file is gitignored)
mkdir -p enviornment
cat > enviornment/enviornment.py <<EOF
import os
RESEND_API_KEY = os.environ.get('RESEND_API_KEY', '')
NEON_URL = os.environ.get('NEON_URL', '')
NEON_USERNAME = os.environ.get('NEON_USERNAME', '')
NEON_PASSWORD = os.environ.get('NEON_PASSWORD', '')
HF_TOKEN = os.environ.get('HF_TOKEN', '')
EOF

# Upgrade pip
pip install --upgrade pip

# Install packages with pre-built wheels only (no compilation)
pip install --only-binary=:all: pandas numpy || pip install pandas numpy

# Install rest of requirements
pip install -r requirements.txt
