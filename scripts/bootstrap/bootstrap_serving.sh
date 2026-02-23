#!/bin/bash
set -e

VENV="/workspace/.venv_serving"

if [ ! -d "$VENV" ]; then

  python3 -m venv "$VENV"

  source "$VENV/bin/activate"

  pip install --upgrade pip

  pip install -r env/serving_requirements.txt

  deactivate
fi

echo "Serving environment ready."