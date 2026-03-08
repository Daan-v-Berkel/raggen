#!/usr/bin/env bash
set -e

VENV_DIR=".venv-clean"

if [[ -n "$VIRTUAL_ENV" ]]; then
  echo "Detected active virtual environment: $VIRTUAL_ENV"
  echo "Deactivating..."
  deactivate || true
fi

echo "Cleaning old venv..."
rm -rf $VENV_DIR

echo "Creating new venv..."
python3 -m venv $VENV_DIR

echo "Activating venv..."
source $VENV_DIR/bin/activate

echo "Installing requirements..."
pip install -r requirements.txt

echo "Installing project in editable mode..."
pip install -e .

echo "Running tests..."
pytest -q

echo "Testing CLI..."
rag -h

echo "✅ Environment check completed successfully"
