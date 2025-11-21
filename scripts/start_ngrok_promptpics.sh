#!/usr/bin/env bash
set -euo pipefail

echo "Starting ngrok tunnel 'promptpics' (app.promptpics.ai → localhost:7080)..."
ngrok start promptpics
