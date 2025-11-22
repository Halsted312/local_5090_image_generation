#!/usr/bin/env bash
set -euo pipefail

echo "Starting ngrok tunnel 'promptpics' (app.promptpics.ai → localhost:7999)..."

/usr/local/bin/ngrok start promptpics
