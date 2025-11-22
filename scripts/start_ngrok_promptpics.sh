#!/usr/bin/env bash
set -euo pipefail
echo "Starting ngrok HTTP tunnel (app.promptpics.ai -> localhost:7999)..."
NGROK_BIN="$(command -v ngrok)"
if [ -z "$NGROK_BIN" ]; then
  echo "ngrok binary not found in PATH"
  exit 1
fi
"$NGROK_BIN" http 7999 --hostname=app.promptpics.ai
