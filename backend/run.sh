#!/usr/bin/env bash
# Start the LimbScan server (HTTPS on :8443). Open the printed URL on your phone (same Wi-Fi).
cd "$(dirname "$0")"
exec .venv/bin/python -m server.app
