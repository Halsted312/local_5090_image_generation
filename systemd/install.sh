#!/bin/bash
# Install flexy-face systemd services
# Run with: sudo ./install.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing flexy-face systemd services..."

# Copy service files
cp "$SCRIPT_DIR/flexy-face.service" /etc/systemd/system/
cp "$SCRIPT_DIR/flexy-face-watchdog.service" /etc/systemd/system/
cp "$SCRIPT_DIR/flexy-face-watchdog.timer" /etc/systemd/system/

# Reload systemd
systemctl daemon-reload

# Enable services
systemctl enable flexy-face.service
systemctl enable flexy-face-watchdog.timer

# Start services
systemctl start flexy-face.service
systemctl start flexy-face-watchdog.timer

echo ""
echo "✓ Installation complete!"
echo ""
echo "Status:"
systemctl status flexy-face.service --no-pager || true
echo ""
echo "Timers:"
systemctl list-timers | grep flexy || true
