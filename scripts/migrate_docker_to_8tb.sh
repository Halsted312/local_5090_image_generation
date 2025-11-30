#!/bin/bash
# Docker Migration Script - Run as root!
# Moves /var/lib/docker to /mnt/8tb_data/docker

set -e

echo "============================================"
echo "Docker Migration to 8TB Drive"
echo "============================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: This script must be run as root"
    echo "Run: sudo bash scripts/migrate_docker_to_8tb.sh"
    exit 1
fi

TARGET="/mnt/8tb_data/docker"
SOURCE="/var/lib/docker"
BACKUP="/var/lib/docker.bak"

echo ""
echo "Source: $SOURCE"
echo "Target: $TARGET"
echo ""

# Check target drive
if [ ! -d "/mnt/8tb_data" ]; then
    echo "ERROR: /mnt/8tb_data does not exist"
    exit 1
fi

echo "Step 1/6: Stopping Docker..."
systemctl stop docker docker.socket containerd 2>/dev/null || true
sleep 2

echo "Step 2/6: Creating daemon.json..."
mkdir -p /etc/docker
cat > /etc/docker/daemon.json << 'EOF'
{
  "data-root": "/mnt/8tb_data/docker"
}
EOF
echo "  Created /etc/docker/daemon.json"

echo "Step 3/6: Creating target directory..."
mkdir -p "$TARGET"

echo "Step 4/6: Moving data (this may take 5-10 minutes)..."
echo "  Starting rsync..."
rsync -aP --info=progress2 "$SOURCE/" "$TARGET/"

echo ""
echo "Step 5/6: Backing up old directory..."
if [ -d "$BACKUP" ]; then
    rm -rf "$BACKUP"
fi
mv "$SOURCE" "$BACKUP"

echo "Step 6/6: Starting Docker..."
systemctl start docker

echo ""
echo "============================================"
echo "Migration Complete!"
echo "============================================"
echo ""
echo "Docker data-root is now: $TARGET"
echo "Old data backed up to: $BACKUP"
echo ""
echo "Verify with: docker info | grep 'Docker Root Dir'"
echo ""
echo "To delete backup (after verifying everything works):"
echo "  rm -rf $BACKUP"
