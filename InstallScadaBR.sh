#!/bin/bash

set -e

echo "===== SCADA BR (Scada-LTS) INSTALL START ====="

# Update system
echo "[1/6] Updating system..."
sudo apt update

# Install Java
echo "[2/6] Installing Java..."
sudo apt install -y openjdk-11-jdk wget unzip

# Check Java
java -version

# Create directory
echo "[3/6] Creating SCADA directory..."
mkdir -p ~/scadaBR
cd ~/scadaBR

# Download Scada-LTS
echo "[4/6] Downloading Scada-LTS..."
wget -O scadalts.war https://github.com/SCADA-LTS/Scada-LTS/releases/download/v2.7.8/scadalts.war

# Open port (for cloud VM)
echo "[5/6] Opening port 8080..."
sudo apt install -y ufw
sudo ufw allow 8080 || true

# Run SCADA BR
echo "[6/6] Starting SCADA BR..."
echo "======================================"
echo "SCADA BR will start now..."
echo "Open in browser:"
echo "http://<your-vm-ip>:8080"
echo "Login: admin / admin"
echo "======================================"

java -jar scadalts.war
