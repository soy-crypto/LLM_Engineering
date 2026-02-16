#!/usr/bin/env bash

set -e

echo "========================================"
echo "🚀 LLM Inference Host Setup"
echo "========================================"

# -------------------------------------------------
# 0️⃣ Prevent running inside Docker
# -------------------------------------------------
if [ -f /.dockerenv ]; then
  echo "❌ This script must be run on the HOST machine."
  echo "You are inside a Docker container."
  exit 1
fi

# -------------------------------------------------
# 1️⃣ Check Ubuntu
# -------------------------------------------------
if ! grep -qi ubuntu /etc/os-release; then
  echo "❌ Only Ubuntu 22.04 / 24.04 supported."
  exit 1
fi

echo "✅ Ubuntu detected."

# -------------------------------------------------
# 2️⃣ Determine privilege mode
# -------------------------------------------------
if [ "$EUID" -eq 0 ]; then
  SUDO=""
  echo "ℹ Running as root."
else
  if command -v sudo &> /dev/null; then
    SUDO="sudo"
    echo "ℹ Using sudo."
  else
    echo "❌ sudo not found. Run as root or install sudo."
    exit 1
  fi
fi

# -------------------------------------------------
# 3️⃣ Install NVIDIA Driver (if missing)
# -------------------------------------------------
if command -v nvidia-smi &> /dev/null; then
  echo "✅ NVIDIA driver already installed."
else
  echo "🔧 Installing NVIDIA driver..."
  $SUDO apt update
  $SUDO ubuntu-drivers autoinstall
  echo ""
  echo "⚠️ Driver installed. Please reboot:"
  echo "    sudo reboot"
  exit 0
fi

echo "🔍 Verifying nvidia-smi..."
nvidia-smi || { echo "❌ Driver not working."; exit 1; }

# -------------------------------------------------
# 4️⃣ Install Docker (if missing)
# -------------------------------------------------
if command -v docker &> /dev/null; then
  echo "✅ Docker already installed."
else
  echo "🔧 Installing Docker..."

  $SUDO apt remove -y docker docker-engine docker.io containerd runc || true
  $SUDO apt update
  $SUDO apt install -y ca-certificates curl gnupg lsb-release

  $SUDO mkdir -p /etc/apt/keyrings

  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    $SUDO gpg --dearmor -o /etc/apt/keyrings/docker.gpg

  echo \
  "deb [arch=$(dpkg --print-architecture) \
  signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  $SUDO tee /etc/apt/sources.list.d/docker.list > /dev/null

  $SUDO apt update
  $SUDO apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

  $SUDO systemctl enable docker
  $SUDO systemctl start docker
fi

echo "🔍 Docker version:"
docker --version

# -------------------------------------------------
# 5️⃣ Install NVIDIA Container Toolkit
# -------------------------------------------------
if dpkg -l | grep -q nvidia-container-toolkit; then
  echo "✅ NVIDIA Container Toolkit already installed."
else
  echo "🔧 Installing NVIDIA Container Toolkit..."

  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    $SUDO gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

  curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

  $SUDO apt update
  $SUDO apt install -y nvidia-container-toolkit

  $SUDO nvidia-ctk runtime configure --runtime=docker
  $SUDO systemctl restart docker
fi

# -------------------------------------------------
# 6️⃣ Verify GPU inside Docker
# -------------------------------------------------
echo "🔍 Testing GPU inside Docker..."

docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi

echo ""
echo "========================================"
echo "✅ Host environment setup complete."
echo "========================================"
echo ""
echo "Next steps:"
echo "  docker login nvcr.io"
echo "  docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3"
echo "  docker run --gpus all -it -v \$PWD:/workspace nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3"
