#!/usr/bin/env bash

set -e

echo "========================================"
echo "🚀 LLM Inference Environment Bootstrap"
echo "========================================"

# -----------------------------
# 0️⃣ Check OS
# -----------------------------
if ! grep -qi ubuntu /etc/os-release; then
  echo "❌ This script supports Ubuntu only."
  exit 1
fi

echo "✅ Ubuntu detected."

# -----------------------------
# 1️⃣ Install NVIDIA Driver
# -----------------------------
if command -v nvidia-smi &> /dev/null; then
  echo "✅ NVIDIA driver already installed."
else
  echo "🔧 Installing NVIDIA driver..."
  sudo apt update
  sudo ubuntu-drivers autoinstall
  echo "⚠️ Reboot required after driver install."
  echo "Run: sudo reboot"
  exit 0
fi

echo "🔍 Verifying nvidia-smi..."
nvidia-smi || { echo "❌ Driver not working."; exit 1; }

# -----------------------------
# 2️⃣ Install Docker
# -----------------------------
if command -v docker &> /dev/null; then
  echo "✅ Docker already installed."
else
  echo "🔧 Installing Docker..."

  sudo apt remove -y docker docker-engine docker.io containerd runc || true
  sudo apt update
  sudo apt install -y ca-certificates curl gnupg lsb-release

  sudo mkdir -p /etc/apt/keyrings

  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

  echo \
  "deb [arch=$(dpkg --print-architecture) \
  signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

  sudo apt update
  sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

  sudo systemctl enable docker
  sudo systemctl start docker
fi

echo "🔍 Docker version:"
docker --version

# -----------------------------
# 3️⃣ Install NVIDIA Container Toolkit
# -----------------------------
if dpkg -l | grep -q nvidia-container-toolkit; then
  echo "✅ NVIDIA Container Toolkit already installed."
else
  echo "🔧 Installing NVIDIA Container Toolkit..."

  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

  curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

  sudo apt update
  sudo apt install -y nvidia-container-toolkit

  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi

# -----------------------------
# 4️⃣ Test GPU in Docker
# -----------------------------
echo "🔍 Testing GPU inside Docker..."

docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi

echo "========================================"
echo "✅ Environment setup complete."
echo "========================================"

echo ""
echo "Next Steps:"
echo "1️⃣ docker login nvcr.io"
echo "2️⃣ docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3"
echo "3️⃣ Run TensorRT-LLM container"
