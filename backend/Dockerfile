# SAM-Audio Pipeline - CUDA-enabled Docker Image (DGX/Devel Optimized)
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV DECORD_GPU=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    build-essential \
    cmake \
    g++ \
    ninja-build \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavdevice-dev \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default and install pip properly for 3.12
# We use get-pip.py because the system python3-pip depends on 'distutils' which was removed in 3.12
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && python3 -m pip install --upgrade pip setuptools wheel

# Create working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
# 1. Install torch, torchaudio, and torchcodec with explicit CUDA 12.6 index
# Aligned for modern DGX servers (CUDA 12.8/13.0)
RUN python3 -m pip install --no-cache-dir \
    torch>=2.5.1 torchaudio>=2.5.1 torchcodec \
    --extra-index-url https://download.pytorch.org/whl/cu126

# 2. Build decord from source (Git)
RUN git clone --recursive https://github.com/dmlc/decord.git /tmp/decord \
    && cd /tmp/decord \
    && mkdir build && cd build \
    && cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc) \
    && cd ../python \
    && python3 setup.py install \
    && rm -rf /tmp/decord

# 3. Install the rest of the requirements
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create temp directories
RUN mkdir -p /tmp/sam-audio/uploads /tmp/sam-audio/results

# Expose port
EXPOSE 8011

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8011/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8011"]
