# SAM-Audio Pipeline - NVIDIA DGX ARM64 Optimized
# Solution for perception-models (needs Python 3.11+) + decord (no ARM64 wheels) conflict
# Strategy: Use Python 3.11 via deadsnakes + build decord from source
FROM nvcr.io/nvidia/pytorch:24.08-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV DECORD_GPU=0

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3.11-distutils \
    ninja-build \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavdevice-dev \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11 and make it default
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip and install build tools
RUN python3.11 -m pip install --upgrade pip setuptools wheel pybind11 numpy

# Create working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Build decord from source (no ARM64 wheels available for Python 3.11)
# Must be built before sam-audio which has perception-models dependency
RUN git clone --recursive https://github.com/dmlc/decord.git /tmp/decord \
    && cd /tmp/decord \
    && mkdir build && cd build \
    && cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc) \
    && cd ../python \
    && python3.11 -m pip install . \
    && rm -rf /tmp/decord

# Verify decord installation
RUN python3.11 -c "import decord; print('decord installed successfully')"

# Install PyTorch ecosystem for Python 3.11 FIRST (torchcodec depends on it)
RUN python3.11 -m pip install --no-cache-dir \
    torch==2.5.0 \
    torchaudio==2.5.0 \
    torchvision==0.20.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Verify PyTorch CUDA
RUN python3.11 -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# Build torchcodec v0.1.0 from source (AFTER PyTorch is installed)
ENV I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1
RUN git clone --branch v0.1.0 --depth 1 https://github.com/pytorch/torchcodec.git /tmp/torchcodec \
    && cd /tmp/torchcodec \
    && python3.11 -m pip install --no-build-isolation . \
    && rm -rf /tmp/torchcodec

# Install sam-audio and API requirements
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create temp directories
RUN mkdir -p /tmp/sam-audio/uploads /tmp/sam-audio/results

# Expose port
EXPOSE 8011

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8011/health || exit 1

# Run the application with Python 3.11
CMD ["python3.11", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8011"]
