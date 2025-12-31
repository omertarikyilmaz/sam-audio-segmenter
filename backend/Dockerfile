# SAM-Audio Pipeline - NVIDIA DGX ARM64 Optimized
# Strategy: Build decord from source FIRST, then install perception-models
FROM nvcr.io/nvidia/pytorch:24.08-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ninja-build \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavdevice-dev \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel cmake

# Create working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Verify base image Python and PyTorch
RUN python3 -c "import torch; print('Python:', __import__('sys').version); print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# Build torchcodec v0.1.0 from source
ENV I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1
RUN git clone --branch v0.1.0 --depth 1 https://github.com/pytorch/torchcodec.git /tmp/torchcodec \
    && cd /tmp/torchcodec \
    && python3 -m pip install --no-build-isolation . \
    && rm -rf /tmp/torchcodec

# BUILD DECORD FROM SOURCE FIRST - No ARM64 wheels exist!
# This must be done BEFORE perception-models which depends on it
RUN git clone --recursive https://github.com/dmlc/decord.git /tmp/decord \
    && cd /tmp/decord \
    && mkdir build && cd build \
    && cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc) \
    && cd ../python \
    && python3 -m pip install . \
    && rm -rf /tmp/decord

# Verify decord installation
RUN python3 -c "import decord; print('decord installed successfully')"

# Now install perception-models (decord is already available)
RUN python3 -m pip install --no-cache-dir --ignore-requires-python \
    "perception-models @ git+https://github.com/facebookresearch/perception_models@unpin-deps"

# Install sam-audio and other requirements
RUN python3 -m pip install --no-cache-dir \
    "sam-audio @ git+https://github.com/facebookresearch/sam-audio.git" \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    python-multipart>=0.0.6 \
    pydub>=0.25.1 \
    soundfile>=0.12.1 \
    huggingface-hub>=0.19.0

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
