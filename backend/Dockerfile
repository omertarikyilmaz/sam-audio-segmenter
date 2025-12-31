# SAM-Audio Pipeline - NVIDIA DGX ARM64
# Strategy: Use base image's Python 3.10 + PyTorch (proven CUDA working)
# Build problematic deps from source, bypass Python version checks
FROM nvcr.io/nvidia/pytorch:24.08-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

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

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel cmake ninja pybind11

# Verify base image Python and PyTorch (already working on ARM64 with CUDA)
RUN python3 -c "import torch; print('Python:', __import__('sys').version); print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# Create working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Build torchcodec from source
ENV I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1
RUN git clone --branch v0.1.0 --depth 1 https://github.com/pytorch/torchcodec.git /tmp/torchcodec \
    && cd /tmp/torchcodec \
    && python3 -m pip install --no-build-isolation . \
    && rm -rf /tmp/torchcodec

# Build decord from source (no ARM64 wheels exist)
RUN git clone --recursive https://github.com/dmlc/decord.git /tmp/decord \
    && cd /tmp/decord \
    && mkdir build && cd build \
    && cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc) \
    && cd ../python \
    && python3 -m pip install . \
    && rm -rf /tmp/decord

# Verify decord
RUN python3 -c "import decord; print('decord OK')"

# Build xformers from source (CUDA_HOME is set above)
RUN python3 -m pip install --no-cache-dir numpy \
    && python3 -m pip install -v --no-build-isolation git+https://github.com/facebookresearch/xformers.git

# Install perception-models with --ignore-requires-python (code works on 3.10)
RUN python3 -m pip install --no-cache-dir --ignore-requires-python \
    "perception-models @ git+https://github.com/facebookresearch/perception_models@unpin-deps"

# Install sam-audio and API dependencies
RUN python3 -m pip install --no-cache-dir --ignore-requires-python \
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
