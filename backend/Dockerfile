# SAM-Audio Pipeline - NVIDIA DGX ARM64 Optimized
# Reddit-recommended approach: Python 3.11 + PyTorch nightly for aarch64
FROM nvcr.io/nvidia/pytorch:24.08-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies + Miniconda
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
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install micromamba (faster than conda, no TOS issues)
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/mamba
RUN micromamba create -n sam python=3.11 -c conda-forge -y
ENV PATH="/opt/mamba/envs/sam/bin:${PATH}"

# Activate sam environment for all subsequent commands
SHELL ["/bin/bash", "-c"]
ENV CONDA_DEFAULT_ENV=sam

# Install PyTorch with CUDA from conda-forge/pytorch channels (has aarch64 builds)
RUN micromamba install -n sam -c pytorch -c nvidia -c conda-forge \
    pytorch torchvision torchaudio pytorch-cuda=12.4 -y

# Verify PyTorch installation
RUN python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# Install build tools
RUN pip install --upgrade pip setuptools wheel cmake ninja pybind11

# Build torchcodec from source
ENV I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1
RUN git clone --branch v0.1.0 --depth 1 https://github.com/pytorch/torchcodec.git /tmp/torchcodec \
    && cd /tmp/torchcodec \
    && pip install --no-build-isolation . \
    && rm -rf /tmp/torchcodec

# Build decord from source (no ARM64 wheels)
RUN git clone --recursive https://github.com/dmlc/decord.git /tmp/decord \
    && cd /tmp/decord \
    && mkdir build && cd build \
    && cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc) \
    && cd ../python \
    && pip install -e . \
    && rm -rf /tmp/decord/.git

# Verify decord
RUN python -c "import decord; print('decord OK')"

# Build xformers from source (with CUDA_HOME set)
RUN pip install -v --no-build-isolation git+https://github.com/facebookresearch/xformers.git

# Install perception-models (Python 3.11 satisfies >=3.11 requirement)
RUN pip install --no-cache-dir \
    "perception-models @ git+https://github.com/facebookresearch/perception_models@unpin-deps"

# Install sam-audio and API dependencies
RUN pip install --no-cache-dir \
    "sam-audio @ git+https://github.com/facebookresearch/sam-audio.git" \
    fastapi>=0.104.0 \
    uvicorn>=0.24.0 \
    python-multipart>=0.0.6 \
    pydub>=0.25.1 \
    soundfile>=0.12.1 \
    huggingface-hub>=0.19.0

# Create working directory
WORKDIR /app

# Copy application code
COPY . .

# Create temp directories
RUN mkdir -p /tmp/sam-audio/uploads /tmp/sam-audio/results

# Expose port
EXPOSE 8011

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8011/health || exit 1

# Run the application (PATH already includes sam env)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8011"]
