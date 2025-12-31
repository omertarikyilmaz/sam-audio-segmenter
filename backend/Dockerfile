# SAM-Audio Pipeline - NVIDIA DGX ARM64 (aarch64) Optimized
# Final Build: User's sequence + ARM64 specific fixes (decord build + xformers stub)
FROM nvcr.io/nvidia/pytorch:24.08-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install basic system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    build-essential \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install micromamba (fast, aarch64 support)
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/mamba

# Create Python 3.11 environment + Install FFmpeg (as per user's success list)
RUN micromamba create -n sam_audio python=3.11 -c conda-forge "ffmpeg<8" -y
ENV PATH="/opt/mamba/envs/sam_audio/bin:${PATH}"

# Activation for subsequent RUNs
SHELL ["/bin/bash", "-c"]
ENV CONDA_DEFAULT_ENV=sam_audio

# 1. Install PyTorch with CUDA (Heavy download - Cache prioritized)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. Install core dependencies (User's list)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "numpy>=1.24.0,<2.0" protobuf==3.20.3 scipy scikit-learn "transformers>=4.54.0" einops pydub torchdiffeq safetensors

# 3. BUILD DECORD FROM SOURCE
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ ! -d "/tmp/decord" ]; then git clone --recursive https://github.com/dmlc/decord.git /tmp/decord; fi && \
    cd /tmp/decord && \
    mkdir -p build && cd build && \
    cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    cd ../python && \
    pip install .

# 4. BYPASS XFORMERS
RUN mkdir -p /opt/mamba/envs/sam_audio/lib/python3.11/site-packages/xformers/ops \
    && echo "# xformers stub" > /opt/mamba/envs/sam_audio/lib/python3.11/site-packages/xformers/__init__.py \
    && echo "class MemoryEfficientAttentionOp: pass" > /opt/mamba/envs/sam_audio/lib/python3.11/site-packages/xformers/ops/__init__.py \
    && echo "memory_efficient_attention = None" >> /opt/mamba/envs/sam_audio/lib/python3.11/site-packages/xformers/ops/__init__.py

# 5. Install Git dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install git+https://github.com/facebookresearch/dacvae.git \
    git+https://github.com/facebookresearch/ImageBind.git \
    git+https://github.com/lematt1991/CLAP.git \
    git+https://github.com/facebookresearch/perception_models@unpin-deps

# 6. Install final packages
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install audiobox_aesthetics \
    git+https://github.com/facebookresearch/sam-audio#egg=sam_audio

# 7. Install API requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install fastapi>=0.104.0 uvicorn>=0.24.0 python-multipart>=0.0.6 huggingface-hub>=0.19.0

# 8. Torchcodec
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torchcodec || ( \
    git clone --branch v0.1.0 --depth 1 https://github.com/pytorch/torchcodec.git /tmp/torchcodec && \
    cd /tmp/torchcodec && \
    pip install --no-build-isolation . && \
    rm -rf /tmp/torchcodec )

# Create application directory
WORKDIR /app
COPY . .

# Create temp directories
RUN mkdir -p /tmp/sam-audio/uploads /tmp/sam-audio/results

# Expose port
EXPOSE 8011

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8011"]
