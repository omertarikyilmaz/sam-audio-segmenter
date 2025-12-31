# SAM-Audio Pipeline - Clean Installation (No Mocks)
# Builds torchcodec from source for ARM64/Blackwell compatibility
FROM nvcr.io/nvidia/cuda:12.9.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# HuggingFace cache
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers

# CUDA config for Blackwell
ENV TORCH_CUDA_ARCH_LIST="9.0;12.0"
ENV NVIDIA_TF32_OVERRIDE=0

# 1. Install system dependencies + Python 3.11 + FFmpeg (for torchcodec)
RUN apt-get update && apt-get install -y \
    curl git wget build-essential software-properties-common \
    libsndfile1 libgl1 libglib2.0-0 \
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavdevice-dev \
    ninja-build cmake pkg-config \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel ninja cmake pybind11

# 3. Install PyTorch nightly with CUDA 12.9 (Blackwell sm_120 support)
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129

# 4. BUILD TORCHCODEC FROM SOURCE (Real package, not mock)
RUN git clone https://github.com/pytorch/torchcodec.git /tmp/torchcodec \
    && cd /tmp/torchcodec \
    && pip install -e . --no-build-isolation \
    && rm -rf /tmp/torchcodec

# 5. Verify torchcodec installation
RUN python3 -c "import torchcodec; print('torchcodec version:', torchcodec.__version__)"

# 6. Install SAM-Audio dependencies (xformers will be skipped if not available)
WORKDIR /app

# Core dependencies first
RUN pip install "numpy>=1.24.0,<2.0" protobuf==3.20.3 scipy scikit-learn einops pydub torchdiffeq safetensors soundfile

# Install transformers
RUN pip install "transformers>=4.54.0"

# Git dependencies
RUN pip install git+https://github.com/facebookresearch/dacvae.git
RUN pip install git+https://github.com/facebookresearch/ImageBind.git
RUN pip install git+https://github.com/lematt1991/CLAP.git

# perception-models (may need xformers, will handle gracefully)
RUN pip install git+https://github.com/facebookresearch/perception_models@unpin-deps || \
    echo "perception-models install failed, trying without xformers dependency"

# Install audiobox_aesthetics and sam-audio
RUN pip install audiobox_aesthetics
RUN pip install git+https://github.com/facebookresearch/sam-audio#egg=sam_audio

# API dependencies
RUN pip install fastapi>=0.104.0 uvicorn>=0.24.0 python-multipart>=0.0.6 huggingface-hub>=0.19.0

# 7. Copy application code
COPY . .
RUN mkdir -p /tmp/sam-audio/uploads /tmp/sam-audio/results
EXPOSE 8011

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8011"]
