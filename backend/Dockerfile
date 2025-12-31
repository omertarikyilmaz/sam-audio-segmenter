# SAM-Audio Pipeline - NVIDIA DGX Blackwell (GB10) FULL GPU SUPPORT
# Uses CUDA 12.9 base + PyTorch 2.7 nightly with sm_120 (Blackwell) support
FROM nvcr.io/nvidia/cuda:12.9.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
# Ensure HuggingFace uses the mounted volume for model cache
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers

# 1. Install system dependencies + Python 3.11
RUN apt-get update && apt-get install -y \
    curl git wget build-essential software-properties-common \
    libsndfile1 libgl1 libglib2.0-0 ffmpeg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# 3. Install PyTorch 2.7+ nightly with CUDA 12.8+ (has sm_120 Blackwell support)
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 4. Verify PyTorch + CUDA
RUN python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# 5. Create mock packages script
WORKDIR /app
COPY create_mocks.py .
RUN python3 create_mocks.py

# 6. Install SAM-Audio dependencies
RUN pip install "numpy>=1.24.0,<2.0" protobuf==3.20.3 scipy scikit-learn "transformers>=4.54.0" einops pydub torchdiffeq safetensors && \
    pip install git+https://github.com/facebookresearch/dacvae.git \
    git+https://github.com/facebookresearch/ImageBind.git \
    git+https://github.com/lematt1991/CLAP.git \
    git+https://github.com/facebookresearch/perception_models@unpin-deps && \
    pip install audiobox_aesthetics git+https://github.com/facebookresearch/sam-audio#egg=sam_audio && \
    pip install fastapi>=0.104.0 uvicorn>=0.24.0 python-multipart>=0.0.6 huggingface-hub>=0.19.0

# 7. Copy application code
COPY . .
RUN mkdir -p /tmp/sam-audio/uploads /tmp/sam-audio/results
EXPOSE 8011

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8011"]
