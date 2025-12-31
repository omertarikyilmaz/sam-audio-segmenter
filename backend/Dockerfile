# SAM-Audio Pipeline - Soundfile-based Audio Saving (No torchcodec build needed)
# Uses minimal mocks for import compatibility, soundfile for actual audio I/O
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

# 1. Install system dependencies + Python 3.11
RUN apt-get update && apt-get install -y \
    curl git wget build-essential software-properties-common \
    libsndfile1 libgl1 libglib2.0-0 ffmpeg \
    ninja-build cmake pkg-config \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# 3. Install PyTorch nightly cu129 (Blackwell sm_120 support)
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129

# 4. Create MINIMAL mocks for import compatibility only
# Actual audio I/O uses soundfile in processor.py
RUN SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    mkdir -p $SITE_PACKAGES/torchcodec/decoders $SITE_PACKAGES/torchcodec/encoders && \
    echo 'class AudioDecoder: pass' > $SITE_PACKAGES/torchcodec/decoders/__init__.py && \
    echo 'class VideoDecoder: pass' >> $SITE_PACKAGES/torchcodec/decoders/__init__.py && \
    echo 'class AudioEncoder: pass' > $SITE_PACKAGES/torchcodec/encoders/__init__.py && \
    echo '__version__="0.0.0"; from .decoders import *; from .encoders import *' > $SITE_PACKAGES/torchcodec/__init__.py && \
    mkdir -p $SITE_PACKAGES/torchcodec-0.0.0.dist-info && \
    echo "Name: torchcodec\nVersion: 0.0.0" > $SITE_PACKAGES/torchcodec-0.0.0.dist-info/METADATA && \
    mkdir -p $SITE_PACKAGES/xformers/ops/fmha && \
    echo 'class AttentionBias: pass' > $SITE_PACKAGES/xformers/ops/__init__.py && \
    echo 'class BlockDiagonalMask: pass' > $SITE_PACKAGES/xformers/ops/fmha/__init__.py && \
    echo '__version__="0.0.0"' > $SITE_PACKAGES/xformers/__init__.py && \
    mkdir -p $SITE_PACKAGES/xformers-0.0.0.dist-info && \
    echo "Name: xformers\nVersion: 0.0.0" > $SITE_PACKAGES/xformers-0.0.0.dist-info/METADATA && \
    mkdir -p $SITE_PACKAGES/decord && \
    echo 'class VideoReader: pass\nclass AudioReader: pass\ndef cpu(i=0): return 0\ndef gpu(i=0): return 1' > $SITE_PACKAGES/decord/__init__.py && \
    mkdir -p $SITE_PACKAGES/decord-0.0.0.dist-info && \
    echo "Name: decord\nVersion: 0.0.0" > $SITE_PACKAGES/decord-0.0.0.dist-info/METADATA

WORKDIR /app

# 5. Install core dependencies (soundfile is critical for audio saving)
RUN pip install "numpy>=1.24.0,<2.0" protobuf==3.20.3 scipy scikit-learn einops pydub torchdiffeq safetensors soundfile

# 6. Install transformers
RUN pip install "transformers>=4.54.0"

# 7. Install git dependencies one by one
RUN pip install git+https://github.com/facebookresearch/dacvae.git
RUN pip install git+https://github.com/facebookresearch/ImageBind.git
RUN pip install git+https://github.com/lematt1991/CLAP.git
RUN pip install git+https://github.com/facebookresearch/perception_models@unpin-deps
RUN pip install audiobox_aesthetics
RUN pip install git+https://github.com/facebookresearch/sam-audio#egg=sam_audio

# 8. API dependencies
RUN pip install fastapi>=0.104.0 uvicorn>=0.24.0 python-multipart>=0.0.6 huggingface-hub>=0.19.0

# 9. Copy application code
COPY . .
RUN mkdir -p /tmp/sam-audio/uploads /tmp/sam-audio/results
EXPOSE 8011

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8011"]
