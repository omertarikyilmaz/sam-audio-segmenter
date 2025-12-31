# SAM-Audio Pipeline - NVIDIA DGX ARM64
# TOTAL MOCK STRATEGY: Bypass all problematic aarch64 binary builds
FROM nvcr.io/nvidia/pytorch:24.08-py3

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system dependencies (Minimal)
RUN apt-get update && apt-get install -y \
    curl git wget build-essential libsndfile1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Setup Python 3.11 environment via micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/mamba
RUN micromamba create -n sam_audio python=3.11 -c conda-forge "ffmpeg<8" -y
ENV PATH="/opt/mamba/envs/sam_audio/bin:${PATH}"
SHELL ["/bin/bash", "-c"]

# 3. Install PyTorch (Official aarch64 wheels)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. THE ULTIMATE MOCK STEP: satisfy all imports without building anything
RUN SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])") && \
    # xformers mock
    mkdir -p $SITE_PACKAGES/xformers/ops/fmha && \
    echo 'print("NOTE: Mock xformers loaded")' > $SITE_PACKAGES/xformers/__init__.py && \
    echo '__version__ = "0.0.0"' >> $SITE_PACKAGES/xformers/__init__.py && \
    echo 'class AttentionBias: pass' > $SITE_PACKAGES/xformers/ops/__init__.py && \
    echo 'class LowerTriangularMask(AttentionBias): pass' >> $SITE_PACKAGES/xformers/ops/__init__.py && \
    echo 'def memory_efficient_attention(*args, **kwargs): return None' >> $SITE_PACKAGES/xformers/ops/__init__.py && \
    echo 'from . import fmha' >> $SITE_PACKAGES/xformers/ops/__init__.py && \
    echo 'class BlockDiagonalMask: pass' > $SITE_PACKAGES/xformers/ops/fmha/__init__.py && \
    echo 'class AttentionBias: pass' >> $SITE_PACKAGES/xformers/ops/fmha/__init__.py && \
    echo 'def memory_efficient_attention(*args, **kwargs): return None' >> $SITE_PACKAGES/xformers/ops/fmha/__init__.py && \
    mkdir -p $SITE_PACKAGES/xformers-0.0.0.dist-info && \
    echo -e "Metadata-Version: 2.1\nName: xformers\nVersion: 0.0.0" > $SITE_PACKAGES/xformers-0.0.0.dist-info/METADATA && \
    # torchcodec mock
    mkdir -p $SITE_PACKAGES/torchcodec/decoders $SITE_PACKAGES/torchcodec/samplers && \
    echo 'print("NOTE: Mock torchcodec loaded")' > $SITE_PACKAGES/torchcodec/__init__.py && \
    echo '__version__ = "0.10.0"' >> $SITE_PACKAGES/torchcodec/__init__.py && \
    echo 'class VideoDecoder: def __init__(self, *args, **kwargs): pass' > $SITE_PACKAGES/torchcodec/decoders/__init__.py && \
    echo 'class AudioDecoder: def __init__(self, *args, **kwargs): pass' >> $SITE_PACKAGES/torchcodec/decoders/__init__.py && \
    echo 'class ClipSampler: pass' > $SITE_PACKAGES/torchcodec/samplers/__init__.py && \
    mkdir -p $SITE_PACKAGES/torchcodec-0.10.0.dist-info && \
    echo -e "Metadata-Version: 2.1\nName: torchcodec\nVersion: 0.10.0" > $SITE_PACKAGES/torchcodec-0.10.0.dist-info/METADATA && \
    # decord mock
    mkdir -p $SITE_PACKAGES/decord && \
    echo 'print("NOTE: Mock decord loaded")' > $SITE_PACKAGES/decord/__init__.py && \
    echo 'class VideoReader: def __init__(self, *args, **kwargs): pass' >> $SITE_PACKAGES/decord/__init__.py && \
    echo 'class AudioReader: def __init__(self, *args, **kwargs): pass' >> $SITE_PACKAGES/decord/__init__.py && \
    echo 'def cpu(i=0): return 0' >> $SITE_PACKAGES/decord/__init__.py && \
    echo 'def gpu(i=0): return 1' >> $SITE_PACKAGES/decord/__init__.py && \
    mkdir -p $SITE_PACKAGES/decord-0.6.0.dist-info && \
    echo -e "Metadata-Version: 2.1\nName: decord\nVersion: 0.6.0" > $SITE_PACKAGES/decord-0.6.0.dist-info/METADATA

# 5. Install dependencies into the environment
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "numpy>=1.24.0,<2.0" protobuf==3.20.3 scipy scikit-learn "transformers>=4.54.0" einops pydub torchdiffeq safetensors && \
    pip install git+https://github.com/facebookresearch/dacvae.git \
    git+https://github.com/facebookresearch/ImageBind.git \
    git+https://github.com/lematt1991/CLAP.git \
    git+https://github.com/facebookresearch/perception_models@unpin-deps && \
    pip install audiobox_aesthetics git+https://github.com/facebookresearch/sam-audio#egg=sam_audio && \
    pip install fastapi>=0.104.0 uvicorn>=0.24.0 python-multipart>=0.0.6 huggingface-hub>=0.19.0

WORKDIR /app
COPY . .
RUN mkdir -p /tmp/sam-audio/uploads /tmp/sam-audio/results
EXPOSE 8011
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8011"]
