# SAM-Audio Pipeline - NVIDIA DGX ARM64
# Strategy: COMPREHENSIVE MOCKS for problematic binary dependencies
FROM nvcr.io/nvidia/pytorch:24.08-py3

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl git wget build-essential libsndfile1 libgl1 libglib2.0-0 cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/mamba
RUN micromamba create -n sam_audio python=3.11 -c conda-forge "ffmpeg<8" -y
ENV PATH="/opt/mamba/envs/sam_audio/bin:${PATH}"

SHELL ["/bin/bash", "-c"]
ENV CONDA_DEFAULT_ENV=sam_audio

# 1. Install PyTorch (Official aarch64 wheels)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. CORE MOCKING STEP - This solves the binary incompatibility once and for all
# We create the structure expected by SAM-Audio exactly as in the script you shared
RUN SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])") && \
    echo "Creating mocks in: $SITE_PACKAGES" && \
    # XFORMERS MOCK
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
    # TORCHCODEC MOCK
    mkdir -p $SITE_PACKAGES/torchcodec/decoders $SITE_PACKAGES/torchcodec/samplers && \
    echo 'print("NOTE: Mock torchcodec loaded")' > $SITE_PACKAGES/torchcodec/__init__.py && \
    echo '__version__ = "0.10.0"' >> $SITE_PACKAGES/torchcodec/__init__.py && \
    echo 'class VideoDecoder: def __init__(self, *args, **kwargs): pass' > $SITE_PACKAGES/torchcodec/decoders/__init__.py && \
    echo 'class AudioDecoder: def __init__(self, *args, **kwargs): pass' >> $SITE_PACKAGES/torchcodec/decoders/__init__.py && \
    echo 'class ClipSampler: pass' > $SITE_PACKAGES/torchcodec/samplers/__init__.py

# 3. Install core dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "numpy>=1.24.0,<2.0" protobuf==3.20.3 scipy scikit-learn "transformers>=4.54.0" einops pydub torchdiffeq safetensors

# 4. Build decord from source (Still needed and usually builds fine)
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ ! -d "/tmp/decord" ]; then git clone --recursive https://github.com/dmlc/decord.git /tmp/decord; fi && \
    cd /tmp/decord && \
    mkdir -p build && cd build && \
    cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    cd ../python && \
    pip install .

# 5. Git dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install git+https://github.com/facebookresearch/dacvae.git \
    git+https://github.com/facebookresearch/ImageBind.git \
    git+https://github.com/lematt1991/CLAP.git \
    git+https://github.com/facebookresearch/perception_models@unpin-deps

# 6. Final install
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install audiobox_aesthetics \
    git+https://github.com/facebookresearch/sam-audio#egg=sam_audio && \
    pip install fastapi>=0.104.0 uvicorn>=0.24.0 python-multipart>=0.0.6 huggingface-hub>=0.19.0

WORKDIR /app
COPY . .
RUN mkdir -p /tmp/sam-audio/uploads /tmp/sam-audio/results
EXPOSE 8011
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8011"]
