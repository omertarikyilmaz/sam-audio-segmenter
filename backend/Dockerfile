# SAM-Audio Pipeline - NVIDIA DGX Blackwell (GB10) Optimized
# Using 24.12 for Blackwell support and native Python 3.12
FROM nvcr.io/nvidia/pytorch:24.12-py3

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    curl git wget build-essential libsndfile1 libgl1 libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Python 3.12 is ALREADY the system default in 24.12-py3
# No need for Micromamba/Conda - staying in system Python for stability

# 3. MOUNT CACHE PILL - BuildKit cache for remaining pip installs
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

# 4. THE ULTIMATE MOCK STEP: satisfy all imports without building problematic binaries
RUN SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    echo "Creating mocks in: $SITE_PACKAGES" && \
    # XFORMERS MOCK
    mkdir -p $SITE_PACKAGES/xformers/ops/fmha && \
    cat <<EOF > $SITE_PACKAGES/xformers/__init__.py
print("NOTE: Mock xformers loaded to bypass binary incompatibility")
__version__ = "0.0.0"
EOF
    cat <<EOF > $SITE_PACKAGES/xformers/ops/__init__.py
class AttentionBias: pass
class LowerTriangularMask(AttentionBias): pass
def memory_efficient_attention(*args, **kwargs): return None
from . import fmha
EOF
    cat <<EOF > $SITE_PACKAGES/xformers/ops/fmha/__init__.py
class BlockDiagonalMask: pass
class AttentionBias: pass
def memory_efficient_attention(*args, **kwargs): return None
EOF
    # xformers dist-info
    mkdir -p $SITE_PACKAGES/xformers-0.0.0.dist-info && \
    echo -e "Metadata-Version: 2.1\nName: xformers\nVersion: 0.0.0" > $SITE_PACKAGES/xformers-0.0.0.dist-info/METADATA && \
    # TORCHCODEC MOCK
    mkdir -p $SITE_PACKAGES/torchcodec/decoders $SITE_PACKAGES/torchcodec/samplers && \
    cat <<EOF > $SITE_PACKAGES/torchcodec/__init__.py
print("NOTE: Mock torchcodec loaded to bypass binary incompatibility")
__version__ = "0.10.0"
EOF
    cat <<EOF > $SITE_PACKAGES/torchcodec/decoders/__init__.py
class VideoDecoder:
    def __init__(self, *args, **kwargs): pass
class AudioDecoder:
    def __init__(self, *args, **kwargs): pass
EOF
    cat <<EOF > $SITE_PACKAGES/torchcodec/samplers/__init__.py
class ClipSampler: pass
EOF
    # torchcodec dist-info
    mkdir -p $SITE_PACKAGES/torchcodec-0.10.0.dist-info && \
    echo -e "Metadata-Version: 2.1\nName: torchcodec\nVersion: 0.10.0" > $SITE_PACKAGES/torchcodec-0.10.0.dist-info/METADATA && \
    # decord mock
    mkdir -p $SITE_PACKAGES/decord && \
    cat <<EOF > $SITE_PACKAGES/decord/__init__.py
print("NOTE: Mock decord loaded")
class VideoReader:
    def __init__(self, *args, **kwargs): pass
class AudioReader:
    def __init__(self, *args, **kwargs): pass
def cpu(i=0): return 0
def gpu(i=0): return 1
EOF
    mkdir -p $SITE_PACKAGES/decord-0.6.0.dist-info && \
    echo -e "Metadata-Version: 2.1\nName: decord\nVersion: 0.6.0" > $SITE_PACKAGES/decord-0.6.0.dist-info/METADATA

# 5. Install SAM-Audio dependencies (pip will use system torch in 24.12)
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

# In 24.12, system python is python3.12
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8011"]
