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

# 2. Upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

# 3. Create working directory and copy mock script
WORKDIR /app
COPY create_mocks.py .

# 4. Create mock packages using Python script (avoids Dockerfile syntax issues)
RUN python3 create_mocks.py

# 5. Install SAM-Audio dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "numpy>=1.24.0,<2.0" protobuf==3.20.3 scipy scikit-learn "transformers>=4.54.0" einops pydub torchdiffeq safetensors && \
    pip install git+https://github.com/facebookresearch/dacvae.git \
    git+https://github.com/facebookresearch/ImageBind.git \
    git+https://github.com/lematt1991/CLAP.git \
    git+https://github.com/facebookresearch/perception_models@unpin-deps && \
    pip install audiobox_aesthetics git+https://github.com/facebookresearch/sam-audio#egg=sam_audio && \
    pip install fastapi>=0.104.0 uvicorn>=0.24.0 python-multipart>=0.0.6 huggingface-hub>=0.19.0

# 6. Copy application code
COPY . .
RUN mkdir -p /tmp/sam-audio/uploads /tmp/sam-audio/results
EXPOSE 8011

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8011"]
