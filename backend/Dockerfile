# SAM-Audio Pipeline - Official NVIDIA DGX Image (ARM/aarch64 Optimized)
FROM nvcr.io/nvidia/pytorch:24.12-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV DECORD_GPU=0

# Install additional system dependencies
# Note: Basic build tools, ffmpeg, and python are already in the NVIDIA image
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

# Create working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# 1. Build torchcodec from source (Git)
# Fixes "No module named torchcodec.decoders" by ensuring matching binaries for ARM
# Use --no-build-isolation to use NGC's pre-installed PyTorch
RUN git clone https://github.com/pytorch/torchcodec.git /tmp/torchcodec \
    && cd /tmp/torchcodec \
    && python3 -m pip install --no-build-isolation . \
    && rm -rf /tmp/torchcodec

# 2. Final verification of dependencies
RUN python3 -c "import torch; import torchcodec.decoders; print('Verified: All DGX dependencies are correctly installed')"

# 4. Install the rest of the requirements
RUN python3 -m pip install --no-cache-dir -r requirements.txt

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
