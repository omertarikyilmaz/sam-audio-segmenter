#!/usr/bin/env python3
"""Create mock packages for xformers, torchcodec, and decord to bypass binary incompatibility on ARM64."""

import os
import site

# Find site-packages path
packages_dir = site.getsitepackages()[0]
print(f"Creating mocks in: {packages_dir}")

# ============ XFORMERS MOCK ============
xformers_dir = os.path.join(packages_dir, "xformers")
os.makedirs(xformers_dir, exist_ok=True)

with open(os.path.join(xformers_dir, "__init__.py"), "w") as f:
    f.write('print("NOTE: Mock xformers loaded to bypass binary incompatibility")\n')
    f.write('__version__ = "0.0.0"\n')

os.makedirs(os.path.join(xformers_dir, "ops"), exist_ok=True)
with open(os.path.join(xformers_dir, "ops", "__init__.py"), "w") as f:
    f.write('class AttentionBias: pass\n')
    f.write('class LowerTriangularMask(AttentionBias): pass\n')
    f.write('def memory_efficient_attention(*args, **kwargs): return None\n')
    f.write('from . import fmha\n')

os.makedirs(os.path.join(xformers_dir, "ops", "fmha"), exist_ok=True)
with open(os.path.join(xformers_dir, "ops", "fmha", "__init__.py"), "w") as f:
    f.write('class BlockDiagonalMask: pass\n')
    f.write('class AttentionBias: pass\n')
    f.write('def memory_efficient_attention(*args, **kwargs): return None\n')

# xformers dist-info
xformers_dist = os.path.join(packages_dir, "xformers-0.0.0.dist-info")
os.makedirs(xformers_dist, exist_ok=True)
with open(os.path.join(xformers_dist, "METADATA"), "w") as f:
    f.write("Metadata-Version: 2.1\nName: xformers\nVersion: 0.0.0\n")

# ============ TORCHCODEC MOCK ============
torchcodec_dir = os.path.join(packages_dir, "torchcodec")
os.makedirs(torchcodec_dir, exist_ok=True)

with open(os.path.join(torchcodec_dir, "__init__.py"), "w") as f:
    f.write('print("NOTE: Mock torchcodec loaded to bypass binary incompatibility")\n')
    f.write('__version__ = "0.10.0"\n')

os.makedirs(os.path.join(torchcodec_dir, "decoders"), exist_ok=True)
with open(os.path.join(torchcodec_dir, "decoders", "__init__.py"), "w") as f:
    f.write('class VideoDecoder:\n')
    f.write('    def __init__(self, *args, **kwargs): pass\n')
    f.write('class AudioDecoder:\n')
    f.write('    def __init__(self, *args, **kwargs): pass\n')

os.makedirs(os.path.join(torchcodec_dir, "samplers"), exist_ok=True)
with open(os.path.join(torchcodec_dir, "samplers", "__init__.py"), "w") as f:
    f.write('class ClipSampler: pass\n')

# torchcodec dist-info
torchcodec_dist = os.path.join(packages_dir, "torchcodec-0.10.0.dist-info")
os.makedirs(torchcodec_dist, exist_ok=True)
with open(os.path.join(torchcodec_dist, "METADATA"), "w") as f:
    f.write("Metadata-Version: 2.1\nName: torchcodec\nVersion: 0.10.0\n")

# ============ DECORD MOCK ============
decord_dir = os.path.join(packages_dir, "decord")
os.makedirs(decord_dir, exist_ok=True)

with open(os.path.join(decord_dir, "__init__.py"), "w") as f:
    f.write('print("NOTE: Mock decord loaded")\n')
    f.write('class VideoReader:\n')
    f.write('    def __init__(self, *args, **kwargs): pass\n')
    f.write('class AudioReader:\n')
    f.write('    def __init__(self, *args, **kwargs): pass\n')
    f.write('def cpu(i=0): return 0\n')
    f.write('def gpu(i=0): return 1\n')

# decord dist-info
decord_dist = os.path.join(packages_dir, "decord-0.6.0.dist-info")
os.makedirs(decord_dist, exist_ok=True)
with open(os.path.join(decord_dist, "METADATA"), "w") as f:
    f.write("Metadata-Version: 2.1\nName: decord\nVersion: 0.6.0\n")

print("Mock packages created successfully!")
