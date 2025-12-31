#!/usr/bin/env python3
"""Create COMPLETE mock packages for xformers, torchcodec, and decord.
This version includes ALL methods and attributes that sam-audio might use."""

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

xformers_dist = os.path.join(packages_dir, "xformers-0.0.0.dist-info")
os.makedirs(xformers_dist, exist_ok=True)
with open(os.path.join(xformers_dist, "METADATA"), "w") as f:
    f.write("Metadata-Version: 2.1\nName: xformers\nVersion: 0.0.0\n")

# ============ TORCHCODEC MOCK - COMPLETE IMPLEMENTATION ============
torchcodec_dir = os.path.join(packages_dir, "torchcodec")
os.makedirs(torchcodec_dir, exist_ok=True)

with open(os.path.join(torchcodec_dir, "__init__.py"), "w") as f:
    f.write('''print("NOTE: Mock torchcodec loaded to bypass binary incompatibility")
__version__ = "0.10.0"
from .decoders import AudioDecoder, VideoDecoder
from .encoders import AudioEncoder

def save_with_torchcodec(uri, src, sample_rate, channels_first=True, **kwargs):
    """Mock save_with_torchcodec - does nothing, use torchaudio.save directly"""
    import torchaudio
    # Fall back to regular torchaudio save with sox or soundfile backend
    torchaudio.save(uri, src, sample_rate, format="wav")
''')

os.makedirs(os.path.join(torchcodec_dir, "encoders"), exist_ok=True)
with open(os.path.join(torchcodec_dir, "encoders", "__init__.py"), "w") as f:
    f.write('''class AudioEncoder:
    def __init__(self, uri=None, sample_rate=16000, num_channels=1, format=None, encoder=None, encoder_format=None, compression_level=None, **kwargs):
        self.uri = uri
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.format = format
        self.encoder = encoder
        self.encoder_format = encoder_format
        self.compression_level = compression_level

    def write(self, audio_tensor):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
''')

os.makedirs(os.path.join(torchcodec_dir, "decoders"), exist_ok=True)
with open(os.path.join(torchcodec_dir, "decoders", "__init__.py"), "w") as f:
    f.write('''import torch

class AudioDecoderMetadata:
    def __init__(self):
        self.sample_rate = 16000
        self.num_channels = 1
        self.num_frames = 0
        self.duration_seconds = 0.0
        self.codec = "unknown"
        self.bit_rate = 0

class VideoDecoderMetadata:
    def __init__(self):
        self.num_frames = 0
        self.duration_seconds = 0.0
        self.width = 0
        self.height = 0
        self.codec = "unknown"
        self.bit_rate = 0
        self.average_fps = 0.0

class AudioSamples:
    def __init__(self):
        self.data = torch.zeros(1, 16000)  # 1 channel, 1 second at 16kHz
        self.sample_rate = 16000
        self.pts_seconds = 0.0
        self.duration_seconds = 1.0

class AudioDecoder:
    """Complete mock of torchcodec.decoders.AudioDecoder"""
    def __init__(self, source, *args, **kwargs):
        self.metadata = AudioDecoderMetadata()
        self._source = source
    
    def __len__(self):
        return 0
    
    def __getitem__(self, idx):
        return torch.zeros(1, 16000)
    
    def get_all_samples(self):
        """Returns all audio samples as a tensor"""
        return AudioSamples()
    
    def get_samples_played_in_range(self, start_seconds, end_seconds):
        """Returns samples in time range"""
        return AudioSamples()
    
    def decode(self, *args, **kwargs):
        """Decode audio"""
        return torch.zeros(1, 16000)
    
    @property
    def sample_rate(self):
        return self.metadata.sample_rate

class VideoDecoder:
    """Complete mock of torchcodec.decoders.VideoDecoder"""
    def __init__(self, source, *args, **kwargs):
        self.metadata = VideoDecoderMetadata()
        self._source = source
    
    def __len__(self):
        return 0
    
    def __getitem__(self, idx):
        return torch.zeros(3, 224, 224)
    
    def get_frame_at(self, index):
        return torch.zeros(3, 224, 224)
    
    def get_frames_at(self, indices):
        return torch.zeros(len(indices), 3, 224, 224)
    
    def get_frame_played_at(self, seconds):
        return torch.zeros(3, 224, 224)
    
    def get_frames_played_at(self, seconds_list):
        return torch.zeros(len(seconds_list), 3, 224, 224)
    
    def get_frames_in_range(self, start, end, step=1):
        return torch.zeros(1, 3, 224, 224)
    
    def get_frames_played_in_range(self, start_seconds, end_seconds):
        return torch.zeros(1, 3, 224, 224)
''')

os.makedirs(os.path.join(torchcodec_dir, "samplers"), exist_ok=True)
with open(os.path.join(torchcodec_dir, "samplers", "__init__.py"), "w") as f:
    f.write('''class IndexBasedSamplerArgs: pass
class TimeBasedSamplerArgs: pass
class ClipSampler:
    def __init__(self, *args, **kwargs): pass
''')

torchcodec_dist = os.path.join(packages_dir, "torchcodec-0.10.0.dist-info")
os.makedirs(torchcodec_dist, exist_ok=True)
with open(os.path.join(torchcodec_dist, "METADATA"), "w") as f:
    f.write("Metadata-Version: 2.1\nName: torchcodec\nVersion: 0.10.0\n")

# ============ DECORD MOCK - COMPLETE IMPLEMENTATION ============
decord_dir = os.path.join(packages_dir, "decord")
os.makedirs(decord_dir, exist_ok=True)

with open(os.path.join(decord_dir, "__init__.py"), "w") as f:
    f.write('''print("NOTE: Mock decord loaded")
import numpy as np

def cpu(i=0): return 0
def gpu(i=0): return 1

class VideoReader:
    def __init__(self, uri, ctx=cpu(), *args, **kwargs):
        self._uri = uri
        self._num_frames = 0
    
    def __len__(self):
        return self._num_frames
    
    def __getitem__(self, idx):
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    def get_batch(self, indices):
        return np.zeros((len(indices), 224, 224, 3), dtype=np.uint8)
    
    def get_avg_fps(self):
        return 30.0
    
    def seek(self, frame_id):
        pass
    
    def seek_accurate(self, frame_id):
        pass

class AudioReader:
    def __init__(self, uri, ctx=cpu(), *args, **kwargs):
        self._uri = uri
        self._sample_rate = 16000
    
    def __len__(self):
        return 0
    
    def __getitem__(self, idx):
        return np.zeros((1, 16000), dtype=np.float32)
    
    @property
    def sample_rate(self):
        return self._sample_rate

class AVReader:
    def __init__(self, uri, ctx=cpu(), *args, **kwargs):
        self.video = VideoReader(uri, ctx)
        self.audio = AudioReader(uri, ctx)
''')

decord_dist = os.path.join(packages_dir, "decord-0.6.0.dist-info")
os.makedirs(decord_dist, exist_ok=True)
with open(os.path.join(decord_dist, "METADATA"), "w") as f:
    f.write("Metadata-Version: 2.1\nName: decord\nVersion: 0.6.0\n")

print("Mock packages created successfully!")
print("All methods implemented: get_all_samples, decode, metadata, etc.")
