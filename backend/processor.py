"""
SAM-Audio Pipeline - Audio Source Separation using Meta's SAM-Audio Model

This module provides audio chunking and processing capabilities for
separating specific sounds from audio mixtures using text prompts.
"""
import os
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

import torch
import torchaudio
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

logger = logging.getLogger(__name__)

# Constants
CHUNK_DURATION_SECONDS = 45  # 45 seconds - balanced for 24GB VRAM
SAMPLE_RATE = 16000  # SAM-Audio uses 16kHz
MAX_AUDIO_DURATION = 65 * 60  # 65 minutes in seconds

# Silence detection settings
SILENCE_THRESH_DB = -40  # dB threshold for silence detection
MIN_SILENCE_LEN_MS = 500  # Minimum silence length to be considered (ms)
MIN_SPEECH_LEN_MS = 1000  # Minimum speech segment length to keep (ms)
GAP_BETWEEN_SEGMENTS_MS = 1000  # 1 second gap between segments
SEGMENT_PADDING_MS = 100  # Padding to add before/after each segment
CHUNK_OVERLAP_SECONDS = 5  # Overlap between chunks to prevent word cutting

def remove_silence_and_concatenate(
    audio: AudioSegment,
    silence_thresh: int = SILENCE_THRESH_DB,
    min_silence_len: int = MIN_SILENCE_LEN_MS,
    min_speech_len: int = MIN_SPEECH_LEN_MS,
    gap_duration: int = GAP_BETWEEN_SEGMENTS_MS,
    merge_threshold_ms: int = 200  # Merge segments if gap is less than this
) -> AudioSegment:
    """
    Remove silent parts from audio and concatenate non-silent segments with gaps.
    Smart merging: if two segments are close together (< merge_threshold_ms apart),
    they are merged without a gap to prevent cutting words at boundaries.
    
    Args:
        audio: Input audio segment
        silence_thresh: Silence threshold in dB
        min_silence_len: Minimum silence length to split on (ms)
        min_speech_len: Minimum speech segment length to keep (ms)
        gap_duration: Duration of silence gap between segments (ms)
        merge_threshold_ms: Merge consecutive segments if gap is less than this
        
    Returns:
        Cleaned audio with silent parts removed and segments concatenated
    """
    logger.info(f"Detecting non-silent segments (threshold: {silence_thresh}dB)...")
    
    # Detect non-silent segments
    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )
    
    if not nonsilent_ranges:
        logger.warning("No non-silent segments found!")
        return audio
    
    logger.info(f"Found {len(nonsilent_ranges)} raw non-silent segments")
    
    # Step 1: Merge consecutive segments that are close together
    # This prevents cutting words that span batch boundaries
    merged_ranges = []
    current_start, current_end = nonsilent_ranges[0]
    
    for i in range(1, len(nonsilent_ranges)):
        next_start, next_end = nonsilent_ranges[i]
        gap_between = next_start - current_end
        
        if gap_between <= merge_threshold_ms:
            # Merge: extend current segment to include the next one
            current_end = next_end
            logger.debug(f"Merging segments (gap={gap_between}ms)")
        else:
            # Save current merged segment and start a new one
            merged_ranges.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    
    # Don't forget the last segment
    merged_ranges.append((current_start, current_end))
    
    logger.info(f"After merging consecutive segments: {len(merged_ranges)} segments")
    
    # Step 2: Filter out short segments, add padding, and concatenate with gaps
    gap = AudioSegment.silent(duration=gap_duration)
    result = AudioSegment.empty()
    segments_kept = 0
    audio_duration_ms = len(audio)
    
    for i, (start_ms, end_ms) in enumerate(merged_ranges):
        segment_duration = end_ms - start_ms
        
        # Skip very short segments (likely noise)
        if segment_duration < min_speech_len:
            logger.debug(f"Skipping short segment {i+1}: {segment_duration}ms")
            continue
        
        # Add padding to preserve word boundaries (100ms before and after)
        padded_start = max(0, start_ms - SEGMENT_PADDING_MS)
        padded_end = min(audio_duration_ms, end_ms + SEGMENT_PADDING_MS)
        
        segment = audio[padded_start:padded_end]
        
        # Only add gap between segments, not at the start
        if len(result) > 0:
            result += gap
        
        result += segment
        segments_kept += 1
        
        logger.debug(f"Kept segment {i+1}: {padded_start/1000:.1f}s - {padded_end/1000:.1f}s (padded from {start_ms/1000:.1f}s-{end_ms/1000:.1f}s)")
    
    original_duration = len(audio) / 1000
    result_duration = len(result) / 1000
    reduction = (1 - result_duration / original_duration) * 100
    
    logger.info(f"Silence removal: {original_duration:.1f}s → {result_duration:.1f}s ({reduction:.1f}% reduced, {segments_kept} segments)")
    
    return result


class SAMAudioProcessor:
    """Handles audio chunking and SAM-Audio model inference"""
    
    def __init__(self, model_name: str = "facebook/sam-audio-small"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._loaded = False
        
    async def load_model(self):
        """Load SAM-Audio model and processor with memory optimization"""
        if self._loaded:
            return
            
        logger.info(f"Loading SAM-Audio model: {self.model_name}")
        
        try:
            from sam_audio import SAMAudio, SAMAudioProcessor as SAMProcessor
            
            # Load model
            self.model = SAMAudio.from_pretrained(self.model_name)
            
            # Disable memory-heavy components (rankers and span predictor)
            # This reduces VRAM usage from ~24GB to ~10GB
            # See: https://huggingface.co/facebook/sam-audio-small/discussions
            logger.info("Disabling rankers and span predictor for memory optimization...")
            self.model.visual_ranker = None
            self.model.text_ranker = None
            self.model.span_predictor = None
            self.model.span_predictor_transform = None
            
            # Move to device and set to eval mode
            self.model = self.model.to(self.device).eval()
            self.processor = SAMProcessor.from_pretrained(self.model_name)
            self._loaded = True
            
            logger.info(f"Model loaded successfully on {self.device} (memory-optimized)")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Could not load SAM-Audio model: {str(e)}")
    
    def chunk_audio(
        self, 
        audio_path: Path, 
        chunk_duration: int = CHUNK_DURATION_SECONDS,
        overlap: int = CHUNK_OVERLAP_SECONDS
    ) -> List[Tuple[Path, float, float, float, float]]:
        """
        Split audio file into overlapping chunks for processing
        
        Args:
            audio_path: Path to input audio file
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds
            
        Returns:
            List of tuples: (chunk_path, start_time, end_time, trim_start, trim_end)
            trim_start/trim_end indicate how much to trim from merged output
        """
        logger.info(f"Chunking audio: {audio_path}")
        
        # Load audio using pydub (handles various formats)
        audio = AudioSegment.from_file(str(audio_path))
        duration_ms = len(audio)
        duration_sec = duration_ms / 1000
        
        if duration_sec > MAX_AUDIO_DURATION:
            raise ValueError(f"Audio too long: {duration_sec:.1f}s (max: {MAX_AUDIO_DURATION}s)")
        
        logger.info(f"Audio duration: {duration_sec:.1f} seconds")
        
        # Create temporary directory for chunks
        chunk_dir = Path(tempfile.mkdtemp(prefix="sam_chunks_"))
        chunks = []
        
        chunk_duration_ms = chunk_duration * 1000
        overlap_ms = overlap * 1000
        step_ms = chunk_duration_ms - overlap_ms  # Step size between chunks
        
        # Calculate number of chunks
        num_chunks = max(1, int(np.ceil((duration_ms - overlap_ms) / step_ms)))
        
        logger.info(f"Splitting into {num_chunks} chunks of {chunk_duration}s with {overlap}s overlap")
        
        for i in range(num_chunks):
            start_ms = i * step_ms
            end_ms = min(start_ms + chunk_duration_ms, duration_ms)
            
            # For first chunk, no trim at start; for last chunk, no trim at end
            # For middle chunks, trim half the overlap from each side when merging
            trim_start_ms = 0 if i == 0 else overlap_ms // 2
            trim_end_ms = 0 if i == num_chunks - 1 else overlap_ms // 2
            
            chunk = audio[start_ms:end_ms]
            
            # Export as WAV (SAM-Audio format)
            chunk_path = chunk_dir / f"chunk_{i:04d}.wav"
            chunk.export(str(chunk_path), format="wav")
            
            chunks.append((
                chunk_path, 
                start_ms / 1000, 
                end_ms / 1000,
                trim_start_ms / 1000,
                trim_end_ms / 1000
            ))
            
        logger.info(f"Created {len(chunks)} overlapping chunks in {chunk_dir}")
        return chunks
    
    async def process_chunk(
        self, 
        chunk_path: Path, 
        prompt: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single audio chunk with SAM-Audio
        
        Args:
            chunk_path: Path to audio chunk
            prompt: Text description of sound to isolate
            
        Returns:
            Tuple of (target_audio, residual_audio) tensors
        """
        if not self._loaded:
            await self.load_model()
        
        logger.info(f"Processing chunk: {chunk_path.name} with prompt: '{prompt}'")
        
        # Process with SAM-Audio
        inputs = self.processor(
            audios=[str(chunk_path)], 
            descriptions=[prompt]
        ).to(self.device)
        
        with torch.inference_mode():
            # predict_spans=False to avoid using span predictor (reduces memory)
            result = self.model.separate(inputs, predict_spans=False)
        
        # Extract tensors
        target = result.target[0].cpu()
        residual = result.residual[0].cpu()
        
        return target, residual
    
    def merge_chunks(
        self, 
        chunks: List[Tuple[torch.Tensor, torch.Tensor, float, float]],
        sample_rate: int = SAMPLE_RATE
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merge processed chunks back together, trimming overlapping portions
        
        Args:
            chunks: List of (target, residual, trim_start, trim_end) tuples
            sample_rate: Sample rate for calculating trim samples
            
        Returns:
            Tuple of (merged_target, merged_residual) tensors
        """
        if not chunks:
            raise ValueError("No chunks to merge")
            
        if len(chunks) == 1:
            target, residual, _, _ = chunks[0]
            return target, residual
        
        logger.info(f"Merging {len(chunks)} chunks with overlap trimming")
        
        trimmed_targets = []
        trimmed_residuals = []
        
        for i, (target, residual, trim_start, trim_end) in enumerate(chunks):
            # Convert trim times to samples
            trim_start_samples = int(trim_start * sample_rate)
            trim_end_samples = int(trim_end * sample_rate) if trim_end > 0 else 0
            
            # Trim the tensors
            if trim_end_samples > 0:
                trimmed_target = target[..., trim_start_samples:-trim_end_samples] if trim_end_samples > 0 else target[..., trim_start_samples:]
                trimmed_residual = residual[..., trim_start_samples:-trim_end_samples] if trim_end_samples > 0 else residual[..., trim_start_samples:]
            else:
                trimmed_target = target[..., trim_start_samples:]
                trimmed_residual = residual[..., trim_start_samples:]
            
            trimmed_targets.append(trimmed_target)
            trimmed_residuals.append(trimmed_residual)
            
            logger.debug(f"Chunk {i+1}: trimmed {trim_start}s from start, {trim_end}s from end")
        
        merged_target = torch.cat(trimmed_targets, dim=-1)
        merged_residual = torch.cat(trimmed_residuals, dim=-1)
        
        return merged_target, merged_residual
    
    async def separate(
        self, 
        audio_path: Path, 
        prompt: str,
        progress_callback: Optional[callable] = None,
        remove_silence: bool = True
    ) -> Tuple[Path, Path, Path, Optional[Path]]:
        """
        Full separation pipeline: chunk → process → merge → remove silence → save
        
        Args:
            audio_path: Path to input audio file
            prompt: Text description of sound to isolate
            progress_callback: Optional callback(current, total, message)
            remove_silence: Whether to create a cleaned version with silence removed
            
        Returns:
            Tuple of paths: (original_path, target_path, residual_path, cleaned_path)
        """
        # Ensure model is loaded
        await self.load_model()
        
        # Create output directory
        output_dir = Path(tempfile.mkdtemp(prefix="sam_output_"))
        
        # Step 1: Chunk the audio with overlap
        if progress_callback:
            progress_callback(0, 100, "Chunking audio with overlap...")
            
        chunks_info = self.chunk_audio(audio_path)
        num_chunks = len(chunks_info)
        
        # Step 2: Process each chunk
        processed_chunks = []
        for i, (chunk_path, start, end, trim_start, trim_end) in enumerate(chunks_info):
            if progress_callback:
                progress_callback(
                    int((i / num_chunks) * 70) + 10, 
                    100, 
                    f"Processing chunk {i+1}/{num_chunks} ({start:.1f}s - {end:.1f}s)"
                )
            
            target, residual = await self.process_chunk(chunk_path, prompt)
            processed_chunks.append((target, residual, trim_start, trim_end))
            
            # Clean up chunk file
            chunk_path.unlink()
        
        # Clean up chunk directory
        chunks_info[0][0].parent.rmdir()
        
        # Step 3: Merge chunks
        if progress_callback:
            progress_callback(80, 100, "Merging chunks...")
            
        merged_target, merged_residual = self.merge_chunks(processed_chunks)
        
        # Step 4: Load original for comparison
        original_waveform, orig_sr = torchaudio.load(str(audio_path))
        if orig_sr != self.processor.audio_sampling_rate:
            original_waveform = torchaudio.functional.resample(
                original_waveform, orig_sr, self.processor.audio_sampling_rate
            )
        
        # Make mono if stereo
        if original_waveform.shape[0] > 1:
            original_waveform = original_waveform.mean(dim=0, keepdim=True)
        else:
            original_waveform = original_waveform[0:1]
        
        # Step 5: Save outputs
        if progress_callback:
            progress_callback(85, 100, "Saving results...")
        
        sr = self.processor.audio_sampling_rate
        
        original_path = output_dir / "original.wav"
        target_path = output_dir / "isolated.wav"
        residual_path = output_dir / "residual.wav"
        cleaned_path = None
        
        torchaudio.save(str(original_path), original_waveform, sr)
        torchaudio.save(str(target_path), merged_target.unsqueeze(0), sr)
        torchaudio.save(str(residual_path), merged_residual.unsqueeze(0), sr)
        
        # Step 6: Remove silence from isolated (human voices) if requested
        if remove_silence:
            if progress_callback:
                progress_callback(90, 100, "Removing silence from speech content...")
            
            try:
                # Load isolated (human voices) as pydub AudioSegment
                isolated_audio = AudioSegment.from_wav(str(target_path))
                
                # Remove silence and concatenate with gaps
                cleaned_audio = remove_silence_and_concatenate(isolated_audio)
                
                # Save cleaned version
                cleaned_path = output_dir / "cleaned.wav"
                cleaned_audio.export(str(cleaned_path), format="wav")
                
                logger.info(f"Cleaned audio saved: {cleaned_path}")
            except Exception as e:
                logger.error(f"Failed to remove silence: {str(e)}")
                cleaned_path = None
        
        if progress_callback:
            progress_callback(100, 100, "Complete!")
        
        logger.info(f"Separation complete. Output directory: {output_dir}")
        
        return original_path, target_path, residual_path, cleaned_path


# Singleton instance for reuse
_processor_instance: Optional[SAMAudioProcessor] = None


def get_processor() -> SAMAudioProcessor:
    """Get or create SAM-Audio processor instance"""
    global _processor_instance
    if _processor_instance is None:
        model_name = os.getenv("SAM_MODEL", "facebook/sam-audio-small")
        _processor_instance = SAMAudioProcessor(model_name=model_name)
    return _processor_instance
