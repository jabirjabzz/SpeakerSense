import os
import torch
import logging
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple, List
from tqdm import tqdm
from dataclasses import dataclass
import argparse
from pyannote.audio import Pipeline
from pyannote.core import SlidingWindowFeature
import scipy.io.wavfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('speaker_separation.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingProgress:
    """Data class to track processing progress and status"""
    total_chunks: int = 0
    processed_chunks: int = 0
    current_stage: str = ""
    status: str = "not_started"
    error_message: Optional[str] = None

    def update(self, chunks_processed: int = 0, stage: Optional[str] = None,
               status: Optional[str] = None, error_message: Optional[str] = None):
        """Update progress tracking information"""
        if chunks_processed:
            self.processed_chunks += chunks_processed
        if stage:
            self.current_stage = stage
        if status:
            self.status = status
        if error_message is not None:
            self.error_message = error_message

    @property
    def progress_percentage(self) -> float:
        """Calculate current progress as a percentage"""
        if self.total_chunks == 0:
            return 0.0
        return (self.processed_chunks / self.total_chunks) * 100

    def __str__(self) -> str:
        """String representation of current progress"""
        progress_str = (f"Stage: {self.current_stage} | "
                       f"Progress: {self.progress_percentage:.1f}% | "
                       f"Status: {self.status}")
        if self.error_message:
            progress_str += f" | Error: {self.error_message}"
        return progress_str


class LongAudioSpeakerSeparation:
    def __init__(self, model_name: str, token: str, device: str = 'cuda',
                 sample_rate: int = 16000, progress_callback: Optional[Callable] = None):
        # ... (other initialization code)

        try:
            self.pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)
            self.pipeline.to(self.device)
            logger.info(f"Model loaded on {self.device}")

            if self.device.type == 'cuda':
                try:
                    max_mem_bytes = torch.cuda.get_device_properties(self.device).total_memory
                    available_memory_factor = 0.7
                    max_mem_available = int(max_mem_bytes * available_memory_factor)

                    # KEY CHANGE: Get an example input shape
                    example_input = torch.randn(1, self.sample_rate).to(self.device) # 1 second of audio
                    with torch.no_grad():
                        _ = self.pipeline({"waveform": example_input, "sample_rate": self.sample_rate}) # Run a dummy forward pass

                    # Get input shape from the model's first layer
                    input_shape = self.pipeline.model.modules.model.in_channels * self.sample_rate # changed this line
                    model_memory_footprint = input_shape * 4

                    self.max_chunk_size = int(max_mem_available / model_memory_footprint)

                    if self.max_chunk_size < self.sample_rate:
                        self.max_chunk_size = self.sample_rate

                    self.chunk_duration = self.max_chunk_size / self.sample_rate
                    logger.info(f"Calculated chunk duration: {self.chunk_duration:.2f} seconds based on available memory.")

                except RuntimeError as e:
                    logger.error(f"Error getting CUDA memory info: {e}. Falling back to default chunk size.")
                    self.chunk_duration = 30.0
                    self.max_chunk_size = int(self.chunk_duration * self.sample_rate)
                except Exception as e:
                    logger.error(f"Error during chunk size calculation: {e}. Falling back to default chunk size.")
                    self.chunk_duration = 30.0
                    self.max_chunk_size = int(self.chunk_duration * self.sample_rate)
            else:
                self.chunk_duration = 30.0
                self.max_chunk_size = int(self.chunk_duration * self.sample_rate)

            self.overlap_duration = 5.0
            self.chunk_size = int(self.chunk_duration * self.sample_rate)
            self.overlap_size = int(self.overlap_duration * self.sample_rate)
            self.hop_size = self.chunk_size - self.overlap_size
            logger.info(f"Initializing with chunk duration: {self.chunk_duration}s, overlap: {self.overlap_duration}s")

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def _apply_crossfade(self, chunk1: torch.Tensor, chunk2: torch.Tensor,
                        overlap_size: int) -> torch.Tensor:
        """Apply linear crossfade between two chunks"""
        fade_in = torch.linspace(0, 1, overlap_size, device=chunk1.device)
        fade_out = torch.linspace(1, 0, overlap_size, device=chunk1.device)
        
        overlap1 = chunk1[..., -overlap_size:]
        overlap2 = chunk2[..., :overlap_size]
        
        crossfaded = overlap1 * fade_out + overlap2 * fade_in
        return torch.cat([chunk1[..., :-overlap_size], crossfaded])

    def _process_single_chunk(self, chunk: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Process a single chunk and return separated sources and metadata"""
        try:
            with torch.inference_mode():
                # Ensure chunk has correct shape (channel, time)
                if chunk.dim() == 1:
                    chunk = chunk.unsqueeze(0)  # Add channel dimension
                elif chunk.dim() == 3:
                    chunk = chunk.squeeze(0)  # Remove batch dimension if present
                
                # Verify shape is correct
                if not (chunk.dim() == 2 and chunk.size(0) == 1):
                    raise ValueError(f"Unexpected chunk shape: {chunk.shape}. Expected shape: (1, time)")
                
                logger.debug(f"Processing chunk with shape: {chunk.shape}")
                diarization, sources = self.pipeline({
                    "waveform": chunk,
                    "sample_rate": self.sample_rate
                })
                
                # Extract speaker embeddings for consistency
                embeddings = self._extract_speaker_embeddings(sources)
                
                return sources, {
                    "diarization": diarization,
                    "speaker_embeddings": embeddings
                }
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            raise

    def _extract_speaker_embeddings(self, sources: torch.Tensor) -> torch.Tensor:
        """Extract speaker embeddings for speaker consistency"""
        # Use .data to access the tensor from SlidingWindowFeature
        if isinstance(sources, SlidingWindowFeature):
            sources_tensor = sources.data
        else:
            sources_tensor = sources

        # Convert numpy array to torch tensor if necessary
        if isinstance(sources_tensor, np.ndarray):
            sources_tensor = torch.from_numpy(sources_tensor)

        # Ensure tensor is on the correct device
        sources_tensor = sources_tensor.to(self.device)

        # Simple embedding extraction - mean of each source along time dimension
        return torch.mean(sources_tensor, dim=1)


    def _align_speakers_across_chunks(self, current_embeddings: torch.Tensor,
                                    previous_embeddings: torch.Tensor) -> List[int]:
        """Align speakers between chunks based on embedding similarity"""
        similarity_matrix = torch.nn.functional.cosine_similarity(
            current_embeddings.unsqueeze(1),
            previous_embeddings.unsqueeze(0)
        )
        return torch.argmax(similarity_matrix, dim=1).tolist()

    def process_long_audio(self, audio_path: str, output_dir: str):
        """Process a long audio file by chunking with overlap"""
        logger.info(f"Processing long audio file: {audio_path}")
        self.progress.update(stage="Loading audio", status="processing")
        
        try:
            # Load and preprocess audio
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
                logger.info(f"Resampled audio from {sr}Hz to {self.sample_rate}Hz")
            
            # Convert to mono if necessary
            if waveform.dim() == 2 and waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                logger.info("Converted stereo audio to mono")
            elif waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            # Calculate chunks
            total_samples = waveform.size(-1)
            num_chunks = (total_samples - self.overlap_size) // self.hop_size + 1
            
            logger.info(f"Processing {num_chunks} chunks with {self.overlap_size} samples overlap")
            
            # Initialize storage
            processed_sources = []
            previous_embeddings = None
            speaker_mapping = None
            
            self.progress.total_chunks = num_chunks
            
            # Process chunks
            with tqdm(total=num_chunks, desc="Processing chunks") as pbar:
                for i in range(num_chunks):
                    self.progress.update(stage=f"Processing chunk {i+1}/{num_chunks}")
                    
                    # Extract chunk
                    start = i * self.hop_size
                    end = min(start + self.chunk_size, total_samples)
                    chunk = waveform[:, start:end]
                    
                    # Pad if necessary
                    if chunk.size(-1) < self.chunk_size:
                        chunk = torch.nn.functional.pad(
                            chunk, (0, self.chunk_size - chunk.size(-1))
                        )
                    
                    # Process chunk
                    chunk = chunk.to(self.device)
                    sources, metadata = self._process_single_chunk(chunk)
                    
                    # Align speakers with previous chunk
                    if previous_embeddings is not None:
                        speaker_mapping = self._align_speakers_across_chunks(
                            metadata["speaker_embeddings"],
                            previous_embeddings
                        )
                        sources = sources[speaker_mapping]
                    
                    previous_embeddings = metadata["speaker_embeddings"]
                    
                    # Apply crossfade if not first chunk
                    if processed_sources and self.overlap_size > 0:
                        sources = self._apply_crossfade(
                            processed_sources[-1],
                            sources,
                            self.overlap_size
                        )
                    
                    processed_sources.append(sources)
                    
                    self.progress.update(chunks_processed=1)
                    if self.progress_callback:
                        self.progress_callback(self.progress)
                    
                    pbar.update(1)
                    
                    # Memory management
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
            
            # Save results
            self.progress.update(stage="Saving results", status="saving")
            os.makedirs(output_dir, exist_ok=True)
            final_sources = torch.cat(processed_sources, dim=-1)
            
            for spk_idx in range(final_sources.size(0)):
                output_path = os.path.join(output_dir, f"speaker_{spk_idx+1}.wav")
                torchaudio.save(
                    output_path,
                    final_sources[spk_idx].cpu().unsqueeze(0),
                    self.sample_rate
                )
                logger.info(f"Saved speaker {spk_idx+1} to {output_path}")
            
            self.progress.update(stage="Completed", status="success")
            if self.progress_callback:
                self.progress_callback(self.progress)
            
        except Exception as e:
            self.progress.update(status="failed", error_message=str(e))
            logger.error("Processing failed", exc_info=True)
            if self.progress_callback:
                self.progress_callback(self.progress)
            raise RuntimeError(f"Processing failed: {str(e)}") from e


def progress_handler(progress: ProcessingProgress):
    """Example progress callback function"""
    print(f"\rProgress: {progress}", end='')


def main():
    parser = argparse.ArgumentParser(description="Long Audio Speaker Separation")
    parser.add_argument('--input_path', type=str, required=True,
                      help='Path to input audio file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory')
    parser.add_argument('--chunk_duration', type=float, default=30.0,
                      help='Chunk duration in seconds')
    parser.add_argument('--overlap_duration', type=float, default=5.0,
                      help='Overlap duration in seconds')
    parser.add_argument('--model_name', type=str, default="pyannote/speech-separation-ami-1.0",
                      help='HuggingFace model name')
    parser.add_argument('--token', type=str, required=True,
                      help='HuggingFace token')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    try:
        separator = LongAudioSpeakerSeparation(
            model_name=args.model_name,
            token=args.token,
            device=args.device,
            chunk_duration=args.chunk_duration,
            overlap_duration=args.overlap_duration,
            progress_callback=progress_handler
        )
        
        separator.process_long_audio(args.input_path, args.output_dir)
        logger.info("Processing completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return 1
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
