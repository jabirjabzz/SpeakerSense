import os
import torch
import logging
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple, List, Union
from tqdm import tqdm
from dataclasses import dataclass
import argparse
from pyannote.audio import Pipeline
from pyannote.core import SlidingWindowFeature

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
               status: Optional[str] = None, error_message: Optional[str] = None) -> None:
        """
        Update progress tracking information
        
        Args:
            chunks_processed: Number of new chunks processed
            stage: Current processing stage
            status: Current status
            error_message: Error message if any
        """
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
                 sample_rate: int = 16000, chunk_duration: float = 30.0,
                 overlap_duration: float = 5.0, progress_callback: Optional[Callable] = None):
        """
        Initialize the speaker separation system.
        
        Args:
            model_name: Name of the pretrained model
            token: HuggingFace access token
            device: Device to run the model on ('cuda' or 'cpu')
            sample_rate: Audio sample rate
            chunk_duration: Duration of each processing chunk in seconds
            overlap_duration: Duration of overlap between chunks in seconds
            progress_callback: Optional callback function for progress updates
            
        Raises:
            RuntimeError: If model loading fails
            ValueError: If invalid parameters are provided
        """
        if overlap_duration >= chunk_duration:
            raise ValueError("Overlap duration must be less than chunk duration")
            
        self.model_name = model_name
        self.token = token
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.sample_rate = sample_rate
        self.progress_callback = progress_callback
        self.progress = ProcessingProgress()
        
        logger.info(f"Initializing with device: {self.device}")
        
        try:
            # Initialize the pipeline
            self.pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)
            self.pipeline.to(self.device)
            logger.info(f"Model loaded on {self.device}")
            
            # Set chunk parameters
            self.chunk_duration = chunk_duration
            self.overlap_duration = overlap_duration
            self.chunk_size = int(self.chunk_duration * self.sample_rate)
            self.overlap_size = int(self.overlap_duration * self.sample_rate)
            self.hop_size = self.chunk_size - self.overlap_size
            
            logger.info(f"Initialized with chunk duration: {self.chunk_duration}s, overlap: {self.overlap_duration}s")
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def _apply_crossfade(self, chunk1: torch.Tensor, chunk2: torch.Tensor,
                        overlap_size: int) -> torch.Tensor:
        """
        Apply linear crossfade between two chunks
        
        Args:
            chunk1: First audio chunk
            chunk2: Second audio chunk
            overlap_size: Size of overlap region
            
        Returns:
            torch.Tensor: Crossfaded audio
            
        Raises:
            RuntimeError: If crossfade operation fails
            ValueError: If chunk shapes are incompatible
        """
        if chunk1.shape[:-1] != chunk2.shape[:-1]:
            raise ValueError(f"Incompatible chunk shapes: {chunk1.shape} and {chunk2.shape}")
            
        fade_in = torch.linspace(0, 1, overlap_size, device=chunk1.device)
        fade_out = torch.linspace(1, 0, overlap_size, device=chunk1.device)
        
        # Add dimensions to match audio shape
        for _ in range(chunk1.dim() - 1):
            fade_in = fade_in.unsqueeze(0)
            fade_out = fade_out.unsqueeze(0)
        
        # Extract overlapping regions
        overlap1 = chunk1[..., -overlap_size:]
        overlap2 = chunk2[..., :overlap_size]
        
        logger.debug(f"Crossfading chunks of shapes {chunk1.shape}, {chunk2.shape}")
        
        try:
            # Apply crossfade
            crossfaded = overlap1 * fade_out + overlap2 * fade_in
            result = torch.cat([chunk1[..., :-overlap_size], crossfaded, chunk2[..., overlap_size:]], dim=-1)
            return result
        except RuntimeError as e:
            logger.error(f"Crossfade failed: {str(e)}")
            raise

    def _process_single_chunk(self, chunk: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Process a single chunk and return separated sources and metadata
        
        Args:
            chunk: Audio chunk to process
            
        Returns:
            Tuple[torch.Tensor, Dict]: Processed audio and metadata
            
        Raises:
            ValueError: If chunk has invalid shape
            RuntimeError: If processing fails
        """
        try:
            with torch.inference_mode():
                # Ensure chunk has correct shape (channel, time)
                if chunk.dim() == 1:
                    chunk = chunk.unsqueeze(0)
                elif chunk.dim() > 2:
                    chunk = chunk.squeeze()
                
                if chunk.dim() != 2 or chunk.size(0) != 1:
                    raise ValueError(f"Invalid chunk shape: {chunk.shape}. Expected shape: (1, time)")
                
                logger.debug(f"Processing chunk with shape: {chunk.shape}")
                diarization = self.pipeline({
                    "waveform": chunk,
                    "sample_rate": self.sample_rate
                })
                
                embeddings = self._extract_speaker_embeddings(chunk)
                
                return chunk, {
                    "diarization": diarization,
                    "speaker_embeddings": embeddings
                }
        except Exception as e:
            logger.error(f"Chunk processing failed: {str(e)}")
            raise RuntimeError(f"Chunk processing failed: {str(e)}")

    def _extract_speaker_embeddings(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embeddings from audio waveform
        
        Args:
            waveform: Input audio waveform
            
        Returns:
            torch.Tensor: Speaker embeddings
        """
        with torch.no_grad():
            # TODO: Implement actual speaker embedding extraction
            embeddings = torch.mean(waveform, dim=-1)
            return embeddings

    def _align_speakers_across_chunks(self, current_embeddings: torch.Tensor,
                                    previous_embeddings: torch.Tensor) -> List[int]:
        """
        Align speakers between chunks based on embedding similarity
        
        Args:
            current_embeddings: Current chunk speaker embeddings
            previous_embeddings: Previous chunk speaker embeddings
            
        Returns:
            List[int]: Speaker mapping indices
        """
        from scipy.optimize import linear_sum_assignment
        import numpy as np

        # Move tensors to CPU for numpy operations
        current_embeddings = current_embeddings.cpu()
        previous_embeddings = previous_embeddings.cpu()

        # Ensure embeddings are 2D
        if current_embeddings.dim() == 1:
            current_embeddings = current_embeddings.unsqueeze(0)
        if previous_embeddings.dim() == 1:
            previous_embeddings = previous_embeddings.unsqueeze(0)

        # Compute cosine similarity
        similarity_matrix = torch.nn.functional.cosine_similarity(
            current_embeddings.unsqueeze(1),
            previous_embeddings.unsqueeze(0),
            dim=2
        ).numpy()

        # Handle 1D case
        if similarity_matrix.ndim == 1:
            similarity_matrix = similarity_matrix.reshape(1, -1)

        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        return col_ind.tolist()

    def process_long_audio(self, audio_path: Union[str, Path], output_dir: Union[str, Path]) -> None:
        """
        Process a long audio file by chunking with overlap
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory for output files
            
        Raises:
            RuntimeError: If processing fails
            FileNotFoundError: If input file doesn't exist
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        logger.info(f"Processing audio file: {audio_path}")
        self.progress.update(stage="Loading audio", status="processing")
        
        try:
            # Load and preprocess audio
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
                logger.info(f"Resampled audio from {sr}Hz to {self.sample_rate}Hz")
            
            # Convert to mono if necessary
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                logger.info("Converted stereo audio to mono")
            elif waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            # Calculate chunks
            total_samples = waveform.size(-1)
            num_chunks = max(1, (total_samples - self.overlap_size) // self.hop_size + 1)
            
            logger.info(f"Processing {num_chunks} chunks with {self.overlap_size} samples overlap")
            
            # Initialize storage
            processed_sources = []
            previous_embeddings = None
            
            self.progress.total_chunks = num_chunks
            
            # Process chunks
            with tqdm(total=num_chunks, desc="Processing chunks") as pbar:
                for i in range(num_chunks):
                    self.progress.update(stage=f"Processing chunk {i+1}/{num_chunks}")
                    logger.info(f"Processing chunk {i+1}/{num_chunks}")

                    # Extract chunk
                    start = i * self.hop_size
                    end = min(start + self.chunk_size, total_samples)
                    chunk = waveform[:, start:end].to(self.device)
                    
                    # Pad if necessary
                    if chunk.size(-1) < self.chunk_size:
                        pad_size = self.chunk_size - chunk.size(-1)
                        chunk = torch.nn.functional.pad(chunk, (0, pad_size))
                        logger.debug(f"Padded chunk to shape: {chunk.shape}")
                    
                    # Process chunk
                    sources, metadata = self._process_single_chunk(chunk)
                    
                    # Align speakers if not first chunk
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
                    logger.info(f"Completed chunk {i+1}/{num_chunks}")

                    # Memory management
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            # Save results
            self.progress.update(stage="Saving results", status="saving")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Concatenate all processed chunks
            final_sources = torch.cat(processed_sources, dim=-1)
            logger.info(f"Final sources shape: {final_sources.shape}")
            
            # Save each speaker's audio
            for spk_idx in range(final_sources.size(0)):
                output_path = output_dir / f"speaker_{spk_idx+1}.wav"
                torchaudio.save(
                    str(output_path),
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
            raise RuntimeError(f"Processing failed: {str(e)}")

def progress_handler(progress: ProcessingProgress) -> None:
    """Example progress callback function"""
    print(f"\rProgress: {progress}", end='')

def main() -> int:
    parser = argparse.ArgumentParser(description="Long Audio Speaker Separation")
    parser.add_argument('--input_path', type=str, required=True, help='Path to input audio file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output audio files')
    parser.add_argument('--token', type=str, required=True, help='HuggingFace access token')
    parser.add_argument('--chunk_duration', type=float, default=30.0, help='Duration of each chunk in seconds')
    parser.add_argument('--overlap_duration', type=float, default=5.0, help='Duration of overlap between chunks in seconds')
    parser.add_argument('--model_name', type=str, default="pyannote/speech-separation-ami-1.0", help='Model name')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (cuda or cpu)')

    args = parser.parse_args()

    try:
        separator = LongAudioSpeakerSeparation(
            model_name=args.model_name,
            token=args.token,
            device=args.device,
            sample_rate=16000,
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
    exit(main())
