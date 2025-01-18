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
        if self.total_chunks == 0:
            return 0.0
        return (self.processed_chunks / self.total_chunks) * 100

    def __str__(self) -> str:
        progress_str = (f"Stage: {self.current_stage} | "
                       f"Progress: {self.progress_percentage:.1f}% | "
                       f"Status: {self.status}")
        if self.error_message:
            progress_str += f" | Error: {self.error_message}"
        return progress_str

class LongAudioSpeakerSeparation:
    def __init__(self, token: str, device: str = 'cuda',
                 sample_rate: int = 16000, chunk_duration: float = 30.0,
                 overlap_duration: float = 5.0, progress_callback: Optional[Callable] = None):
        """
        Initialize the speaker separation system using the dynamic PyAnnote pipeline.
        """
        if overlap_duration >= chunk_duration:
            raise ValueError("Overlap duration must be less than chunk duration")
            
        self.token = token
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.sample_rate = sample_rate
        self.progress_callback = progress_callback
        self.progress = ProcessingProgress()
        
        logger.info(f"Initializing with device: {self.device}")
        
        try:
            # Initialize the dynamic speaker diarization pipeline
            self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)
            logger.info(f"Dynamic speaker diarization pipeline loaded on {self.device}")
            
            self.chunk_duration = chunk_duration
            self.overlap_duration = overlap_duration
            self.chunk_size = int(self.chunk_duration * self.sample_rate)
            self.overlap_size = int(self.overlap_duration * self.sample_rate)
            self.hop_size = self.chunk_size - self.overlap_size
            
            logger.info(f"Initialized with chunk duration: {self.chunk_duration}s, overlap: {self.overlap_duration}s")
            
        except Exception as e:
            logger.error(f"Pipeline loading failed: {str(e)}")
            raise RuntimeError(f"Pipeline loading failed: {str(e)}")

    def process_long_audio(self, audio_path: Union[str, Path], output_dir: Union[str, Path]) -> None:
        """
        Process a long audio file using the dynamic speaker diarization pipeline.
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        logger.info(f"Processing audio file: {audio_path}")
        self.progress.update(stage="Loading audio", status="processing")
        
        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
                logger.info(f"Resampled audio from {sr}Hz to {self.sample_rate}Hz")
            
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                logger.info("Converted stereo audio to mono")
            elif waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            total_samples = waveform.size(-1)
            num_chunks = max(1, (total_samples - self.overlap_size) // self.hop_size + 1)
            
            logger.info(f"Processing {num_chunks} chunks with {self.overlap_size} samples overlap")
            
            self.progress.total_chunks = num_chunks
            
            with tqdm(total=num_chunks, desc="Processing chunks") as pbar:
                for i in range(num_chunks):
                    self.progress.update(stage=f"Processing chunk {i+1}/{num_chunks}")
                    logger.info(f"Processing chunk {i+1}/{num_chunks}")

                    start = i * self.hop_size
                    end = min(start + self.chunk_size, total_samples)
                    chunk = waveform[:, start:end].to(self.device)
                    
                    if chunk.size(-1) < self.chunk_size:
                        pad_size = self.chunk_size - chunk.size(-1)
                        chunk = torch.nn.functional.pad(chunk, (0, pad_size))
                        logger.debug(f"Padded chunk to shape: {chunk.shape}")
                    
                    diarization = self.pipeline({
                        "waveform": chunk,
                        "sample_rate": self.sample_rate
                    })
                    logger.debug(f"Diarization result: {diarization}")

                    # Save results
                    self.progress.update(stage="Saving results", status="saving")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"chunk_{i+1}.rttm"
                    diarization.write_rttm(output_path)
                    logger.info(f"Saved diarization results to {output_path}")
                    
                    self.progress.update(chunks_processed=1)
                    if self.progress_callback:
                        self.progress_callback(self.progress)
                    
                    pbar.update(1)
                    logger.info(f"Completed chunk {i+1}/{num_chunks}")

                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
            
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
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (cuda or cpu)')

    args = parser.parse_args()

    try:
        separator = LongAudioSpeakerSeparation(
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
