import os
import torch
import logging
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, Callable, Optional
from tqdm import tqdm
from dataclasses import dataclass
from speechbrain.pretrained import SepformerSeparation
from asteroid.models import ConvTasNet
from asteroid.utils import tensors_to_device

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

    def update(self, chunks_processed: int = 0, stage: Optional[str] = None, status: Optional[str] = None):
        """Update progress tracking information"""
        if chunks_processed:
            self.processed_chunks += chunks_processed
        if stage:
            self.current_stage = stage
        if status:
            self.status = status

    @property
    def progress_percentage(self) -> float:
        """Calculate current progress as a percentage"""
        if self.total_chunks == 0:
            return 0.0
        return (self.processed_chunks / self.total_chunks) * 100

    def __str__(self) -> str:
        """String representation of current progress"""
        return (f"Stage: {self.current_stage} | "
                f"Progress: {self.progress_percentage:.1f}% | "
                f"Status: {self.status}")


class SpeakerSeparator:
    def __init__(self, 
                 model_type: str = 'sepformer', 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 progress_callback: Optional[Callable[[ProcessingProgress], None]] = None):
        """
        Initialize the speaker separator with progress tracking.
        
        Args:
            model_type: Type of separation model ('sepformer' or 'convtasnet')
            device: Computing device to use ('cuda' or 'cpu')
            progress_callback: Optional callback function to receive progress updates
        """
        self.device = device
        self.model_type = model_type
        self.progress_callback = progress_callback
        self.progress = ProcessingProgress()
        
        if model_type not in ['sepformer', 'convtasnet']:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        try:
            self.progress.update(stage="Initializing model", status="loading")
            
            if model_type == 'sepformer':
                self.model = SepformerSeparation.from_hparams(
                    source="speechbrain/sepformer-wsj02mix",
                    savedir="pretrained_models/sepformer-wsj02mix",
                    run_opts={"device": device}
                )
            else:
                self.model = ConvTasNet.from_pretrained("ConvTasNet_LibriMix_sep_clean").to(device)
            
            self.progress.update(status="ready")
            logger.info(f"Initialized {model_type} model on {device}")
            
            if self.progress_callback:
                self.progress_callback(self.progress)
                
        except Exception as e:
            self.progress.update(status="failed", error_message=str(e))
            logger.error(f"Model initialization failed: {str(e)}")
            if self.progress_callback:
                self.progress_callback(self.progress)
            raise RuntimeError(f"Failed to initialize {model_type} model: {str(e)}")

    def pad_or_trim(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """Adjust audio length to match target length"""
        if len(audio) > target_length:
            return audio[:target_length]
        elif len(audio) < target_length:
            return np.pad(audio, (0, target_length - len(audio)))
        return audio

    def separate_chunk(self, audio_chunk: torch.Tensor, sr: int) -> torch.Tensor:
        """Separate a single audio chunk using the model"""
        with torch.no_grad():
            if self.model_type == 'sepformer':
                separated = self.model.separate_batch(audio_chunk.unsqueeze(0))
            else:
                separated = self.model(audio_chunk.unsqueeze(0))
            return separated.squeeze(0)

    def process_in_chunks(self, audio: np.ndarray, sr: int, chunk_size: int) -> np.ndarray:
        """
        Process large audio files in manageable chunks with progress tracking.
        """
        chunk_samples = chunk_size * sr
        num_chunks = (len(audio) + chunk_samples - 1) // chunk_samples
        separated_chunks = []
        
        self.progress.total_chunks = num_chunks
        self.progress.processed_chunks = 0
        self.progress.update(stage="Processing audio chunks", status="processing")
        
        try:
            # Create progress bar
            pbar = tqdm(total=num_chunks, desc="Processing chunks", unit="chunk")
            
            for i in range(num_chunks):
                start = i * chunk_samples
                end = min((i + 1) * chunk_samples, len(audio))
                chunk = torch.tensor(audio[start:end], device=self.device).float()
                
                # Process chunk
                separated_chunk = self.separate_chunk(chunk, sr).cpu().numpy()
                separated_chunks.append(separated_chunk)
                
                # Update progress
                self.progress.update(chunks_processed=1)
                if self.progress_callback:
                    self.progress_callback(self.progress)
                
                # Update progress bar
                pbar.update(1)
                
                # Clear CUDA cache if using GPU
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            pbar.close()
            return np.concatenate(separated_chunks, axis=1)
            
        except Exception as e:
            self.progress.update(status="failed", error_message=str(e))
            if self.progress_callback:
                self.progress_callback(self.progress)
            logger.error(f"Chunk processing failed: {str(e)}")
            raise RuntimeError(f"Failed to process audio chunks: {str(e)}")
        finally:
            if self.device == 'cuda':
                torch.cuda.empty_cache()

    def separate_speakers(self, input_path: str, output_dir: str, num_speakers: int = 2) -> Dict[str, str]:
        """
        Separate speakers with progress tracking.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if num_speakers < 1:
            raise ValueError("Number of speakers must be at least 1")
        
        logger.info(f"Processing: {input_path}")
        self.progress.update(stage="Starting separation", status="initializing")
        
        output_dir = Path(output_dir)
        try:
            output_dir.mkdir(exist_ok=True, parents=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create output directory: {str(e)}")

        try:
            # Load audio
            self.progress.update(stage="Loading audio file", status="loading")
            try:
                audio, sr = librosa.load(input_path, sr=16000)
                if len(audio) == 0:
                    raise ValueError("Empty audio file")
                logger.info(f"Loaded audio: {len(audio) / sr:.2f} seconds")
            except Exception as e:
                raise RuntimeError(f"Failed to load audio file: {str(e)}")

            # Process audio in chunks
            separated_signals = self.process_in_chunks(audio, sr, chunk_size=10)

            # Save separated audio
            self.progress.update(stage="Saving separated audio", status="saving")
            output_files = {}
            
            for idx in range(min(num_speakers, separated_signals.shape[0])):
                speaker_audio = separated_signals[idx]
                output_path = output_dir / f"speaker_{idx + 1}.wav"
                
                try:
                    sf.write(str(output_path), speaker_audio, sr)
                    output_files[f"speaker_{idx + 1}"] = str(output_path)
                    logger.info(f"Saved speaker {idx + 1} to {output_path}")
                except Exception as e:
                    logger.error(f"Failed to save speaker {idx + 1}: {str(e)}")
                    continue

            if not output_files:
                raise RuntimeError("No speaker audio files were successfully saved")

            self.progress.update(stage="Completed", status="success")
            if self.progress_callback:
                self.progress_callback(self.progress)
                
            return output_files

        except Exception as e:
            self.progress.update(status="failed", error_message=str(e))
            if self.progress_callback:
                self.progress_callback(self.progress)
            logger.error(f"Separation failed: {str(e)}")
            raise


def progress_handler(progress: ProcessingProgress):
    """Example progress callback function"""
    print(f"\rProgress: {progress}")


def main():
    separator = None
    try:
        input_path = r"D:\Movie Dubbing\Extracted\Fast 480 Dirilis.Ertugrul.S01e01-1.mp3"
        output_dir = r"D:\Movie Dubbing\Audio layers\speech"
        
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            return 1
            
        # Initialize with progress callback
        separator = SpeakerSeparator(
            model_type='sepformer',
            progress_callback=progress_handler
        )
        
        separated_files = separator.separate_speakers(
            input_path=input_path,
            output_dir=output_dir,
            num_speakers=2
        )
        
        logger.info("Processing completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return 1
    finally:
        if separator and separator.device == 'cuda':
            torch.cuda.empty_cache()


if __name__ == "__main__":
    exit(main())