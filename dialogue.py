import os
import torch
import logging
import torchaudio
from pathlib import Path
from typing import Dict, Callable, Optional
from tqdm import tqdm
from dataclasses import dataclass

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


class SpeakerSeparator:
    def __init__(self, 
                 num_speakers: int = 4,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 progress_callback: Optional[Callable[[ProcessingProgress], None]] = None):
        """
        Initialize the speaker separator using TorchAudio's implementation.
        
        Args:
            num_speakers: Number of speakers to separate
            device: Computing device to use ('cuda' or 'cpu')
            progress_callback: Optional callback function for progress updates
        """
        self.device = device
        self.num_speakers = num_speakers
        self.progress_callback = progress_callback
        self.progress = ProcessingProgress()
        self.sample_rate = 16000  # Standard sample rate for speech processing
        
        try:
            self.progress.update(stage="Initializing model", status="loading")
            
            # Initialize model using torch.hub
            self.model = torch.hub.load('pyannote/pyannote-audio', 
                                      'sourceformer',
                                      source='local')
            self.model.to(device)
            
            self.progress.update(status="ready")
            logger.info(f"Initialized Sourceformer model on {device}")
            
            if self.progress_callback:
                self.progress_callback(self.progress)
                
        except Exception as e:
            self.progress.update(status="failed", error_message=str(e))
            logger.error(f"Model initialization failed: {str(e)}")
            if self.progress_callback:
                self.progress_callback(self.progress)
            raise RuntimeError(f"Failed to initialize Sourceformer model: {str(e)}")

    def pad_or_trim(self, audio: torch.Tensor, target_length: int) -> torch.Tensor:
        """Adjust audio length to match target length"""
        current_length = audio.size(-1)
        if current_length > target_length:
            return audio[..., :target_length]
        elif current_length < target_length:
            padding = torch.zeros((*audio.shape[:-1], target_length - current_length), 
                                dtype=audio.dtype, device=audio.device)
            return torch.cat([audio, padding], dim=-1)
        return audio

    def separate_chunk(self, audio_chunk: torch.Tensor) -> torch.Tensor:
        """Separate a single audio chunk"""
        with torch.no_grad():
            # Ensure input is in the correct format (batch, channel, time)
            if audio_chunk.dim() == 1:
                audio_chunk = audio_chunk.unsqueeze(0).unsqueeze(0)
            elif audio_chunk.dim() == 2:
                audio_chunk = audio_chunk.unsqueeze(0)
            
            # Process through the model
            separated = self.model(audio_chunk)
            return separated

    def process_in_chunks(self, audio: torch.Tensor, chunk_size: int = 32000) -> torch.Tensor:
        """Process audio in chunks with overlap"""
        overlap = chunk_size // 2
        hop_size = chunk_size - overlap
        
        num_chunks = (audio.size(-1) - overlap) // hop_size + 1
        separated_chunks = []
        
        self.progress.total_chunks = num_chunks
        self.progress.processed_chunks = 0
        self.progress.update(stage="Processing audio chunks", status="processing")
        
        try:
            pbar = tqdm(total=num_chunks, desc="Processing chunks", unit="chunk")
            
            # Process each chunk with overlap
            for i in range(num_chunks):
                start = i * hop_size
                end = min(start + chunk_size, audio.size(-1))
                
                chunk = audio[..., start:end]
                chunk = self.pad_or_trim(chunk, chunk_size)
                
                # Process chunk
                separated_chunk = self.separate_chunk(chunk)
                
                # Apply crossfade for overlapping regions
                if i > 0:  # Apply fade-in to the beginning
                    fade_in = torch.linspace(0, 1, overlap, device=self.device)
                    separated_chunk[..., :overlap] *= fade_in
                    
                if i < num_chunks - 1:  # Apply fade-out to the end
                    fade_out = torch.linspace(1, 0, overlap, device=self.device)
                    separated_chunk[..., -overlap:] *= fade_out
                
                separated_chunks.append(separated_chunk)
                
                # Update progress
                self.progress.update(chunks_processed=1)
                if self.progress_callback:
                    self.progress_callback(self.progress)
                
                pbar.update(1)
                
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            pbar.close()
            
            # Combine chunks with overlap-add
            output_length = audio.size(-1)
            final_output = torch.zeros((self.num_speakers, output_length), device=self.device)
            
            current_position = 0
            for separated_chunk in separated_chunks:
                chunk_length = min(chunk_size, output_length - current_position)
                final_output[..., current_position:current_position + chunk_length] += \
                    separated_chunk[..., :chunk_length]
                current_position += hop_size
            
            return final_output
            
        except Exception as e:
            self.progress.update(status="failed", error_message=str(e))
            if self.progress_callback:
                self.progress_callback(self.progress)
            logger.error(f"Chunk processing failed: {str(e)}")
            raise RuntimeError(f"Failed to process audio chunks: {str(e)}")
        finally:
            if self.device == 'cuda':
                torch.cuda.empty_cache()

    def separate_speakers(self, input_path: str, output_dir: str) -> Dict[str, str]:
        """Separate speakers from an audio file"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
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
                waveform, sr = torchaudio.load(input_path)
                if sr != self.sample_rate:
                    waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
                
                if waveform.size(0) > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                waveform = waveform.to(self.device)
                
                if waveform.size(-1) == 0:
                    raise ValueError("Empty audio file")
                    
                logger.info(f"Loaded audio: {waveform.size(-1) / self.sample_rate:.2f} seconds")
                
            except Exception as e:
                raise RuntimeError(f"Failed to load audio file: {str(e)}")

            # Process audio in chunks
            separated_signals = self.process_in_chunks(waveform)

            # Save separated audio
            self.progress.update(stage="Saving separated audio", status="saving")
            output_files = {}
            
            for idx in range(separated_signals.shape[0]):
                speaker_audio = separated_signals[idx].cpu()
                output_path = output_dir / f"speaker_{idx + 1}.wav"
                
                try:
                    torchaudio.save(str(output_path), speaker_audio.unsqueeze(0), self.sample_rate)
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
            
        separator = SpeakerSeparator(
            num_speakers=4,
            progress_callback=progress_handler
        )
        
        separated_files = separator.separate_speakers(
            input_path=input_path,
            output_dir=output_dir
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