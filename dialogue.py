import os
import torch
import logging
import torchaudio
from pathlib import Path
from typing import Dict, Callable, Optional
from tqdm import tqdm
from dataclasses import dataclass
import argparse

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

from pyannote.audio import Pipeline

import os
import torch
from pyannote.audio import Pipeline
import scipy.io.wavfile
import torchaudio
from typing import Tuple

class SpeakerSeparationPipeline:
    def __init__(self, model_name: str, token: str, device: str = 'cuda'):
        """
        Initialize the PyAnnote speech separation pipeline.
        
        Args:
            model_name: Hugging Face model identifier.
            token: Hugging Face access token.
            device: Device to run the model on ('cuda' or 'cpu').
        """
        self.pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.pipeline.to(self.device)
        print(f"Model loaded on {self.device}")

    def process_audio_file(self, audio_file_path: str) -> Tuple:
        """
        Process an audio file for speaker diarization and separation.
        
        Args:
            audio_file_path: Path to the input audio file.
        
        Returns:
            diarization: Diarization output.
            sources: Separated audio sources.
        """
        waveform, sample_rate = torchaudio.load(audio_file_path)
        diarization, sources = self.pipeline({"waveform": waveform, "sample_rate": sample_rate})
        return diarization, sources

    def save_outputs(self, diarization, sources, output_dir: str):
        """
        Save diarization and separated sources to disk.
        
        Args:
            diarization: Diarization output.
            sources: Separated audio sources.
            output_dir: Directory to save output files.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save diarization in RTTM format
        diarization_file = os.path.join(output_dir, "diarization.rttm")
        with open(diarization_file, "w") as rttm:
            diarization.write_rttm(rttm)
        print(f"Diarization saved to {diarization_file}")

        # Save separated sources
        for s, speaker in enumerate(diarization.labels()):
            source_file = os.path.join(output_dir, f"{speaker}.wav")
            scipy.io.wavfile.write(source_file, 16000, sources.data[:, s])
            print(f"Speaker {s} source saved to {source_file}")




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
            if audio_chunk.dim() == 1:
                audio_chunk = audio_chunk.unsqueeze(0).unsqueeze(0)
            elif audio_chunk.dim() == 2:
                audio_chunk = audio_chunk.unsqueeze(0)
            return self.model(audio_chunk)

    def process_in_chunks(self, audio: torch.Tensor, chunk_size: int = 80000) -> Dict[str, torch.Tensor]:
        """Process audio in 5-second chunks with overlap"""
        hop_size = chunk_size  # No overlap for 5s chunks
        num_chunks = (audio.size(-1) + hop_size - 1) // hop_size
        diarization_results = []
        separation_results = []
        
        self.progress.total_chunks = num_chunks
        self.progress.processed_chunks = 0
        self.progress.update(stage="Processing audio chunks", status="processing")

        try:
            pbar = tqdm(total=num_chunks, desc="Processing chunks", unit="chunk")
            for i in range(num_chunks):
                start = i * hop_size
                end = min(start + chunk_size, audio.size(-1))
                chunk = audio[..., start:end]
                chunk = self.pad_or_trim(chunk, chunk_size)

                with torch.inference_mode():
                    diarization, sources = self.model(chunk.unsqueeze(0))  # Batch dimension added
                    diarization_results.append(diarization)
                    separation_results.append(sources.squeeze(0))  # Remove batch dimension

                self.progress.update(chunks_processed=1)
                if self.progress_callback:
                    self.progress_callback(self.progress)
                pbar.update(1)

                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            pbar.close()

            return {
                'diarization': torch.cat(diarization_results, dim=1),  # Concatenate along time axis
                'sources': torch.cat(separation_results, dim=0)       # Concatenate along sample axis
            }

        except Exception as e:
            self.progress.update(status="failed", error_message=str(e))
            logger.error("Chunk processing failed", exc_info=True)
            if self.progress_callback:
                self.progress_callback(self.progress)
            raise RuntimeError("Failed to process audio chunks") from e


    def separate_speakers(self, input_path: str, output_dir: str) -> Dict[str, str]:
        """Separate speakers from an audio file"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        logger.info(f"Processing: {input_path}")
        self.progress.update(stage="Starting separation", status="initializing")
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        try:
            waveform, sr = torchaudio.load(input_path)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            waveform = waveform.to(self.device)

            if waveform.size(-1) == 0:
                raise ValueError("Empty audio file")
            separated_signals = self.process_in_chunks(waveform)

            output_files = {}
            for idx in range(separated_signals.shape[0]):
                speaker_audio = separated_signals[idx].cpu()
                output_path = output_dir / f"speaker_{idx + 1}.wav"
                torchaudio.save(str(output_path), speaker_audio.unsqueeze(0), self.sample_rate)
                output_files[f"speaker_{idx + 1}"] = str(output_path)
                logger.info(f"Saved speaker {idx + 1} to {output_path}")

            if not output_files:
                raise RuntimeError("No speaker audio files were successfully saved")

            self.progress.update(stage="Completed", status="success")
            if self.progress_callback:
                self.progress_callback(self.progress)
            return output_files

        except Exception as e:
            self.progress.update(status="failed", error_message=str(e))
            logger.error("Separation failed", exc_info=True)
            if self.progress_callback:
                self.progress_callback(self.progress)
            raise RuntimeError("Separation failed") from e


def progress_handler(progress: ProcessingProgress):
    """Example progress callback function"""
    print(f"\rProgress: {progress}", end='')


def main():
    parser = argparse.ArgumentParser(description="Speaker Separation Tool")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input audio file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the separated audio files')
    parser.add_argument('--num_speakers', type=int, default=4, help='Number of speakers to separate')
    args = parser.parse_args()

    separator = None
    try:
        separator = SpeakerSeparationPipeline(
            num_speakers=args.num_speakers,
            progress_callback=progress_handler
        )
        separator.separate_speakers(
            input_path=args.input_path,
            output_dir=args.output_dir
        )
        logger.info("Processing completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return 1
    finally:
        if separator and separator.device == 'cuda':
            torch.cuda.empty_cache()
