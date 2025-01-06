import os
import torch
import logging
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Dict
from speechbrain.pretrained import SepformerSeparation
from asteroid.models import ConvTasNet
from asteroid.utils import tensors_to_device

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('speaker_separation.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class SpeakerSeparator:
    def __init__(self, model_type: str = 'sepformer', device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_type = model_type
        
        try:
            if model_type == 'sepformer':
                self.model = SepformerSeparation.from_hparams(
                    source="speechbrain/sepformer-wsj02mix",
                    savedir="pretrained_models/sepformer-wsj02mix",
                    run_opts={"device": device}
                )
            else:
                self.model = ConvTasNet.from_pretrained("ConvTasNet_LibriMix_sep_clean").to(device)
            
            logger.info(f"Initialized {model_type} model on {device}")
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise

    def pad_or_trim(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """Adjust audio length to match target length"""
        if len(audio) > target_length:
            return audio[:target_length]
        elif len(audio) < target_length:
            return np.pad(audio, (0, target_length - len(audio)))
        return audio

    def separate_speakers(
        self, 
        input_path: str, 
        output_dir: str,
        num_speakers: int = 2,
        min_voice_length: float = 0.5
    ) -> Dict[str, str]:
        """
        Separate speakers while maintaining original audio length
        """
        logger.info(f"Processing: {input_path}")
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        try:
            # Load audio
            audio, sr = librosa.load(input_path, sr=16000)
            original_length = len(audio)
            logger.info(f"Loaded audio: {len(audio)/sr:.2f} seconds")

            # Separate speakers
            if self.model_type == 'sepformer':
                est_sources = self.model.separate_file(input_path)
                separated_signals = est_sources.cpu().numpy()
            else:
                with torch.no_grad():
                    waveform = torch.from_numpy(audio).float().to(self.device)
                    separated_signals = self.model(waveform[None])[0].cpu().numpy()

            # Process each speaker
            output_files = {}
            for idx, signal in enumerate(separated_signals[:num_speakers]):
                # Ensure consistent length
                aligned_signal = self.pad_or_trim(signal, original_length)
                
                # Voice activity detection
                voice_intervals = librosa.effects.split(
                    aligned_signal,
                    top_db=20,
                    frame_length=2048,
                    hop_length=512
                )
                
                # Filter short segments
                valid_intervals = [
                    (start, end) for start, end in voice_intervals
                    if (end - start) / sr >= min_voice_length
                ]
                
                if valid_intervals:
                    output_path = output_dir / f"speaker_{idx + 1}.wav"
                    sf.write(str(output_path), aligned_signal, sr)
                    output_files[f"speaker_{idx + 1}"] = str(output_path)
                    logger.info(f"Saved speaker {idx + 1} to {output_path}")

            return output_files

        except Exception as e:
            logger.error(f"Separation failed: {str(e)}")
            raise

def main():
    try:
        separator = SpeakerSeparator(model_type='sepformer')
        
        input_path = "path/to/your/audio.wav"
        output_dir = "path/to/output/directory"
        
        separated_files = separator.separate_speakers(
            input_path=input_path,
            output_dir=output_dir,
            num_speakers=2,
            min_voice_length=0.5
        )
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())