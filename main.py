import os
import argparse
import logging
import torch
from typing import Dict, Optional
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from demucs.pretrained import get_model
from demucs.apply import apply_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors"""
    pass

class AudioSeparator:
    """
    A class for separating audio files into different stems and enhancing vocal tracks.
    Uses Demucs for source separation and implements custom vocal enhancement.
    """
    
    SUPPORTED_FORMATS = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')
    SUPPORTED_MODELS = ('htdemucs', 'htdemucs_ft', 'mdx_extra', 'mdx_extra_q')
    
    def __init__(
        self,
        model_name: str = 'htdemucs',
        sample_rate: int = 44100,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the AudioSeparator with specified parameters.
        
        Args:
            model_name: Type of separation model
            sample_rate: Output audio sample rate
            device: Device to run separation on ('cuda' or 'cpu')
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type. Choose from: {self.SUPPORTED_MODELS}")
        
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.device = device
        
        try:
            self.model = get_model(model_name)
            self.model.to(device)
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
            
        logger.info(f"Initialized AudioSeparator with {model_name} model on {device}")
    
    def validate_audio_file(self, file_path: str) -> None:
        """Validate the input audio file exists and has a supported format."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
            
        if not any(file_path.lower().endswith(fmt) for fmt in self.SUPPORTED_FORMATS):
            raise ValueError(f"Unsupported audio format. Supported formats: {self.SUPPORTED_FORMATS}")
    
    def separate_audio(
        self,
        input_path: str,
        output_dir: str,
        filename_prefix: Optional[str] = None
    ) -> Dict[str, str]:
        """Separate the audio file into different stems."""
        try:
            self.validate_audio_file(input_path)
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load audio
            audio, sr = librosa.load(input_path, sr=self.sample_rate, mono=False)
            if audio.ndim == 1:
                audio = audio[None]  # Add channel dimension
            
            # Convert to torch tensor
            audio_tensor = torch.tensor(audio)[None]  # Add batch dimension
            
            # Separate
            sources = apply_model(self.model, audio_tensor, device=self.device)[0]
            sources = sources.cpu().numpy()
            
            # Save separated tracks
            stem_names = ['vocals', 'drums', 'bass', 'other']
            output_files = {}
            
            for source, name in zip(sources, stem_names):
                output_path = output_dir / f"{filename_prefix + '_' if filename_prefix else ''}{name}.wav"
                sf.write(str(output_path), source.T, self.sample_rate)
                output_files[name] = str(output_path)
            
            logger.info(f"Separation completed: {len(output_files)} stems created")
            return output_files
            
        except Exception as e:
            logger.error(f"Separation failed: {str(e)}")
            raise AudioProcessingError(f"Failed to separate audio: {str(e)}")
    
    @staticmethod
    def enhance_vocals(
        input_path: str,
        output_path: str,
        noise_reduction_strength: float = 0.3,
        num_bands: int = 4
    ) -> None:
        """Enhance the separated vocals using multi-band spectral gating."""
        if not 0 <= noise_reduction_strength <= 1:
            raise ValueError("noise_reduction_strength must be between 0 and 1")
        
        try:
            logger.info(f"Enhancing vocals: {os.path.basename(input_path)}")
            
            audio, sr = librosa.load(input_path)
            D = librosa.stft(audio)
            magnitude, phase = librosa.magphase(D)
            
            freqs = librosa.fft_frequencies(sr=sr)
            band_edges = np.logspace(
                np.log10(20),
                np.log10(sr/2),
                num_bands + 1
            )
            
            for i in range(num_bands):
                mask = (freqs >= band_edges[i]) & (freqs < band_edges[i+1])
                
                if np.any(mask):
                    band_magnitude = magnitude[mask, :]
                    threshold = np.median(band_magnitude) * noise_reduction_strength
                    magnitude[mask, :] = np.maximum(
                        band_magnitude - threshold,
                        band_magnitude * 0.1
                    )
            
            D_enhanced = magnitude * phase
            audio_enhanced = librosa.istft(D_enhanced)
            audio_enhanced = librosa.util.normalize(audio_enhanced)
            
            sf.write(output_path, audio_enhanced, sr)
            logger.info(f"Vocal enhancement completed: {os.path.basename(output_path)}")
            
        except Exception as e:
            logger.error(f"Vocal enhancement failed: {str(e)}")
            raise AudioProcessingError(f"Failed to enhance vocals: {str(e)}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Audio separation and enhancement tool')
    
    parser.add_argument('--input', required=True,
                      help='Input audio file path')
    parser.add_argument('--output-dir', required=True,
                      help='Output directory path')
    parser.add_argument('--model', default='htdemucs',
                      choices=['htdemucs', 'htdemucs_ft', 'mdx_extra', 'mdx_extra_q'],
                      help='Model type for separation')
    parser.add_argument('--prefix', default=None,
                      help='Optional prefix for output filenames')
    parser.add_argument('--enhance-vocals', action='store_true',
                      help='Apply vocal enhancement')
    parser.add_argument('--noise-reduction', type=float, default=0.3,
                      help='Noise reduction strength (0.0 to 1.0)')
    
    return parser.parse_args()

def main():
    try:
        separator = AudioSeparator(model_name='htdemucs')
        
        input_file = Path("D:/Movie Dubbing/Extracted/Fast 480 Dirilis.Ertugrul.S01e01-1.mp3")
        output_dir = Path("D:/Movie Dubbing/Audio layers")
        
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return
        
        separated_files = separator.separate_audio(str(input_file), str(output_dir))
        
        enhanced_vocals_path = output_dir / 'enhanced_vocals.wav'
        separator.enhance_vocals(
            separated_files['vocals'],
            str(enhanced_vocals_path),
            noise_reduction_strength=0.3
        )
        
        logger.info("Processing completed successfully!")
        logger.info(f"Enhanced vocals saved to: {enhanced_vocals_path}")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
