import os
import argparse
import logging
from typing import Dict, Optional
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from spleeter.separator import Separator

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
    
    This implementation uses the Spleeter library for source separation and implements
    custom vocal enhancement using spectral gating and multi-band processing.
    """
    
    SUPPORTED_FORMATS = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')
    SUPPORTED_MODELS = ('2stems', '4stems', '5stems')
    
    def __init__(
        self,
        model_type: str = '2stems',
        sample_rate: int = 44100,
        bitrate: str = '128k',
        use_multiprocess: bool = True
    ):
        """
        Initialize the AudioSeparator with specified parameters.
        
        Args:
            model_type: Type of separation model ('2stems', '4stems', '5stems')
            sample_rate: Output audio sample rate
            bitrate: Output audio bitrate
            use_multiprocess: Whether to use multiple processes for separation
            
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type. Choose from: {self.SUPPORTED_MODELS}")
        
        self.model_type = model_type
        self.sample_rate = sample_rate
        
        # Initialize the Spleeter separator
        try:
            self.separator = Separator(
                model_type,
                multiprocess=use_multiprocess,
                stft_backend='tensorflow',
                sample_rate=sample_rate,
                bitrate=bitrate
            )
        except Exception as e:
            logger.error(f"Failed to initialize Separator: {str(e)}")
            raise
            
        logger.info(f"Initialized AudioSeparator with {model_type} model")
    
    def validate_audio_file(self, file_path: str) -> None:
        """
        Validate the input audio file exists and has a supported format.
        
        Args:
            file_path: Path to the input audio file
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
        """
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
        """
        Separate the audio file into different stems.
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory to save separated audio files
            filename_prefix: Optional prefix for output filenames
            
        Returns:
            Dictionary containing paths to separated audio files
            
        Raises:
            AudioProcessingError: If separation fails
        """
        try:
            # Validate input file
            self.validate_audio_file(input_path)
            
            # Create output directory
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Add prefix to output path if provided
            base_output_path = output_dir
            if filename_prefix:
                base_output_path = output_dir / filename_prefix
            
            logger.info(f"Starting separation of {os.path.basename(input_path)}")
            
            # Perform separation
            self.separator.separate_to_file(
                input_path,
                str(base_output_path)
            )
            
            # Create dictionary of output paths
            output_files = {
                'vocals': str(output_dir / 'vocals.wav'),
                'accompaniment': str(output_dir / 'accompaniment.wav')
            }
            
            if self.model_type in ('4stems', '5stems'):
                output_files.update({
                    'drums': str(output_dir / 'drums.wav'),
                    'bass': str(output_dir / 'bass.wav'),
                    'other': str(output_dir / 'other.wav')
                })
            
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
        """
        Enhance the separated vocals using multi-band spectral gating.
        
        Args:
            input_path: Path to input vocals file
            output_path: Path to save enhanced vocals
            noise_reduction_strength: Strength of noise reduction (0.0 to 1.0)
            num_bands: Number of frequency bands for processing
            
        Raises:
            ValueError: If parameters are invalid
            AudioProcessingError: If enhancement fails
        """
        if not 0 <= noise_reduction_strength <= 1:
            raise ValueError("noise_reduction_strength must be between 0 and 1")
        
        try:
            logger.info(f"Enhancing vocals: {os.path.basename(input_path)}")
            
            # Load the audio file
            audio, sr = librosa.load(input_path)
            
            # Compute the spectrogram
            D = librosa.stft(audio)
            magnitude, phase = librosa.magphase(D)
            
            # Calculate frequency bands (logarithmically spaced)
            freqs = librosa.fft_frequencies(sr=sr)
            band_edges = np.logspace(
                np.log10(20),
                np.log10(sr/2),
                num_bands + 1
            )
            
            # Process each frequency band
            for i in range(num_bands):
                # Find frequencies in this band
                mask = (freqs >= band_edges[i]) & (freqs < band_edges[i+1])
                
                if np.any(mask):
                    # Calculate threshold for this band
                    band_magnitude = magnitude[mask, :]
                    threshold = np.median(band_magnitude) * noise_reduction_strength
                    
                    # Apply spectral gating
                    magnitude[mask, :] = np.maximum(
                        band_magnitude - threshold,
                        band_magnitude * 0.1  # Retain some minimal signal
                    )
            
            # Reconstruct the signal
            D_enhanced = magnitude * phase
            audio_enhanced = librosa.istft(D_enhanced)
            
            # Normalize audio
            audio_enhanced = librosa.util.normalize(audio_enhanced)
            
            # Save the enhanced audio
            sf.write(output_path, audio_enhanced, sr)
            logger.info(f"Vocal enhancement completed: {os.path.basename(output_path)}")
            
        except Exception as e:
            logger.error(f"Vocal enhancement failed: {str(e)}")
            raise AudioProcessingError(f"Failed to enhance vocals: {str(e)}")
    
    def __del__(self):
        """Cleanup method to free resources"""
        if hasattr(self, 'separator'):
            try:
                self.separator.clear_pool()
            except Exception as e:
                logger.warning(f"Failed to clean up separator: {str(e)}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Audio separation and enhancement tool')
    
    parser.add_argument('--input', required=True,
                      help='Input audio file path')
    parser.add_argument('--output-dir', required=True,
                      help='Output directory path')
    parser.add_argument('--model', default='2stems',
                      choices=['2stems', '4stems', '5stems'],
                      help='Model type for separation')
    parser.add_argument('--prefix', default=None,
                      help='Optional prefix for output filenames')
    parser.add_argument('--enhance-vocals', action='store_true',
                      help='Apply vocal enhancement')
    parser.add_argument('--noise-reduction', type=float, default=0.3,
                      help='Noise reduction strength (0.0 to 1.0)')
    
    return parser.parse_args()

def main():

def main():
    """Main function to demonstrate the audio separation process."""
    try:
        # Initialize the separator
        separator = AudioSeparator(model_type='2stems')
        
        # Define paths using pathlib for safe path handling
        input_file = Path("D:/Movie Dubbing/Extracted/Fast 480 Dirilis.Ertugrul.S01e01-1.mp3")
        output_dir = Path("D:/Movie Dubbing/Audio layers")
        
        # Verify the input file exists before processing
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return
        
        # Perform separation
        separated_files = separator.separate_audio(input_file, output_dir)
        
        # Enhance the vocals
        enhanced_vocals_path = output_dir / 'enhanced_vocals.wav'
        separator.enhance_vocals(
            separated_files['vocals'],
            enhanced_vocals_path,
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