import os
import argparse
import logging
import torch
from typing import Dict, Optional
from pathlib import Path
import time
import numpy as np
import librosa
import soundfile as sf
from demucs.pretrained import get_model
from demucs.apply import apply_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('audio_processing.log')
    ]
)
logger = logging.getLogger(__name__)

class AudioProcessingError(Exception):
    pass

class AudioSeparator:
    SUPPORTED_FORMATS = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')
    SUPPORTED_MODELS = ('htdemucs', 'htdemucs_ft', 'mdx_extra', 'mdx_extra_q')
    
    def __init__(
        self,
        model_name: str = 'htdemucs',
        sample_rate: int = 44100,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        start_time = time.time()
        logger.info(f"Initializing AudioSeparator with model: {model_name} on {device}")
        
        if model_name not in self.SUPPORTED_MODELS:
            logger.error(f"Invalid model: {model_name}")
            raise ValueError(f"Unsupported model type. Choose from: {self.SUPPORTED_MODELS}")
        
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.device = device
        
        try:
            model_load_start = time.time()
            self.model = get_model(model_name)
            self.model.to(device)
            logger.info(f"Model loaded in {time.time() - model_load_start:.2f} seconds")
        except Exception as e:
            logger.exception("Model initialization failed")
            raise
            
        logger.info(f"AudioSeparator initialized in {time.time() - start_time:.2f} seconds")
    
    def validate_audio_file(self, file_path: str) -> None:
        logger.debug(f"Validating file: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Input file not found: {file_path}")
            
        if not any(file_path.lower().endswith(fmt) for fmt in self.SUPPORTED_FORMATS):
            logger.error(f"Invalid format: {os.path.splitext(file_path)[1]}")
            raise ValueError(f"Unsupported audio format. Supported: {self.SUPPORTED_FORMATS}")
        
        logger.debug("File validation successful")
    
    def separate_audio(
        self,
        input_path: str,
        output_dir: str,
        filename_prefix: Optional[str] = None
    ) -> Dict[str, str]:
        start_time = time.time()
        logger.info(f"Starting separation of {os.path.basename(input_path)}")
        
        try:
            self.validate_audio_file(input_path)
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Output directory created: {output_dir}")
            
            load_start = time.time()
            audio, sr = librosa.load(input_path, sr=self.sample_rate, mono=False)
            logger.info(f"Audio loaded in {time.time() - load_start:.2f} seconds")
            
            if audio.ndim == 1:
                audio = audio[None]
            
            process_start = time.time()
            audio_tensor = torch.tensor(audio)[None]
            sources = apply_model(self.model, audio_tensor, device=self.device)[0]
            sources = sources.cpu().numpy()
            logger.info(f"Separation completed in {time.time() - process_start:.2f} seconds")
            
            stem_names = ['vocals', 'drums', 'bass', 'other']
            output_files = {}
            
            save_start = time.time()
            for source, name in zip(sources, stem_names):
                output_path = output_dir / f"{filename_prefix + '_' if filename_prefix else ''}{name}.wav"
                sf.write(str(output_path), source.T, self.sample_rate)
                output_files[name] = str(output_path)
                logger.info(f"Saved {name} to {output_path}")
            
            logger.info(f"All stems saved in {time.time() - save_start:.2f} seconds")
            logger.info(f"Total separation time: {time.time() - start_time:.2f} seconds")
            
            return output_files
            
        except Exception as e:
            logger.exception("Separation failed")
            raise AudioProcessingError(f"Failed to separate audio: {str(e)}")
    
    @staticmethod
    def enhance_vocals(
        input_path: str,
        output_path: str,
        noise_reduction_strength: float = 0.3,
        num_bands: int = 4
    ) -> None:
        start_time = time.time()
        logger.info(f"Starting vocal enhancement: {os.path.basename(input_path)}")
        
        if not 0 <= noise_reduction_strength <= 1:
            logger.error(f"Invalid noise reduction strength: {noise_reduction_strength}")
            raise ValueError("noise_reduction_strength must be between 0 and 1")
        
        try:
            load_start = time.time()
            audio, sr = librosa.load(input_path)
            logger.info(f"Audio loaded in {time.time() - load_start:.2f} seconds")
            
            process_start = time.time()
            D = librosa.stft(audio)
            magnitude, phase = librosa.magphase(D)
            
            freqs = librosa.fft_frequencies(sr=sr)
            band_edges = np.logspace(np.log10(20), np.log10(sr/2), num_bands + 1)
            
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
            logger.info(f"Enhancement processing completed in {time.time() - process_start:.2f} seconds")
            
            save_start = time.time()
            sf.write(output_path, audio_enhanced, sr)
            logger.info(f"Enhanced audio saved in {time.time() - save_start:.2f} seconds")
            
            logger.info(f"Total enhancement time: {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.exception("Vocal enhancement failed")
            raise AudioProcessingError(f"Failed to enhance vocals: {str(e)}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Audio separation and enhancement tool')
    
    parser.add_argument('--input', required=True, help='Input audio file path')
    parser.add_argument('--output-dir', required=True, help='Output directory path')
    parser.add_argument('--model', default='htdemucs',
                      choices=['htdemucs', 'htdemucs_ft', 'mdx_extra', 'mdx_extra_q'],
                      help='Model type for separation')
    parser.add_argument('--prefix', default=None, help='Optional prefix for output filenames')
    parser.add_argument('--enhance-vocals', action='store_true', help='Apply vocal enhancement')
    parser.add_argument('--noise-reduction', type=float, default=0.3,
                      help='Noise reduction strength (0.0 to 1.0)')
    
    return parser.parse_args()

def main():
    start_time = time.time()
    logger.info("Starting audio processing pipeline")
    
    try:
        init_start = time.time()
        separator = AudioSeparator(model_name='htdemucs')
        logger.info(f"Separator initialized in {time.time() - init_start:.2f} seconds")
        
        input_file = Path("D:/Movie Dubbing/Extracted/Fast 480 Dirilis.Ertugrul.S01e01-1.mp3")
        output_dir = Path("D:/Movie Dubbing/Audio layers")
        
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            return 1
        
        separation_start = time.time()
        separated_files = separator.separate_audio(str(input_file), str(output_dir))
        logger.info(f"Separation completed in {time.time() - separation_start:.2f} seconds")
        
        enhancement_start = time.time()
        enhanced_vocals_path = output_dir / 'enhanced_vocals.wav'
        separator.enhance_vocals(
            separated_files['vocals'],
            str(enhanced_vocals_path),
            noise_reduction_strength=0.3
        )
        logger.info(f"Enhancement completed in {time.time() - enhancement_start:.2f} seconds")
        
        total_time = time.time() - start_time
        logger.info(f"Total processing pipeline completed in {total_time:.2f} seconds")
        
    except Exception as e:
        logger.exception("Main processing pipeline failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())