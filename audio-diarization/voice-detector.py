import torch
import torchaudio
import numpy as np
from typing import List, Tuple
import sys
import os

def check_gpu_availability():
    """Check if CUDA is available and print GPU info"""
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your NVIDIA GPU and PyTorch CUDA installation.")
        return False
    
    print(f"GPU available: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")
    return True

def load_silero_vad_model():
    """Load Silero VAD model and move it to GPU"""
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False  # Set to True if you want to always download the latest version
    )
    
    # Move model to GPU
    model = model.cuda()
    
    # Get utility functions
    get_speech_timestamps, read_audio, save_audio, VADIterator, collect_chunks = utils
    
    return model, get_speech_timestamps, read_audio

def preprocess_audio(wav_file: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Load and preprocess audio file for VAD
    Returns tensor on GPU
    """
    try:
        # Load audio file
        waveform, sample_rate = torchaudio.load(wav_file)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
        
        # Move to GPU and ensure correct shape
        waveform = waveform.squeeze().cuda()
        
        return waveform, target_sr
    
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None, None

def detect_voice_activity(model, get_speech_timestamps, audio_tensor: torch.Tensor, 
                         sample_rate: int = 16000, threshold: float = 0.5) -> List[dict]:
    """
    Detect voice activity using Silero VAD
    """
    # Ensure audio is on GPU
    audio_tensor = audio_tensor.cuda()
    
    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        audio_tensor, 
        model, 
        threshold=threshold,
        sampling_rate=sample_rate
    )
    
    return speech_timestamps

def analyze_voice_presence(speech_timestamps: List[dict], audio_duration: float) -> dict:
    """
    Analyze voice presence and return detailed results
    """
    if not speech_timestamps:
        return {
            'has_voice': False,
            'total_voice_duration': 0.0,
            'voice_percentage': 0.0,
            'speech_segments': 0
        }
    
    total_voice_duration = 0.0
    for segment in speech_timestamps:
        segment_duration = (segment['end'] - segment['start']) / 16000  # Convert samples to seconds
        total_voice_duration += segment_duration
    
    voice_percentage = (total_voice_duration / audio_duration) * 100
    
    return {
        'has_voice': True,
        'total_voice_duration': total_voice_duration,
        'voice_percentage': voice_percentage,
        'speech_segments': len(speech_timestamps)
    }

def main(wav_file_path: str, threshold: float = 0.5):
    """
    Main function to detect voice in a WAV file
    """
    print(f"Analyzing: {wav_file_path}")
    print(f"VAD threshold: {threshold}")
    print("-" * 50)
    
    # Check GPU availability
    if not check_gpu_availability():
        sys.exit(1)
    
    try:
        # Load model
        print("Loading Silero VAD model...")
        model, get_speech_timestamps, read_audio = load_silero_vad_model()
        
        # Preprocess audio
        print("Loading and preprocessing audio...")
        audio_tensor, sample_rate = preprocess_audio(wav_file_path)
        
        if audio_tensor is None:
            print("Failed to load audio file.")
            return
        
        # Calculate audio duration
        audio_duration = len(audio_tensor) / sample_rate
        print(f"Audio duration: {audio_duration:.2f} seconds")
        
        # Detect voice activity
        print("Detecting voice activity...")
        speech_timestamps = detect_voice_activity(
            model, get_speech_timestamps, audio_tensor, sample_rate, threshold
        )
        
        # Analyze results
        results = analyze_voice_presence(speech_timestamps, audio_duration)
        
        # Print results
        print("\n" + "="*50)
        print("VOICE DETECTION RESULTS")
        print("="*50)
        print(f"Voice detected: {'YES' if results['has_voice'] else 'NO'}")
        print(f"Total voice duration: {results['total_voice_duration']:.2f} seconds")
        print(f"Voice percentage: {results['voice_percentage']:.2f}%")
        print(f"Number of speech segments: {results['speech_segments']}")
        
        # Print segment details if voice detected
        if results['has_voice']:
            print(f"\nSpeech segments (threshold: {threshold}):")
            for i, segment in enumerate(speech_timestamps):
                start_sec = segment['start'] / 16000
                end_sec = segment['end'] / 16000
                duration = end_sec - start_sec
                print(f"  Segment {i+1}: {start_sec:.2f}s - {end_sec:.2f}s (duration: {duration:.2f}s)")
        
        return results
        
    except Exception as e:
        print(f"Error during voice detection: {e}")
        return None

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python voice_detector.py <path_to_wav_file> [threshold]")
        print("Example: python voice_detector.py audio.wav 0.5")
        sys.exit(1)
    
    wav_file = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    # Verify file exists
    if not os.path.exists(wav_file):
        print(f"Error: File '{wav_file}' not found.")
        sys.exit(1)
    
    # Run voice detection
    results = main(wav_file, threshold)
    
    # Exit with appropriate code
    sys.exit(0 if results and results['has_voice'] else 1)