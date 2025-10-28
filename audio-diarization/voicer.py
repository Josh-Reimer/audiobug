import torch
import torchaudio
import numpy as np
from typing import List, Tuple, Optional
import sys
import os
from pathlib import Path

def check_gpu_availability():
    """Check if CUDA is available and print GPU info"""
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your NVIDIA GPU and PyTorch CUDA installation.")
        return False
    
    print(f"GPU available: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")
    return True

def load_silero_vad_model():
    """Load Silero VAD model and utility functions"""
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False
    )
    
    # Move model to GPU
    model = model.cuda()
    
    # Get utility functions
    get_speech_timestamps, read_audio, save_audio, VADIterator, collect_chunks = utils
    
    return model, get_speech_timestamps, read_audio, VADIterator, collect_chunks

def preprocess_audio(wav_file: str, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """
    Load and preprocess audio file for VAD
    Returns tensor on GPU and original sample rate
    """
    try:
        # Load audio file
        waveform, original_sample_rate = torchaudio.load(wav_file)
        
        # Store original for saving clips
        original_waveform = waveform.clone()
        
        # Convert to mono if stereo for VAD processing
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary for VAD
        if original_sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
        
        # Move to GPU and ensure correct shape
        waveform = waveform.squeeze().cuda()
        
        return waveform, original_sample_rate, original_waveform
    
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None, None, None

def ensure_voices_folder():
    """Create voices folder if it doesn't exist"""
    voices_dir = Path("voices")
    voices_dir.mkdir(exist_ok=True)
    return voices_dir

def detect_sentences_with_eos(audio_tensor: torch.Tensor, 
                             model, 
                             get_speech_timestamps,
                             eos_pause_ms: int = 800,
                             threshold: float = 0.3) -> List[dict]:
    """
    Detect complete sentences using Silero VAD with end-of-sentence pause detection
    """
    # Ensure audio is on GPU
    audio_tensor = audio_tensor.cuda()
    
    # Use batch processing with end-of-sentence pause detection
    speech_timestamps = get_speech_timestamps(
        audio_tensor, 
        model, 
        threshold=threshold,
        sampling_rate=16000,
        min_speech_duration_ms=300,      # Minimum speech duration
        min_silence_duration_ms=eos_pause_ms,  # Pause duration for EOS
        speech_pad_ms=300,               # Padding around speech
        window_size_samples=512,         # Processing window
        return_seconds=False
    )
    
    return speech_timestamps

def merge_sentence_segments(segments: List[dict], max_gap_ms: int = 300) -> List[dict]:
    """
    Merge segments that are close together (likely same sentence with brief pauses)
    """
    if not segments:
        return []
    
    max_gap_samples = int(max_gap_ms * 16000 / 1000)  # Convert to samples
    
    merged_segments = []
    current_segment = segments[0].copy()
    
    for i in range(1, len(segments)):
        current_end = current_segment['end']
        next_start = segments[i]['start']
        next_end = segments[i]['end']
        
        # If the gap is small, merge segments (brief pause within sentence)
        if next_start - current_end <= max_gap_samples:
            current_segment['end'] = next_end
        else:
            # Significant gap - treat as separate sentences
            merged_segments.append(current_segment)
            current_segment = segments[i].copy()
    
    # Add the last segment
    merged_segments.append(current_segment)
    
    return merged_segments

def add_context_to_sentences(segments: List[dict], audio_length: int,
                           pre_context_ms: int = 200, post_context_ms: int = 500) -> List[dict]:
    """
    Add contextual padding to sentence segments
    """
    pre_context_samples = int(pre_context_ms * 16000 / 1000)
    post_context_samples = int(post_context_ms * 16000 / 1000)
    
    contextual_segments = []
    
    for segment in segments:
        contextual_segment = segment.copy()
        
        # Add pre-context (before sentence starts)
        contextual_segment['start'] = max(0, segment['start'] - pre_context_samples)
        
        # Add post-context (after sentence ends)
        contextual_segment['end'] = min(audio_length, segment['end'] + post_context_samples)
        
        contextual_segments.append(contextual_segment)
    
    return contextual_segments

def filter_short_sentences(segments: List[dict], min_duration_ms: int = 800) -> List[dict]:
    """
    Filter out very short segments that are unlikely to be complete sentences
    """
    min_duration_samples = int(min_duration_ms * 16000 / 1000)
    
    return [seg for seg in segments if (seg['end'] - seg['start']) >= min_duration_samples]

def save_sentence_clips(original_waveform: torch.Tensor, sample_rate: int, 
                       sentence_segments: List[dict], audio_file_path: str) -> List[str]:
    """
    Save complete sentence clips to the voices folder
    """
    voices_dir = ensure_voices_folder()
    audio_filename = Path(audio_file_path).stem
    saved_files = []
    
    # Convert timestamps from VAD sample rate (16kHz) to original sample rate
    conversion_ratio = sample_rate / 16000
    
    for i, sentence in enumerate(sentence_segments):
        try:
            # Convert sentence boundaries to original sample rate
            start_sample = int(sentence['start'] * conversion_ratio)
            end_sample = int(sentence['end'] * conversion_ratio)
            
            # Ensure we don't exceed audio length
            start_sample = min(start_sample, original_waveform.shape[-1] - 1)
            end_sample = min(end_sample, original_waveform.shape[-1])
            
            # Extract the audio segment
            sentence_audio = original_waveform[:, start_sample:end_sample]
            
            # Generate filename with duration info
            start_time = sentence['start'] / 16000
            end_time = sentence['end'] / 16000
            duration = end_time - start_time
            output_filename = f"{voices_dir}/{audio_filename}_sentence_{i+1:03d}_{duration:.1f}s.wav"
            
            # Save the sentence using torchaudio.save
            torchaudio.save(output_filename, sentence_audio, sample_rate)
            saved_files.append(output_filename)
            
            print(f"  Sentence {i+1}: {start_time:.2f}s - {end_time:.2f}s (duration: {duration:.2f}s)")
            
        except Exception as e:
            print(f"  Error saving sentence {i+1}: {e}")
    
    return saved_files

def analyze_sentences(sentence_segments: List[dict], audio_duration: float) -> dict:
    """
    Analyze sentence detection results
    """
    if not sentence_segments:
        return {
            'has_speech': False,
            'total_speech_duration': 0.0,
            'speech_percentage': 0.0,
            'sentence_count': 0,
            'avg_sentence_duration': 0.0
        }
    
    total_speech_duration = 0.0
    sentence_durations = []
    
    for sentence in sentence_segments:
        duration = (sentence['end'] - sentence['start']) / 16000
        total_speech_duration += duration
        sentence_durations.append(duration)
    
    speech_percentage = (total_speech_duration / audio_duration) * 100
    avg_sentence_duration = np.mean(sentence_durations) if sentence_durations else 0
    
    return {
        'has_speech': True,
        'total_speech_duration': total_speech_duration,
        'speech_percentage': speech_percentage,
        'sentence_count': len(sentence_segments),
        'avg_sentence_duration': avg_sentence_duration,
        'sentence_durations': sentence_durations
    }

def main(wav_file_path: str, eos_pause_ms: int = 800, min_sentence_duration_ms: int = 800,
         merge_gap_ms: int = 300, pre_context_ms: int = 200, post_context_ms: int = 500,
         threshold: float = 0.3):
    """
    Main function to detect complete sentences in a WAV file
    """
    print(f"Analyzing: {wav_file_path}")
    print(f"End-of-sentence pause: {eos_pause_ms}ms")
    print(f"Minimum sentence duration: {min_sentence_duration_ms}ms")
    print(f"Merge gap (within sentence): {merge_gap_ms}ms")
    print(f"Pre-context: {pre_context_ms}ms, Post-context: {post_context_ms}ms")
    print(f"VAD threshold: {threshold}")
    print("-" * 50)
    
    # Check GPU availability
    if not check_gpu_availability():
        sys.exit(1)
    
    try:
        # Load model
        print("Loading Silero VAD model...")
        model, get_speech_timestamps, read_audio, VADIterator, collect_chunks = load_silero_vad_model()
        
        # Preprocess audio
        print("Loading and preprocessing audio...")
        audio_tensor, original_sample_rate, original_waveform = preprocess_audio(wav_file_path)
        
        if audio_tensor is None:
            print("Failed to load audio file.")
            return
        
        # Calculate audio duration
        audio_length_samples = len(audio_tensor)
        audio_duration = audio_length_samples / 16000
        print(f"Audio duration: {audio_duration:.2f} seconds")
        print(f"Original sample rate: {original_sample_rate} Hz")
        
        # Detect speech segments with end-of-sentence detection
        print("Detecting sentences with EOS...")
        speech_segments = detect_sentences_with_eos(
            audio_tensor, model, get_speech_timestamps, eos_pause_ms, threshold
        )
        
        print(f"Initial speech segments detected: {len(speech_segments)}")
        
        # Process segments to form complete sentences
        if speech_segments:
            # Merge brief pauses within sentences
            merged_segments = merge_sentence_segments(speech_segments, merge_gap_ms)
            print(f"After merging brief pauses: {len(merged_segments)} segments")
            
            # Filter out very short segments
            sentence_segments = filter_short_sentences(merged_segments, min_sentence_duration_ms)
            print(f"After duration filtering: {len(sentence_segments)} sentences")
            
            # Add contextual padding
            final_sentences = add_context_to_sentences(sentence_segments, audio_length_samples, 
                                                     pre_context_ms, post_context_ms)
            print(f"Final sentences after context padding: {len(final_sentences)}")
        else:
            final_sentences = []
        
        # Save sentence clips
        saved_files = []
        if final_sentences:
            print(f"\nSaving complete sentences to 'voices' folder...")
            saved_files = save_sentence_clips(
                original_waveform, 
                original_sample_rate, 
                final_sentences, 
                wav_file_path
            )
            print(f"Successfully saved {len(saved_files)} complete sentences")
        
        # Analyze and print results
        results = analyze_sentences(final_sentences, audio_duration)
        
        print("\n" + "="*50)
        print("SENTENCE DETECTION RESULTS")
        print("="*50)
        print(f"Speech detected: {'YES' if results['has_speech'] else 'NO'}")
        if results['has_speech']:
            print(f"Number of complete sentences: {results['sentence_count']}")
            print(f"Total speech duration: {results['total_speech_duration']:.2f}s")
            print(f"Speech percentage: {results['speech_percentage']:.2f}%")
            print(f"Average sentence duration: {results['avg_sentence_duration']:.2f}s")
            
            # Print sentence statistics
            if results['sentence_durations']:
                print(f"Sentence duration range: {min(results['sentence_durations']):.2f}s - {max(results['sentence_durations']):.2f}s")
        else:
            print("No complete sentences detected in the audio file.")
        
        results['saved_clips'] = saved_files
        return results
        
    except Exception as e:
        print(f"Error during sentence detection: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python sentence_detector.py <path_to_wav_file> [eos_pause_ms] [min_sentence_duration_ms] [merge_gap_ms] [pre_context_ms] [post_context_ms] [threshold]")
        print("Example: python sentence_detector.py audio.wav 800 800 300 200 500 0.3")
        print("\nParameters:")
        print("  eos_pause_ms: Pause duration to detect end-of-sentence (default: 800)")
        print("  min_sentence_duration_ms: Minimum sentence length (default: 800)")
        print("  merge_gap_ms: Max gap to merge within sentence (default: 300)")
        print("  pre_context_ms: Padding before sentence (default: 200)")
        print("  post_context_ms: Padding after sentence (default: 500)")
        print("  threshold: VAD sensitivity 0.0-1.0 (default: 0.3)")
        sys.exit(1)
    
    wav_file = sys.argv[1]
    eos_pause_ms = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    min_sentence_duration_ms = int(sys.argv[3]) if len(sys.argv) > 3 else 800
    merge_gap_ms = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    pre_context_ms = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    post_context_ms = int(sys.argv[6]) if len(sys.argv) > 6 else 500
    threshold = float(sys.argv[7]) if len(sys.argv) > 7 else 0.3
    
    # Verify file exists
    if not os.path.exists(wav_file):
        print(f"Error: File '{wav_file}' not found.")
        sys.exit(1)
    
    # Run sentence detection
    results = main(wav_file, eos_pause_ms, min_sentence_duration_ms, merge_gap_ms, pre_context_ms, post_context_ms, threshold)
    
    # Exit with appropriate code
    sys.exit(0 if results and results['has_speech'] else 1)