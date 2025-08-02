#!/usr/bin/env python3
"""
Voice Activity Detection for MP3 files in a folder.
Processes short MP3 files (3-5 seconds) and detects which ones contain human voice.
"""

import os
import glob
from pathlib import Path
import webrtcvad
from pydub import AudioSegment
import numpy as np

def setup_vad(aggressiveness=2):
    """
    Setup WebRTC VAD with specified aggressiveness level.
    
    Args:
        aggressiveness (int): 0-3, where 3 is most aggressive at filtering non-speech
    
    Returns:
        webrtcvad.Vad: Configured VAD instance
    """
    return webrtcvad.Vad(aggressiveness)

def load_and_prepare_audio(mp3_path):
    """
    Load MP3 file and convert to format suitable for VAD.
    
    Args:
        mp3_path (str): Path to MP3 file
    
    Returns:
        bytes: Raw audio data in 16kHz, 16-bit, mono format
        int: Duration in milliseconds
    """
    try:
        # Load MP3 file
        audio = AudioSegment.from_mp3(mp3_path)
        
        # Convert to 16kHz, 16-bit, mono (required by WebRTC VAD)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        
        # Get raw audio data
        raw_data = audio.raw_data
        duration_ms = len(audio)
        
        return raw_data, duration_ms
    
    except Exception as e:
        print(f"Error loading {mp3_path}: {e}")
        return None, 0

def detect_voice_in_frames(vad, audio_data, sample_rate=16000, frame_duration_ms=30):
    """
    Process audio in frames and detect voice activity.
    
    Args:
        vad: WebRTC VAD instance
        audio_data (bytes): Raw audio data
        sample_rate (int): Sample rate (must be 8000, 16000, 32000, or 48000)
        frame_duration_ms (int): Frame duration in ms (must be 10, 20, or 30)
    
    Returns:
        tuple: (voice_detected, voice_percentage, total_frames)
    """
    frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # 2 bytes per sample
    voice_frames = 0
    total_frames = 0
    
    # Process audio in chunks
    for i in range(0, len(audio_data) - frame_size + 1, frame_size):
        frame = audio_data[i:i + frame_size]
        
        # Skip if frame is too short
        if len(frame) < frame_size:
            continue
            
        try:
            # Check if frame contains speech
            if vad.is_speech(frame, sample_rate):
                voice_frames += 1
            total_frames += 1
        except Exception as e:
            # Skip problematic frames
            continue
    
    if total_frames == 0:
        return False, 0.0, 0
    
    voice_percentage = (voice_frames / total_frames) * 100
    voice_detected = voice_percentage > 10  # Consider voice detected if >10% of frames contain speech
    
    return voice_detected, voice_percentage, total_frames

def process_mp3_folder(folder_path, output_file=None, aggressiveness=2):
    """
    Process all MP3 files in a folder and detect voice activity.
    
    Args:
        folder_path (str): Path to folder containing MP3 files
        output_file (str): Optional path to save results to a text file
        aggressiveness (int): VAD aggressiveness level (0-3)
    """
    # Setup VAD
    vad = setup_vad(aggressiveness)
    
    # Find all MP3 files
    mp3_files = glob.glob(os.path.join(folder_path, "*.mp3"))
    
    if not mp3_files:
        print(f"No MP3 files found in {folder_path}")
        return
    
    print(f"Processing {len(mp3_files)} MP3 files...")
    print(f"VAD Aggressiveness Level: {aggressiveness}")
    print("-" * 80)
    
    results = []
    voice_files = []
    no_voice_files = []
    
    for mp3_file in sorted(mp3_files):
        filename = os.path.basename(mp3_file)
        print(f"Processing: {filename}")
        
        # Load and prepare audio
        audio_data, duration_ms = load_and_prepare_audio(mp3_file)
        
        if audio_data is None:
            print(f"  ❌ Failed to load audio")
            continue
        
        # Detect voice activity
        voice_detected, voice_percentage, total_frames = detect_voice_in_frames(vad, audio_data)
        
        # Store results
        result = {
            'filename': filename,
            'duration_s': duration_ms / 1000,
            'voice_detected': voice_detected,
            'voice_percentage': voice_percentage,
            'total_frames': total_frames
        }
        results.append(result)
        
        # Categorize files
        if voice_detected:
            voice_files.append(filename)
            status = "🎤 VOICE DETECTED"
        else:
            no_voice_files.append(filename)
            status = "🔇 NO VOICE"
        
        print(f"  {status} ({voice_percentage:.1f}% voice frames, {duration_ms/1000:.1f}s)")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {len(results)}")
    print(f"Files with voice: {len(voice_files)}")
    print(f"Files without voice: {len(no_voice_files)}")
    
    if voice_files:
        print(f"\nFiles with VOICE detected:")
        for filename in voice_files:
            print(f"  ✓ {filename}")
    
    if no_voice_files:
        print(f"\nFiles with NO VOICE:")
        for filename in no_voice_files:
            print(f"  • {filename}")
    
    # Save results to file if requested
    if output_file:
        save_results_to_file(results, output_file, voice_files, no_voice_files)

def save_results_to_file(results, output_file, voice_files, no_voice_files):
    """Save detection results to a text file."""
    try:
        with open(output_file, 'w') as f:
            f.write("Voice Activity Detection Results\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 50 + "\n")
            for result in results:
                f.write(f"File: {result['filename']}\n")
                f.write(f"  Duration: {result['duration_s']:.1f}s\n")
                f.write(f"  Voice Detected: {'YES' if result['voice_detected'] else 'NO'}\n")
                f.write(f"  Voice Percentage: {result['voice_percentage']:.1f}%\n")
                f.write(f"  Frames Analyzed: {result['total_frames']}\n\n")
            
            f.write("\nSUMMARY:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total files: {len(results)}\n")
            f.write(f"With voice: {len(voice_files)}\n")
            f.write(f"Without voice: {len(no_voice_files)}\n\n")
            
            if voice_files:
                f.write("FILES WITH VOICE:\n")
                for filename in voice_files:
                    f.write(f"  {filename}\n")
                f.write("\n")
            
            if no_voice_files:
                f.write("FILES WITHOUT VOICE:\n")
                for filename in no_voice_files:
                    f.write(f"  {filename}\n")
        
        print(f"\nResults saved to: {output_file}")
    
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    """Main function with example usage."""
    # Example usage
    folder_path = input("Enter the path to your MP3 folder: ").strip()
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    # Ask for aggressiveness level
    print("\nVAD Aggressiveness levels:")
    print("0 - Least aggressive (may include more non-speech)")
    print("1 - Less aggressive")
    print("2 - Moderate (recommended)")
    print("3 - Most aggressive (strict speech detection)")
    
    try:
        aggressiveness = int(input("Enter aggressiveness level (0-3, default 2): ").strip() or "2")
        if aggressiveness not in [0, 1, 2, 3]:
            aggressiveness = 2
            print("Invalid level, using default (2)")
    except:
        aggressiveness = 2
        print("Invalid input, using default (2)")
    
    # Ask if user wants to save results
    save_results = input("Save results to file? (y/n, default n): ").strip().lower()
    output_file = None
    if save_results in ['y', 'yes']:
        output_file = os.path.join(folder_path, "voice_detection_results.txt")
    
    # Process the folder
    process_mp3_folder(folder_path, output_file, aggressiveness)

if __name__ == "__main__":
    # Required packages check
    try:
        import webrtcvad
        from pydub import AudioSegment
    except ImportError as e:
        print("Missing required packages. Please install them:")
        print("pip install webrtcvad pydub")
        print("\nNote: You may also need to install ffmpeg for MP3 support:")
        print("- Windows: Download from https://ffmpeg.org/")
        print("- macOS: brew install ffmpeg")
        print("- Linux: sudo apt-get install ffmpeg")
        exit(1)
    
    main()