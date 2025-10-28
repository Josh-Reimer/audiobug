#!/usr/bin/env python3
"""
Voice-Activated Audio Recorder
Automatically starts recording when human voice is detected and stops when silence is detected.
"""

import pyaudio
import webrtcvad
import wave
import threading
import time
import os
from datetime import datetime
import collections
import json

class VoiceActivatedRecorder:
    def __init__(self, 
                 aggressiveness=2, 
                 sample_rate=16000, 
                 frame_duration_ms=30,
                 silence_timeout=2.0,
                 min_recording_duration=1.0,
                 output_dir="recordings"):
        """
        Initialize Voice-Activated Recorder.
        
        Args:
            aggressiveness (int): VAD sensitivity (0-3)
            sample_rate (int): Audio sample rate
            frame_duration_ms (int): Frame duration in ms
            silence_timeout (float): Seconds of silence before stopping recording
            min_recording_duration (float): Minimum recording length in seconds
            output_dir (str): Directory to save recordings
        """
        # VAD settings
        self.aggressiveness = aggressiveness
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.frame_bytes = self.frame_size * 2
        
        # Recording settings
        self.silence_timeout = silence_timeout
        self.min_recording_duration = min_recording_duration
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad(aggressiveness)
        
        # Audio settings
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.chunk_size = self.frame_size
        
        # State management
        self.is_recording = False
        self.is_listening = False
        self.last_speech_time = 0
        self.recording_start_time = 0
        self.current_recording_frames = []
        self.current_filename = None
        
        # Pre-recording buffer (to catch the beginning of speech)
        self.pre_buffer_duration = 0.5  # seconds
        self.pre_buffer_frames = int(self.pre_buffer_duration * 1000 / frame_duration_ms)
        self.pre_buffer = collections.deque(maxlen=self.pre_buffer_frames)
        
        # Statistics
        self.stats = {
            'total_recordings': 0,
            'total_recording_time': 0,
            'session_start': time.time(),
            'last_recording': None
        }
        
        # Threading
        self.running = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Settings for recording quality
        self.auto_gain = True
        self.noise_suppression = True

    def list_audio_devices(self):
        """List available audio input devices."""
        print("Available audio input devices:")
        print("-" * 50)
        
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                print(f"Device {i}: {device_info['name']}")
                print(f"  Max Input Channels: {device_info['maxInputChannels']}")
                print(f"  Default Sample Rate: {device_info['defaultSampleRate']}")
                print()

    def start_listening(self, device_index=None):
        """Start listening for voice activity."""
        try:
            self.stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.running = True
            self.is_listening = True
            self.stream.start_stream()
            
            print("✅ Voice-activated recorder started!")
            print(f"📁 Recordings will be saved to: {os.path.abspath(self.output_dir)}")
            print(f"🔧 Settings:")
            print(f"   - VAD Aggressiveness: {self.aggressiveness}")
            print(f"   - Silence Timeout: {self.silence_timeout}s")
            print(f"   - Min Recording Length: {self.min_recording_duration}s")
            print(f"   - Sample Rate: {self.sample_rate}Hz")
            print("\n🎙️  Listening for voice... (speak to start recording)")
            
        except Exception as e:
            print(f"❌ Error starting audio stream: {e}")
            return False
        
        return True

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Process each audio frame."""
        if len(in_data) == self.frame_bytes and self.is_listening:
            # Always add to pre-buffer
            self.pre_buffer.append(in_data)
            
            # Process the frame
            self._process_audio_frame(in_data)
        
        return (None, pyaudio.paContinue)

    def _process_audio_frame(self, frame_data):
        """Process audio frame for voice activity and recording."""
        current_time = time.time()
        
        try:
            # Check for speech
            is_speech = self.vad.is_speech(frame_data, self.sample_rate)
            
            if is_speech:
                self.last_speech_time = current_time
                
                # Start recording if not already recording
                if not self.is_recording:
                    self._start_recording()
                
                # Add frame to current recording
                if self.is_recording:
                    self.current_recording_frames.append(frame_data)
            
            else:
                # Add frame to recording if we're currently recording
                if self.is_recording:
                    self.current_recording_frames.append(frame_data)
                    
                    # Check if we should stop recording due to silence
                    silence_duration = current_time - self.last_speech_time
                    if silence_duration >= self.silence_timeout:
                        self._stop_recording()
            
        except Exception as e:
            # Skip problematic frames
            pass

    def _start_recording(self):
        """Start a new recording session."""
        if self.is_recording:
            return
        
        self.is_recording = True
        self.recording_start_time = time.time()
        self.current_recording_frames = []
        
        # Add pre-buffer frames to catch the beginning of speech
        self.current_recording_frames.extend(list(self.pre_buffer))
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_filename = f"voice_recording_{timestamp}.wav"
        
        print(f"\n🔴 Recording started: {self.current_filename}")

    def _stop_recording(self):
        """Stop current recording and save to file."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        recording_duration = time.time() - self.recording_start_time
        
        # Check if recording meets minimum duration
        if recording_duration < self.min_recording_duration:
            print(f"⚠️  Recording too short ({recording_duration:.1f}s), discarding...")
            self.current_recording_frames = []
            return
        
        # Save the recording
        if self.current_recording_frames:
            filepath = os.path.join(self.output_dir, self.current_filename)
            success = self._save_recording(filepath, self.current_recording_frames)
            
            if success:
                print(f"💾 Recording saved: {self.current_filename} ({recording_duration:.1f}s)")
                
                # Update statistics
                self.stats['total_recordings'] += 1
                self.stats['total_recording_time'] += recording_duration
                self.stats['last_recording'] = {
                    'filename': self.current_filename,
                    'duration': recording_duration,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Save metadata
                self._save_metadata(filepath, recording_duration)
            else:
                print(f"❌ Failed to save recording: {self.current_filename}")
        
        self.current_recording_frames = []
        print("🎙️  Listening for voice...")

    def _save_recording(self, filepath, frames):
        """Save audio frames to WAV file."""
        try:
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))
            return True
        except Exception as e:
            print(f"Error saving recording: {e}")
            return False

    def _save_metadata(self, audio_filepath, duration):
        """Save recording metadata to JSON file."""
        metadata_filepath = audio_filepath.replace('.wav', '_metadata.json')
        
        metadata = {
            'filename': os.path.basename(audio_filepath),
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat(),
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'vad_aggressiveness': self.aggressiveness,
            'silence_timeout': self.silence_timeout,
            'min_duration': self.min_recording_duration
        }
        
        try:
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")

    def print_stats(self):
        """Print recording statistics."""
        session_duration = time.time() - self.stats['session_start']
        
        print(f"\n📊 Recording Session Statistics:")
        print(f"   Session Duration: {session_duration:.1f}s")
        print(f"   Total Recordings: {self.stats['total_recordings']}")
        print(f"   Total Recording Time: {self.stats['total_recording_time']:.1f}s")
        
        if self.stats['total_recordings'] > 0:
            avg_duration = self.stats['total_recording_time'] / self.stats['total_recordings']
            print(f"   Average Recording Length: {avg_duration:.1f}s")
            efficiency = (self.stats['total_recording_time'] / session_duration) * 100
            print(f"   Recording Efficiency: {efficiency:.1f}%")
        
        if self.stats['last_recording']:
            last = self.stats['last_recording']
            print(f"   Last Recording: {last['filename']} ({last['duration']:.1f}s)")
        
        print(f"   Currently: {'🔴 RECORDING' if self.is_recording else '🎙️ LISTENING'}")

    def list_recordings(self):
        """List all recordings in the output directory."""
        recordings = []
        
        for filename in os.listdir(self.output_dir):
            if filename.endswith('.wav'):
                filepath = os.path.join(self.output_dir, filename)
                stat = os.stat(filepath)
                
                # Try to load metadata
                metadata_file = filepath.replace('.wav', '_metadata.json')
                duration = None
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            duration = metadata.get('duration_seconds')
                    except:
                        pass
                
                recordings.append({
                    'filename': filename,
                    'size_mb': stat.st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(stat.st_mtime),
                    'duration': duration
                })
        
        if not recordings:
            print("📂 No recordings found.")
            return
        
        # Sort by modification time (newest first)
        recordings.sort(key=lambda x: x['modified'], reverse=True)
        
        print(f"\n📂 Recordings in {self.output_dir}:")
        print("-" * 80)
        
        for rec in recordings:
            duration_str = f"({rec['duration']:.1f}s)" if rec['duration'] else ""
            modified_str = rec['modified'].strftime("%Y-%m-%d %H:%M:%S")
            print(f"🎵 {rec['filename']}")
            print(f"   Size: {rec['size_mb']:.1f}MB | Modified: {modified_str} {duration_str}")

    def run_interactive(self, device_index=None):
        """Run in interactive mode with commands."""
        if not self.start_listening(device_index):
            return
        
        print("\nCommands:")
        print("  's' - Show statistics")
        print("  'l' - List recordings")
        print("  'p' - Pause/Resume listening")
        print("  'c' - Clear statistics")
        print("  'q' - Quit")
        print("\nPress Enter after each command...")
        
        try:
            while self.running:
                try:
                    command = input().strip().lower()
                    
                    if command == 'q':
                        break
                    elif command == 's':
                        self.print_stats()
                    elif command == 'l':
                        self.list_recordings()
                    elif command == 'p':
                        if self.is_listening:
                            self.is_listening = False
                            print("⏸️  Listening paused")
                        else:
                            self.is_listening = True
                            print("▶️  Listening resumed")
                    elif command == 'c':
                        self.stats = {
                            'total_recordings': 0,
                            'total_recording_time': 0,
                            'session_start': time.time(),
                            'last_recording': None
                        }
                        print("🗑️  Statistics cleared")
                    elif command == '':
                        continue
                    else:
                        print("❓ Unknown command. Use 's', 'l', 'p', 'c', or 'q'")
                        
                except (EOFError, KeyboardInterrupt):
                    break
                    
        except KeyboardInterrupt:
            pass
        
        # Stop any ongoing recording
        if self.is_recording:
            self._stop_recording()
        
        print("\n🛑 Stopping recorder...")
        self.stop()

    def run_continuous(self, duration=None, device_index=None):
        """Run continuously for specified duration."""
        if not self.start_listening(device_index):
            return
        
        print(f"Running for {duration}s..." if duration else "Running continuously (Press Ctrl+C to stop)")
        
        start_time = time.time()
        
        try:
            while self.running:
                if duration and (time.time() - start_time) >= duration:
                    break
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        # Stop any ongoing recording
        if self.is_recording:
            self._stop_recording()
        
        print("\n🛑 Stopping recorder...")
        self.stop()
        self.print_stats()

    def stop(self):
        """Stop the recorder."""
        self.running = False
        self.is_listening = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()

def main():
    """Main function with setup interface."""
    print("🎙️  Voice-Activated Recorder Setup")
    print("=" * 50)
    
    # Configuration
    config = {}
    
    # VAD aggressiveness
    print("VAD Aggressiveness (how strict voice detection is):")
    print("0 - Least strict (may record background noise)")
    print("1 - Less strict")
    print("2 - Moderate (recommended)")
    print("3 - Most strict (only clear speech)")
    
    try:
        config['aggressiveness'] = int(input("Enter level (0-3, default 2): ").strip() or "2")
        if config['aggressiveness'] not in [0, 1, 2, 3]:
            config['aggressiveness'] = 2
    except:
        config['aggressiveness'] = 2
    
    # Silence timeout
    try:
        timeout_input = input("Silence timeout in seconds (default 2.0): ").strip()
        config['silence_timeout'] = float(timeout_input) if timeout_input else 2.0
    except:
        config['silence_timeout'] = 2.0
    
    # Minimum recording duration
    try:
        min_dur_input = input("Minimum recording duration in seconds (default 1.0): ").strip()
        config['min_recording_duration'] = float(min_dur_input) if min_dur_input else 1.0
    except:
        config['min_recording_duration'] = 1.0
    
    # Output directory
    output_dir = input("Output directory (default 'recordings'): ").strip() or "recordings"
    config['output_dir'] = output_dir
    
    # Create recorder
    recorder = VoiceActivatedRecorder(**config)
    
    # Show devices
    print("\n" + "=" * 50)
    recorder.list_audio_devices()
    
    # Device selection
    try:
        device_input = input("Enter device number (or press Enter for default): ").strip()
        device_index = int(device_input) if device_input else None
    except:
        device_index = None
    
    # Mode selection
    print("\nSelect mode:")
    print("1 - Interactive mode (with commands)")
    print("2 - Continuous mode (run for specified time)")
    
    try:
        mode = input("Enter mode (1 or 2, default 1): ").strip() or "1"
    except:
        mode = "1"
    
    # Run recorder
    try:
        if mode == "2":
            try:
                duration_input = input("Duration in seconds (or press Enter for unlimited): ").strip()
                duration = float(duration_input) if duration_input else None
            except:
                duration = None
            
            recorder.run_continuous(duration, device_index)
        else:
            recorder.run_interactive(device_index)
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        recorder.stop()

if __name__ == "__main__":
    # Check dependencies
    try:
        import pyaudio
        import webrtcvad
    except ImportError:
        print("❌ Missing required packages. Install with:")
        print("pip install pyaudio webrtcvad")
        exit(1)
    
    main()