#!/usr/bin/env python3
"""
Real-time Voice Activity Detection from microphone input.
Shows live feedback when human voice is detected.
"""

import pyaudio
import webrtcvad
import threading
import time
import sys
import collections
from datetime import datetime
import wave
import os

class RealTimeVAD:
    def __init__(self, aggressiveness=2, sample_rate=16000, frame_duration_ms=30):
        """
        Initialize real-time Voice Activity Detector.
        
        Args:
            aggressiveness (int): VAD sensitivity (0-3, 3 = most strict)
            sample_rate (int): Audio sample rate (8000, 16000, 32000, or 48000)
            frame_duration_ms (int): Frame duration in ms (10, 20, or 30)
        """
        # VAD settings
        self.aggressiveness = aggressiveness
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.frame_bytes = self.frame_size * 2  # 2 bytes per sample (16-bit)
        
        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad(aggressiveness)
        
        # Audio stream settings
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.chunk_size = self.frame_size
        
        # Buffers and state
        self.audio_buffer = collections.deque(maxlen=10)  # Keep last 10 frames
        self.is_speaking = False
        self.last_speech_time = 0
        self.total_frames = 0
        self.voice_frames = 0
        
        # Recording functionality
        self.is_recording = False
        self.recorded_frames = []
        self.recording_filename = None
        
        # Threading
        self.running = False
        self.audio_thread = None
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'voice_frames': 0,
            'session_start': time.time(),
            'total_speaking_time': 0,
            'last_activity': None
        }
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None

    def list_audio_devices(self):
        """List available audio input devices."""
        print("Available audio input devices:")
        print("-" * 50)
        
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                print(f"Device {i}: {device_info['name']}")
                print(f"  Channels: {device_info['maxInputChannels']}")
                print(f"  Sample Rate: {device_info['defaultSampleRate']}")
                print()

    def start_stream(self, device_index=None):
        """Start the audio input stream."""
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
            self.stream.start_stream()
            print(f"✅ Audio stream started successfully!")
            print(f"📊 Sample Rate: {self.sample_rate}Hz")
            print(f"🎯 Frame Duration: {self.frame_duration_ms}ms")
            print(f"🔧 VAD Aggressiveness: {self.aggressiveness}")
            
        except Exception as e:
            print(f"❌ Error starting audio stream: {e}")
            return False
        
        return True

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream - processes each audio frame."""
        if len(in_data) == self.frame_bytes:
            # Add frame to buffer for processing
            self.audio_buffer.append(in_data)
            
            # Process the frame
            self._process_audio_frame(in_data)
        
        return (None, pyaudio.paContinue)

    def _process_audio_frame(self, frame_data):
        """Process a single audio frame for voice activity."""
        try:
            # Check if frame contains speech
            is_speech = self.vad.is_speech(frame_data, self.sample_rate)
            
            # Update statistics
            self.stats['total_frames'] += 1
            if is_speech:
                self.stats['voice_frames'] += 1
                self.stats['last_activity'] = time.time()
            
            # Update speaking state
            current_time = time.time()
            
            if is_speech:
                if not self.is_speaking:
                    self.is_speaking = True
                    self._on_speech_started()
                self.last_speech_time = current_time
                
            else:
                # Stop speaking if no speech detected for 500ms
                if self.is_speaking and (current_time - self.last_speech_time) > 0.5:
                    self.is_speaking = False
                    self._on_speech_stopped()
            
            # Add to recording if active
            if self.is_recording:
                self.recorded_frames.append(frame_data)
                
        except Exception as e:
            # Skip problematic frames
            pass

    def _on_speech_started(self):
        """Called when speech detection starts."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\r🎤 [{timestamp}] SPEAKING... ", end="", flush=True)

    def _on_speech_stopped(self):
        """Called when speech detection stops."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        voice_percentage = (self.stats['voice_frames'] / max(self.stats['total_frames'], 1)) * 100
        print(f"\r🔇 [{timestamp}] SILENT     (Voice: {voice_percentage:.1f}%)", flush=True)

    def start_recording(self, filename=None):
        """Start recording audio to a WAV file."""
        if self.is_recording:
            print("⚠️  Already recording!")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"voice_recording_{timestamp}.wav"
        
        self.recording_filename = filename
        self.recorded_frames = []
        self.is_recording = True
        print(f"🔴 Started recording to: {filename}")

    def stop_recording(self):
        """Stop recording and save to WAV file."""
        if not self.is_recording:
            print("⚠️  Not currently recording!")
            return
        
        self.is_recording = False
        
        if self.recorded_frames:
            # Save recorded audio to WAV file
            try:
                with wave.open(self.recording_filename, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(b''.join(self.recorded_frames))
                
                duration = len(self.recorded_frames) * self.frame_duration_ms / 1000
                print(f"💾 Recording saved: {self.recording_filename} ({duration:.1f}s)")
                
            except Exception as e:
                print(f"❌ Error saving recording: {e}")
        else:
            print("⚠️  No audio recorded!")

    def print_stats(self):
        """Print current statistics."""
        if self.stats['total_frames'] == 0:
            return
        
        current_time = time.time()
        session_duration = current_time - self.stats['session_start']
        voice_percentage = (self.stats['voice_frames'] / self.stats['total_frames']) * 100
        
        print(f"\n📊 Session Statistics:")
        print(f"   Duration: {session_duration:.1f}s")
        print(f"   Total Frames: {self.stats['total_frames']}")
        print(f"   Voice Frames: {self.stats['voice_frames']} ({voice_percentage:.1f}%)")
        print(f"   Currently: {'🎤 SPEAKING' if self.is_speaking else '🔇 SILENT'}")
        
        if self.stats['last_activity']:
            time_since_activity = current_time - self.stats['last_activity']
            print(f"   Last Activity: {time_since_activity:.1f}s ago")

    def run_interactive(self, device_index=None):
        """Run interactive mode with keyboard commands."""
        print("🎙️  Real-time Voice Activity Detector")
        print("=" * 50)
        
        if not self.start_stream(device_index):
            return
        
        print("\nCommands:")
        print("  's' - Show statistics")
        print("  'r' - Start/stop recording")
        print("  'd' - List audio devices")
        print("  'q' - Quit")
        print("\nListening... (speak into your microphone)")
        print("-" * 50)
        
        try:
            while self.running:
                try:
                    # Non-blocking input check
                    import select
                    import sys
                    
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        command = input().strip().lower()
                        
                        if command == 'q':
                            break
                        elif command == 's':
                            self.print_stats()
                        elif command == 'r':
                            if self.is_recording:
                                self.stop_recording()
                            else:
                                self.start_recording()
                        elif command == 'd':
                            self.list_audio_devices()
                        
                except (EOFError, KeyboardInterrupt):
                    break
                except:
                    # Handle systems without select module (Windows)
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            pass
        
        print("\n🛑 Stopping...")
        self.stop()

    def run_simple(self, duration=None, device_index=None):
        """Run simple mode - just show voice activity for specified duration."""
        print("🎙️  Real-time Voice Activity Detector")
        print("=" * 50)
        
        if not self.start_stream(device_index):
            return
        
        print(f"Listening for {duration}s... (Press Ctrl+C to stop early)" if duration else "Listening... (Press Ctrl+C to stop)")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            while self.running:
                if duration and (time.time() - start_time) >= duration:
                    break
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            pass
        
        print("\n🛑 Stopping...")
        self.stop()
        self.print_stats()

    def stop(self):
        """Stop the voice detector."""
        self.running = False
        
        if self.is_recording:
            self.stop_recording()
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()

def main():
    """Main function with user interface."""
    print("🎙️  Real-time Voice Activity Detector Setup")
    print("=" * 50)
    
    # VAD aggressiveness selection
    print("VAD Aggressiveness levels:")
    print("0 - Least aggressive (may include background noise)")
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
    
    # Create VAD instance
    vad = RealTimeVAD(aggressiveness=aggressiveness)
    
    # Show available devices
    print("\n" + "=" * 50)
    vad.list_audio_devices()
    
    # Device selection
    try:
        device_input = input("Enter device number (or press Enter for default): ").strip()
        device_index = int(device_input) if device_input else None
    except:
        device_index = None
        print("Using default audio device")
    
    # Mode selection
    print("\nSelect mode:")
    print("1 - Interactive mode (with commands)")
    print("2 - Simple mode (just voice detection)")
    
    try:
        mode = input("Enter mode (1 or 2, default 1): ").strip() or "1"
    except:
        mode = "1"
    
    # Run the detector
    try:
        if mode == "2":
            # Simple mode - ask for duration
            try:
                duration_input = input("Duration in seconds (or press Enter for unlimited): ").strip()
                duration = float(duration_input) if duration_input else None
            except:
                duration = None
            
            vad.run_simple(duration, device_index)
        else:
            # Interactive mode
            vad.run_interactive(device_index)
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        vad.stop()

if __name__ == "__main__":
    # Check required packages
    try:
        import pyaudio
        import webrtcvad
    except ImportError as e:
        print("❌ Missing required packages. Please install them:")
        print("pip install pyaudio webrtcvad")
        print("\nNote: On some systems you may need:")
        print("- Windows: May need Visual C++ Build Tools")
        print("- macOS: brew install portaudio")
        print("- Linux: sudo apt-get install portaudio19-dev python3-pyaudio")
        sys.exit(1)
    
    main()