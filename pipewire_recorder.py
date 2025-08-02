#!/usr/bin/env python3
"""
PipeWire USB Microphone Sound Level Monitor and MP3 Recorder
Uses pw-record and system monitoring for better PipeWire compatibility
Now with proper decibel calculation like the PyGame example
"""

import subprocess
import threading
import time
import os
import signal
import argparse
import audioop
import math
from datetime import datetime
import numpy as np
import wave

class PipeWireRecorder:
    def __init__(self, threshold_db=-40, min_recording_seconds=2, use_sound_monitoring=True, interval_seconds=5):
        self.threshold_db = threshold_db  # Now using actual dB values (e.g., -40 dB)
        self.min_recording_seconds = min_recording_seconds
        self.use_sound_monitoring = use_sound_monitoring
        self.interval_seconds = interval_seconds
        self.usb_device = None
        self.is_monitoring = False
        self.is_recording = False
        self.recording_process = None
        self.monitor_process = None
        self.current_filename = None
        self.recording_start_time = None
        
        # Find USB microphone device
        self._find_usb_device()
    
    def _find_usb_device(self):
        """Find USB microphone device using PipeWire commands"""
        print("Searching for USB microphones using PipeWire...")
        
        # Method 1: Try pactl (PulseAudio/PipeWire compatibility layer)
        usb_devices = []
        try:
            result = subprocess.run(['pactl', 'list', 'sources'], 
                                  capture_output=True, text=True, check=True)
            
            current_device = {}
            for line in result.stdout.split('\n'):
                line = line.strip()
                
                if line.startswith('Source #'):
                    # Save previous device if it was USB
                    if current_device and self._is_usb_microphone(current_device):
                        usb_devices.append(current_device)
                    current_device = {'index': line}
                    
                elif line.startswith('Name: '):
                    current_device['name'] = line.split('Name: ')[1]
                elif line.startswith('Description: '):
                    current_device['description'] = line.split('Description: ')[1]
                elif 'device.class = "Audio/Source"' in line:
                    current_device['is_source'] = True
            
            # Check last device
            if current_device and self._is_usb_microphone(current_device):
                usb_devices.append(current_device)
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("pactl not available, trying alternative method...")
        
        # Method 2: Try wpctl (WirePlumber control)
        if not usb_devices:
            try:
                result = subprocess.run(['wpctl', 'status'], 
                                      capture_output=True, text=True, check=True)
                
                lines = result.stdout.split('\n')
                in_sources_section = False
                
                for line in lines:
                    if 'Sources:' in line:
                        in_sources_section = True
                        continue
                    elif ('Sinks:' in line or 'Clients:' in line or 
                          'Video' in line or 'Settings' in line):
                        in_sources_section = False
                        continue
                    
                    if in_sources_section and line.strip():
                        # Parse wpctl format: "   │  *   65. PCM2902 Audio Codec Analog Mono     [vol: 1.00]"
                        if ('pcm' in line.lower() or 'codec' in line.lower() or 'usb' in line.lower()):
                            # Exclude unwanted devices
                            if not any(exclude in line.lower() for exclude in ['midi', 'bridge', 'monitor', 'loopback']):
                                
                                # Extract device ID and name
                                # Look for pattern like "65. PCM2902 Audio Codec"
                                import re
                                match = re.search(r'(\d+)\.\s+([^[]+)', line)
                                if match:
                                    device_id = match.group(1)
                                    device_name = match.group(2).strip()
                                    
                                    usb_devices.append({
                                        'name': device_id,  # Use numeric ID for wpctl
                                        'description': device_name,
                                        'line': line.strip(),
                                        'method': 'wpctl'
                                    })
                                    print(f"Found via wpctl: ID {device_id} - {device_name}")
                
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("wpctl not available")
        
        # Method 3: Direct device name matching
        if not usb_devices:
            print("Trying known USB audio device patterns...")
            
            # Test the device ID we found, plus common patterns
            test_devices = [
                "65",  # The ID we saw in wpctl output
                "@DEFAULT_AUDIO_SOURCE@",
                "alsa_input.usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device-00.analog-mono",
            ]
            
            for device in test_devices:
                print(f"Testing device: {device}")
                if self._test_device(device):
                    usb_devices.append({
                        'name': device,
                        'description': f"USB Audio Device (tested: {device})",
                        'tested': True
                    })
                    print(f"✓ Working device found: {device}")
                    break
                else:
                    print(f"✗ Device {device} failed test")
        
        # Select the best device
        if usb_devices:
            print(f"\nFound {len(usb_devices)} USB audio device(s):")
            for i, device in enumerate(usb_devices):
                desc = device.get('description', device.get('name', 'Unknown'))
                print(f"  {i}: {desc}")
            
            # Use the first (and likely best) device
            selected_device = usb_devices[0]
            self.usb_device = selected_device['name']
            print(f"Selected: {selected_device.get('description', self.usb_device)}")
        else:
            print("No USB audio devices found")
            self._list_all_sources()
            raise Exception("No USB microphone device found")
    
    def _is_usb_microphone(self, device):
        """Check if device is likely a USB microphone"""
        name = device.get('name', '').lower()
        desc = device.get('description', '').lower()
        
        # Look for USB audio patterns
        usb_patterns = ['usb', 'pcm', 'codec', 'analog']
        exclude_patterns = ['midi', 'bridge', 'monitor', 'loopback']
        
        text = f"{name} {desc}"
        
        has_usb_pattern = any(pattern in text for pattern in usb_patterns)
        has_exclude_pattern = any(pattern in text for pattern in exclude_patterns)
        
        return has_usb_pattern and not has_exclude_pattern
    
    def _test_device(self, device_name):
        """Test if a device name works with pw-record"""
        try:
            cmd = ['pw-record', '--target', device_name, '/dev/null']
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE)
            time.sleep(0.5)  # Give it more time to start
            process.terminate()
            process.wait()
            
            # Check stderr for specific errors
            stderr_output = process.stderr.read().decode() if process.stderr else ""
            
            # If pw-record starts successfully, it should accept termination gracefully
            success = (process.returncode in [0, -15] or 
                      "failed to create stream" not in stderr_output.lower())
            
            if success:
                print(f"✓ Device {device_name} test successful")
            else:
                print(f"✗ Device {device_name} test failed: {stderr_output.strip()}")
            
            return success
        except Exception as e:
            print(f"✗ Device {device_name} test error: {e}")
            return False
    
    def _list_all_sources(self):
        """List all available audio sources for debugging"""
        print("\nAll available audio sources:")
        
        try:
            result = subprocess.run(['pactl', 'list', 'sources', 'short'], 
                                  capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if line.strip():
                    print(f"  {line}")
        except:
            try:
                result = subprocess.run(['wpctl', 'status'], 
                                      capture_output=True, text=True)
                print("Available sources from wpctl status:")
                in_sources = False
                for line in result.stdout.split('\n'):
                    if 'Sources:' in line:
                        in_sources = True
                    elif 'Sinks:' in line:
                        in_sources = False
                    elif in_sources and line.strip():
                        print(f"  {line}")
            except:
                print("Could not list audio sources")
    
    def calculate_decibel(self, amplitude):
        """Convert amplitude to decibels (same calculation as PyGame example)"""
        if amplitude <= 0:
            return -120  # Silence
        
        # Calculate dB relative to maximum possible amplitude (32767 for 16-bit signed)
        try:
            db = 20 * math.log10(amplitude / 32767)
        except ValueError:
            return -120
        
        # Scale to a more meaningful range (0 = loudest, -120 = quietest)
        return max(db, -120)
    
    def _get_current_decibels(self):
        """Get current decibel level from the microphone using proper dB calculation"""
        try:
            # Use a temporary file approach instead of stdout reading
            temp_file = "/tmp/audio_sample.wav"
            
            # Record a very short sample (0.2 seconds) to a file
            if self.usb_device.isdigit():
                cmd = ['timeout', '0.2', 'pw-record', '--target', self.usb_device, 
                       '--format', 's16', '--rate', '44100', '--channels', '1', temp_file]
            else:
                cmd = ['timeout', '0.2', 'pw-record', '--target', self.usb_device, 
                       '--format', 's16', '--rate', '44100', '--channels', '1', temp_file]
            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Read the recorded file
            if os.path.exists(temp_file):
                try:
                    with wave.open(temp_file, 'rb') as wav_file:
                        # Read all frames
                        frames = wav_file.readframes(wav_file.getnframes())
                        
                        if len(frames) >= 2:
                            # Calculate maximum amplitude like in PyGame example
                            max_amplitude = audioop.max(frames, 2)  # 2 = sample width for 16-bit
                            
                            # Apply same scaling as PyGame example
                            scaled_amplitude = max_amplitude // 100
                            
                            # Calculate decibels
                            current_db = self.calculate_decibel(scaled_amplitude)
                            
                            # Clean up temp file
                            os.remove(temp_file)
                            
                            return current_db
                            
                except Exception as e:
                    print(f"Error reading temp file: {e}")
                
                # Clean up temp file if it exists
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            # Fallback method: try direct stdout reading with better buffering
            return self._get_decibels_fallback()
            
        except Exception as e:
            if self.usb_device.isdigit():
                print(f"Temp file method failed, trying default source...")
                self.usb_device = "@DEFAULT_AUDIO_SOURCE@"
                return self._get_current_decibels()
            else:
                print(f"Error getting decibels: {e}")
        
        return -120  # Return silence level on error
    
    def _get_decibels_fallback(self):
        """Fallback method using direct stdout reading with better handling"""
        try:
            # Use pw-record with stdout but with better parameters
            if self.usb_device.isdigit():
                cmd = ['pw-record', '--target', self.usb_device, '--format', 's16', 
                       '--rate', '44100', '--channels', '1', '-']
            else:
                cmd = ['pw-record', '--target', self.usb_device, '--format', 's16', 
                       '--rate', '44100', '--channels', '1', '-']
            
            # Use unbuffered reading and a shorter timeout
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            
            # Give pw-record a moment to start
            time.sleep(0.1)
            
            # Read a single larger chunk (0.1 seconds worth of data)
            # 44100 samples/sec * 2 bytes/sample * 0.1 sec = 8820 bytes
            chunk_size = 8820
            
            try:
                # Use select to check if data is available (Linux/Unix only)
                import select
                ready, _, _ = select.select([process.stdout], [], [], 0.2)  # 0.2 second timeout
                
                if ready:
                    audio_data = process.stdout.read(chunk_size)
                    
                    if len(audio_data) >= 2:
                        # Calculate maximum amplitude
                        max_amplitude = audioop.max(audio_data, 2)
                        
                        # Apply scaling like PyGame example
                        scaled_amplitude = max_amplitude // 100
                        
                        # Calculate decibels
                        current_db = self.calculate_decibel(scaled_amplitude)
                        
                        process.terminate()
                        process.wait()
                        
                        return current_db
                        
            except ImportError:
                # Fallback for systems without select
                audio_data = process.stdout.read(chunk_size)
                
                if len(audio_data) >= 2:
                    max_amplitude = audioop.max(audio_data, 2)
                    scaled_amplitude = max_amplitude // 100
                    current_db = self.calculate_decibel(scaled_amplitude)
                    
                    process.terminate()
                    process.wait()
                    
                    return current_db
            
            process.terminate()
            process.wait()
            
        except Exception as e:
            print(f"Fallback method error: {e}")
        
        return -120
    
    def _start_recording(self, duration=None):
        """Start recording using pw-record"""
        if not self.is_recording:
            self.is_recording = True
            self.recording_start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if duration:
                self.current_filename = f"interval_{timestamp}_{duration}s.wav"
            else:
                self.current_filename = f"recording_{timestamp}.wav"
            
            # Start pw-record process with appropriate target format
            if self.usb_device.isdigit():
                # Use numeric device ID
                cmd = ['pw-record', '--target', self.usb_device, 
                       '--format', 's16', '--rate', '44100', '--channels', '1',
                       self.current_filename]
            else:
                # Use device name
                cmd = ['pw-record', '--target', self.usb_device, 
                       '--format', 's16', '--rate', '44100', '--channels', '1',
                       self.current_filename]
            
            self.recording_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                                    stderr=subprocess.PIPE)
            
            if duration:
                print(f"🔴 Started {duration}s interval recording: {self.current_filename}")
            else:
                print(f"🔴 Started recording: {self.current_filename}")
    
    def _stop_recording(self):
        """Stop recording and convert to MP3"""
        if self.is_recording and self.recording_process:
            self.is_recording = False
            
            # Stop the recording process
            self.recording_process.terminate()
            self.recording_process.wait()
            
            print(f"⏹️  Stopped recording: {self.current_filename}")
            
            # Convert to MP3 in background thread
            threading.Thread(target=self._convert_to_mp3, 
                           args=(self.current_filename,)).start()
    
    def _convert_to_mp3(self, wav_filename):
        """Convert WAV to MP3 and clean up"""
        try:
            mp3_filename = wav_filename.replace('.wav', '.mp3')
            
            cmd = ['ffmpeg', '-i', wav_filename, '-acodec', 'libmp3lame', 
                   '-b:a', '128k', mp3_filename, '-y']
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                os.remove(wav_filename)  # Remove WAV file
                print(f"💾 Saved recording: {mp3_filename}")
            else:
                print(f"Error converting to MP3: {result.stderr}")
                print(f"WAV file kept as: {wav_filename}")
                
        except Exception as e:
            print(f"Error converting recording: {e}")
    
    def start_monitoring(self):
        """Start monitoring sound levels or interval recording"""
        if not self.usb_device:
            raise Exception("No USB device selected")
        
        if self.use_sound_monitoring:
            self._start_sound_level_monitoring()
        else:
            self._start_interval_recording()
    
    def _start_sound_level_monitoring(self):
        """Start monitoring sound levels with proper dB calculation"""
        print(f"Starting sound level monitoring...")
        print(f"Device: {self.usb_device}")
        print(f"dB threshold: {self.threshold_db} dB")
        print(f"Minimum recording duration: {self.min_recording_seconds} seconds")
        print("Press Ctrl+C to stop")
        
        self.is_monitoring = True
        consecutive_low_readings = 0
        min_consecutive_readings = 3
        max_db_seen = -120
        
        try:
            while self.is_monitoring:
                # Get current decibel level
                current_db = self._get_current_decibels()
                
                # Track maximum dB seen
                if current_db > max_db_seen:
                    max_db_seen = current_db
                
                # Print current level with visual indicators
                status = '🔴 RECORDING' if self.is_recording else '⚪'
                
                # Create a dB level bar (range from -80 to 0 dB for visualization)
                db_range = 80  # -80 to 0 dB range
                normalized_db = max(0, min(db_range, current_db + db_range))  # Scale -80..0 to 0..80
                bar_length = 30
                filled_length = int(bar_length * normalized_db / db_range)
                db_bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                print(f"\rLevel: {current_db:6.1f} dB [{db_bar}] Max: {max_db_seen:6.1f} dB {status}", 
                      end='', flush=True)
                
                # Check if we should start or stop recording based on dB threshold
                if current_db > self.threshold_db:
                    consecutive_low_readings = 0
                    if not self.is_recording:
                        self._start_recording()
                else:
                    if self.is_recording:
                        consecutive_low_readings += 1
                        
                        # Check if minimum recording time has passed
                        recording_duration = time.time() - self.recording_start_time
                        min_duration_met = recording_duration >= self.min_recording_seconds
                        
                        # Stop recording after consecutive low readings AND minimum duration
                        if consecutive_low_readings >= min_consecutive_readings and min_duration_met:
                            self._stop_recording()
                            consecutive_low_readings = 0
                
                time.sleep(0.2)  # Check every 200ms
                
        except KeyboardInterrupt:
            print("\n\nStopping monitoring...")
            
        finally:
            self.cleanup()
    
    def _start_interval_recording(self):
        """Start interval-based recording"""
        print(f"Starting interval recording...")
        print(f"Device: {self.usb_device}")
        print(f"Recording {self.interval_seconds} second intervals")
        print("Press Ctrl+C to stop")
        
        self.is_monitoring = True
        recording_count = 0
        
        try:
            while self.is_monitoring:
                recording_count += 1
                print(f"\n📹 Starting recording #{recording_count}")
                
                # Start recording
                self._start_recording(duration=self.interval_seconds)
                
                # Wait for the specified duration
                time.sleep(self.interval_seconds)
                
                # Stop recording
                self._stop_recording()
                
                # Brief pause between recordings
                print("⏸️  Pausing...")
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\nStopping interval recording...")
            
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up processes"""
        self.is_monitoring = False
        
        if self.is_recording and self.recording_process:
            print("Stopping current recording...")
            self._stop_recording()
        
        print("Cleanup complete")

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import audioop
        import math
        import wave
    except ImportError:
        print("Missing audioop, math, or wave (should be built-in with Python)")
        return False
    
    # Check for timeout command
    try:
        subprocess.run(['timeout', '--help'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("timeout command not found. Install coreutils package")
        return False
    
    # Check for pw-record
    try:
        subprocess.run(['pw-record', '--help'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("pw-record not found. Install PipeWire tools:")
        print("  sudo apt-get install pipewire-pulse pipewire-utils")
        return False
    
    # Check for ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ffmpeg not found. Install with: sudo apt-get install ffmpeg")
        return False
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='PipeWire USB Microphone Recorder with Proper dB Calculation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 pipewire_recorder.py                    # Sound level monitoring mode (default)
  python3 pipewire_recorder.py --interval         # 5-second interval recording
  python3 pipewire_recorder.py --interval -d 10   # 10-second interval recording  
  python3 pipewire_recorder.py -t -30 -m 3        # -30dB threshold, 3-second minimum recording
  python3 pipewire_recorder.py -t -50 -m 1        # Very sensitive: -50dB threshold, 1-second minimum
        """
    )
    
    parser.add_argument('--interval', '-i', action='store_true',
                        help='Use interval recording instead of sound level monitoring')
    parser.add_argument('--duration', '-d', type=int, default=5,
                        help='Duration of each interval recording in seconds (default: 5)')
    parser.add_argument('--threshold', '-t', type=int, default=-40,
                        help='dB threshold for monitoring mode (default: -40 dB, range: -120 to 0)')
    parser.add_argument('--min-duration', '-m', type=int, default=2,
                        help='Minimum recording duration in seconds for monitoring mode (default: 2)')
    
    args = parser.parse_args()
    
    if not check_dependencies():
        return
    
    # Validate dB threshold
    if args.threshold > 0 or args.threshold < -120:
        print("Warning: dB threshold should be between -120 and 0 dB")
        print("Common values: -60 dB (quiet), -40 dB (normal), -20 dB (loud)")
    
    try:
        # Create recorder with specified parameters
        recorder = PipeWireRecorder(
            threshold_db=args.threshold,
            min_recording_seconds=args.min_duration,
            use_sound_monitoring=not args.interval,
            interval_seconds=args.duration
        )
        
        # Start monitoring or interval recording
        recorder.start_monitoring()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()