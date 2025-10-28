import pyaudio
import wave
import os
from datetime import datetime
import time

# Configuration
DEVICE_INDEX = None  # Will auto-detect USB microphone
CHANNELS = 1
RATE = 48000
CHUNK = 8192  # Increased buffer size to prevent overflow
RECORD_SECONDS = 6 * 60 * 60  # 6 hours
OUTPUT_DIR = "6hour-watch"
FORMAT = pyaudio.paInt16

def find_usb_microphone():
    """Find the USB microphone device index."""
    p = pyaudio.PyAudio()
    usb_device = None
    
    print("Available audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"{i}: {info['name']} - Inputs: {info['maxInputChannels']}")
        
        # Look for USB audio device with input channels
        if "usb" in info['name'].lower() and info['maxInputChannels'] > 0:
            usb_device = i
            print(f"Found USB microphone at index {i}: {info['name']}")
    
    p.terminate()
    return usb_device

def record_audio(device_index, duration, filename):
    """Record audio for specified duration and write incrementally to file."""
    p = pyaudio.PyAudio()
    
    # Get sample width before opening stream
    sample_width = p.get_sample_size(FORMAT)
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=CHUNK)
    
    print(f"Recording started: {filename}")
    
    # Open wave file before recording so partial file exists immediately
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    
    # Calculate total chunks needed
    total_chunks = int(RATE / CHUNK * duration)
    overflow_count = 0
    
    for i in range(total_chunks):
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            wf.writeframes(data)
            
            # Flush underlying file periodically so host sees data
            if i % (int(RATE / CHUNK * 60)) == 0:  # every ~60 seconds
                try:
                    wf._file.flush()
                except Exception:
                    pass
            
            # Print progress every 5 minutes
            if i % int(RATE / CHUNK * 300) == 0 and i > 0:
                elapsed = i / (RATE / CHUNK)
                print(f"Recording progress: {elapsed / 3600:.1f} hours (overflows: {overflow_count})")
        except IOError as e:
            # Handle overflow by inserting silence
            if getattr(e, 'errno', None) == pyaudio.paInputOverflowed:
                overflow_count += 1
                silence = b'\x00' * (CHUNK * CHANNELS * sample_width)
                wf.writeframes(silence)
            else:
                wf.close()
                stream.stop_stream()
                stream.close()
                p.terminate()
                raise
    
    print(f"Recording finished (total overflows: {overflow_count})")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf.close()
    
    print(f"Saved: {filename}")

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find USB microphone
    device_index = find_usb_microphone()
    
    if device_index is None:
        print("ERROR: No USB microphone found!")
        return
    
    print(f"Using device index: {device_index}")
    print(f"Recording 6-hour intervals to '{OUTPUT_DIR}' directory")
    
    # Continuous recording loop
    while True:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(OUTPUT_DIR, f"recording_{timestamp}.wav")
        
        try:
            record_audio(device_index, RECORD_SECONDS, filename)
        except Exception as e:
            print(f"Error during recording: {e}")
            print("Waiting 60 seconds before retrying...")
            time.sleep(60)
        
        # Small pause between recordings
        time.sleep(1)

if __name__ == "__main__":
    main()