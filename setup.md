# USB Microphone Recorder Setup Guide

## Project Structure
```
audio-recorder/
├── Dockerfile
├── docker-compose.yml
├── recorder.py
└── 6hour-watch/          (created automatically)
```

## Setup Steps

1. **Create project directory and files:**
```bash
mkdir audio-recorder
cd audio-recorder
```

2. **Save the three artifacts:**
   - Save `recorder.py` (Python script)
   - Save `Dockerfile`
   - Save `docker-compose.yml`

3. **Build the Docker image:**
```bash
docker-compose build
```

4. **Run the container:**
```bash
docker-compose up -d
```

## Usage

**View logs:**
```bash
docker-compose logs -f
```

**Stop recording:**
```bash
docker-compose down
```

**Restart after stop:**
```bash
docker-compose up -d
```

## How It Works

- The script automatically detects your USB microphone (the C-Media device)
- Records in 6-hour intervals continuously
- Saves files as `recording_YYYYMMDD_HHMMSS.wav` in the `6hour-watch` folder
- Recordings are stored on your host machine (not just in the container)
- Progress updates every 5 minutes
- Automatically restarts if there's an error

## Troubleshooting

**If the USB microphone isn't detected:**

1. Check available devices on your host:
```bash
arecord -l
```

2. You may need to adjust the device detection in `recorder.py` if your device name differs

**PipeWire/PulseAudio issues:**

If you're using PipeWire (like your system), you might need to expose the socket:
```bash
docker run --rm -it \
  --device /dev/snd \
  -v $(pwd)/6hour-watch:/app/6hour-watch \
  -v /run/user/$(id -u)/pulse:/run/user/1000/pulse \
  usb-microphone-recorder
```

**Check recordings:**
```bash
ls -lh 6hour-watch/
```

## File Format

- Format: WAV (uncompressed)
- Sample rate: 48000 Hz
- Channels: Mono (1 channel)
- Bit depth: 16-bit
- Expected file size: ~1.2 GB per 6-hour recording