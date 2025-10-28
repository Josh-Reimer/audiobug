FROM python:3.11-slim

# Install system dependencies for audio and build tools
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    portaudio19-dev \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir pyaudio

# Copy the recording script
COPY recorder.py .

# Create ALSA config to reduce warnings
RUN echo "pcm.!default { type hw card 0 }" > /etc/asound.conf && \
    echo "ctl.!default { type hw card 0 }" >> /etc/asound.conf

# Run the recorder
CMD ["python", "-u", "recorder.py"]