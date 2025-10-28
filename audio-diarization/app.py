import os
import sys
import json
import time
import datetime
import hashlib
from pathlib import Path
import whisper
from tqdm import tqdm
from pyannote.audio import Pipeline
import ffmpeg
from dotenv import load_dotenv

# --------------------------- Configuration --------------------------- #
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")  # Replace with your Hugging Face token
WHISPER_MODEL = "large"

# --------------------------- Argument Parsing --------------------------- #

if len(sys.argv) < 2:
    print("Usage: python transcribe_diarize.py <input_audio_file> [--force] [--srt]")
    sys.exit(1)

input_file = Path(sys.argv[1])
force = "--force" in sys.argv
export_srt = "--srt" in sys.argv

if not input_file.exists():
    print(f"Error: File not found – {input_file}")
    sys.exit(1)

# --------------------------- Utility: File Hash --------------------------- #

def file_hash(path, block_size=65536):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            hasher.update(chunk)
    return hasher.hexdigest()[:12]  # Shorten to 12 characters for brevity

# --------------------------- File Conversion --------------------------- #

converted_wav = input_file.with_suffix(".wav")
if input_file.suffix.lower() != ".wav":
    print(f"Converting {input_file.name} to WAV format...")
    try:
        ffmpeg.input(str(input_file)).output(str(converted_wav), ac=1, ar=16000).run(quiet=True, overwrite_output=True)
        print(f"Converted to: {converted_wav}")
    except Exception as e:
        print(f"Error during ffmpeg conversion: {e}")
        sys.exit(1)
else:
    converted_wav = input_file

# --------------------------- Output File Naming --------------------------- #

hash_id = file_hash(input_file)
base_name = input_file.stem
prefix = f"{base_name}_{hash_id}"

transcript_json = input_file.parent / f"{prefix}_whisper_transcript.json"
final_transcript_txt = input_file.parent / f"{prefix}_diarized_transcript.txt"
srt_output = input_file.parent / f"{prefix}_diarized_transcript.srt"

# --------------------------- Whisper Transcription --------------------------- #

print("Loading Whisper model...")
start_time = time.time()
whisper_model = whisper.load_model(WHISPER_MODEL)
print(f"Whisper model loaded in {time.time() - start_time:.1f} seconds")

if transcript_json.exists() and not force:
    print(f"Using cached transcription: {transcript_json}")
    with open(transcript_json, "r", encoding="utf-8") as f:
        result = json.load(f)
else:
    print("Running Whisper transcription (this may take several minutes)...")
    start_time = time.time()
    result = whisper_model.transcribe(str(converted_wav), verbose=False)  # Show progress bar
    with open(transcript_json, "w", encoding="utf-8") as f:
        json.dump(result, f)
    print(f"Transcription completed and saved in {time.time() - start_time:.1f} seconds")

# --------------------------- Pyannote Diarization --------------------------- #

print("Loading speaker diarization pipeline...")
try:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=HUGGINGFACE_TOKEN
    )
except Exception as e:
    print("Failed to load diarization model. Make sure you've accepted model access:")
    print(" - https://huggingface.co/pyannote/speaker-diarization")
    print(" - https://huggingface.co/pyannote/segmentation")
    print(" - https://huggingface.co/pyannote/embedding")
    print(" - https://huggingface.co/pyannote/clustering")
    print(f"Error: {e}")
    sys.exit(1)

print("Running diarization (this may take a few minutes)...")
start_time = time.time()
diarization = pipeline(str(converted_wav))
print(f"Diarization completed in {time.time() - start_time:.1f} seconds")

# --------------------------- Output Alignment --------------------------- #

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))[2:]

def format_srt_timestamp(seconds):
    td = datetime.timedelta(seconds=seconds)
    return str(td)[:-3].replace('.', ',').rjust(12, '0')

print("Aligning transcript with diarization...")
speaker_segments = []
for turn, _, speaker in tqdm(diarization.itertracks(yield_label=True), desc="Speaker Segments"):
    speaker_segments.append({
        "start": turn.start,
        "end": turn.end,
        "speaker": speaker
    })

aligned_output = []
for segment in tqdm(result['segments'], desc="Merging Segments"):
    start = segment['start']
    text = segment['text'].strip()
    for seg in speaker_segments:
        if seg["start"] <= start < seg["end"]:
            speaker = seg["speaker"]
            timestamp = format_time(start)
            aligned_output.append({
                "speaker": speaker,
                "timestamp": timestamp,
                "start": start,
                "end": segment['end'],
                "text": text
            })
            break

# Save to plain text
with open(final_transcript_txt, "w", encoding="utf-8") as f:
    for line in aligned_output:
        f.write(f"[{line['speaker']}] {line['timestamp']} – {line['text']}\n")
print(f"Transcript saved to: {final_transcript_txt}")

# Save to SRT format
if export_srt:
    print("Generating SRT file...")
    with open(srt_output, "w", encoding="utf-8") as f:
        for idx, entry in enumerate(aligned_output, 1):
            f.write(f"{idx}\n")
            f.write(f"{format_srt_timestamp(entry['start'])} --> {format_srt_timestamp(entry['end'])}\n")
            f.write(f"{entry['speaker']}: {entry['text']}\n\n")
    print(f"SRT file saved to: {srt_output}")