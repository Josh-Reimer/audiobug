import whisper
import torch
import sys

def transcribe_wav_to_text(wav_file_path, model_size="large"):
    """
    Transcribe a .wav file to text using Whisper model on CUDA.
    
    Args:
        wav_file_path (str): Path to the input .wav file
        model_size (str): Whisper model size (tiny, base, small, medium, large)
    
    Returns:
        str: Transcribed text
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and CUDA installed.")
    
    try:
        # Load the Whisper model with specified size, move to CUDA
        model = whisper.load_model(model_size, device="cuda")
        
        # Transcribe the audio file
        result = model.transcribe(wav_file_path, fp16=True)
        
        # Return the transcribed text
        return result["text"]
    
    except FileNotFoundError:
        return f"Error: The file {wav_file_path} was not found."
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    # Check if a file path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python transcribe_wav.py <path_to_wav_file>")
        sys.exit(1)
    
    wav_file = sys.argv[1]
    transcription = transcribe_wav_to_text(wav_file)
    print("Transcription:")
    print(transcription)