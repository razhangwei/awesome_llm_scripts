import requests
from litellm import LiteLLM
import os
from pydub import AudioSegment

def download_audio(url, filename):
    """Downloads the audio file from the given URL and saves it to the specified filename."""
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def transcribe_audio(filename):
    """Transcribes the audio file using the Groq/Whisper-Large-v3 model via LiteLLM."""
    model = LiteLLM("groq/whisper-large-v3")
    with open(filename, 'rb') as f:
        audio_data = f.read()
    transcript = model.transcribe(audio_data)
    return transcript

def compress_audio(filename, target_size_mb=25):
    """Compresses the audio file if it exceeds the target size."""
    audio = AudioSegment.from_file(filename)
    file_size_mb = os.path.getsize(filename) / (1024 * 1024)
    if file_size_mb > target_size_mb:
        compression_ratio = target_size_mb / file_size_mb
        compressed_audio = audio.set_frame_rate(int(audio.frame_rate * compression_ratio))
        compressed_audio.export(filename, format=audio.file_ext.replace('.', ''))
        print(f"Audio file compressed to {target_size_mb} MB.")

def main():
    """Main function to download, transcribe, and save the transcript."""
    url = input("Enter the URL of the audio file: ")
    filename = os.path.basename(url)  # Extract filename from URL
    download_audio(url, filename)
    compress_audio(filename)  # Compress if needed
    transcript = transcribe_audio(filename)
    print("Transcript segments:")
    for segment in transcript.segments:
        print(f"Start: {segment.start}, End: {segment.end}, Text: {segment.text}")
    with open("transcript.txt", "w") as f:
        f.write(transcript.text)
    print("Transcript saved to transcript.txt")

if __name__ == "__main__":
    main()
