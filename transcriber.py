import requests
import litellm
import os
from pydub import AudioSegment
import click
import logging
import json
import mlx_whisper 


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def download_audio(url: str, filename: str) -> None:
    """
    Downloads the audio file from the given URL and saves it to the specified filename.

    Args:
        url: The URL of the audio file.
        filename: The filename to save the audio file to.
    """
    logging.info(f"Downloading audio from {url} to {filename}")
    if os.path.exists(filename):
        logging.info(f"Audio file already exists at {filename}. Skipping download.")
        return

    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    logging.info(f"Audio downloaded successfully.")


def transcribe_audio(filename: str, model: str) -> list[dict]:
    """
    Transcribes the audio file using the specified model via LiteLLM.

    Args:
        filename: The path to the audio file.
        model: The name of the model to use for transcription.

    Returns:
        A list of dictionaries containing the transcript segments.
    """
    logging.info(f"Transcribing {filename} with {model}")
    if model.startswith("mlx-community/"):
        result = mlx_whisper.transcribe(
            filename,
            path_or_hf_repo=model, 
            verbose=False, 
        )
        logging.info(f"Transcription complete. Returned {len(result['segments'])} segments.")
        return result["segments"]
    else:
        with open(filename, "rb") as f:
            logging.info(f"Transcribing {filename} using LiteLLM")
            transcript = litellm.transcription(
                model=model,
                file=f,
                temperature=0,
                langfuse="en",
                response_format="verbose_json",
            )
            logging.info(f"Transcription complete. Returned {len(transcript.segments)} segments.")
        return transcript.segments


def compress_audio(filename: str, target_size_mb: int = 25) -> None:
    """
    Compresses the audio file if it exceeds the target size.

    Args:
        filename: The path to the audio file.
        target_size_mb: The target size in megabytes (default is 25).
    """
    audio = AudioSegment.from_file(filename)
    file_size_mb = os.path.getsize(filename) / (1024 * 1024)
    if file_size_mb > target_size_mb:
        compression_ratio = target_size_mb / file_size_mb
        compressed_audio = audio.set_frame_rate(
            int(audio.frame_rate * compression_ratio)
        )
        extension = filename.split(".")[-1]
        compressed_audio.export(filename, format=extension)
        logging.info(f"Audio file compressed to {target_size_mb} MB.")




@click.command()
@click.argument("audio_url", type=str)
@click.option(
    "-m",
    "--model",
    default="mlx-community/distil-whisper-large-v3",
    help="Name of the model to use (default: mlx-community/distil-whisper-large-v3). "
         "MLX models will be run locally",
)
def main(audio_url: str, model: str) -> None:
    """
    Main function to download, transcribe, and save the transcript.
    """

    filename = "data/" + os.path.basename(audio_url)  # Extract filename from URL
    download_audio(audio_url, filename)
    compress_audio(filename)  # Compress if needed
    segments = transcribe_audio(filename, model)

    logging.info("Transcript segments:")
    for segment in segments:
        logging.info(
            f"Start: {segment['start']}, End: {segment['end']}, Text: {segment['text']}"
        )

    # dump the segments to a json file
    transcript_filename = filename[:filename.rfind(".")] + "_transcript.json"
    with open(transcript_filename, "w") as f:
        json.dump(segments, f, indent=4)
        logging.info(f"Transcript saved to {transcript_filename}.")


if __name__ == "__main__":
    main()


