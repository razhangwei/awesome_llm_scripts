# Audio Transcriber

This script downloads an audio file from a given URL, compresses it if necessary, and transcribes it using a specified model.

## Requirements

- Python 3.x
- Required libraries (install via pip):
  - requests
  - litellm
  - pydub
  - mlx_whisper

## How to Run

1. Open your command line interface (Terminal, Command Prompt, etc.).
2. Navigate to the directory where the `transcriber.py` file is located.
3. Run the script using the following command:

   ```bash
   python transcriber.py <audio_url> [-m <model_name>]
   ```

   - Replace `<audio_url>` with the URL of the audio file you want to transcribe.
   - Optionally, specify a model name using the `-m` flag. If not specified, it defaults to `mlx-community/distil-whisper-large-v3`.

## Example

