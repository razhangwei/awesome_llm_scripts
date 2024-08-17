import unittest
from unittest.mock import patch, mock_open
import transcriber

class TestTranscribeAudio(unittest.TestCase):

    @patch('transcriber.litellm.transcription')
    @patch('transcriber.mlx_whisper.transcribe')
    @patch('builtins.open', new_callable=mock_open, read_data=b'some audio data')
    def test_transcribe_audio_litellm(self, mock_open, mock_mlx_transcribe, mock_litellm_transcription):
        # Arrange
        mock_litellm_transcription.return_value.segments = [{'start': 0, 'end': 1, 'text': 'Hello world'}]
        filename = 'test_audio.wav'
        model = 'litellm-model'

        # Act
        segments = transcriber.transcribe_audio(filename, model)

        # Assert
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]['text'], 'Hello world')
        mock_litellm_transcription.assert_called_once()

    @patch('transcriber.mlx_whisper.transcribe')
    @patch('builtins.open', new_callable=mock_open, read_data=b'some audio data')
    def test_transcribe_audio_mlx(self, mock_open, mock_mlx_transcribe):
        # Arrange
        mock_mlx_transcribe.return_value = {'segments': [{'start': 0, 'end': 1, 'text': 'Hello from MLX'}]}
        filename = 'test_audio.wav'
        model = 'mlx-community/test-model'

        # Act
        segments = transcriber.transcribe_audio(filename, model)

        # Assert
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]['text'], 'Hello from MLX')
        mock_mlx_transcribe.assert_called_once()

if __name__ == '__main__':
    unittest.main()
