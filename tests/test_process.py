import unittest
from unittest.mock import patch, mock_open
from click.testing import CliRunner
from process import summarize  # Assuming this is how your module is imported

class TestSummarize(unittest.TestCase):
    @patch('process.completion')
    def test_valid_transcript(self, mock_completion):
        mock_completion.return_value = [
            {'choices': [{'delta': {'content': 'Summary of the transcript.'}}]}
        ]
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open('test_transcript.json', 'w') as f:
                f.write('[{"text": "This is a test transcript."}, {"text": "It has multiple segments."}]')
            
            result = runner.invoke(summarize, ['test_transcript.json'])
            
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.output.strip(), 'Summary of the transcript.')

    def test_transcript_segments_no_text(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open('empty_transcript.json', 'w') as f:
                f.write('[{"text": ""}, {"text": ""}]')
            
            result = runner.invoke(summarize, ['empty_transcript.json'])
            
            self.assertEqual(result.exit_code, 1)
            # Add an assertion here for the expected output of an empty transcript

if __name__ == '__main__':
    unittest.main()