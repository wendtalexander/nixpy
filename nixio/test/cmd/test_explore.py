import os
import io
import sys
import pytest
import tempfile
# from IPython import embed
import nixio as nix
import numpy as np
from unittest.mock import patch, MagicMock

# Import the module you want to test
# Assuming the code you showed is in a file called explore.py
import nixio.cmd.explore as explore  # adjust the import to match your actual module name

class TestProgressFunction:
    @patch('sys.stderr', new_callable=io.StringIO)
    def test_progress_function(self, mock_stderr):
        """Test that the progress function writes the expected output to stderr."""
        explore.progress(50, 100, status='Testing')
        output = mock_stderr.getvalue()
        
        # Check that the output contains expected elements
        assert '[' in output
        assert ']' in output
        assert '50.00%' in output
        assert 'Testing' in output
        
    @patch('sys.stderr', new_callable=io.StringIO)
    def test_progress_zero(self, mock_stderr):
        """Test the progress function with a count of 0."""
        explore.progress(0, 100, status='Starting')
        output = mock_stderr.getvalue()
        assert '0.00%' in output
        
    @patch('sys.stderr', new_callable=io.StringIO)
    def test_progress_complete(self, mock_stderr):
        """Test the progress function with count equal to total."""
        explore.progress(100, 100, status='Complete')
        output = mock_stderr.getvalue()
        assert '100.00%' in output
