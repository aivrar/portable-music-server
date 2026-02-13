"""Shared fixtures for music module tests."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a clean temporary directory."""
    return tmp_path


@pytest.fixture
def sample_audio_mono():
    """1-second mono 44100 Hz sine wave as numpy array."""
    import numpy as np
    sr = 44100
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio, sr


@pytest.fixture
def sample_audio_stereo():
    """1-second stereo 44100 Hz sine wave as numpy array."""
    import numpy as np
    sr = 44100
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 554 * t)
    audio = np.column_stack([left, right])
    return audio, sr


@pytest.fixture
def sample_wav_file(tmp_path, sample_audio_mono):
    """Write a mono WAV file and return its path."""
    import soundfile as sf
    audio, sr = sample_audio_mono
    path = tmp_path / "test.wav"
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return str(path), sr


@pytest.fixture
def sample_stereo_wav_file(tmp_path, sample_audio_stereo):
    """Write a stereo WAV file and return its path."""
    import soundfile as sf
    audio, sr = sample_audio_stereo
    path = tmp_path / "test_stereo.wav"
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return str(path), sr
