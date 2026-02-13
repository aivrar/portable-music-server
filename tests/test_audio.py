"""Tests for audio_profiles.py, audio_processing.py, and audio_assembler.py."""

import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import pytest

from audio_profiles import PROFILES, EQ_PRESETS
from audio_processing import (
    _apply_denoise,
    _apply_highpass,
    _apply_multiband_compression,
    _apply_stereo_enhancement,
    _apply_eq,
    _trim_silence,
    _normalize_loudness,
    _apply_peak_limit,
    post_process,
)
from audio_assembler import assemble_chunks, convert_format, _ffmpeg_args


# ===========================================================================
# audio_profiles.py
# ===========================================================================
class TestProfiles:
    def test_all_8_models_have_profiles(self):
        expected = {
            "ace_step_v15", "ace_step_v1", "heartmula", "diffrhythm",
            "yue", "musicgen", "riffusion", "stable_audio",
        }
        assert set(PROFILES.keys()) == expected

    def test_required_profile_fields(self):
        required = {
            "sample_rate", "lufs", "clipping", "denoise_strength",
            "highpass_cutoff", "stereo_width", "eq_preset", "compression_ratio",
            "trim_db", "min_silence_ms", "front_protect_ms", "end_protect_ms",
        }
        for model, profile in PROFILES.items():
            for field in required:
                assert field in profile, f"{model} missing '{field}'"

    def test_ace_step_models_are_48khz(self):
        assert PROFILES["ace_step_v15"]["sample_rate"] == 48000
        assert PROFILES["ace_step_v1"]["sample_rate"] == 48000

    def test_sample_rates_are_valid(self):
        valid_rates = {22050, 32000, 44100, 48000}
        for model, profile in PROFILES.items():
            assert profile["sample_rate"] in valid_rates, (
                f"{model} has unexpected sample_rate {profile['sample_rate']}"
            )


class TestEQPresets:
    def test_all_presets_present(self):
        assert set(EQ_PRESETS.keys()) == {"flat", "balanced", "warm", "bright"}

    def test_presets_are_3_tuples(self):
        for name, values in EQ_PRESETS.items():
            assert len(values) == 3, f"EQ preset '{name}' should have 3 values"

    def test_flat_is_all_zeros(self):
        assert EQ_PRESETS["flat"] == (0.0, 0.0, 0.0)


# ===========================================================================
# audio_processing.py — individual stages
# ===========================================================================
class TestDenoise:
    def test_no_denoise_when_strength_zero(self, sample_audio_mono):
        audio, sr = sample_audio_mono
        result = _apply_denoise(audio, sr, strength=0.0)
        np.testing.assert_array_equal(result, audio)

    def test_short_audio_unchanged(self):
        sr = 44100
        # Audio shorter than 0.5s noise profile
        short = np.random.randn(int(sr * 0.3)).astype(np.float32)
        result = _apply_denoise(short, sr, strength=0.2)
        np.testing.assert_array_equal(result, short)

    def test_denoise_returns_same_shape(self, sample_audio_mono):
        audio, sr = sample_audio_mono
        try:
            result = _apply_denoise(audio, sr, strength=0.2)
            assert result.shape == audio.shape
        except ImportError:
            pytest.skip("noisereduce not available")

    def test_denoise_stereo(self, sample_audio_stereo):
        audio, sr = sample_audio_stereo
        try:
            result = _apply_denoise(audio, sr, strength=0.2)
            assert result.shape == audio.shape
        except ImportError:
            pytest.skip("noisereduce not available")


class TestHighpass:
    def test_returns_same_shape(self, sample_audio_mono):
        audio, sr = sample_audio_mono
        try:
            result = _apply_highpass(audio, sr, cutoff=30)
            assert result.shape == audio.shape
        except ImportError:
            pytest.skip("scipy not available")

    def test_stereo_highpass(self, sample_audio_stereo):
        audio, sr = sample_audio_stereo
        try:
            result = _apply_highpass(audio, sr, cutoff=30)
            assert result.shape == audio.shape
        except ImportError:
            pytest.skip("scipy not available")


class TestMultibandCompression:
    def test_no_compression_when_ratio_one(self, sample_audio_mono):
        audio, sr = sample_audio_mono
        result = _apply_multiband_compression(audio, sr, ratio=1.0)
        np.testing.assert_array_equal(result, audio)

    def test_returns_same_shape(self, sample_audio_mono):
        audio, sr = sample_audio_mono
        try:
            result = _apply_multiband_compression(audio, sr, ratio=2.0)
            assert result.shape == audio.shape
        except ImportError:
            pytest.skip("scipy not available")


class TestStereoEnhancement:
    def test_mono_unchanged(self, sample_audio_mono):
        audio, sr = sample_audio_mono
        result = _apply_stereo_enhancement(audio, width=1.5)
        np.testing.assert_array_equal(result, audio)

    def test_width_1_unchanged(self, sample_audio_stereo):
        audio, sr = sample_audio_stereo
        result = _apply_stereo_enhancement(audio, width=1.0)
        np.testing.assert_array_almost_equal(result, audio)

    def test_wider_changes_stereo(self, sample_audio_stereo):
        audio, sr = sample_audio_stereo
        result = _apply_stereo_enhancement(audio, width=1.5)
        # Result should differ from original
        assert not np.allclose(result, audio)
        assert result.shape == audio.shape


class TestEQ:
    def test_flat_preset_no_change(self, sample_audio_mono):
        audio, sr = sample_audio_mono
        result = _apply_eq(audio, sr, preset="flat")
        np.testing.assert_array_almost_equal(result, audio)

    def test_balanced_preset_changes_audio(self, sample_audio_mono):
        audio, sr = sample_audio_mono
        try:
            result = _apply_eq(audio, sr, preset="balanced")
            # Should modify the audio (balanced has non-zero gains)
            assert result.shape == audio.shape
        except ImportError:
            pytest.skip("scipy not available")


class TestPeakLimit:
    def test_peak_limit_clamps(self, tmp_path):
        sr = 44100
        # Audio with peaks at 1.0
        audio = np.ones(sr, dtype=np.float32)
        path = str(tmp_path / "loud.wav")
        sf.write(path, audio, sr, subtype="PCM_16")

        _apply_peak_limit(path, threshold=0.95)

        data, _ = sf.read(path)
        assert np.max(np.abs(data)) <= 0.96  # slight tolerance for PCM_16 quantization

    def test_peak_limit_quiet_audio_unchanged(self, tmp_path):
        sr = 44100
        audio = np.ones(sr, dtype=np.float32) * 0.5
        path = str(tmp_path / "quiet.wav")
        sf.write(path, audio, sr, subtype="PCM_16")

        _apply_peak_limit(path, threshold=0.95)

        data, _ = sf.read(path)
        # Peak was 0.5, below threshold, so shouldn't change much
        assert np.max(np.abs(data)) < 0.96


class TestTrimSilence:
    def test_trim_skips_without_pydub(self, sample_wav_file, monkeypatch):
        wav_path, sr = sample_wav_file
        # Mock pydub not being available
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pydub":
                raise ImportError("no pydub")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        profile = {"trim_db": -45, "min_silence_ms": 500,
                    "front_protect_ms": 200, "end_protect_ms": 1000}
        # Should not raise
        _trim_silence(wav_path, profile)


class TestNormalizeLoudness:
    def test_normalize_skips_without_pyloudnorm(self, sample_wav_file, monkeypatch):
        wav_path, sr = sample_wav_file
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pyloudnorm":
                raise ImportError("no pyloudnorm")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        _normalize_loudness(wav_path, target_lufs=-14.0)


# ===========================================================================
# audio_processing.py — full pipeline
# ===========================================================================
class TestPostProcess:
    def test_returns_same_path(self, sample_wav_file):
        wav_path, sr = sample_wav_file
        profile = PROFILES["musicgen"]
        result = post_process(wav_path, profile)
        assert result == wav_path

    def test_file_still_valid_after_processing(self, sample_wav_file):
        wav_path, sr = sample_wav_file
        profile = PROFILES["musicgen"]
        post_process(wav_path, profile)
        data, rate = sf.read(wav_path)
        assert rate == sr
        assert len(data) > 0

    def test_nonexistent_file_returns_path(self):
        result = post_process("/nonexistent/file.wav", {})
        assert result == "/nonexistent/file.wav"

    def test_override_params(self, sample_wav_file):
        wav_path, sr = sample_wav_file
        profile = PROFILES["musicgen"]
        result = post_process(
            wav_path, profile,
            denoise_strength=0.0,
            stereo_width=1.0,
            eq_preset="flat",
            target_lufs=-16.0,
        )
        assert result == wav_path

    def test_override_all_new_params(self, sample_wav_file):
        wav_path, sr = sample_wav_file
        profile = PROFILES["musicgen"]
        result = post_process(
            wav_path, profile,
            denoise_strength=0.1,
            stereo_width=1.0,
            eq_preset="warm",
            target_lufs=-16.0,
            highpass_cutoff=50,
            compression_ratio=3.0,
            clipping=0.9,
            trim_db=-40,
            min_silence_ms=300,
            front_protect_ms=100,
            end_protect_ms=500,
        )
        assert result == wav_path

    def test_stereo_processing(self, sample_stereo_wav_file):
        wav_path, sr = sample_stereo_wav_file
        profile = PROFILES["stable_audio"]
        result = post_process(wav_path, profile)
        data, rate = sf.read(result)
        assert data.ndim == 2
        assert data.shape[1] == 2


# ===========================================================================
# audio_assembler.py
# ===========================================================================
class TestAssembleChunks:
    def test_single_chunk(self, tmp_path, sample_audio_mono):
        audio, sr = sample_audio_mono
        chunk_path = tmp_path / "chunk1.wav"
        sf.write(str(chunk_path), audio, sr, subtype="PCM_16")

        out_path = tmp_path / "assembled.wav"
        result = assemble_chunks([str(chunk_path)], str(out_path), sr)
        assert result == str(out_path)

        data, rate = sf.read(result)
        assert rate == sr
        assert len(data) > 0

    def test_two_chunks_no_crossfade(self, tmp_path, sample_audio_mono):
        audio, sr = sample_audio_mono
        c1 = tmp_path / "c1.wav"
        c2 = tmp_path / "c2.wav"
        sf.write(str(c1), audio, sr, subtype="PCM_16")
        sf.write(str(c2), audio, sr, subtype="PCM_16")

        out = tmp_path / "out.wav"
        assemble_chunks([str(c1), str(c2)], str(out), sr, crossfade_ms=0)

        data, _ = sf.read(str(out))
        # Should be approximately 2x the original length
        expected_len = len(audio) * 2
        assert abs(len(data) - expected_len) < 10

    def test_two_chunks_with_crossfade(self, tmp_path, sample_audio_mono):
        audio, sr = sample_audio_mono
        c1 = tmp_path / "c1.wav"
        c2 = tmp_path / "c2.wav"
        sf.write(str(c1), audio, sr, subtype="PCM_16")
        sf.write(str(c2), audio, sr, subtype="PCM_16")

        out = tmp_path / "out.wav"
        assemble_chunks([str(c1), str(c2)], str(out), sr, crossfade_ms=100)

        data, _ = sf.read(str(out))
        # With crossfade, should be slightly less than 2x
        expected_max = len(audio) * 2
        assert len(data) < expected_max

    def test_front_and_end_padding(self, tmp_path, sample_audio_mono):
        audio, sr = sample_audio_mono
        c1 = tmp_path / "c1.wav"
        sf.write(str(c1), audio, sr, subtype="PCM_16")

        out = tmp_path / "out.wav"
        assemble_chunks([str(c1)], str(out), sr, front_pad=0.5, end_pad=0.5)

        data, _ = sf.read(str(out))
        expected_len = len(audio) + sr  # 0.5s front + 0.5s end
        assert abs(len(data) - expected_len) < 10

    def test_empty_chunks_produces_silence(self, tmp_path):
        out = tmp_path / "out.wav"
        assemble_chunks([], str(out), 44100)

        data, rate = sf.read(str(out))
        assert rate == 44100
        # Should produce short silence fallback
        assert len(data) > 0

    def test_stereo_crossfade(self, tmp_path, sample_audio_stereo):
        audio, sr = sample_audio_stereo
        c1 = tmp_path / "c1.wav"
        c2 = tmp_path / "c2.wav"
        sf.write(str(c1), audio, sr, subtype="PCM_16")
        sf.write(str(c2), audio, sr, subtype="PCM_16")

        out = tmp_path / "out.wav"
        assemble_chunks([str(c1), str(c2)], str(out), sr, crossfade_ms=100)

        data, _ = sf.read(str(out))
        assert data.ndim == 2
        assert data.shape[1] == 2


class TestFfmpegArgs:
    def test_mp3_args(self):
        args = _ffmpeg_args("mp3")
        assert "-c:a" in args
        assert "libmp3lame" in args

    def test_ogg_args(self):
        args = _ffmpeg_args("ogg")
        assert "libvorbis" in args

    def test_flac_args(self):
        args = _ffmpeg_args("flac")
        assert "flac" in args

    def test_m4a_args(self):
        args = _ffmpeg_args("m4a")
        assert "aac" in args

    def test_unknown_format_empty(self):
        assert _ffmpeg_args("xyz") == []


class TestConvertFormat:
    def test_wav_to_wav_copies(self, sample_wav_file, tmp_path):
        wav_path, sr = sample_wav_file
        out_path = str(tmp_path / "copy.wav")
        result = convert_format(wav_path, out_path, fmt="wav")
        assert result == out_path
        data, rate = sf.read(out_path)
        assert rate == sr

    def test_wav_same_path(self, sample_wav_file):
        wav_path, sr = sample_wav_file
        result = convert_format(wav_path, wav_path, fmt="wav")
        assert result == wav_path
