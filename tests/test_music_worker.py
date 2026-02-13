"""Tests for music_worker.py — dispatch tables and endpoint logic.

Note: music_worker.py parses args at module level, so we must set sys.argv
before importing. Tests focus on structural correctness and the API endpoints.
"""

import sys
import base64
import io
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Module import helper — sets sys.argv to satisfy argparse
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def import_music_worker():
    """Patch sys.argv so music_worker can be imported without real CLI args."""
    original_argv = sys.argv
    sys.argv = ["music_worker.py", "--model", "musicgen", "--port", "9199", "--device", "cpu"]
    try:
        # Also need to stub uvicorn.run to prevent auto-start
        with patch.dict("sys.modules", {}):
            pass
        yield
    finally:
        sys.argv = original_argv


def _get_worker_module():
    """Import music_worker with patched argv."""
    original_argv = sys.argv
    sys.argv = ["music_worker.py", "--model", "musicgen", "--port", "9199", "--device", "cpu"]
    try:
        import importlib
        import music_worker
        importlib.reload(music_worker)
        return music_worker
    finally:
        sys.argv = original_argv


# ---------------------------------------------------------------------------
# Dispatch table structure
# ---------------------------------------------------------------------------
class TestDispatchTables:
    def test_loaders_has_all_models(self):
        mw = _get_worker_module()
        expected = {
            "musicgen", "ace_step_v15", "ace_step_v1", "heartmula",
            "diffrhythm", "yue", "riffusion", "stable_audio",
        }
        assert set(mw.LOADERS.keys()) == expected

    def test_inferencers_has_all_models(self):
        mw = _get_worker_module()
        expected = {
            "musicgen", "ace_step_v15", "ace_step_v1", "heartmula",
            "diffrhythm", "yue", "riffusion", "stable_audio",
        }
        assert set(mw.INFERENCERS.keys()) == expected

    def test_loaders_match_inferencers(self):
        mw = _get_worker_module()
        assert set(mw.LOADERS.keys()) == set(mw.INFERENCERS.keys())

    def test_all_loaders_are_callable(self):
        mw = _get_worker_module()
        for name, loader in mw.LOADERS.items():
            assert callable(loader), f"Loader for {name} is not callable"

    def test_all_inferencers_are_callable(self):
        mw = _get_worker_module()
        for name, inferencer in mw.INFERENCERS.items():
            assert callable(inferencer), f"Inferencer for {name} is not callable"


# ---------------------------------------------------------------------------
# Generic loader/inferencer
# ---------------------------------------------------------------------------
class TestGenericFallback:
    def test_generic_loader(self):
        mw = _get_worker_module()
        result = mw._load_generic("test_model", "cpu")
        assert result["model_id"] == "test_model"
        assert result["loaded"] is True

    def test_generic_inferencer_returns_silence(self):
        mw = _get_worker_module()
        model = {"model_id": "test", "device": "cpu", "loaded": True}
        audio, sr = mw._infer_generic(model, {"duration": 2}, "cpu")
        assert sr == 44100
        assert len(audio) == 44100 * 2
        assert np.all(audio == 0)


# ---------------------------------------------------------------------------
# VRAM helper
# ---------------------------------------------------------------------------
class TestVramHelper:
    def test_vram_info_no_gpu(self):
        mw = _get_worker_module()
        info = mw._get_vram_info("cpu")
        assert info["vram_used_mb"] == 0
        assert info["vram_total_mb"] == 0


# ---------------------------------------------------------------------------
# Spectrogram to audio (Riffusion helper)
# ---------------------------------------------------------------------------
class TestSpectrogramToAudio:
    """Test _spectrogram_image_to_audio with mocked riffusion/torchaudio."""

    def _mock_spectrogram_deps(self):
        """Set up mocks for _spectrogram_image_to_audio's internal imports."""
        # Mock SpectrogramParams - frozen dataclass with known defaults
        mock_params = MagicMock()
        mock_params.power_for_image = 0.25
        mock_params.stereo = False
        mock_params.n_fft = 17640
        mock_params.num_frequencies = 512
        mock_params.sample_rate = 44100
        mock_params.min_frequency = 0
        mock_params.max_frequency = 10000
        mock_params.mel_scale_norm = None
        mock_params.mel_scale_type = "htk"
        mock_params.num_griffin_lim_iters = 32
        mock_params.win_length = 4410
        mock_params.hop_length = 441

        mock_spec_params_mod = MagicMock()
        mock_spec_params_mod.SpectrogramParams.return_value = mock_params

        # Mock image_util.spectrogram_from_image → returns (1, 512, time) array
        mock_image_util = MagicMock()
        mock_image_util.spectrogram_from_image.return_value = \
            np.random.rand(1, 512, 100).astype(np.float32) * 1000

        # Mock torch
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.hann_window = MagicMock()

        # Build a fake waveform with peak > 1 to verify normalization
        fake_waveform = np.random.randn(44100).astype(np.float32) * 2.0

        # Mock chain: waveform[0].cpu().numpy() → fake_waveform
        mock_cpu_result = MagicMock()
        mock_cpu_result.numpy.return_value = fake_waveform
        mock_channel = MagicMock()
        mock_channel.cpu.return_value = mock_cpu_result

        mock_waveform = MagicMock()
        mock_waveform.__getitem__ = MagicMock(return_value=mock_channel)

        # GriffinLim instance returns waveform tensor
        mock_griffin_lim = MagicMock(return_value=mock_waveform)
        mock_griffin_lim.to.return_value = mock_griffin_lim

        # InverseMelScale instance returns linear amplitudes
        mock_inv_mel = MagicMock()
        mock_inv_mel.to.return_value = mock_inv_mel

        mock_torchaudio = MagicMock()
        mock_torchaudio.transforms.InverseMelScale.return_value = mock_inv_mel
        mock_torchaudio.transforms.GriffinLim.return_value = mock_griffin_lim

        mock_torch.from_numpy.return_value.to.return_value = MagicMock()

        return {
            "torch": mock_torch,
            "torchaudio": mock_torchaudio,
            "torchaudio.transforms": mock_torchaudio.transforms,
            "riffusion": MagicMock(),
            "riffusion.spectrogram_params": mock_spec_params_mod,
            "riffusion.util": MagicMock(),
            "riffusion.util.image_util": mock_image_util,
        }

    def test_returns_float32_array(self):
        mw = _get_worker_module()
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        modules = self._mock_spectrogram_deps()
        with patch.dict("sys.modules", modules):
            audio = mw._spectrogram_image_to_audio(img, 44100)
        assert audio.dtype == np.float32
        assert len(audio) > 0

    def test_output_is_normalized(self):
        mw = _get_worker_module()
        from PIL import Image
        img = Image.fromarray(np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8))
        modules = self._mock_spectrogram_deps()
        with patch.dict("sys.modules", modules):
            audio = mw._spectrogram_image_to_audio(img, 44100)
        peak = np.max(np.abs(audio))
        assert peak <= 1.0


# ---------------------------------------------------------------------------
# FastAPI endpoints (via TestClient)
# ---------------------------------------------------------------------------
class TestWorkerEndpoints:
    def _get_client(self):
        mw = _get_worker_module()
        from fastapi.testclient import TestClient
        return TestClient(mw.app), mw

    def test_health_idle(self):
        client, mw = self._get_client()
        mw._model_loaded = False
        mw._loading = False
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "idle"
        assert data["model"] == "musicgen"

    def test_health_loading(self):
        client, mw = self._get_client()
        mw._model_loaded = False
        mw._loading = True
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "loading"
        mw._loading = False

    def test_health_ready(self):
        client, mw = self._get_client()
        mw._model_loaded = True
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ready"
        mw._model_loaded = False

    def test_load_calls_loader(self):
        client, mw = self._get_client()
        mw._model_loaded = False
        mw._model = None

        mock_model = {"loaded": True}
        with patch.dict(mw.LOADERS, {"musicgen": lambda device: mock_model}):
            resp = client.post("/load")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "loaded"
        assert mw._model_loaded is True
        # Reset
        mw._model_loaded = False
        mw._model = None

    def test_load_already_loaded(self):
        client, mw = self._get_client()
        mw._model_loaded = True
        resp = client.post("/load")
        data = resp.json()
        assert data["status"] == "already_loaded"
        mw._model_loaded = False

    def test_infer_without_load_returns_503(self):
        client, mw = self._get_client()
        mw._model_loaded = False
        resp = client.post("/infer", json={"params": {}})
        assert resp.status_code == 503

    def test_infer_returns_audio(self):
        client, mw = self._get_client()
        mw._model_loaded = True
        mw._model = {"loaded": True}

        fake_audio = np.zeros(44100, dtype=np.float32)

        def fake_inferencer(model, params, device):
            return fake_audio, 44100

        with patch.dict(mw.INFERENCERS, {"musicgen": fake_inferencer}):
            resp = client.post("/infer", json={"params": {"prompt": "test"}})

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "audio_base64" in data
        assert data["sample_rate"] == 44100
        # Reset
        mw._model_loaded = False
        mw._model = None

    def test_unload_frees_model(self):
        client, mw = self._get_client()
        mw._model_loaded = True
        mw._model = {"loaded": True}

        resp = client.post("/unload")
        data = resp.json()
        assert data["status"] == "unloaded"
        assert mw._model_loaded is False
        assert mw._model is None

    def test_unload_when_not_loaded(self):
        client, mw = self._get_client()
        mw._model_loaded = False
        resp = client.post("/unload")
        data = resp.json()
        assert data["status"] == "not_loaded"


# ---------------------------------------------------------------------------
# YuE loader path correctness
# ---------------------------------------------------------------------------
class TestYuELoaderPaths:
    """Verify _load_yue references inference/ subdirectory for tokenizer and xcodec."""

    def test_tokenizer_path_uses_inference_dir(self):
        """tokenizer_path must be under inference/, not repo root."""
        import inspect
        mw = _get_worker_module()
        source = inspect.getsource(mw._load_yue)
        # Should reference inference_path for tokenizer, not repo_path
        assert 'inference_path / "mm_tokenizer_v0.2_hf"' in source
        assert 'repo_path / "mm_tokenizer_v0.2_hf"' not in source

    def test_xcodec_path_uses_inference_dir(self):
        """xcodec_path must be under inference/, not repo root."""
        import inspect
        mw = _get_worker_module()
        source = inspect.getsource(mw._load_yue)
        # Should reference inference_path for xcodec, not repo_path
        assert 'inference_path / "xcodec_mini_infer"' in source
        assert 'repo_path / "xcodec_mini_infer"' not in source

    def test_inference_path_defined_before_use(self):
        """inference_path must be defined from repo_path / 'inference'."""
        import inspect
        mw = _get_worker_module()
        source = inspect.getsource(mw._load_yue)
        assert 'inference_path = repo_path / "inference"' in source

    def test_chdir_to_inference_before_xcodec(self):
        """CWD must be set to inference_path before SoundStream construction.

        XCodec's SoundStream uses a hardcoded relative path
        './xcodec_mini_infer/semantic_ckpts/...' so CWD must be inference/.
        """
        import inspect
        mw = _get_worker_module()
        source = inspect.getsource(mw._load_yue)
        chdir_pos = source.find('os.chdir(str(inference_path))')
        soundstream_pos = source.find('from models.soundstream_hubert_new')
        assert chdir_pos != -1, "os.chdir(str(inference_path)) not found in _load_yue"
        assert soundstream_pos != -1, "SoundStream import not found in _load_yue"
        assert chdir_pos < soundstream_pos, "os.chdir must come before SoundStream import"

    def test_attn_fallback_chain_includes_sdpa(self):
        """Attention fallback must try sdpa before eager for performance."""
        import inspect
        mw = _get_worker_module()
        source = inspect.getsource(mw._load_yue)
        assert 'scaled_dot_product_attention' in source, \
            "SDPA check missing from _load_yue attention fallback"
        sdpa_pos = source.find('"sdpa"')
        eager_pos = source.find('"eager"')
        assert sdpa_pos != -1, "sdpa attn_impl not found"
        assert eager_pos != -1, "eager attn_impl not found"
        assert sdpa_pos < eager_pos, "sdpa must be tried before eager"
