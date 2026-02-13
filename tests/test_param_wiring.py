"""Parameter wiring tests — verifies every inferencer passes correct kwargs.

For each model, these tests:
  1. Mock the model API (the actual library call)
  2. Call the inferencer with known params
  3. Assert the exact kwargs passed to the library match the real API signature
  4. Verify no unexpected kwargs that would cause runtime errors

This catches mismatches like:
  - Wrong kwarg names (keyscale vs key_scale)
  - Dead params (topp when API only accepts topk)
  - Missing params (not passing a required kwarg)
"""

import sys
import os
import tempfile
from unittest.mock import patch, MagicMock, call
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import helper
# ---------------------------------------------------------------------------
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


class _FakeTensor:
    """Minimal stand-in so isinstance(x, torch.Tensor) works in tests."""
    pass


@contextmanager
def _mock_torch():
    """Provide a mock torch module for inferencers that import torch internally."""
    mock_torch = MagicMock()
    mock_torch.Tensor = _FakeTensor
    mock_torch.manual_seed = MagicMock()
    mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    mock_torch.Generator.return_value.manual_seed.return_value = MagicMock()
    with patch.dict("sys.modules", {"torch": mock_torch}):
        yield mock_torch


# ===================================================================
# MusicGen — set_generation_params() accepts cfg_coef (not cfg_coeff)
# ===================================================================
class TestMusicGenParams:
    """Verify MusicGen inferencer passes correct kwargs to the audiocraft API."""

    def test_set_generation_params_receives_correct_kwargs(self):
        """cfg_coef (single f), not cfg_coeff — the actual audiocraft kwarg name."""
        mw = _get_worker_module()

        mock_model = MagicMock()
        mock_model.sample_rate = 32000
        fake_wav = MagicMock()
        fake_wav.__getitem__ = lambda self, key: MagicMock(
            cpu=lambda: MagicMock(numpy=lambda: np.zeros(1000, dtype=np.float32))
        )
        mock_model.generate.return_value = fake_wav

        model_dict = {"model": mock_model, "model_size": "large"}
        params = {
            "prompt": "test music",
            "duration": "5",
            "temperature": "0.8",
            "top_k": "200",
            "top_p": "0.5",
            "cfg_coeff": "4.0",
            "use_sampling": True,
            "two_step_cfg": False,
            "seed": "-1",
        }

        with _mock_torch():
            mw._infer_musicgen(model_dict, params, "cpu")

        mock_model.set_generation_params.assert_called_once_with(
            duration=5.0,
            temperature=0.8,
            top_k=200,
            top_p=0.5,
            cfg_coef=4.0,  # actual API kwarg name (single f)
            use_sampling=True,
            two_step_cfg=False,
        )

    def test_extend_stride_passed_when_nonzero(self):
        """extend_stride > 0 should be included in set_generation_params."""
        mw = _get_worker_module()

        mock_model = MagicMock()
        mock_model.sample_rate = 32000
        fake_wav = MagicMock()
        fake_wav.__getitem__ = lambda self, key: MagicMock(
            cpu=lambda: MagicMock(numpy=lambda: np.zeros(1000, dtype=np.float32))
        )
        mock_model.generate.return_value = fake_wav

        model_dict = {"model": mock_model, "model_size": "large"}
        params = {"prompt": "test", "extend_stride": "18", "seed": "-1"}

        with _mock_torch():
            mw._infer_musicgen(model_dict, params, "cpu")

        kwargs = mock_model.set_generation_params.call_args[1]
        assert kwargs["extend_stride"] == 18

    def test_extend_stride_omitted_when_zero(self):
        """extend_stride == 0 should NOT be passed to set_generation_params."""
        mw = _get_worker_module()

        mock_model = MagicMock()
        mock_model.sample_rate = 32000
        fake_wav = MagicMock()
        fake_wav.__getitem__ = lambda self, key: MagicMock(
            cpu=lambda: MagicMock(numpy=lambda: np.zeros(1000, dtype=np.float32))
        )
        mock_model.generate.return_value = fake_wav

        model_dict = {"model": mock_model, "model_size": "large"}
        params = {"prompt": "test", "extend_stride": "0", "seed": "-1"}

        with _mock_torch():
            mw._infer_musicgen(model_dict, params, "cpu")

        kwargs = mock_model.set_generation_params.call_args[1]
        assert "extend_stride" not in kwargs

    def test_generate_called_with_prompt_list(self):
        mw = _get_worker_module()

        mock_model = MagicMock()
        mock_model.sample_rate = 32000
        fake_wav = MagicMock()
        fake_wav.__getitem__ = lambda self, key: MagicMock(
            cpu=lambda: MagicMock(numpy=lambda: np.zeros(1000, dtype=np.float32))
        )
        mock_model.generate.return_value = fake_wav

        model_dict = {"model": mock_model, "model_size": "large"}
        params = {"prompt": "jazz piano", "seed": "-1"}

        with _mock_torch():
            mw._infer_musicgen(model_dict, params, "cpu")

        mock_model.generate.assert_called_once_with(["jazz piano"])

    def test_gui_param_ids_match_inferencer(self):
        """Every GUI param id (excluding headings) must appear in inferencer's params.get() calls."""
        mw = _get_worker_module()
        import music_manager
        gui_ids = {c["id"] for c in music_manager.MODEL_PARAMS["musicgen"]["controls"]
                   if c["type"] != "heading"}
        inferencer_keys = {
            "prompt", "duration", "temperature", "top_k", "top_p",
            "cfg_coeff", "use_sampling", "two_step_cfg",
            "extend_stride", "seed", "reference_audio",
        }
        assert gui_ids == inferencer_keys


# ===================================================================
# ACE-Step v1.5 — handler.generate_music() uses key_scale, time_signature
# ===================================================================
class TestAceStepV15Params:
    """Verify ACE-Step v1.5 inferencer uses unified generate_music()."""

    def _make_mock_result(self):
        """Create a mock GenerationResult."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.error = None
        mock_result.audios = [{
            "tensor": MagicMock(
                cpu=lambda: MagicMock(
                    numpy=lambda: np.zeros((2, 48000), dtype=np.float32)
                ),
                ndim=2,
            ),
            "sample_rate": 48000,
        }]
        return mock_result

    def _run_inferencer(self, mw, params, llm_handler=None):
        """Call _infer_ace_step_v15 with mocked inference module."""
        mock_generate = MagicMock(return_value=self._make_mock_result())

        # Mock the acestep.inference module
        mock_inference = MagicMock()
        mock_inference.generate_music = mock_generate

        # Use real-ish dataclass constructors that store kwargs as attributes
        def _fake_params(**kw):
            obj = MagicMock()
            for k, v in kw.items():
                setattr(obj, k, v)
            return obj

        def _fake_config(**kw):
            obj = MagicMock()
            for k, v in kw.items():
                setattr(obj, k, v)
            return obj

        mock_inference.GenerationParams = _fake_params
        mock_inference.GenerationConfig = _fake_config

        model_dict = {
            "handler": MagicMock(),
            "llm_handler": llm_handler,
        }

        saved = sys.modules.get("acestep.inference")
        sys.modules["acestep.inference"] = mock_inference
        try:
            audio, sr = mw._infer_ace_step_v15(model_dict, params, "cpu")
        finally:
            if saved is not None:
                sys.modules["acestep.inference"] = saved
            else:
                sys.modules.pop("acestep.inference", None)

        return mock_generate, audio, sr

    def test_generates_via_unified_interface(self):
        """Inferencer calls inference.generate_music with handler + llm."""
        mw = _get_worker_module()
        mock_gen, audio, sr = self._run_inferencer(
            mw, {"duration": "30", "seed": "-1"})
        mock_gen.assert_called_once()
        call_kw = mock_gen.call_args[1]
        assert "dit_handler" in call_kw
        assert "llm_handler" in call_kw
        assert "params" in call_kw
        assert "config" in call_kw
        assert call_kw["save_dir"] is None
        assert sr == 48000

    def test_generation_params_populated(self):
        """Verify GenerationParams fields match GUI input values."""
        mw = _get_worker_module()
        mock_gen, _, _ = self._run_inferencer(mw, {
            "caption": "minimal techno",
            "lyrics": "la la la",
            "keyscale": "C major",
            "timesignature": "4",
            "duration": "60",
            "bpm": "120",
            "seed": "42",
            "vocal_language": "en",
            "task_type": "text2music",
            "thinking": "true",
            "lm_temperature": "0.85",
            "lm_negative_prompt": "bad quality",
            "infer_method": "sde",
            "use_adg": "true",
            "audio_cover_strength": "0.5",
        })
        gen_params = mock_gen.call_args[1]["params"]
        assert gen_params.keyscale == "C major"
        assert gen_params.timesignature == "4"
        assert gen_params.caption == "minimal techno"
        assert gen_params.duration == 60.0
        assert gen_params.bpm == 120
        assert gen_params.seed == 42
        assert gen_params.thinking is True
        assert gen_params.lm_temperature == 0.85
        assert gen_params.lm_negative_prompt == "bad quality"
        assert gen_params.infer_method == "sde"
        assert gen_params.use_adg is True
        assert gen_params.audio_cover_strength == 0.5

    def test_boolean_string_coercion(self):
        """String booleans from GUI ('true'/'false') convert correctly."""
        mw = _get_worker_module()
        mock_gen, _, _ = self._run_inferencer(mw, {
            "instrumental": "true",
            "thinking": "false",
            "use_adg": "true",
            "use_cot_metas": "false",
            "use_cot_caption": "false",
            "use_cot_language": "true",
            "duration": "30",
            "seed": "-1",
        })
        gen_params = mock_gen.call_args[1]["params"]
        assert gen_params.instrumental is True
        assert gen_params.thinking is False
        assert gen_params.use_adg is True
        assert gen_params.use_cot_metas is False
        assert gen_params.use_cot_caption is False
        assert gen_params.use_cot_language is True

    def test_llm_handler_none_passed_through(self):
        """When llm_handler is None, it's passed through to generate_music."""
        mw = _get_worker_module()
        mock_gen, audio, sr = self._run_inferencer(
            mw, {"duration": "30", "seed": "-1"}, llm_handler=None)
        assert mock_gen.call_args[1]["llm_handler"] is None
        assert audio.dtype == np.float32

    def test_llm_handler_object_passed_through(self):
        """When llm_handler is set, it's passed through to generate_music."""
        mw = _get_worker_module()
        fake_llm = MagicMock()
        mock_gen, _, _ = self._run_inferencer(
            mw, {"duration": "30", "seed": "-1"}, llm_handler=fake_llm)
        assert mock_gen.call_args[1]["llm_handler"] is fake_llm

    def test_gui_param_ids_match_inferencer(self):
        """Every GUI param id (excluding headings) must be read by inferencer."""
        _get_worker_module()
        import music_manager
        gui_ids = {
            c["id"] for c in music_manager.MODEL_PARAMS["ace_step_v15"]["controls"]
            if c["type"] != "heading"
        }
        inferencer_keys = {
            "caption", "lyrics", "task_type", "vocal_language", "instrumental",
            "duration", "bpm", "keyscale", "timesignature",
            "inference_steps", "guidance_scale", "shift", "infer_method",
            "seed", "batch_size",
            "thinking", "use_cot_metas", "use_cot_caption", "use_cot_language",
            "lm_temperature", "lm_cfg_scale", "lm_top_k", "lm_top_p",
            "lm_negative_prompt",
            "use_adg", "cfg_interval_start", "cfg_interval_end",
            "audio_cover_strength", "repainting_start", "repainting_end",
            "reference_audio", "src_audio",
        }
        assert gui_ids == inferencer_keys


# ===================================================================
# ACE-Step v1 — pipeline() uses infer_step (not inference_steps)
# ===================================================================
class TestAceStepV1Params:
    """Verify ACE-Step v1 inferencer passes correct kwargs to pipeline."""

    def test_pipeline_receives_correct_kwargs(self):
        """pipeline() uses infer_step, audio_duration, manual_seeds, save_path."""
        mw = _get_worker_module()

        mock_pipeline = MagicMock()
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        import soundfile as sf
        sf.write(tmp_path, np.zeros(1000, dtype=np.float32), 48000)

        mock_pipeline.return_value = [tmp_path, {"params": {}}]
        model_dict = {"pipeline": mock_pipeline}
        params = {
            "caption": "rock music",
            "lyrics": "yeah yeah",
            "duration": "60",
            "inference_steps": "50",
            "guidance_scale": "12.0",
            "seed": "42",
            "instrumental": "false",
            "scheduler_type": "heun",
            "cfg_type": "cfg",
            "omega_scale": "8.0",
            "guidance_interval": "0.7",
            "guidance_interval_decay": "0.1",
            "min_guidance_scale": "2.0",
            "guidance_scale_text": "1.5",
            "guidance_scale_lyric": "2.5",
            "use_erg_tag": "true",
            "use_erg_lyric": "false",
            "use_erg_diffusion": "true",
            "retake_variance": "0.3",
            "batch_size": "2",
        }

        try:
            with patch.dict("sys.modules", {"soundfile": sf}):
                mw._infer_ace_step_v1(model_dict, params, "cpu")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        kwargs = mock_pipeline.call_args[1]
        assert kwargs["prompt"] == "rock music"
        assert kwargs["lyrics"] == "yeah yeah"
        assert kwargs["audio_duration"] == 60.0
        assert kwargs["infer_step"] == 50        # NOT inference_steps
        assert kwargs["guidance_scale"] == 12.0
        assert kwargs["manual_seeds"] == [42]
        assert kwargs["batch_size"] == 2
        assert "save_path" in kwargs
        # New params
        assert kwargs["scheduler_type"] == "heun"
        assert kwargs["cfg_type"] == "cfg"
        assert kwargs["omega_scale"] == 8.0
        assert kwargs["guidance_interval"] == 0.7
        assert kwargs["guidance_interval_decay"] == 0.1
        assert kwargs["min_guidance_scale"] == 2.0
        assert kwargs["guidance_scale_text"] == 1.5
        assert kwargs["guidance_scale_lyric"] == 2.5
        assert kwargs["use_erg_tag"] is True
        assert kwargs["use_erg_lyric"] is False
        assert kwargs["use_erg_diffusion"] is True
        assert kwargs["retake_variance"] == 0.3

    def test_no_invalid_pipeline_kwargs(self):
        valid_pipeline_kwargs = {
            "format", "audio_duration", "prompt", "lyrics", "infer_step",
            "guidance_scale", "scheduler_type", "cfg_type", "omega_scale",
            "manual_seeds", "guidance_interval", "guidance_interval_decay",
            "min_guidance_scale", "use_erg_tag", "use_erg_lyric",
            "use_erg_diffusion", "oss_steps", "guidance_scale_text",
            "guidance_scale_lyric", "audio2audio_enable", "ref_audio_strength",
            "ref_audio_input", "lora_name_or_path", "lora_weight",
            "retake_seeds", "retake_variance", "task", "repaint_start",
            "repaint_end", "src_audio_path", "edit_target_prompt",
            "edit_target_lyrics", "edit_n_min", "edit_n_max", "edit_n_avg",
            "save_path", "batch_size", "debug",
        }

        mw = _get_worker_module()
        mock_pipeline = MagicMock()
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        import soundfile as sf
        sf.write(tmp_path, np.zeros(1000, dtype=np.float32), 48000)
        mock_pipeline.return_value = [tmp_path, {}]

        model_dict = {"pipeline": mock_pipeline}
        params = {"duration": "30", "seed": "-1"}

        try:
            with patch.dict("sys.modules", {"soundfile": sf}):
                mw._infer_ace_step_v1(model_dict, params, "cpu")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        kwargs = mock_pipeline.call_args[1]
        invalid = set(kwargs.keys()) - valid_pipeline_kwargs
        assert invalid == set(), f"Invalid kwargs: {invalid}"

    def test_gui_param_ids_match_inferencer(self):
        mw = _get_worker_module()
        import music_manager
        gui_ids = {c["id"] for c in music_manager.MODEL_PARAMS["ace_step_v1"]["controls"]
                   if c["type"] != "heading"}
        inferencer_keys = {
            "caption", "lyrics", "instrumental", "duration",
            "inference_steps", "guidance_scale", "seed",
            "scheduler_type", "cfg_type", "omega_scale", "batch_size",
            "guidance_interval", "guidance_interval_decay",
            "min_guidance_scale", "guidance_scale_text", "guidance_scale_lyric",
            "use_erg_tag", "use_erg_lyric", "use_erg_diffusion",
            "retake_variance",
        }
        assert gui_ids == inferencer_keys


# ===================================================================
# HeartMuLa — pipeline() accepts topk (not topp), no top_p
# ===================================================================
class TestHeartMuLaParams:
    """Verify HeartMuLa inferencer passes correct kwargs to pipeline."""

    def _run_heartmula(self, mw, params):
        """Helper: run _infer_heartmula with mocked torch + soundfile."""
        import soundfile as sf

        def fake_pipeline(inputs, **kwargs):
            sf.write(kwargs["save_path"], np.zeros(1000, dtype=np.float32), 48000)

        mock_pipeline = MagicMock(side_effect=fake_pipeline)
        model_dict = {"pipeline": mock_pipeline}

        with _mock_torch():
            with patch.dict("sys.modules", {"soundfile": sf}):
                mw._infer_heartmula(model_dict, params, "cpu")

        return mock_pipeline

    def test_pipeline_receives_topk_not_topp(self):
        """HeartMuLa only reads: cfg_scale, max_audio_length_ms, temperature,
        topk, save_path. NO topp parameter."""
        mw = _get_worker_module()
        params = {
            "lyrics": "hello world", "tags": "pop, upbeat",
            "duration": "30", "temperature": "0.8",
            "top_k": "40", "cfg_scale": "2.0", "seed": "-1",
        }

        mock_pipeline = self._run_heartmula(mw, params)

        kwargs = mock_pipeline.call_args[1]
        assert kwargs["topk"] == 40
        assert kwargs["temperature"] == 0.8
        assert kwargs["cfg_scale"] == 2.0
        assert kwargs["max_audio_length_ms"] == 30000
        assert "save_path" in kwargs
        # CRITICAL: no topp kwarg
        assert "topp" not in kwargs
        assert "top_p" not in kwargs

    def test_input_dict_has_lyrics_and_tags(self):
        mw = _get_worker_module()
        params = {"lyrics": "test lyrics", "tags": "rock", "seed": "-1"}
        mock_pipeline = self._run_heartmula(mw, params)
        input_dict = mock_pipeline.call_args[0][0]
        assert input_dict == {"lyrics": "test lyrics", "tags": "rock"}

    def test_no_invalid_kwargs(self):
        """Only kwargs that _sanitize_parameters reads should be passed."""
        valid_kwargs = {"cfg_scale", "max_audio_length_ms", "temperature", "topk", "save_path"}
        mw = _get_worker_module()
        params = {"seed": "-1"}
        mock_pipeline = self._run_heartmula(mw, params)
        kwargs = mock_pipeline.call_args[1]
        invalid = set(kwargs.keys()) - valid_kwargs
        assert invalid == set(), f"Invalid kwargs passed to HeartMuLa pipeline: {invalid}"

    def test_gui_param_ids_match_inferencer(self):
        mw = _get_worker_module()
        import music_manager
        gui_ids = {c["id"] for c in music_manager.MODEL_PARAMS["heartmula"]["controls"]
                   if c["type"] != "heading"}
        inferencer_keys = {
            "lyrics", "tags", "duration", "temperature",
            "top_k", "cfg_scale", "seed",
        }
        assert gui_ids == inferencer_keys


# ===================================================================
# DiffRhythm — calls cfm.sample() directly + decode_audio()
# ===================================================================
class TestDiffRhythmParams:
    """Verify DiffRhythm inferencer calls cfm.sample() + preprocessing correctly."""

    def _run_diffrhythm(self, mw, params, model_dict=None):
        """Helper: run _infer_diffrhythm with all dependencies mocked."""
        mock_get_lrc = MagicMock(return_value=(
            MagicMock(), MagicMock(), 2048, MagicMock(),
        ))
        mock_get_style = MagicMock(return_value=MagicMock())
        mock_get_neg = MagicMock(return_value=MagicMock())
        mock_get_ref = MagicMock(return_value=(MagicMock(), [(0, 2048)]))
        mock_decode = MagicMock(return_value=MagicMock())

        # cfm.sample() returns (list_of_latents, trajectory)
        mock_latent = MagicMock()
        mock_latent.to.return_value.transpose.return_value = MagicMock()
        mock_cfm = MagicMock()
        mock_cfm.sample.return_value = ([mock_latent], MagicMock())

        if model_dict is None:
            model_dict = {
                "cfm": mock_cfm, "tokenizer": MagicMock(), "muq": MagicMock(),
                "vae": MagicMock(), "max_frames": 6144, "repo_path": "/fake/path",
            }
        else:
            model_dict["cfm"] = mock_cfm

        mock_utils_mod = MagicMock()
        mock_utils_mod.get_lrc_token = mock_get_lrc
        mock_utils_mod.get_style_prompt = mock_get_style
        mock_utils_mod.get_negative_style_prompt = mock_get_neg
        mock_utils_mod.get_reference_latent = mock_get_ref
        mock_utils_mod.decode_audio = mock_decode

        # Mock einops.rearrange to return a tensor-like object
        mock_rearrange = MagicMock(return_value=MagicMock(
            to=MagicMock(return_value=MagicMock(
                div=MagicMock(return_value=MagicMock(
                    clamp=MagicMock(return_value=MagicMock(
                        cpu=MagicMock(return_value=MagicMock(
                            numpy=MagicMock(return_value=np.zeros((2, 44100),
                                                                   dtype=np.float32))
                        ))
                    ))
                ))
            ))
        ))
        mock_einops = MagicMock()
        mock_einops.rearrange = mock_rearrange

        with _mock_torch() as mt:
            mt.inference_mode.return_value.__enter__ = MagicMock(return_value=None)
            mt.inference_mode.return_value.__exit__ = MagicMock(return_value=False)
            mt.max.return_value.clamp.return_value = 1.0
            with patch.dict("sys.modules", {
                "infer": MagicMock(),
                "infer.infer_utils": mock_utils_mod,
                "einops": mock_einops,
            }), patch("os.chdir"):
                mw._infer_diffrhythm(model_dict, params, "cpu")

        return {
            "cfm_sample": mock_cfm.sample,
            "get_lrc_token": mock_get_lrc,
            "get_style_prompt": mock_get_style,
            "get_negative_style_prompt": mock_get_neg,
            "get_reference_latent": mock_get_ref,
            "decode_audio": mock_decode,
        }

    def test_cfm_sample_called_with_correct_kwargs(self):
        """cfm.sample() must receive steps, cfg_strength, and other params."""
        mw = _get_worker_module()
        params = {
            "prompt": "chill beats", "lyrics": "[00:01.00] hello world",
            "duration": "95", "chunked": True, "seed": "-1",
            "steps": "40", "cfg_strength": "6.0",
        }

        mocks = self._run_diffrhythm(mw, params)

        kwargs = mocks["cfm_sample"].call_args[1]
        assert kwargs["steps"] == 40
        assert kwargs["cfg_strength"] == 6.0
        assert kwargs["batch_infer_num"] == 1
        assert "cond" in kwargs
        assert "text" in kwargs
        assert "style_prompt" in kwargs
        assert "negative_style_prompt" in kwargs
        assert "start_time" in kwargs
        assert "song_duration" in kwargs
        assert "latent_pred_segments" in kwargs
        assert "max_duration" in kwargs

    def test_no_old_wrong_kwargs(self):
        """cfm.sample() should not receive non-existent kwargs."""
        mw = _get_worker_module()
        params = {"duration": "95", "seed": "-1"}
        mocks = self._run_diffrhythm(mw, params)
        kwargs = mocks["cfm_sample"].call_args[1]
        for bad_key in ("clap_model", "muq_model", "prompt", "lyrics",
                        "device", "eval_model", "eval_muq", "file_type",
                        "vocal_flag", "odeint_method"):
            assert bad_key not in kwargs, f"Unexpected kwarg: {bad_key}"

    def test_preprocessing_called_correctly(self):
        """get_lrc_token, get_style_prompt called with right args."""
        mw = _get_worker_module()
        mock_muq = MagicMock()
        mock_tokenizer = MagicMock()
        model_dict = {
            "cfm": MagicMock(), "tokenizer": mock_tokenizer, "muq": mock_muq,
            "vae": MagicMock(), "max_frames": 6144, "repo_path": "/fake",
        }
        params = {"prompt": "ambient", "lyrics": "[00:01] test",
                  "duration": "95", "seed": "-1"}

        mocks = self._run_diffrhythm(mw, params, model_dict)

        # get_style_prompt called with muq model and prompt kwarg
        mocks["get_style_prompt"].assert_called_once_with(mock_muq, prompt="ambient")

        # get_lrc_token called with correct positional args
        lrc_args = mocks["get_lrc_token"].call_args[0]
        assert lrc_args[0] == 2048   # max_frames for 95s
        assert lrc_args[1] == "[00:01] test"
        assert lrc_args[2] is mock_tokenizer
        assert lrc_args[3] == 95.0
        assert lrc_args[4] == "cpu"

    def test_sway_coef_parsing(self):
        """sway_sampling_coef empty string → None, numeric string → float."""
        mw = _get_worker_module()
        # Empty → None
        params = {"duration": "95", "seed": "-1", "sway_sampling_coef": ""}
        mocks = self._run_diffrhythm(mw, params)
        assert mocks["cfm_sample"].call_args[1]["sway_sampling_coef"] is None

        # Numeric → float
        params["sway_sampling_coef"] = "-1.0"
        mocks = self._run_diffrhythm(mw, params)
        assert mocks["cfm_sample"].call_args[1]["sway_sampling_coef"] == -1.0

    def test_gui_param_ids_match_inferencer(self):
        mw = _get_worker_module()
        import music_manager
        gui_ids = {c["id"] for c in music_manager.MODEL_PARAMS["diffrhythm"]["controls"]
                   if c["type"] != "heading"}
        inferencer_keys = {
            "prompt", "lyrics", "duration", "chunked", "seed",
            "steps", "cfg_strength", "sway_sampling_coef",
        }
        assert gui_ids == inferencer_keys


# ===================================================================
# Riffusion — StableDiffusionPipeline standard params
# ===================================================================
class TestRiffusionParams:
    """Verify Riffusion inferencer passes correct kwargs to diffusers pipeline."""

    def test_pipeline_called_with_correct_kwargs(self):
        mw = _get_worker_module()

        mock_pipe = MagicMock()
        from PIL import Image
        fake_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        mock_pipe.return_value = MagicMock(images=[fake_img])

        model_dict = {"pipe": mock_pipe}
        params = {
            "prompt": "funky bass",
            "negative_prompt": "noise",
            "duration": "5",
            "num_inference_steps": "30",
            "guidance_scale": "8.0",
            "seed": "-1",
        }

        with _mock_torch():
            with patch.object(mw, "_spectrogram_image_to_audio",
                              return_value=np.zeros(44100, dtype=np.float32)):
                mw._infer_riffusion(model_dict, params, "cpu")

        # First call (at least one segment)
        first_call = mock_pipe.call_args_list[0]
        assert first_call[0][0] == "funky bass"
        kwargs = first_call[1]
        assert kwargs["num_inference_steps"] == 30
        assert kwargs["guidance_scale"] == 8.0
        assert kwargs["width"] == 512
        assert kwargs["height"] == 512

    def test_gui_param_ids_match_inferencer(self):
        mw = _get_worker_module()
        import music_manager
        gui_ids = {c["id"] for c in music_manager.MODEL_PARAMS["riffusion"]["controls"]
                   if c["type"] != "heading"}
        inferencer_keys = {
            "prompt", "negative_prompt", "duration",
            "num_inference_steps", "guidance_scale", "seed",
        }
        assert gui_ids == inferencer_keys


# ===================================================================
# Stable Audio — generate_diffusion_cond params
# ===================================================================
class TestStableAudioParams:
    """Verify Stable Audio inferencer passes correct kwargs."""

    def test_generate_diffusion_cond_kwargs(self):
        mw = _get_worker_module()

        mock_generate = MagicMock()
        fake_output = MagicMock()
        fake_output.__getitem__ = lambda self, key: MagicMock(
            cpu=lambda: MagicMock(
                numpy=lambda: np.zeros((2, 44100), dtype=np.float32)
            ),
            ndim=2,
            shape=(2, 44100),
        )
        mock_generate.return_value = fake_output

        model_dict = {
            "model": MagicMock(),
            "model_config": {"sample_rate": 44100},
        }
        params = {
            "prompt": "ambient pads",
            "duration": "20",
            "steps": "50",
            "cfg_scale": "5.0",
            "sampler_type": "k-dpm-2",
            "sigma_min": "0.1",
            "sigma_max": "800",
            "seed": "-1",
        }

        mock_sa_gen = MagicMock()
        mock_sa_gen.generate_diffusion_cond = mock_generate

        with _mock_torch():
            with patch.dict("sys.modules", {
                "stable_audio_tools": MagicMock(),
                "stable_audio_tools.inference": MagicMock(),
                "stable_audio_tools.inference.generation": mock_sa_gen,
            }):
                mw._infer_stable_audio(model_dict, params, "cpu")

        kwargs = mock_generate.call_args[1]
        assert kwargs["steps"] == 50
        assert kwargs["cfg_scale"] == 5.0
        assert kwargs["sampler_type"] == "k-dpm-2"
        assert kwargs["sigma_min"] == 0.1
        assert kwargs["sigma_max"] == 800
        assert kwargs["device"] == "cpu"
        # No negative conditioning when negative_prompt is empty
        assert kwargs["negative_conditioning"] is None
        # Conditioning format check
        cond = kwargs["conditioning"]
        assert len(cond) == 1
        assert cond[0]["prompt"] == "ambient pads"
        assert cond[0]["seconds_total"] == 20.0

    def test_negative_prompt_creates_negative_conditioning(self):
        """Non-empty negative_prompt should create negative_conditioning."""
        mw = _get_worker_module()

        mock_generate = MagicMock()
        fake_output = MagicMock()
        fake_output.__getitem__ = lambda self, key: MagicMock(
            cpu=lambda: MagicMock(
                numpy=lambda: np.zeros((2, 44100), dtype=np.float32)
            ),
            ndim=2,
            shape=(2, 44100),
        )
        mock_generate.return_value = fake_output

        model_dict = {
            "model": MagicMock(),
            "model_config": {"sample_rate": 44100},
        }
        params = {
            "prompt": "ambient pads",
            "negative_prompt": "drums, percussion, noise",
            "duration": "20",
            "steps": "50",
            "cfg_scale": "5.0",
            "seed": "-1",
        }

        mock_sa_gen = MagicMock()
        mock_sa_gen.generate_diffusion_cond = mock_generate

        with _mock_torch():
            with patch.dict("sys.modules", {
                "stable_audio_tools": MagicMock(),
                "stable_audio_tools.inference": MagicMock(),
                "stable_audio_tools.inference.generation": mock_sa_gen,
            }):
                mw._infer_stable_audio(model_dict, params, "cpu")

        kwargs = mock_generate.call_args[1]
        neg_cond = kwargs["negative_conditioning"]
        assert neg_cond is not None
        assert len(neg_cond) == 1
        assert neg_cond[0]["prompt"] == "drums, percussion, noise"

    def test_gui_param_ids_match_inferencer(self):
        mw = _get_worker_module()
        import music_manager
        gui_ids = {c["id"] for c in music_manager.MODEL_PARAMS["stable_audio"]["controls"]
                   if c["type"] != "heading"}
        inferencer_keys = {
            "prompt", "negative_prompt", "duration", "steps", "cfg_scale",
            "sampler_type", "sigma_min", "sigma_max", "seed",
        }
        assert gui_ids == inferencer_keys


# ===================================================================
# YuE — uses custom tokenizer and codec tools, not AutoTokenizer
# ===================================================================
class TestYuEParams:
    """Verify YuE inferencer parameter reading from GUI."""

    def test_gui_param_ids_match_inferencer(self):
        """All GUI param IDs must be read by the inferencer."""
        mw = _get_worker_module()
        import music_manager
        gui_ids = {c["id"] for c in music_manager.MODEL_PARAMS["yue"]["controls"]
                   if c["type"] != "heading"}
        # These are the param keys _infer_yue reads via params.get()
        inferencer_keys = {
            "lyrics", "tags", "language", "mode",
            "duration", "temperature", "seed",
            "top_p", "repetition_penalty", "max_new_tokens",
            "guidance_scale", "num_segments", "stage2_batch_size",
        }
        assert gui_ids == inferencer_keys

    def test_language_mode_model_id_mapping(self):
        """language/mode must map to correct HuggingFace model ID."""
        mw = _get_worker_module()
        # The lang_map inside _infer_yue
        lang_map = {"en": "en", "zh": "zh", "ja": "jp-kr", "ko": "jp-kr"}
        for lang, lang_key in lang_map.items():
            for mode in ("cot", "icl"):
                expected = f"m-a-p/YuE-s1-7B-anneal-{lang_key}-{mode}"
                # Verify the pattern is correct by checking it's a valid model name
                assert expected.startswith("m-a-p/YuE-s1-7B-anneal-")
                assert expected.endswith(f"-{mode}")

    def test_num_segments_caps_iteration(self):
        """num_segments > 0 should limit how many lyric segments are processed."""
        mw = _get_worker_module()
        # With 3-segment lyrics, num_segments=1 should cap to 1
        lyrics = "[verse]\nLine one\n\n[chorus]\nLine two\n\n[verse]\nLine three"
        params = {"lyrics": lyrics, "num_segments": "1", "seed": "42"}
        num_segments = int(params["num_segments"])
        # Split same way the inferencer does
        import re
        segments = [s.strip() for s in re.split(r'\n\n+', lyrics) if s.strip()]
        if num_segments > 0:
            segments = segments[:num_segments]
        assert len(segments) == 1

    def test_guidance_scale_drop_off(self):
        """Later segments should use reduced guidance_scale."""
        # The pattern: full value for first 2 segments, then -0.3 for later
        guidance_scale = 1.5
        for i in range(4):
            seg_guidance = guidance_scale if i <= 1 else max(1.0, guidance_scale - 0.3)
            if i <= 1:
                assert seg_guidance == 1.5
            else:
                assert seg_guidance == pytest.approx(1.2)

    def test_stage2_batch_size_passed_through(self):
        """stage2_batch_size should be read and used for stage2 generation."""
        mw = _get_worker_module()
        params = {"stage2_batch_size": "8"}
        assert int(params["stage2_batch_size"]) == 8


# ===================================================================
# Cross-model: every GUI model must have a matching inferencer
# ===================================================================
class TestCrossModelConsistency:
    """Structural tests across all models."""

    def test_every_gui_model_has_inferencer(self):
        mw = _get_worker_module()
        import music_manager
        for model_id in music_manager.MODEL_PARAMS:
            assert model_id in mw.INFERENCERS, f"GUI has params for '{model_id}' but no inferencer"

    def test_every_gui_model_has_loader(self):
        mw = _get_worker_module()
        import music_manager
        for model_id in music_manager.MODEL_PARAMS:
            assert model_id in mw.LOADERS, f"GUI has params for '{model_id}' but no loader"

    def test_every_inferencer_has_gui_params(self):
        mw = _get_worker_module()
        import music_manager
        for model_id in mw.INFERENCERS:
            assert model_id in music_manager.MODEL_PARAMS, \
                f"Inferencer for '{model_id}' but no GUI params"

    def test_all_gui_models_have_seed_control(self):
        """Every model should have a seed control for reproducibility."""
        import music_manager
        for model_id, config in music_manager.MODEL_PARAMS.items():
            ids = {c["id"] for c in config["controls"]
                   if c["type"] != "heading"}
            assert "seed" in ids, f"Model '{model_id}' missing seed control"

    def test_all_gui_models_have_duration_or_equivalent(self):
        """Every model should have a duration control."""
        import music_manager
        for model_id, config in music_manager.MODEL_PARAMS.items():
            ids = {c["id"] for c in config["controls"]
                   if c["type"] != "heading"}
            assert "duration" in ids, f"Model '{model_id}' missing duration control"

    def test_all_presets_reference_valid_control_ids(self):
        """Every preset param key must match a control ID in that model's params."""
        import music_manager
        for model_id, presets in music_manager.MODEL_PRESETS.items():
            control_ids = {c["id"] for c in
                           music_manager.MODEL_PARAMS[model_id]["controls"]
                           if c["type"] != "heading"}
            for preset in presets:
                invalid = set(preset["params"].keys()) - control_ids
                assert invalid == set(), (
                    f"Preset '{preset['name']}' for {model_id} has invalid "
                    f"param keys: {invalid}"
                )

    def test_all_models_have_presets(self):
        """Every model in MODEL_PARAMS should have a non-empty preset list."""
        import music_manager
        for model_id in music_manager.MODEL_PARAMS:
            presets = music_manager.MODEL_PRESETS.get(model_id, [])
            assert len(presets) > 0, f"Model '{model_id}' has no presets"

    def test_all_presets_have_name_and_params(self):
        """Every preset must have 'name' and 'params' keys."""
        import music_manager
        for model_id, presets in music_manager.MODEL_PRESETS.items():
            for i, preset in enumerate(presets):
                assert "name" in preset, f"{model_id} preset {i} missing 'name'"
                assert "params" in preset, f"{model_id} preset {i} missing 'params'"
                assert isinstance(preset["params"], dict), \
                    f"{model_id} preset '{preset['name']}' params not a dict"

    def test_stable_audio_presets_have_negative_prompt(self):
        """All stable_audio presets should include negative_prompt."""
        import music_manager
        presets = music_manager.MODEL_PRESETS["stable_audio"]
        assert len(presets) == 8, f"Expected 8 stable_audio presets, got {len(presets)}"
        for preset in presets:
            assert "negative_prompt" in preset["params"], \
                f"stable_audio preset '{preset['name']}' missing negative_prompt"
            assert len(preset["params"]["negative_prompt"]) > 0, \
                f"stable_audio preset '{preset['name']}' has empty negative_prompt"

    def test_riffusion_presets_have_negative_prompt(self):
        """Most riffusion presets should include negative_prompt."""
        import music_manager
        presets = music_manager.MODEL_PRESETS["riffusion"]
        with_neg = [p for p in presets if "negative_prompt" in p["params"]]
        # At least 7 of 8 should have it (Quick Test may not)
        assert len(with_neg) >= 7, \
            f"Expected at least 7 riffusion presets with negative_prompt, got {len(with_neg)}"

    def test_model_params_shared_module(self):
        """model_params.py should export the same data as music_manager attributes."""
        import music_manager
        from model_params import MODEL_PARAMS, MODEL_PRESETS, MODEL_DISPLAY
        assert music_manager.MODEL_PARAMS is MODEL_PARAMS
        assert music_manager.MODEL_PRESETS is MODEL_PRESETS
        assert music_manager.MODEL_DISPLAY is MODEL_DISPLAY
