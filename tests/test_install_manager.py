"""Tests for install_manager.py — InstallManager, InstallJob, MusicEnvironment."""

import os
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from install_manager import (
    InstallManager, InstallJob, MusicEnvironment, MODEL_SETUP, _CancelledError,
)


# ---------------------------------------------------------------------------
# MODEL_SETUP structure
# ---------------------------------------------------------------------------
class TestModelSetup:
    def test_all_8_models_present(self):
        expected = {
            "ace_step_v15", "ace_step_v1", "heartmula", "diffrhythm",
            "yue", "musicgen", "riffusion", "stable_audio",
        }
        assert set(MODEL_SETUP.keys()) == expected

    def test_each_model_has_required_fields(self):
        required = {"display", "env", "weights_repo", "weights_dir", "gated"}
        for model_id, info in MODEL_SETUP.items():
            for field in required:
                assert field in info, f"{model_id} missing field '{field}'"

    def test_env_names_are_unique(self):
        envs = [info["env"] for info in MODEL_SETUP.values()]
        assert len(envs) == len(set(envs))


# ---------------------------------------------------------------------------
# MusicEnvironment
# ---------------------------------------------------------------------------
class TestMusicEnvironment:
    def test_python_path(self):
        config = MagicMock()
        env = MusicEnvironment("test_env", config)
        assert env.python_path.name == "python.exe"
        assert "test_env" in str(env.python_path)

    def test_pip_path(self):
        config = MagicMock()
        env = MusicEnvironment("test_env", config)
        assert env.pip_path.name == "pip.exe"

    def test_check_status_not_installed(self):
        config = MagicMock()
        env = MusicEnvironment("nonexistent_env_xyz", config)
        # venv_path won't exist
        status = env.check_status()
        assert status == "not_installed"

    def test_check_status_installed_no_verify(self, tmp_path):
        """If venv exists and no verify_package/verify_path, reports installed."""
        config = MagicMock()
        config.verify_package = ""
        config.verify_path = ""

        env = MusicEnvironment("test_env", config)
        # Override venv_path to a real temp dir
        env.venv_path = tmp_path / "test_env"
        scripts = env.venv_path / "Scripts"
        scripts.mkdir(parents=True)
        (scripts / "python.exe").touch()

        status = env.check_status()
        assert status == "installed"


# ---------------------------------------------------------------------------
# InstallJob dataclass
# ---------------------------------------------------------------------------
class TestInstallJob:
    def test_defaults(self):
        job = InstallJob(job_id="abc123", model_id="musicgen", operation="install")
        assert job.status == "pending"
        assert job.progress_pct == 0.0
        assert job.log_lines == []
        assert job.error == ""
        assert job.finished_at is None

    def test_created_at_is_set(self):
        before = time.time()
        job = InstallJob(job_id="abc", model_id="musicgen", operation="install")
        assert job.created_at >= before


# ---------------------------------------------------------------------------
# InstallManager — status
# ---------------------------------------------------------------------------
class TestInstallManagerStatus:
    def test_get_model_status_unknown_model_raises(self):
        mgr = InstallManager()
        with pytest.raises(ValueError, match="Unknown model"):
            mgr.get_model_status("nonexistent_model")

    def test_get_model_status_not_installed(self):
        mgr = InstallManager()
        status = mgr.get_model_status("musicgen")
        # With no venvs or weights on disk, should be not_installed
        assert status["model_id"] == "musicgen"
        assert status["status"] == "not_installed"
        assert status["env_installed"] is False

    def test_get_all_model_status(self):
        mgr = InstallManager()
        all_status = mgr.get_all_model_status()
        assert len(all_status) == 8
        assert "musicgen" in all_status
        assert "stable_audio" in all_status

    def test_get_model_status_includes_active_job(self):
        mgr = InstallManager()
        # Manually inject an active job
        with mgr._lock:
            mgr._active_models["musicgen"] = "fake-job-id"
        status = mgr.get_model_status("musicgen")
        assert status["active_job_id"] == "fake-job-id"


# ---------------------------------------------------------------------------
# InstallManager — job tracking
# ---------------------------------------------------------------------------
class TestInstallManagerJobs:
    def test_get_job_nonexistent(self):
        mgr = InstallManager()
        assert mgr.get_job("nonexistent") is None

    def test_get_job_logs_nonexistent(self):
        mgr = InstallManager()
        assert mgr.get_job_logs("nonexistent") is None

    def test_list_jobs_empty(self):
        mgr = InstallManager()
        assert mgr.list_jobs() == []

    def test_list_jobs_active_only(self):
        mgr = InstallManager()
        # Inject jobs manually
        job1 = InstallJob(job_id="j1", model_id="musicgen", operation="install", status="running")
        job2 = InstallJob(job_id="j2", model_id="riffusion", operation="install", status="completed")
        with mgr._lock:
            mgr._jobs["j1"] = job1
            mgr._jobs["j2"] = job2
        active = mgr.list_jobs(active_only=True)
        assert len(active) == 1
        assert active[0]["job_id"] == "j1"

    def test_job_to_dict(self):
        job = InstallJob(
            job_id="j1", model_id="musicgen", operation="install",
            status="running", progress_pct=50.0, current_step="Installing deps",
        )
        job.log_lines = ["line1", "line2"]
        d = InstallManager._job_to_dict(job)
        assert d["job_id"] == "j1"
        assert d["progress_pct"] == 50.0
        assert d["log_line_count"] == 2
        assert "log_lines" not in d  # logs not included in status dict

    def test_get_job_logs_with_offset(self):
        mgr = InstallManager()
        job = InstallJob(job_id="j1", model_id="musicgen", operation="install")
        job.log_lines = ["line0", "line1", "line2", "line3"]
        with mgr._lock:
            mgr._jobs["j1"] = job
        result = mgr.get_job_logs("j1", offset=2)
        assert result["offset"] == 2
        assert result["total_lines"] == 4
        assert result["lines"] == ["line2", "line3"]


# ---------------------------------------------------------------------------
# InstallManager — install lifecycle
# ---------------------------------------------------------------------------
class TestInstallManagerInstall:
    def test_start_install_unknown_model_raises(self):
        mgr = InstallManager()
        with pytest.raises(ValueError, match="Unknown model"):
            mgr.start_install("nonexistent_model")

    def test_start_install_returns_job_id(self):
        mgr = InstallManager()
        # Patch the install thread so it doesn't actually run subprocess
        with patch.object(mgr, '_run_install'):
            job_id = mgr.start_install("musicgen")
            assert isinstance(job_id, str)
            assert len(job_id) == 12

    def test_concurrent_install_same_model_raises(self):
        mgr = InstallManager()
        with patch.object(mgr, '_run_install'):
            mgr.start_install("musicgen")
            with pytest.raises(RuntimeError, match="already has an active job"):
                mgr.start_install("musicgen")

    def test_different_models_can_install_concurrently(self):
        mgr = InstallManager()
        with patch.object(mgr, '_run_install'):
            j1 = mgr.start_install("musicgen")
            j2 = mgr.start_install("riffusion")
            assert j1 != j2

    def test_start_uninstall_unknown_model_raises(self):
        mgr = InstallManager()
        with pytest.raises(ValueError, match="Unknown model"):
            mgr.start_uninstall("nonexistent_model")

    def test_start_uninstall_returns_job_id(self):
        mgr = InstallManager()
        with patch.object(mgr, '_run_uninstall'):
            job_id = mgr.start_uninstall("musicgen")
            assert isinstance(job_id, str)
            assert len(job_id) == 12


# ---------------------------------------------------------------------------
# InstallManager — _job_log
# ---------------------------------------------------------------------------
class TestJobLog:
    def test_log_appends_lines(self):
        mgr = InstallManager()
        job = InstallJob(job_id="j1", model_id="musicgen", operation="install")
        mgr._job_log(job, "hello")
        mgr._job_log(job, "world")
        assert job.log_lines == ["hello", "world"]

    def test_log_caps_at_max(self):
        mgr = InstallManager()
        job = InstallJob(job_id="j1", model_id="musicgen", operation="install")
        for i in range(250):
            mgr._job_log(job, f"line-{i}")
        assert len(job.log_lines) == mgr.MAX_LOG_LINES

    def test_external_log_callback(self):
        messages = []
        mgr = InstallManager(log_callback=messages.append)
        job = InstallJob(job_id="j1", model_id="musicgen", operation="install")
        mgr._job_log(job, "test message")
        assert "test message" in messages


# ---------------------------------------------------------------------------
# InstallManager — pruning
# ---------------------------------------------------------------------------
class TestJobPruning:
    def test_prune_keeps_max_completed(self):
        mgr = InstallManager()
        mgr.MAX_COMPLETED_JOBS = 5

        # Create 10 completed jobs
        for i in range(10):
            job = InstallJob(
                job_id=f"j{i}", model_id="musicgen", operation="install",
                status="completed", finished_at=float(i),
            )
            with mgr._lock:
                mgr._jobs[f"j{i}"] = job

        mgr._prune_old_jobs()

        with mgr._lock:
            remaining = len(mgr._jobs)
        assert remaining == 5


# ---------------------------------------------------------------------------
# Step tracking
# ---------------------------------------------------------------------------
class TestStepTracking:
    def test_job_has_step_fields(self):
        job = InstallJob(job_id="j1", model_id="musicgen", operation="install")
        assert job.step_index == 0
        assert job.step_total == 0

    def test_job_to_dict_includes_steps(self):
        job = InstallJob(
            job_id="j1", model_id="musicgen", operation="install",
            step_index=3, step_total=7,
        )
        d = InstallManager._job_to_dict(job)
        assert d["step_index"] == 3
        assert d["step_total"] == 7


# ---------------------------------------------------------------------------
# Cancel support
# ---------------------------------------------------------------------------
class TestCancelSupport:
    def test_job_has_cancel_event(self):
        job = InstallJob(job_id="j1", model_id="musicgen", operation="install")
        assert hasattr(job, '_cancel_event')
        assert not job._cancel_event.is_set()

    def test_job_has_current_process(self):
        job = InstallJob(job_id="j1", model_id="musicgen", operation="install")
        assert job._current_process is None

    def test_cancel_install_no_active_job(self):
        mgr = InstallManager()
        result = mgr.cancel_install("musicgen")
        assert result is False

    def test_cancel_install_running_job(self):
        mgr = InstallManager()
        job = InstallJob(job_id="j1", model_id="musicgen", operation="install",
                         status="running")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        job._current_process = mock_proc

        with mgr._lock:
            mgr._jobs["j1"] = job
            mgr._active_models["musicgen"] = "j1"

        result = mgr.cancel_install("musicgen")
        assert result is True
        assert job._cancel_event.is_set()
        mock_proc.kill.assert_called_once()

    def test_cancel_install_completed_job_returns_false(self):
        mgr = InstallManager()
        job = InstallJob(job_id="j1", model_id="musicgen", operation="install",
                         status="completed")
        with mgr._lock:
            mgr._jobs["j1"] = job
            mgr._active_models["musicgen"] = "j1"

        result = mgr.cancel_install("musicgen")
        assert result is False

    def test_cancelled_error_sentinel(self):
        """_CancelledError is catchable and distinguishable."""
        with pytest.raises(_CancelledError):
            raise _CancelledError()


# ---------------------------------------------------------------------------
# Step validation
# ---------------------------------------------------------------------------
class TestStepValidation:
    def test_requirements_file_not_found_raises(self, tmp_path):
        """_execute_step should raise FileNotFoundError for missing -r file."""
        mgr = InstallManager()
        config = MagicMock()
        config.verify_package = ""
        config.verify_path = ""
        env = MusicEnvironment("test_env", config)
        env.venv_path = tmp_path / "test_env"
        scripts = env.venv_path / "Scripts"
        scripts.mkdir(parents=True)
        (scripts / "python.exe").touch()
        (scripts / "pip.exe").touch()

        step = {
            "type": "pip",
            "description": "Install reqs",
            "args": ["install", "-r", "./repos/SomeModel/requirements.txt"],
            "cwd": "venv",
        }

        with pytest.raises(FileNotFoundError, match="Requirements file not found"):
            mgr._execute_step(env, step, lambda msg: None)

    def test_requirements_file_exists_proceeds(self, tmp_path):
        """_execute_step should not raise if requirements file exists."""
        mgr = InstallManager()
        config = MagicMock()
        config.verify_package = ""
        config.verify_path = ""
        env = MusicEnvironment("test_env", config)
        env.venv_path = tmp_path / "test_env"
        scripts = env.venv_path / "Scripts"
        scripts.mkdir(parents=True)
        (scripts / "python.exe").touch()
        (scripts / "pip.exe").touch()

        # Create the requirements file
        repos = env.venv_path / "repos" / "SomeModel"
        repos.mkdir(parents=True)
        (repos / "requirements.txt").write_text("numpy\n")

        step = {
            "type": "pip",
            "description": "Install reqs",
            "args": ["install", "-r", "./repos/SomeModel/requirements.txt"],
            "cwd": "venv",
        }

        # It will call run_pip_install which will fail (no real pip), but
        # should NOT raise FileNotFoundError
        with patch("install_manager.run_pip_install", return_value=(True, "ok")):
            mgr._execute_step(env, step, lambda msg: None)

    def test_optional_step_failure_does_not_raise(self, tmp_path):
        """Steps with optional=True should not raise on failure."""
        mgr = InstallManager()
        config = MagicMock()
        config.verify_package = ""
        config.verify_path = ""
        env = MusicEnvironment("test_env", config)
        env.venv_path = tmp_path / "test_env"
        scripts = env.venv_path / "Scripts"
        scripts.mkdir(parents=True)
        (scripts / "python.exe").touch()
        (scripts / "pip.exe").touch()

        step = {
            "type": "pip",
            "description": "Install optional thing",
            "args": ["install", "flash-attn"],
            "optional": True,
        }

        with patch("install_manager.run_pip_install", return_value=(False, "compilation failed")):
            # Should not raise
            mgr._execute_step(env, step, lambda msg: None)


# ---------------------------------------------------------------------------
# verify_path support
# ---------------------------------------------------------------------------
class TestVerifyPath:
    def test_verify_path_found(self, tmp_path):
        """If verify_path file exists, status is 'installed'."""
        config = MagicMock()
        config.verify_path = "repos/YuE/inference/infer.py"
        config.verify_package = "transformers"

        env = MusicEnvironment("yue_env", config)
        env.venv_path = tmp_path / "yue_env"
        scripts = env.venv_path / "Scripts"
        scripts.mkdir(parents=True)
        (scripts / "python.exe").touch()

        # Create the verify_path file
        infer_dir = env.venv_path / "repos" / "YuE" / "inference"
        infer_dir.mkdir(parents=True)
        (infer_dir / "infer.py").touch()

        status = env.check_status()
        assert status == "installed"

    def test_verify_path_missing(self, tmp_path):
        """If verify_path file does not exist, status is 'partial'."""
        config = MagicMock()
        config.verify_path = "repos/YuE/inference/infer.py"
        config.verify_package = "transformers"

        env = MusicEnvironment("yue_env", config)
        env.venv_path = tmp_path / "yue_env"
        scripts = env.venv_path / "Scripts"
        scripts.mkdir(parents=True)
        (scripts / "python.exe").touch()

        status = env.check_status()
        assert status == "partial"

    def test_verify_path_takes_priority_over_verify_package(self, tmp_path):
        """verify_path is checked first; verify_package is not consulted."""
        config = MagicMock()
        config.verify_path = "repos/YuE/inference/infer.py"
        config.verify_package = "nonexistent_package"

        env = MusicEnvironment("yue_env", config)
        env.venv_path = tmp_path / "yue_env"
        scripts = env.venv_path / "Scripts"
        scripts.mkdir(parents=True)
        (scripts / "python.exe").touch()

        # Create the verify_path file
        infer_dir = env.venv_path / "repos" / "YuE" / "inference"
        infer_dir.mkdir(parents=True)
        (infer_dir / "infer.py").touch()

        # Should be "installed" even though verify_package would fail
        status = env.check_status()
        assert status == "installed"

    def test_yue_config_has_verify_path(self):
        """YuE install config should have verify_path set."""
        from install_configs.yue import YuEConfig
        config = YuEConfig()
        assert config.verify_path == "repos/YuE/inference/infer.py"


# ---------------------------------------------------------------------------
# Pruning includes cancelled
# ---------------------------------------------------------------------------
class TestPruneCancelled:
    def test_prune_includes_cancelled_jobs(self):
        mgr = InstallManager()
        mgr.MAX_COMPLETED_JOBS = 3

        for i in range(6):
            status = "cancelled" if i % 2 == 0 else "completed"
            job = InstallJob(
                job_id=f"j{i}", model_id="musicgen", operation="install",
                status=status, finished_at=float(i),
            )
            with mgr._lock:
                mgr._jobs[f"j{i}"] = job

        mgr._prune_old_jobs()

        with mgr._lock:
            remaining = len(mgr._jobs)
        assert remaining == 3


# ---------------------------------------------------------------------------
# Timeout in base helpers
# ---------------------------------------------------------------------------
class TestBaseTimeouts:
    def test_run_pip_install_has_timeout_param(self):
        """run_pip_install should accept a timeout parameter."""
        from install_configs.base import run_pip_install
        import inspect
        sig = inspect.signature(run_pip_install)
        assert "timeout" in sig.parameters
        assert sig.parameters["timeout"].default == 600

    def test_run_git_clone_has_timeout_param(self):
        """run_git_clone should accept a timeout parameter."""
        from install_configs.base import run_git_clone
        import inspect
        sig = inspect.signature(run_git_clone)
        assert "timeout" in sig.parameters
        assert sig.parameters["timeout"].default == 300

    def test_run_pip_install_has_process_callback(self):
        """run_pip_install should accept a process_callback parameter."""
        from install_configs.base import run_pip_install
        import inspect
        sig = inspect.signature(run_pip_install)
        assert "process_callback" in sig.parameters

    def test_run_git_clone_has_process_callback(self):
        """run_git_clone should accept a process_callback parameter."""
        from install_configs.base import run_git_clone
        import inspect
        sig = inspect.signature(run_git_clone)
        assert "process_callback" in sig.parameters


# ---------------------------------------------------------------------------
# Editable flags
# ---------------------------------------------------------------------------
class TestEditableFlags:
    def test_diffrhythm_not_editable(self):
        from install_configs.diffrhythm import DiffRhythmConfig
        config = DiffRhythmConfig()
        assert config.git_repos[0][1] is False

    def test_yue_not_editable(self):
        from install_configs.yue import YuEConfig
        config = YuEConfig()
        assert config.git_repos[0][1] is False


# ---------------------------------------------------------------------------
# MusicGen av binary install order
# ---------------------------------------------------------------------------
class TestMusicGenAvBinary:
    def test_musicgen_installs_av_binary_before_audiocraft(self):
        """MusicGen config should install av with --only-binary before audiocraft."""
        from install_configs.musicgen import MusicGenConfig
        config = MusicGenConfig()
        steps = config.get_install_steps()

        av_idx = None
        audiocraft_idx = None
        for i, step in enumerate(steps):
            if step["type"] == "pip":
                if "--only-binary" in step["args"] and "av" in step["args"]:
                    av_idx = i
                if "audiocraft" in step["args"]:
                    audiocraft_idx = i

        assert av_idx is not None, "No av binary-only install step found"
        assert audiocraft_idx is not None, "No audiocraft install step found"
        assert av_idx < audiocraft_idx, "av must be installed before audiocraft"


# ---------------------------------------------------------------------------
# start_download support
# ---------------------------------------------------------------------------
class TestStartDownload:
    def test_start_download_unknown_model_raises(self):
        mgr = InstallManager()
        with pytest.raises(ValueError, match="Unknown model"):
            mgr.start_download("nonexistent_model")

    def test_start_download_no_weights_repo_raises(self):
        """Models with no weights_repo should raise ValueError."""
        mgr = InstallManager()
        with pytest.raises(ValueError, match="no downloadable weights"):
            mgr.start_download("ace_step_v15")

    def test_start_download_returns_job_id(self):
        mgr = InstallManager()
        with patch.object(mgr, '_run_download'):
            job_id = mgr.start_download("musicgen")
            assert isinstance(job_id, str)
            assert len(job_id) == 12

    def test_start_download_concurrent_with_install_raises(self):
        """Cannot download if install is active."""
        mgr = InstallManager()
        with patch.object(mgr, '_run_install'):
            mgr.start_install("musicgen")
            with pytest.raises(RuntimeError, match="already has an active job"):
                mgr.start_download("musicgen")

    def test_start_download_creates_download_operation(self):
        mgr = InstallManager()
        with patch.object(mgr, '_run_download'):
            job_id = mgr.start_download("musicgen")
            job = mgr.get_job(job_id)
            assert job["operation"] == "download"

    def test_cancel_works_for_download_job(self):
        """cancel_install should work for download jobs too."""
        mgr = InstallManager()
        job = InstallJob(job_id="j1", model_id="musicgen", operation="download",
                         status="running")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        job._current_process = mock_proc
        with mgr._lock:
            mgr._jobs["j1"] = job
            mgr._active_models["musicgen"] = "j1"
        result = mgr.cancel_install("musicgen")
        assert result is True
        assert job._cancel_event.is_set()
        mock_proc.kill.assert_called_once()

    def test_start_download_no_weights_for_auto_download_models(self):
        """Models that auto-download (no weights_repo) should raise."""
        mgr = InstallManager()
        for model_id in ("ace_step_v15", "ace_step_v1", "diffrhythm", "yue"):
            with pytest.raises(ValueError, match="no downloadable weights"):
                mgr.start_download(model_id)

    def test_start_download_works_for_weight_models(self):
        """Models with weights_repo should accept start_download."""
        mgr = InstallManager()
        for model_id in ("heartmula", "musicgen", "riffusion", "stable_audio"):
            with patch.object(mgr, '_run_download'):
                job_id = mgr.start_download(model_id)
                assert isinstance(job_id, str)
            # Clean up active model so next iteration can proceed
            with mgr._lock:
                mgr._active_models.pop(model_id, None)


# ---------------------------------------------------------------------------
# download_extract step type
# ---------------------------------------------------------------------------
class TestDownloadExtractStep:
    def _make_env(self, tmp_path):
        mgr = InstallManager()
        config = MagicMock()
        config.verify_package = ""
        config.verify_path = ""
        env = MusicEnvironment("test_env", config)
        env.venv_path = tmp_path / "test_env"
        scripts = env.venv_path / "Scripts"
        scripts.mkdir(parents=True)
        (scripts / "python.exe").touch()
        (scripts / "pip.exe").touch()
        return mgr, env

    def test_skips_if_all_present(self, tmp_path):
        """download_extract should skip download when all skip_if paths exist."""
        mgr, env = self._make_env(tmp_path)
        marker = tmp_path / "espeak-ng" / "libespeak-ng.dll"
        marker.parent.mkdir(parents=True)
        marker.touch()

        step = {
            "type": "download_extract",
            "description": "Install espeak-ng",
            "url": "https://example.com/bin.zip",
            "extractions": [
                {"prefix": "espeak-ng/", "target_dir": str(tmp_path),
                 "skip_if": str(marker)},
            ],
        }

        logs = []
        mgr._execute_step(env, step, logs.append)
        assert any("Already present" in l for l in logs)

    def test_legacy_single_format(self, tmp_path):
        """download_extract should support legacy single extract_prefix format."""
        mgr, env = self._make_env(tmp_path)
        marker = tmp_path / "dll"
        marker.touch()

        step = {
            "type": "download_extract",
            "description": "Install something",
            "url": "https://example.com/bin.zip",
            "extract_prefix": "espeak-ng/",
            "target_dir": str(tmp_path / "out"),
            "skip_if_exists": str(marker),
        }

        logs = []
        mgr._execute_step(env, step, logs.append)
        assert any("Already present" in l for l in logs)

    def test_downloads_and_extracts_multi(self, tmp_path):
        """download_extract should download zip and extract multiple prefixes."""
        import zipfile
        mgr, env = self._make_env(tmp_path)

        # Create a small test zip
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(str(zip_path), "w") as zf:
            zf.writestr("espeak-ng/espeak-ng.exe", "fake exe")
            zf.writestr("espeak-ng/libespeak-ng.dll", "fake dll")
            zf.writestr("ffmpeg/bin/ffmpeg.exe", "fake ffmpeg")
            zf.writestr("rubberband/rb.exe", "should not extract")

        target_dir = tmp_path / "out"
        step = {
            "type": "download_extract",
            "description": "Install espeak-ng and ffmpeg",
            "url": "https://example.com/bin.zip",
            "extractions": [
                {"prefix": "espeak-ng/", "target_dir": str(target_dir)},
                {"prefix": "ffmpeg/", "target_dir": str(target_dir)},
            ],
        }

        import shutil
        def fake_retrieve(url, dest):
            shutil.copy(str(zip_path), dest)

        with patch("install_manager.urllib.request.urlretrieve", side_effect=fake_retrieve):
            logs = []
            mgr._execute_step(env, step, logs.append)

        assert (target_dir / "espeak-ng" / "espeak-ng.exe").exists()
        assert (target_dir / "espeak-ng" / "libespeak-ng.dll").exists()
        assert (target_dir / "ffmpeg" / "bin" / "ffmpeg.exe").exists()
        assert not (target_dir / "rubberband").exists()

    def test_partial_skip(self, tmp_path):
        """download_extract should skip already-present extractions but fetch others."""
        import zipfile
        mgr, env = self._make_env(tmp_path)

        # espeak-ng already exists
        espeak_marker = tmp_path / "out" / "espeak-ng" / "libespeak-ng.dll"
        espeak_marker.parent.mkdir(parents=True)
        espeak_marker.touch()

        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(str(zip_path), "w") as zf:
            zf.writestr("espeak-ng/espeak-ng.exe", "fake exe")
            zf.writestr("ffmpeg/bin/ffmpeg.exe", "fake ffmpeg")

        target_dir = tmp_path / "out"
        step = {
            "type": "download_extract",
            "description": "Install espeak-ng and ffmpeg",
            "url": "https://example.com/bin.zip",
            "extractions": [
                {"prefix": "espeak-ng/", "target_dir": str(target_dir),
                 "skip_if": str(espeak_marker)},
                {"prefix": "ffmpeg/", "target_dir": str(target_dir)},
            ],
        }

        import shutil
        def fake_retrieve(url, dest):
            shutil.copy(str(zip_path), dest)

        with patch("install_manager.urllib.request.urlretrieve", side_effect=fake_retrieve):
            logs = []
            mgr._execute_step(env, step, logs.append)

        # ffmpeg should be extracted, espeak-ng should NOT have espeak-ng.exe
        # (only the marker we created exists)
        assert (target_dir / "ffmpeg" / "bin" / "ffmpeg.exe").exists()
        assert not (target_dir / "espeak-ng" / "espeak-ng.exe").exists()


# ---------------------------------------------------------------------------
# DiffRhythm espeak-ng install step
# ---------------------------------------------------------------------------
class TestDiffRhythmEspeakStep:
    def test_diffrhythm_has_download_extract_step(self):
        """DiffRhythm install config should have espeak-ng + ffmpeg as first step."""
        from install_configs.diffrhythm import DiffRhythmConfig
        config = DiffRhythmConfig()
        steps = config.get_install_steps()
        dl_steps = [s for s in steps if s["type"] == "download_extract"]
        assert len(dl_steps) == 1
        # Should be first step
        assert steps[0]["type"] == "download_extract"
        # Multi-extraction: espeak-ng and ffmpeg
        extractions = dl_steps[0]["extractions"]
        prefixes = {e["prefix"] for e in extractions}
        assert "espeak-ng/" in prefixes
        assert "ffmpeg/" in prefixes

    def test_diffrhythm_no_system_notes(self):
        """espeak-ng is now auto-installed, so system_notes should be empty."""
        from install_configs.diffrhythm import DiffRhythmConfig
        config = DiffRhythmConfig()
        assert config.system_notes == ""


# ---------------------------------------------------------------------------
# YuE xcodec clone step
# ---------------------------------------------------------------------------
class TestYuEXcodecStep:
    def test_yue_has_xcodec_clone_step(self):
        """YuE install config should include xcodec_mini_infer HuggingFace clone."""
        from install_configs.yue import YuEConfig
        config = YuEConfig()
        steps = config.get_install_steps()
        xcodec_steps = [s for s in steps if s["type"] == "git_clone"
                        and "xcodec" in s.get("url", "")]
        assert len(xcodec_steps) == 1, "Expected exactly 1 xcodec clone step"

    def test_yue_xcodec_targets_inference_dir(self):
        """xcodec clone step must target repos/YuE/inference, not repos/."""
        from install_configs.yue import YuEConfig
        config = YuEConfig()
        steps = config.get_install_steps()
        xcodec_step = [s for s in steps if s["type"] == "git_clone"
                       and "xcodec" in s.get("url", "")][0]
        assert xcodec_step["target_dir"] == "repos/YuE/inference"

    def test_yue_xcodec_not_editable(self):
        """xcodec is a model repo, not a pip package — editable must be False."""
        from install_configs.yue import YuEConfig
        config = YuEConfig()
        steps = config.get_install_steps()
        xcodec_step = [s for s in steps if s["type"] == "git_clone"
                       and "xcodec" in s.get("url", "")][0]
        assert xcodec_step["editable"] is False

    def test_yue_xcodec_after_main_repo(self):
        """xcodec clone must come after the main YuE repo clone."""
        from install_configs.yue import YuEConfig
        config = YuEConfig()
        steps = config.get_install_steps()
        clone_steps = [s for s in steps if s["type"] == "git_clone"]
        urls = [s["url"] for s in clone_steps]
        yue_idx = next(i for i, u in enumerate(urls) if "YuE.git" in u)
        xcodec_idx = next(i for i, u in enumerate(urls) if "xcodec" in u)
        assert yue_idx < xcodec_idx, "YuE repo must be cloned before xcodec"


# ---------------------------------------------------------------------------
# git_clone target_dir support in InstallManager
# ---------------------------------------------------------------------------
class TestGitCloneTargetDir:
    def _make_env(self, tmp_path):
        mgr = InstallManager()
        config = MagicMock()
        config.verify_package = ""
        config.verify_path = ""
        env = MusicEnvironment("test_env", config)
        env.venv_path = tmp_path / "test_env"
        scripts = env.venv_path / "Scripts"
        scripts.mkdir(parents=True)
        (scripts / "python.exe").touch()
        (scripts / "pip.exe").touch()
        return mgr, env

    def test_git_clone_default_target_is_repos(self, tmp_path):
        """Without target_dir, git_clone should use venv_path/repos."""
        mgr, env = self._make_env(tmp_path)
        step = {
            "type": "git_clone",
            "description": "Clone test repo",
            "url": "https://example.com/test.git",
            "editable": False,
        }
        with patch("install_manager.run_git_clone", return_value=(True, "ok")) as mock_clone:
            mgr._execute_step(env, step, lambda msg: None)
            call_args = mock_clone.call_args
            target_dir = call_args[0][1]  # second positional arg
            assert target_dir == str(env.venv_path / "repos")

    def test_git_clone_custom_target_dir(self, tmp_path):
        """With target_dir, git_clone should use venv_path/target_dir."""
        mgr, env = self._make_env(tmp_path)
        step = {
            "type": "git_clone",
            "description": "Clone into subdir",
            "url": "https://example.com/sub.git",
            "editable": False,
            "target_dir": "repos/YuE/inference",
        }
        with patch("install_manager.run_git_clone", return_value=(True, "ok")) as mock_clone:
            mgr._execute_step(env, step, lambda msg: None)
            call_args = mock_clone.call_args
            target_dir = call_args[0][1]
            assert target_dir == str(env.venv_path / "repos" / "YuE" / "inference")

    def test_git_clone_custom_target_dir_created(self, tmp_path):
        """Custom target_dir should be created (with parents) if it doesn't exist."""
        mgr, env = self._make_env(tmp_path)
        step = {
            "type": "git_clone",
            "description": "Clone into deep subdir",
            "url": "https://example.com/deep.git",
            "editable": False,
            "target_dir": "repos/A/B/C",
        }
        expected_dir = env.venv_path / "repos" / "A" / "B" / "C"
        assert not expected_dir.exists()
        with patch("install_manager.run_git_clone", return_value=(True, "ok")):
            mgr._execute_step(env, step, lambda msg: None)
        assert expected_dir.exists()


# ---------------------------------------------------------------------------
# run_git_clone empty directory handling
# ---------------------------------------------------------------------------
class TestGitCloneEmptyDir:
    def test_empty_dir_is_removed_before_clone(self, tmp_path):
        """Empty placeholder dirs (from parent repo) should be removed so clone works."""
        from install_configs.base import run_git_clone
        clone_target = tmp_path / "repos"
        clone_target.mkdir()
        empty_dir = clone_target / "xcodec_mini_infer"
        empty_dir.mkdir()
        assert empty_dir.exists() and not os.listdir(str(empty_dir))

        with patch("install_configs.base.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.stdout = iter([])
            mock_proc.wait.return_value = None
            mock_proc.returncode = 0
            mock_popen.return_value = mock_proc

            run_git_clone("https://example.com/xcodec_mini_infer",
                          str(clone_target), "pip", editable=False)

            # Should have called git clone (not git pull)
            cmd = mock_popen.call_args[0][0]
            assert "clone" in cmd, f"Expected 'clone' in {cmd}, got 'pull' on empty dir"

    def test_git_dir_present_does_pull(self, tmp_path):
        """Directories with .git should trigger pull, not fresh clone."""
        from install_configs.base import run_git_clone
        clone_target = tmp_path / "repos"
        clone_target.mkdir()
        repo_dir = clone_target / "myrepo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()

        with patch("install_configs.base.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.stdout = iter([])
            mock_proc.wait.return_value = None
            mock_proc.returncode = 0
            mock_popen.return_value = mock_proc

            run_git_clone("https://example.com/myrepo.git",
                          str(clone_target), "pip", editable=False)

            cmd = mock_popen.call_args[0][0]
            assert "pull" in cmd, f"Expected 'pull' for existing repo, got {cmd}"
