"""
Install Manager - Headless install/uninstall/status management for music models.

Shared module used by both the API server (remote control) and the GUI.
Thread-safe job tracking with poll-based progress reporting.
"""

import os
import re
import shutil
import subprocess
import threading
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

# Regex to strip ANSI escape sequences (colors, cursor moves, progress bars)
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07|\r')

import urllib.request
import zipfile

from config import VENVS_DIR, MODELS_DIR, PYTHON_PATH, BASE_DIR
from install_configs import ALL_CONFIGS
from install_configs.base import run_pip_install, run_git_clone, check_package_installed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model setup data (core fields for install logic)
# GUI-only display fields (desc, weights_size, vram) stay in music_manager.py
# ---------------------------------------------------------------------------
MODEL_SETUP = {
    "ace_step_v15": {"display": "ACE-Step v1.5",    "env": "ace_step_env",     "weights_repo": None,                                      "weights_dir": None,           "gated": False},
    "ace_step_v1":  {"display": "ACE-Step v1",       "env": "ace_step_v1_env",  "weights_repo": None,                                      "weights_dir": None,           "gated": False},
    "heartmula":    {"display": "HeartMuLa 3B",      "env": "heartmula_env",    "weights_repo": [("HeartMuLa/HeartMuLaGen", ""), ("HeartMuLa/HeartMuLa-RL-oss-3B-20260123", "HeartMuLa-oss-3B"), ("HeartMuLa/HeartCodec-oss-20260123", "HeartCodec-oss")],  "weights_dir": "heartmula",    "gated": False},
    "diffrhythm":   {"display": "DiffRhythm",        "env": "diffrhythm_env",   "weights_repo": None,                                      "weights_dir": None,           "gated": False},
    "yue":          {"display": "YuE",               "env": "yue_env",          "weights_repo": None,                                      "weights_dir": None,           "gated": False},
    "musicgen":     {"display": "MusicGen",          "env": "musicgen_env",     "weights_repo": "facebook/musicgen-large",                  "weights_dir": "musicgen",     "gated": False},
    "riffusion":    {"display": "Riffusion",         "env": "riffusion_env",    "weights_repo": "riffusion/riffusion-model-v1",             "weights_dir": "riffusion",    "gated": False},
    "stable_audio": {"display": "Stable Audio Open", "env": "stable_audio_env", "weights_repo": "stabilityai/stable-audio-open-1.0",       "weights_dir": "stable-audio", "gated": False},
}


# ---------------------------------------------------------------------------
# MusicEnvironment - represents a single model's virtual environment
# ---------------------------------------------------------------------------
class MusicEnvironment:
    """Represents a single music model virtual environment. No GUI dependencies."""

    def __init__(self, name: str, config):
        self.name = name
        self.config = config
        self.venv_path = VENVS_DIR / name
        self.status = "unknown"
        self.error_message = ""

    @property
    def python_path(self) -> Path:
        return self.venv_path / "Scripts" / "python.exe"

    @property
    def pip_path(self) -> Path:
        return self.venv_path / "Scripts" / "pip.exe"

    def check_status(self) -> str:
        if not self.venv_path.exists() or not self.python_path.exists():
            self.status = "not_installed"
            return self.status
        if self.config.verify_path:
            full = self.venv_path / self.config.verify_path
            if full.exists():
                self.status = "installed"
            else:
                self.status = "partial"
        elif self.config.verify_package:
            if check_package_installed(str(self.pip_path), self.config.verify_package):
                self.status = "installed"
            else:
                self.status = "partial"
        else:
            self.status = "installed"
        return self.status


# ---------------------------------------------------------------------------
# InstallJob - tracks a single install/uninstall operation
# ---------------------------------------------------------------------------
@dataclass
class InstallJob:
    """Tracks a long-running install or uninstall operation."""
    job_id: str
    model_id: str
    operation: str              # "install" | "uninstall" | "download"
    status: str = "pending"     # "pending" | "running" | "completed" | "failed" | "cancelled"
    progress_pct: float = 0.0
    current_step: str = ""
    step_index: int = 0         # current step (1-based)
    step_total: int = 0         # total steps
    log_lines: list = field(default_factory=list)
    error: str = ""
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    _cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _current_process: Optional[subprocess.Popen] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# InstallManager - thread-safe install/uninstall/status management
# ---------------------------------------------------------------------------
class _CancelledError(Exception):
    """Internal sentinel raised when an install job is cancelled."""


class InstallManager:
    """Headless install/uninstall/status manager with thread-safe job tracking."""

    MAX_LOG_LINES = 200
    MAX_COMPLETED_JOBS = 50

    def __init__(self, log_callback: Optional[Callable[[str], None]] = None):
        self._lock = threading.Lock()
        self._jobs: dict[str, InstallJob] = {}
        self._active_models: dict[str, str] = {}  # model_id -> job_id
        self.environments: dict[str, MusicEnvironment] = {}
        self._external_log = log_callback
        self._init_environments()

    def _init_environments(self):
        for name, config in ALL_CONFIGS.items():
            self.environments[name] = MusicEnvironment(name, config)

    # ── Status (synchronous, fast) ───────────────────────────────

    def get_model_status(self, model_id: str) -> dict:
        """Check install status for a single model."""
        if model_id not in MODEL_SETUP:
            raise ValueError(f"Unknown model: {model_id}")

        info = MODEL_SETUP[model_id]
        env_name = info["env"]
        env = self.environments.get(env_name)

        env_status = env.check_status() if env else "not_installed"
        env_ok = env_status == "installed"

        weights_ok = True
        if info["weights_repo"]:
            weights_path = MODELS_DIR / info["weights_dir"]
            try:
                weights_ok = weights_path.exists() and any(weights_path.iterdir())
            except (OSError, StopIteration):
                weights_ok = False

        if env_ok and weights_ok:
            overall = "ready"
        elif env_ok and not weights_ok:
            overall = "env_installed"
        elif not env_ok and weights_ok:
            overall = "weights_only"
        else:
            overall = "not_installed"

        active_job_id = None
        with self._lock:
            active_job_id = self._active_models.get(model_id)

        return {
            "model_id": model_id,
            "display_name": info["display"],
            "status": overall,
            "env_name": env_name,
            "env_installed": env_ok,
            "weights_installed": weights_ok,
            "weights_repo": info["weights_repo"],
            "gated": info["gated"],
            "active_job_id": active_job_id,
        }

    def get_all_model_status(self) -> dict[str, dict]:
        """Check status for all models."""
        return {mid: self.get_model_status(mid) for mid in MODEL_SETUP}

    # ── Job queries ──────────────────────────────────────────────

    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job status (without full logs). Returns None if not found."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return self._job_to_dict(job)

    def get_job_logs(self, job_id: str, offset: int = 0) -> Optional[dict]:
        """Get job log lines starting from offset for incremental fetching."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            lines = job.log_lines[offset:]
            return {
                "job_id": job_id,
                "offset": offset,
                "total_lines": len(job.log_lines),
                "lines": lines,
                "status": job.status,
            }

    def list_jobs(self, active_only: bool = False) -> list[dict]:
        """List all tracked jobs."""
        with self._lock:
            jobs = list(self._jobs.values())
            if active_only:
                jobs = [j for j in jobs if j.status in ("pending", "running")]
            return [self._job_to_dict(j) for j in jobs]

    @staticmethod
    def _job_to_dict(job: InstallJob) -> dict:
        return {
            "job_id": job.job_id,
            "model_id": job.model_id,
            "operation": job.operation,
            "status": job.status,
            "progress_pct": job.progress_pct,
            "current_step": job.current_step,
            "step_index": job.step_index,
            "step_total": job.step_total,
            "error": job.error,
            "created_at": job.created_at,
            "finished_at": job.finished_at,
            "log_line_count": len(job.log_lines),
        }

    # ── Install ──────────────────────────────────────────────────

    def start_install(self, model_id: str) -> str:
        """Start an install job. Returns job_id immediately. Raises if already active."""
        if model_id not in MODEL_SETUP:
            raise ValueError(f"Unknown model: {model_id}")

        with self._lock:
            if model_id in self._active_models:
                existing = self._active_models[model_id]
                raise RuntimeError(
                    f"Model '{model_id}' already has an active job: {existing}"
                )
            job_id = uuid.uuid4().hex[:12]
            job = InstallJob(job_id=job_id, model_id=model_id, operation="install")
            self._jobs[job_id] = job
            self._active_models[model_id] = job_id

        thread = threading.Thread(
            target=self._run_install, args=(job_id,), daemon=True,
            name=f"install-{model_id}-{job_id}",
        )
        thread.start()
        return job_id

    def cancel_install(self, model_id: str) -> bool:
        """Cancel an active install job. Kills the running subprocess if any."""
        with self._lock:
            job_id = self._active_models.get(model_id)
            if not job_id:
                return False
            job = self._jobs.get(job_id)
            if job and job.status == "running":
                job._cancel_event.set()
                proc = job._current_process
                if proc and proc.poll() is None:
                    proc.kill()
                return True
        return False

    # ── Download (weights only) ─────────────────────────────────

    def start_download(self, model_id: str) -> str:
        """Start a weight-only download job. Returns job_id immediately.

        Raises ValueError if model unknown or has no weights_repo.
        Raises RuntimeError if model already has an active job.
        """
        if model_id not in MODEL_SETUP:
            raise ValueError(f"Unknown model: {model_id}")

        info = MODEL_SETUP[model_id]
        if not info["weights_repo"]:
            raise ValueError(
                f"Model '{model_id}' has no downloadable weights "
                "(auto-downloads on first use)"
            )

        with self._lock:
            if model_id in self._active_models:
                existing = self._active_models[model_id]
                raise RuntimeError(
                    f"Model '{model_id}' already has an active job: {existing}"
                )
            job_id = uuid.uuid4().hex[:12]
            job = InstallJob(job_id=job_id, model_id=model_id, operation="download")
            self._jobs[job_id] = job
            self._active_models[model_id] = job_id

        thread = threading.Thread(
            target=self._run_download, args=(job_id,), daemon=True,
            name=f"download-{model_id}-{job_id}",
        )
        thread.start()
        return job_id

    def _run_download(self, job_id: str):
        """Execute weight download in a background thread."""
        with self._lock:
            job = self._jobs[job_id]
        model_id = job.model_id
        info = MODEL_SETUP[model_id]

        def log(msg: str):
            self._job_log(job, msg)

        def _check_cancel():
            if job._cancel_event.is_set():
                job.status = "cancelled"
                job.current_step = "Cancelled by user"
                log("Download cancelled.")
                raise _CancelledError()

        try:
            job.status = "running"
            job.step_total = 1
            job.step_index = 1
            job.current_step = "Downloading weights"

            _check_cancel()

            weights_path = MODELS_DIR / info["weights_dir"]
            try:
                weights_exist = weights_path.exists() and any(weights_path.iterdir())
            except (OSError, StopIteration):
                weights_exist = False

            if weights_exist:
                log(f"Weights already present at {weights_path}.")
                job.progress_pct = 100.0
                job.current_step = "Complete"
                job.status = "completed"
                log("Nothing to download.")
                return

            log(f"Downloading weights from {info['weights_repo']}...")
            self._download_weights(info, log, job=job)

            job.progress_pct = 100.0
            job.current_step = "Complete"
            job.status = "completed"
            log(f"\n{info['display']} weights downloaded!")

        except _CancelledError:
            pass

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.current_step = "Failed"
            log(f"\nDownload failed: {e}")
            logger.error("Download job %s failed: %s", job_id, e, exc_info=True)

        finally:
            job.finished_at = time.time()
            job._current_process = None
            with self._lock:
                self._active_models.pop(model_id, None)
            self._prune_old_jobs()

    # ── Install ──────────────────────────────────────────────────

    def _run_install(self, job_id: str):
        """Execute the full install pipeline in a background thread."""
        with self._lock:
            job = self._jobs[job_id]
        model_id = job.model_id
        info = MODEL_SETUP[model_id]
        env_name = info["env"]
        env = self.environments.get(env_name)

        def log(msg: str):
            self._job_log(job, msg)

        def _check_cancel():
            if job._cancel_event.is_set():
                job.status = "cancelled"
                job.current_step = "Cancelled by user"
                log("Installation cancelled.")
                raise _CancelledError()

        try:
            job.status = "running"

            if env:
                env.check_status()
                if env.status != "installed":
                    steps = env.config.get_install_steps()
                    total_steps = len(steps) + 1  # +1 for venv creation
                    weight_steps = 1 if info["weights_repo"] else 0
                    job.step_total = total_steps + weight_steps

                    # Create venv
                    job.step_index = 1
                    job.current_step = "Creating virtual environment"
                    log(f"Creating venv at {env.venv_path}...")
                    _check_cancel()
                    self._create_venv(env, log)
                    job.progress_pct = (1 / job.step_total) * 100

                    # Run pip/git install steps
                    for i, step in enumerate(steps):
                        _check_cancel()
                        job.step_index = i + 2
                        job.current_step = step["description"]
                        log(f"\n> {step['description']}...")
                        self._execute_step(env, step, log, job=job)
                        job.progress_pct = ((i + 2) / job.step_total) * 100

                    env.status = "installed"
                    if env.config.system_notes:
                        log(f"\n{env.config.system_notes}")
                    log(f"\n{env.config.display_name} environment installed!")
                else:
                    log(f"Environment '{env_name}' already installed.")
                    job.step_total = 1 if info["weights_repo"] else 1
                    job.step_index = 1
                    job.progress_pct = 50.0 if info["weights_repo"] else 90.0

            # Weight download
            _check_cancel()
            if info["weights_repo"]:
                weights_path = MODELS_DIR / info["weights_dir"]
                try:
                    weights_exist = weights_path.exists() and any(weights_path.iterdir())
                except (OSError, StopIteration):
                    weights_exist = False

                if not weights_exist:
                    job.step_index = job.step_total
                    job.current_step = f"Downloading weights"
                    log(f"\nDownloading weights from {info['weights_repo']}...")
                    self._download_weights(info, log, job=job)
                else:
                    log(f"Weights already present at {weights_path}.")
            else:
                log(f"{info['display']} weights auto-download on first use.")

            job.progress_pct = 100.0
            job.current_step = "Complete"
            job.status = "completed"
            log(f"\n{info['display']} installation complete!")

        except _CancelledError:
            pass  # status already set in _check_cancel

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.current_step = "Failed"
            log(f"\nInstallation failed: {e}")
            logger.error("Install job %s failed: %s", job_id, e, exc_info=True)

        finally:
            job.finished_at = time.time()
            job._current_process = None
            with self._lock:
                self._active_models.pop(model_id, None)
            self._prune_old_jobs()

    # ── Uninstall ────────────────────────────────────────────────

    def start_uninstall(self, model_id: str) -> str:
        """Start an uninstall job. Returns job_id immediately."""
        if model_id not in MODEL_SETUP:
            raise ValueError(f"Unknown model: {model_id}")

        with self._lock:
            if model_id in self._active_models:
                existing = self._active_models[model_id]
                raise RuntimeError(
                    f"Model '{model_id}' already has an active job: {existing}"
                )
            job_id = uuid.uuid4().hex[:12]
            job = InstallJob(job_id=job_id, model_id=model_id, operation="uninstall")
            self._jobs[job_id] = job
            self._active_models[model_id] = job_id

        thread = threading.Thread(
            target=self._run_uninstall, args=(job_id,), daemon=True,
            name=f"uninstall-{model_id}-{job_id}",
        )
        thread.start()
        return job_id

    def _run_uninstall(self, job_id: str):
        """Execute uninstall in a background thread."""
        with self._lock:
            job = self._jobs[job_id]
        model_id = job.model_id
        info = MODEL_SETUP[model_id]

        def log(msg: str):
            self._job_log(job, msg)

        try:
            job.status = "running"
            env = self.environments.get(info["env"])

            # Remove weights
            if info.get("weights_dir"):
                weights_path = MODELS_DIR / info["weights_dir"]
                if weights_path.exists():
                    job.current_step = "Removing weights"
                    log(f"Removing weights at {weights_path}...")
                    shutil.rmtree(weights_path)
                    log("Weights removed.")
                    job.progress_pct = 50.0

            # Remove venv
            if env and env.venv_path.exists():
                job.current_step = "Removing virtual environment"
                log(f"Removing venv at {env.venv_path}...")
                shutil.rmtree(env.venv_path)
                env.status = "not_installed"
                log("Environment removed.")

            job.progress_pct = 100.0
            job.current_step = "Complete"
            job.status = "completed"
            log(f"{info['display']} removed.")

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.current_step = "Failed"
            log(f"Uninstall failed: {e}")
            logger.error("Uninstall job %s failed: %s", job_id, e, exc_info=True)

        finally:
            job.finished_at = time.time()
            with self._lock:
                self._active_models.pop(model_id, None)

    # ── Internal helpers ─────────────────────────────────────────

    def _job_log(self, job: InstallJob, msg: str):
        """Append log line to job and optionally forward to external callback."""
        msg = _ANSI_RE.sub('', msg)
        if not msg.strip():
            return
        job.log_lines.append(msg)
        if len(job.log_lines) > self.MAX_LOG_LINES:
            job.log_lines = job.log_lines[-self.MAX_LOG_LINES:]
        if self._external_log:
            try:
                self._external_log(msg)
            except Exception:
                pass

    def _create_venv(self, env: MusicEnvironment, log: Callable):
        VENVS_DIR.mkdir(exist_ok=True)
        if not env.venv_path.exists():
            # Embedded Python doesn't include the venv module, so use virtualenv
            result = subprocess.run(
                [str(PYTHON_PATH), "-m", "virtualenv", str(env.venv_path)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise Exception(f"Failed to create venv: {result.stderr}")
            log("Virtual environment created.")
        else:
            log("Virtual environment already exists.")

    def _execute_step(self, env: MusicEnvironment, step: dict, log: Callable,
                       job: Optional[InstallJob] = None):
        pip_path = str(env.pip_path)

        # Resolve cwd shorthand: "venv" → venv root, "repos" → repos dir
        cwd = step.get("cwd")
        if cwd == "venv":
            cwd = str(env.venv_path)
        elif cwd == "repos":
            cwd = str(env.venv_path / "repos")

        # Callback to store subprocess ref on job for cancel support
        def _track_process(proc):
            if job:
                job._current_process = proc

        def _clear_process():
            if job:
                job._current_process = None

        if step["type"] == "pip":
            # Skip CUDA torch reinstall if already installed correctly
            if step.get("skip_if_cuda_torch") and self._has_cuda_torch(env):
                log("CUDA PyTorch already installed, skipping reinstall.")
                return

            # Validate requirements file exists before running
            args = step["args"]
            if "-r" in args:
                idx = args.index("-r") + 1
                if idx < len(args):
                    req_path = Path(cwd or ".") / args[idx]
                    if not req_path.exists():
                        raise FileNotFoundError(
                            f"Requirements file not found: {req_path}"
                        )

            success, output = run_pip_install(
                pip_path, args, log, cwd=cwd,
                process_callback=_track_process,
            )
            _clear_process()
            if not success:
                if step.get("optional"):
                    log(f"Optional step failed (continuing): {step['description']}")
                    return
                raise Exception(f"Pip install failed: {output}")
        elif step["type"] == "git_clone":
            if "target_dir" in step:
                clone_target = env.venv_path / step["target_dir"]
            else:
                clone_target = env.venv_path / "repos"
            clone_target.mkdir(parents=True, exist_ok=True)
            success, output = run_git_clone(
                step["url"], str(clone_target), pip_path,
                step.get("editable", True), log,
                process_callback=_track_process,
            )
            _clear_process()
            if not success:
                raise Exception(f"Git clone failed: {output}")

        elif step["type"] == "download_extract":
            # Download a zip and extract prefixes to target directories.
            # Supports multi-extraction (one download, multiple prefixes):
            #   "extractions": [{"prefix": "espeak-ng/", "target_dir": "...", "skip_if": "..."}]
            # Also supports legacy single-extraction format:
            #   "extract_prefix", "target_dir", "skip_if_exists"
            url = step["url"]
            extractions = step.get("extractions")
            if not extractions:
                extractions = [{
                    "prefix": step.get("extract_prefix", ""),
                    "target_dir": step["target_dir"],
                    "skip_if": step.get("skip_if_exists"),
                }]

            # Check which extractions are needed
            needed = []
            for ext in extractions:
                skip_path = ext.get("skip_if")
                if skip_path and Path(skip_path).exists():
                    log(f"Already present: {skip_path}, skipping.")
                else:
                    needed.append(ext)

            if not needed:
                return

            log(f"Downloading {url}...")
            zip_path = BASE_DIR / "temp" / f"_dl_{uuid.uuid4().hex[:8]}.zip"
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(url, str(zip_path))
                log(f"Downloaded {zip_path.stat().st_size / 1024 / 1024:.1f} MB")

                with zipfile.ZipFile(str(zip_path), "r") as zf:
                    all_names = zf.namelist()
                    for ext in needed:
                        prefix = ext["prefix"]
                        target = Path(ext["target_dir"])
                        target.mkdir(parents=True, exist_ok=True)
                        members = [n for n in all_names if n.startswith(prefix)]
                        for m in members:
                            zf.extract(m, str(target))
                        log(f"Extracted {len(members)} files to {target / prefix}")
            finally:
                if zip_path.exists():
                    zip_path.unlink()

    @staticmethod
    def _has_cuda_torch(env: MusicEnvironment) -> bool:
        """Check if the venv already has CUDA-enabled PyTorch installed."""
        try:
            result = subprocess.run(
                [str(env.python_path), "-c",
                 "import torch; print(torch.version.cuda or '')"],
                capture_output=True, text=True, timeout=30,
            )
            cuda_ver = result.stdout.strip()
            return result.returncode == 0 and cuda_ver != ""
        except Exception:
            return False

    def _download_weights(self, info: dict, log: Callable,
                           job: Optional[InstallJob] = None):
        """Download model weights via huggingface_hub.snapshot_download.

        weights_repo can be:
          - A string (single repo downloaded to weights_dir)
          - A list of (repo_id, subdir) tuples (each downloaded to weights_dir/subdir)
        """
        hf_python = None
        own_env = VENVS_DIR / info["env"] / "Scripts" / "python.exe"
        if own_env.exists():
            hf_python = str(own_env)
        else:
            for env_name in ALL_CONFIGS.keys():
                env_path = VENVS_DIR / env_name / "Scripts" / "python.exe"
                if env_path.exists():
                    hf_python = str(env_path)
                    break

        if not hf_python:
            raise Exception("No environment with huggingface_hub found.")

        base_dir = MODELS_DIR / info["weights_dir"]
        repos = info["weights_repo"]

        # Normalize to list of (repo_id, subdir) tuples
        if isinstance(repos, str):
            repos = [(repos, "")]

        for repo_id, subdir in repos:
            target_dir = base_dir / subdir if subdir else base_dir
            log(f"Downloading {repo_id} to {target_dir}...")

            download_script = (
                f'import os\n'
                f'os.environ.setdefault("HF_HUB_CACHE", r"{MODELS_DIR / "hub"}")\n'
                f'from huggingface_hub import snapshot_download\n'
                f'snapshot_download(\n'
                f'    repo_id="{repo_id}",\n'
                f'    local_dir=r"{target_dir}",\n'
                f')\n'
                f'print("Download complete!")\n'
            )

            process = subprocess.Popen(
                [hf_python, "-c", download_script],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            if job:
                job._current_process = process
            for line in process.stdout:
                log(line.rstrip())
            process.wait()
            if job:
                job._current_process = None

            if process.returncode != 0:
                raise Exception(f"Weight download failed for {repo_id} (exit code {process.returncode})")

        log(f"Weights downloaded to {base_dir}")

    def _prune_old_jobs(self):
        """Remove oldest completed/failed/cancelled jobs beyond MAX_COMPLETED_JOBS."""
        with self._lock:
            finished = [j for j in self._jobs.values()
                        if j.status in ("completed", "failed", "cancelled")]
            if len(finished) > self.MAX_COMPLETED_JOBS:
                finished.sort(key=lambda j: j.finished_at or 0)
                for j in finished[:len(finished) - self.MAX_COMPLETED_JOBS]:
                    self._jobs.pop(j.job_id, None)
