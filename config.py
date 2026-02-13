"""
Music Module Configuration

Central configuration for the Music Environment Manager with dynamic path resolution.
Supports embedded Python for portable installs.
"""
import shutil
import os
import subprocess
import time
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.resolve()
VENVS_DIR = BASE_DIR / "venvs"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"

# Environment paths
PYTHON_EMBEDDED_DIR = BASE_DIR / "python_embedded"
GIT_PORTABLE_DIR = BASE_DIR / "git_portable"
FFMPEG_DIR = BASE_DIR / "ffmpeg"
ESPEAK_DIR = BASE_DIR / "espeak-ng"


# Dynamic Python path resolution
def _resolve_python_path() -> Path:
    """Find the best available Python executable.

    Priority: embedded Python > system Python
    """
    # 1. Embedded Python (preferred for portable installs)
    embedded = PYTHON_EMBEDDED_DIR / "python.exe"
    if embedded.exists():
        return embedded

    # 2. System Python (last resort)
    system_python = shutil.which("python")
    if system_python:
        return Path(system_python)

    # 3. Return embedded path even if not yet downloaded
    #    (install.bat will create it)
    return embedded


def _resolve_git_path() -> str:
    """Find the best available git executable.

    Priority: portable Git > system Git
    """
    portable_git = GIT_PORTABLE_DIR / "cmd" / "git.exe"
    if portable_git.exists():
        return str(portable_git)
    return "git"


def _resolve_ffmpeg_path() -> str:
    """Find the best available ffmpeg executable.

    Priority: local ffmpeg (multiple possible locations) > system FFmpeg
    """
    # Check our ffmpeg directory for various structures
    possible_paths = [
        FFMPEG_DIR / "ffmpeg.exe",  # Direct in ffmpeg folder
        FFMPEG_DIR / "bin" / "ffmpeg.exe",  # In bin subfolder
    ]

    # Also check for extracted builds with version in name
    if FFMPEG_DIR.exists():
        for subdir in FFMPEG_DIR.iterdir():
            if subdir.is_dir():
                possible_paths.append(subdir / "bin" / "ffmpeg.exe")
                possible_paths.append(subdir / "ffmpeg.exe")

    for path in possible_paths:
        if path.exists():
            return str(path)

    return "ffmpeg"


def _get_ffmpeg_bin_dir() -> Path | None:
    """Get the directory containing ffmpeg binaries."""
    ffmpeg_path = _resolve_ffmpeg_path()
    if ffmpeg_path != "ffmpeg":
        return Path(ffmpeg_path).parent
    return None


PYTHON_PATH = _resolve_python_path()
GIT_PATH = _resolve_git_path()
FFMPEG_PATH = _resolve_ffmpeg_path()
FFMPEG_BIN_DIR = _get_ffmpeg_bin_dir()

# Whether we're running with embedded Python
USE_EMBEDDED = (PYTHON_EMBEDDED_DIR / "python.exe").exists()


# Set up environment variables
def setup_environment():
    """Configure environment variables for HuggingFace and FFmpeg."""
    # HuggingFace cache — redirect downloads but keep token at ~/.cache/huggingface
    # (HF_HOME would override token lookup too, breaking gated-repo auth)
    os.environ.setdefault("HF_HUB_CACHE", str(MODELS_DIR / "hub"))
    os.environ["TORCH_HOME"] = str(MODELS_DIR / "torch")

    # Add FFmpeg to PATH if available
    if FFMPEG_BIN_DIR:
        current_path = os.environ.get("PATH", "")
        if str(FFMPEG_BIN_DIR) not in current_path:
            os.environ["PATH"] = str(FFMPEG_BIN_DIR) + os.pathsep + current_path

    # Add portable Git to PATH if available
    if GIT_PORTABLE_DIR.exists():
        git_cmd_dir = GIT_PORTABLE_DIR / "cmd"
        if git_cmd_dir.exists():
            current_path = os.environ.get("PATH", "")
            if str(git_cmd_dir) not in current_path:
                os.environ["PATH"] = str(git_cmd_dir) + os.pathsep + current_path

    # Set espeak-ng env vars for DiffRhythm's phonemizer
    espeak_dll = ESPEAK_DIR / "libespeak-ng.dll"
    if espeak_dll.exists():
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(espeak_dll)
        os.environ["PHONEMIZER_ESPEAK_PATH"] = str(ESPEAK_DIR)

    # Create local temp directory and clean up stale files from previous runs
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_temp_files()


def cleanup_temp_files(max_age_hours: float = 4.0):
    """Remove temp files older than max_age_hours from the local temp directory."""
    if not TEMP_DIR.exists():
        return
    cutoff = time.time() - (max_age_hours * 3600)
    for entry in TEMP_DIR.iterdir():
        try:
            if entry.is_file() and entry.stat().st_mtime < cutoff:
                entry.unlink()
            elif entry.is_dir() and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            pass


# UI Settings
WINDOW_TITLE = "Music Module - Environment Manager"
WINDOW_SIZE = "1100x800"
APP_VERSION = "1.0.0"

# API Server settings
DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 9150  # 9100 is reserved by Windows Print Spooler (HP JetDirect)

# Worker management
WORKER_PORT_MIN = 9151
WORKER_PORT_MAX = 9249
WORKER_HEALTH_INTERVAL = 10  # seconds
WORKER_STARTUP_TIMEOUT = 120  # seconds (models can take time to load)
WORKER_MAX_HEALTH_FAILURES = 3
WORKER_AUTO_SPAWN = True  # auto-spawn a worker when request comes for unloaded model
WORKER_LOG_DIR = OUTPUT_DIR / "logs"


def _detect_default_device() -> str:
    """Detect whether a CUDA GPU is available, falling back to CPU.

    Uses nvidia-smi (no torch import needed) so config stays lightweight.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return "cuda:0"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "cpu"


WORKER_DEFAULT_DEVICE = _detect_default_device()

# Output / CLAP
GENERATIONS_DIR = OUTPUT_DIR / "generations"
CLAP_PORT = 9250
