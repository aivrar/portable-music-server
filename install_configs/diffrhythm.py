"""DiffRhythm installation configuration.

Note: DiffRhythm's editable install or requirements.txt may pull CPU torch.
CUDA torch is installed last with --force-reinstall to ensure it wins.
espeak-ng is auto-downloaded for DiffRhythm's phonemizer/g2p pipeline.
"""

import os
from pathlib import Path

from .base import InstallConfig
from config import FFMPEG_PATH

# espeak-ng binary archive hosted on user's repo
_ESPEAK_BIN_URL = (
    "https://github.com/aivrar/LocalSoundsAPI/releases/download/v1.0/bin.zip"
)


class DiffRhythmConfig(InstallConfig):
    """Install config for DiffRhythm (diffusion-based full-song generation)."""

    def __init__(self):
        super().__init__()
        self.name = "diffrhythm_env"
        self.display_name = "DiffRhythm"
        self.description = "Diffusion-based full-song generation with lyrics"
        self.verify_package = "muq"
        self.system_notes = ""
        self.pip_packages = []
        self.git_repos = [
            ("https://github.com/ASLP-lab/DiffRhythm.git", False),
        ]

    def get_install_steps(self):
        steps = []

        # espeak-ng + ffmpeg from portable binary archive (single download)
        # espeak-ng is required by phonemizer for DiffRhythm's g2p pipeline
        # ffmpeg is a fallback if the user doesn't have one installed
        base_dir = Path(__file__).parent.parent.resolve()

        # Skip ffmpeg extraction if any local ffmpeg is already resolved
        ffmpeg_skip = FFMPEG_PATH if FFMPEG_PATH != "ffmpeg" else None

        steps.append({
            "type": "download_extract",
            "description": "Installing espeak-ng and ffmpeg",
            "url": _ESPEAK_BIN_URL,
            "extractions": [
                {
                    "prefix": "espeak-ng/",
                    "target_dir": str(base_dir),
                    "skip_if": str(base_dir / "espeak-ng" / "libespeak-ng.dll"),
                },
                {
                    "prefix": "ffmpeg/",
                    "target_dir": str(base_dir),
                    "skip_if": ffmpeg_skip,
                },
            ],
        })

        steps.append({"type": "pip", "description": "Upgrading pip",
                       "args": ["install", "--upgrade", "pip"]})
        steps.append({"type": "pip", "description": "Installing worker dependencies",
                       "args": ["install"] + self.WORKER_PACKAGES})

        for repo_url, editable in self.git_repos:
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            steps.append({"type": "git_clone", "description": f"Cloning {repo_name}",
                           "url": repo_url, "editable": editable})

        steps.append({"type": "pip",
                       "description": "Installing DiffRhythm requirements",
                       "args": ["install", "-r", "./repos/DiffRhythm/requirements.txt"],
                       "cwd": "venv"})

        # CUDA torch last to overwrite any CPU version pulled by deps
        steps.append({"type": "pip",
                       "description": "Installing PyTorch with CUDA (overwriting CPU version)",
                       "args": ["install", "--force-reinstall", "--no-deps",
                                "torch", "torchvision", "torchaudio",
                                "--index-url", "https://download.pytorch.org/whl/cu121"],
                       "skip_if_cuda_torch": True})
        return steps
