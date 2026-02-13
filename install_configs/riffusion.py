"""Riffusion installation configuration.

Install order matters:
  1. CUDA torch from cu124 index
  2. riffusion with --no-deps (it pins old torch/accelerate versions)
  3. riffusion's real dependencies (diffusers, transformers, accelerate, etc.)
"""

from .base import InstallConfig


class RiffusionConfig(InstallConfig):
    """Install config for Riffusion (Stable Diffusion fine-tuned for music)."""

    def __init__(self):
        super().__init__()
        self.name = "riffusion_env"
        self.display_name = "Riffusion"
        self.description = "Stable Diffusion fine-tuned for music spectrogram generation"
        self.verify_package = "riffusion"
        self.pip_packages = []

    def get_install_steps(self):
        steps = []
        steps.append({"type": "pip", "description": "Upgrading pip",
                       "args": ["install", "--upgrade", "pip"]})
        steps.append({"type": "pip", "description": "Installing worker dependencies",
                       "args": ["install"] + self.WORKER_PACKAGES})

        # 1. CUDA torch FIRST
        steps.append({"type": "pip",
                       "description": "Installing PyTorch with CUDA",
                       "args": ["install",
                                "torch", "torchvision", "torchaudio",
                                "--index-url", "https://download.pytorch.org/whl/cu124"],
                       "skip_if_cuda_torch": True})

        # 2. riffusion without deps (avoids old torch/accelerate pins)
        steps.append({"type": "pip", "description": "Installing riffusion (no deps)",
                       "args": ["install", "--no-deps", "riffusion"]})

        # 3. riffusion's real dependencies
        steps.append({"type": "pip",
                       "description": "Installing riffusion dependencies",
                       "args": ["install",
                                "diffusers", "transformers", "accelerate",
                                "pydub", "Pillow", "librosa", "huggingface_hub",
                                "safetensors", "tqdm"]})
        return steps
