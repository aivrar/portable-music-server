"""Stable Audio Open installation configuration.

Install order matters:
  1. CUDA torch from cu124 index
  2. stable-audio-tools with --no-deps (avoids pulling CPU torch)
  3. Real dependencies separately (excluding torch, gradio, wandb)
"""

from .base import InstallConfig


class StableAudioConfig(InstallConfig):
    """Install config for Stable Audio Open (Stability AI's audio generation)."""

    def __init__(self):
        super().__init__()
        self.name = "stable_audio_env"
        self.display_name = "Stable Audio Open"
        self.description = "Stability AI's open-source audio generation model"
        self.verify_package = "stable_audio_tools"
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

        # 2. stable-audio-tools without deps (avoids CPU torch + version conflicts)
        steps.append({"type": "pip",
                       "description": "Installing stable-audio-tools (no deps)",
                       "args": ["install", "--no-deps", "stable-audio-tools"]})

        # 3. Real dependencies (excluding torch/torchaudio, gradio, wandb)
        steps.append({"type": "pip",
                       "description": "Installing stable-audio-tools dependencies",
                       "args": ["install",
                                "alias-free-torch", "auraloss",
                                "descript-audio-codec", "einops", "einops-exts",
                                "ema-pytorch", "encodec", "huggingface_hub",
                                "importlib-resources", "k-diffusion", "laion-clap",
                                "local-attention", "pandas", "prefigure",
                                "pytorch_lightning", "PyWavelets", "safetensors",
                                "sentencepiece", "torchmetrics", "tqdm",
                                "transformers", "v-diffusion-pytorch",
                                "vector-quantize-pytorch", "webdataset"]})
        return steps
