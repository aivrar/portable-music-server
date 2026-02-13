"""MusicGen / AudioCraft installation configuration.

Install order matters:
  1. CUDA torch + xformers as a matched pair from the cu124 index
  2. audiocraft with --no-deps (it pins torch==2.1.0 which conflicts)
  3. audiocraft's remaining dependencies (minus torch/xformers/gradio)
"""

from .base import InstallConfig


class MusicGenConfig(InstallConfig):
    """Install config for MusicGen / AudioCraft (Meta's text-to-music)."""

    def __init__(self):
        super().__init__()
        self.name = "musicgen_env"
        self.display_name = "MusicGen (AudioCraft)"
        self.description = "Meta's text-to-music generation model"
        self.verify_package = "audiocraft"
        self.pip_packages = []

    def get_install_steps(self):
        steps = []
        steps.append({"type": "pip", "description": "Upgrading pip",
                       "args": ["install", "--upgrade", "pip"]})
        steps.append({"type": "pip", "description": "Installing worker dependencies",
                       "args": ["install"] + self.WORKER_PACKAGES})

        # 1. CUDA torch + xformers FIRST — matched pair from the same index
        steps.append({"type": "pip",
                       "description": "Installing PyTorch with CUDA",
                       "args": ["install",
                                "torch", "torchvision", "torchaudio",
                                "--index-url", "https://download.pytorch.org/whl/cu124"],
                       "skip_if_cuda_torch": True})
        steps.append({"type": "pip",
                       "description": "Installing xformers (matched to CUDA torch)",
                       "args": ["install", "xformers",
                                "--index-url", "https://download.pytorch.org/whl/cu124"]})

        # 2. PyAV binary + audiocraft without deps (avoids torch==2.1.0 pin)
        steps.append({"type": "pip", "description": "Installing PyAV (binary only)",
                       "args": ["install", "--only-binary", ":all:", "av"]})
        steps.append({"type": "pip", "description": "Installing audiocraft (no deps)",
                       "args": ["install", "--no-deps", "audiocraft"]})

        # 3. audiocraft's real dependencies (excluding torch, xformers, gradio)
        steps.append({"type": "pip",
                       "description": "Installing audiocraft dependencies",
                       "args": ["install",
                                "einops", "flashy", "hydra-core", "hydra_colorlog",
                                "julius", "num2words", "sentencepiece", "spacy",
                                "huggingface_hub", "tqdm", "transformers",
                                "demucs", "librosa", "torchmetrics",
                                "encodec", "protobuf"]})
        return steps
