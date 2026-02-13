"""YuE installation configuration.

Note: YuE's editable install or requirements.txt may pull CPU torch.
CUDA torch is installed last with --force-reinstall to ensure it wins.
"""

from .base import InstallConfig


class YuEConfig(InstallConfig):
    """Install config for YuE (lyrics-to-song with chain-of-thought, 24GB+ VRAM)."""

    def __init__(self):
        super().__init__()
        self.name = "yue_env"
        self.display_name = "YuE"
        self.description = "Lyrics-to-song with chain-of-thought generation"
        self.verify_package = "transformers"
        self.verify_path = "repos/YuE/inference/infer.py"
        self.system_notes = (
            "Note: YuE requires significant VRAM (24GB+).\n"
            "Flash Attention 2 is recommended for best performance.\n"
            "If flash-attn fails to install, generation will still work\n"
            "but may be slower."
        )
        self.pip_packages = []
        self.git_repos = [
            ("https://github.com/multimodal-art-projection/YuE.git", False),
        ]

    def get_install_steps(self):
        steps = []
        steps.append({"type": "pip", "description": "Upgrading pip",
                       "args": ["install", "--upgrade", "pip"]})
        steps.append({"type": "pip", "description": "Installing worker dependencies",
                       "args": ["install"] + self.WORKER_PACKAGES})
        steps.append({"type": "pip", "description": "Installing transformers stack",
                       "args": ["install", "transformers", "accelerate", "safetensors"]})

        for repo_url, editable in self.git_repos:
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            steps.append({"type": "git_clone", "description": f"Cloning {repo_name}",
                           "url": repo_url, "editable": editable})

        # XCodec model (HuggingFace repo) must be cloned into YuE/inference/
        steps.append({"type": "git_clone",
                       "description": "Cloning xcodec_mini_infer (codec model)",
                       "url": "https://huggingface.co/m-a-p/xcodec_mini_infer",
                       "editable": False,
                       "target_dir": "repos/YuE/inference"})

        steps.append({"type": "pip",
                       "description": "Installing YuE requirements",
                       "args": ["install", "-r", "./repos/YuE/requirements.txt"],
                       "cwd": "venv"})

        # CUDA torch before flash-attn — flash-attn needs CUDA torch to compile
        # cu124 required: torch 2.6+ needed by transformers for safe torch.load
        steps.append({"type": "pip",
                       "description": "Installing PyTorch with CUDA (overwriting CPU version)",
                       "args": ["install", "--force-reinstall", "--no-deps",
                                "torch", "torchvision", "torchaudio",
                                "--index-url", "https://download.pytorch.org/whl/cu124"],
                       "skip_if_cuda_torch": True})

        # Flash Attention — try prebuilt Windows wheel first, fall back to source build
        steps.append({"type": "pip",
                       "description": "Installing Flash Attention (prebuilt wheel)",
                       "args": ["install", "--no-deps",
                                "https://github.com/bdashore3/flash-attention/releases/download/"
                                "v2.7.4.post1/flash_attn-2.7.4.post1+cu124torch2.6.0cxx11abiFALSE"
                                "-cp310-cp310-win_amd64.whl"],
                       "optional": True})
        return steps
