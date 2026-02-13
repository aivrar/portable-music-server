"""HeartMuLa installation configuration.

Note: heartlib pins torch==2.4.1 (CPU) in its pyproject.toml dependencies.
The editable install overwrites CUDA torch with CPU torch from PyPI.
We must re-install CUDA torch AFTER the heartlib editable install.
"""

from .base import InstallConfig


class HeartMuLaConfig(InstallConfig):
    """Install config for HeartMuLa 3B (lyrics-to-music with RL optimization)."""

    def __init__(self):
        super().__init__()
        self.name = "heartmula_env"
        self.display_name = "HeartMuLa 3B"
        self.description = "Lyrics-to-music generation with RL optimization"
        self.verify_package = "heartlib"
        self.pip_packages = []
        self.git_repos = [
            ("https://github.com/HeartMuLa/heartlib.git", True),
        ]

    def get_install_steps(self):
        """Custom install order: heartlib first, then CUDA torch to overwrite CPU version."""
        steps = []

        steps.append({
            "type": "pip",
            "description": "Upgrading pip",
            "args": ["install", "--upgrade", "pip"],
        })

        steps.append({
            "type": "pip",
            "description": "Installing worker dependencies",
            "args": ["install"] + self.WORKER_PACKAGES,
        })

        # Clone and editable-install heartlib (pulls CPU torch as dependency)
        for repo_url, editable in self.git_repos:
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            steps.append({
                "type": "git_clone",
                "description": f"Cloning {repo_name}",
                "url": repo_url,
                "editable": editable,
            })

        # Re-install CUDA torch AFTER heartlib to overwrite CPU version.
        steps.append({
            "type": "pip",
            "description": "Installing PyTorch with CUDA (overwriting CPU version)",
            "args": ["install", "--force-reinstall", "--no-deps",
                     "torch", "torchvision", "torchaudio",
                     "--index-url", "https://download.pytorch.org/whl/cu121"],
            "skip_if_cuda_torch": True,
        })

        return steps
