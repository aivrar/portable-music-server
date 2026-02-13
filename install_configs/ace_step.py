"""ACE-Step v1.5 and v1 installation configurations.

v1.5: requires-python==3.11.* but we use 3.10, so editable install fails.
      We clone without installing, add repo to sys.path at runtime, and
      install dependencies separately.

v1:   Uses ACEStepPipeline from acestep.pipeline_ace_step. Its setup.py
      pulls CPU torch via requirements.txt, so CUDA torch goes last.
"""

from .base import InstallConfig


class AceStepV15Config(InstallConfig):
    """Install config for ACE-Step v1.5 (DiT + LLM, cloned without editable install)."""

    def __init__(self):
        super().__init__()
        self.name = "ace_step_env"
        self.display_name = "ACE-Step v1.5"
        self.description = "High-quality music generation with lyrics support"
        # Can't verify "acestep" — not pip-installed (clone-only, no editable
        # install due to Python 3.11 requirement).  nano-vllm is the last
        # editable install in the chain, so its presence confirms success.
        self.verify_package = "nano-vllm"
        self.pip_packages = []
        # editable=False because pyproject.toml requires Python 3.11
        self.git_repos = [
            ("https://github.com/ace-step/ACE-Step-1.5.git", False),
        ]

    def get_install_steps(self):
        steps = []
        steps.append({"type": "pip", "description": "Upgrading pip",
                       "args": ["install", "--upgrade", "pip"]})
        steps.append({"type": "pip", "description": "Installing worker dependencies",
                       "args": ["install"] + self.WORKER_PACKAGES})

        # Clone repo (no editable install — Python 3.11 requirement blocks it)
        for repo_url, editable in self.git_repos:
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            steps.append({"type": "git_clone", "description": f"Cloning {repo_name}",
                           "url": repo_url, "editable": editable})

        # Install v1.5 dependencies manually (from its pyproject.toml).
        # Pin diffusers==0.30.0 — newer versions need torchao 0.15+ which
        # requires torch 2.7+, but we use torch 2.5.x on cu121.
        steps.append({"type": "pip",
                       "description": "Installing ACE-Step v1.5 dependencies",
                       "args": ["install",
                                "transformers>=4.51.0,<4.58.0", "diffusers==0.30.0",
                                "scipy", "soundfile", "loguru", "einops",
                                "accelerate", "numba", "vector-quantize-pytorch",
                                "toml", "modelscope", "peft"]})

        # Install nano-vllm (local third-party in the cloned repo)
        # cwd="venv" so ./repos/ resolves relative to the venv directory
        steps.append({"type": "pip",
                       "description": "Installing nano-vllm",
                       "args": ["install", "-e", "./repos/ACE-Step-1.5/acestep/third_parts/nano-vllm"],
                       "cwd": "venv"})

        # CUDA torch last to overwrite any CPU version pulled by deps
        # skip_if_cuda_torch: skips entirely when CUDA torch is already present
        steps.append({"type": "pip",
                       "description": "Installing PyTorch with CUDA",
                       "args": ["install", "--force-reinstall", "--no-deps",
                                "torch", "torchvision", "torchaudio",
                                "--index-url", "https://download.pytorch.org/whl/cu121"],
                       "skip_if_cuda_torch": True})
        return steps


class AceStepV1Config(InstallConfig):
    """Install config for ACE-Step v1 (original ACEStepPipeline)."""

    def __init__(self):
        super().__init__()
        self.name = "ace_step_v1_env"
        self.display_name = "ACE-Step v1"
        self.description = "Original ACE-Step music generation"
        self.verify_package = "ace_step"
        self.pip_packages = []
        self.git_repos = [
            ("https://github.com/ace-step/ACE-Step.git", True),
        ]

    def get_install_steps(self):
        steps = []
        steps.append({"type": "pip", "description": "Upgrading pip",
                       "args": ["install", "--upgrade", "pip"]})
        steps.append({"type": "pip", "description": "Installing worker dependencies",
                       "args": ["install"] + self.WORKER_PACKAGES})

        # Clone + editable install (v1 supports Python 3.10)
        for repo_url, editable in self.git_repos:
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            steps.append({"type": "git_clone", "description": f"Cloning {repo_name}",
                           "url": repo_url, "editable": editable})

        # CUDA torch last — v1's requirements.txt pulls CPU torch
        steps.append({"type": "pip",
                       "description": "Installing PyTorch with CUDA (overwriting CPU version)",
                       "args": ["install", "--force-reinstall", "--no-deps",
                                "torch", "torchvision", "torchaudio",
                                "--index-url", "https://download.pytorch.org/whl/cu121"],
                       "skip_if_cuda_torch": True})
        return steps
