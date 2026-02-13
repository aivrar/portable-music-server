"""Base installation configuration and utilities for music environments."""

import re
import subprocess
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Callable, Optional

# Strip ANSI escape sequences from subprocess output
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07|\r')

# Import git path from config if available
try:
    from config import GIT_PATH
except ImportError:
    GIT_PATH = "git"


@dataclass
class InstallConfig:
    """Configuration for a music environment installation."""

    name: str = ""
    display_name: str = ""
    description: str = ""

    # Package installation steps - each item is a pip install command args
    pip_packages: List[List[str]] = field(default_factory=list)

    # Git repositories to clone - list of (url, install_editable) tuples
    git_repos: List[tuple] = field(default_factory=list)

    # Key package to check for verification (pip show)
    verify_package: str = ""

    # Key file path to check for verification (relative to venv dir)
    # Use for clone-only repos that aren't pip-installed
    verify_path: str = ""

    # System dependencies or notes to display
    system_notes: str = ""

    # Packages required by music_worker.py in every model venv
    WORKER_PACKAGES = ["numpy", "uvicorn", "fastapi", "soundfile"]

    def get_install_steps(self) -> List[dict]:
        """Return list of installation steps with descriptions."""
        steps = []

        # Upgrade pip first
        steps.append({
            "type": "pip",
            "description": "Upgrading pip",
            "args": ["install", "--upgrade", "pip"]
        })

        # Install worker runtime dependencies (needed by music_worker.py)
        steps.append({
            "type": "pip",
            "description": "Installing worker dependencies",
            "args": ["install"] + self.WORKER_PACKAGES
        })

        # Add pip package installations
        for pkg_args in self.pip_packages:
            desc = f"Installing {pkg_args[0] if pkg_args else 'packages'}"
            steps.append({
                "type": "pip",
                "description": desc,
                "args": ["install"] + pkg_args
            })

        # Add git clone steps
        for repo_url, editable in self.git_repos:
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            steps.append({
                "type": "git_clone",
                "description": f"Cloning {repo_name}",
                "url": repo_url,
                "editable": editable
            })

        return steps


def run_pip_install(
    pip_path: str,
    args: List[str],
    log_callback: Optional[Callable[[str], None]] = None,
    cwd: Optional[str] = None,
    timeout: int = 600,
    process_callback: Optional[Callable] = None
) -> tuple:
    """Run pip install command.

    Args:
        pip_path: Path to pip executable (or python executable for -m pip)
        args: Arguments for pip (e.g., ["install", "torch"])
        log_callback: Optional callback for logging output
        cwd: Optional working directory for the subprocess
        timeout: Maximum seconds before killing the process (default 600)
        process_callback: Optional callback(process) called after Popen creation

    Returns:
        Tuple of (success: bool, output: str)
    """
    # Use python -m pip instead of pip directly for better compatibility
    python_path = pip_path.replace("pip.exe", "python.exe").replace("pip", "python")
    cmd = [python_path, "-m", "pip", "--no-color"] + args

    if log_callback:
        log_callback(f"Running: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=cwd
        )
        if process_callback:
            process_callback(process)

        output_lines = []
        start = time.monotonic()
        for line in process.stdout:
            if time.monotonic() - start > timeout:
                process.kill()
                process.wait()
                msg = f"Timed out after {timeout}s"
                if log_callback:
                    log_callback(msg)
                return False, msg
            line = _ANSI_RE.sub('', line).rstrip()
            if not line:
                continue
            output_lines.append(line)
            if log_callback:
                log_callback(line)

        process.wait()
        output = "\n".join(output_lines)

        return process.returncode == 0, output

    except Exception as e:
        error_msg = f"Error running pip: {e}"
        if log_callback:
            log_callback(error_msg)
        return False, error_msg


def run_git_clone(
    url: str,
    target_dir: str,
    pip_path: str,
    editable: bool = True,
    log_callback: Optional[Callable[[str], None]] = None,
    timeout: int = 300,
    process_callback: Optional[Callable] = None
) -> tuple:
    """Clone a git repository and optionally install it.

    Args:
        url: Git repository URL
        target_dir: Directory to clone into
        pip_path: Path to pip for editable install
        editable: Whether to run pip install -e
        log_callback: Optional callback for logging
        timeout: Maximum seconds before killing the process (default 300)
        process_callback: Optional callback(process) called after Popen creation

    Returns:
        Tuple of (success: bool, output: str)
    """
    repo_name = url.split("/")[-1].replace(".git", "")
    clone_path = os.path.join(target_dir, repo_name)

    if log_callback:
        log_callback(f"Cloning {url} to {clone_path}")

    try:
        # Check if already cloned (must have .git dir, not just an empty directory)
        if os.path.isdir(os.path.join(clone_path, ".git")):
            if log_callback:
                log_callback(f"Repository already exists at {clone_path}, pulling latest...")
            process = subprocess.Popen(
                [GIT_PATH, "-C", clone_path, "pull"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
        else:
            # Remove empty directory if it exists (e.g. placeholder from parent repo)
            if os.path.isdir(clone_path) and not os.listdir(clone_path):
                os.rmdir(clone_path)
            process = subprocess.Popen(
                [GIT_PATH, "clone", url, clone_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
        if process_callback:
            process_callback(process)

        output_lines = []
        start = time.monotonic()
        for line in process.stdout:
            if time.monotonic() - start > timeout:
                process.kill()
                process.wait()
                msg = f"Git clone timed out after {timeout}s"
                if log_callback:
                    log_callback(msg)
                return False, msg
            line = _ANSI_RE.sub('', line).rstrip()
            if not line:
                continue
            output_lines.append(line)
            if log_callback:
                log_callback(line)

        process.wait()

        if process.returncode != 0:
            return False, "\n".join(output_lines)

        # Run editable install if requested
        if editable and (os.path.exists(os.path.join(clone_path, "setup.py")) or
                         os.path.exists(os.path.join(clone_path, "pyproject.toml"))):
            if log_callback:
                log_callback(f"Installing {repo_name} in editable mode...")

            success, install_output = run_pip_install(
                pip_path,
                ["install", "-e", clone_path],
                log_callback,
                process_callback=process_callback
            )

            if not success:
                return False, install_output

            output_lines.append(install_output)

        return True, "\n".join(output_lines)

    except Exception as e:
        error_msg = f"Error cloning repository: {e}"
        if log_callback:
            log_callback(error_msg)
        return False, error_msg


def check_package_installed(pip_path: str, package_name: str) -> bool:
    """Check if a package is installed in the environment."""
    try:
        result = subprocess.run(
            [pip_path, "show", package_name],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False
