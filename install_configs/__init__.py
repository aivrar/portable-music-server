# Music Module Installation Configurations
from .base import InstallConfig, run_pip_install, run_git_clone
from .ace_step import AceStepV15Config, AceStepV1Config
from .heartmula import HeartMuLaConfig
from .diffrhythm import DiffRhythmConfig
from .yue import YuEConfig
from .musicgen import MusicGenConfig
from .riffusion import RiffusionConfig
from .stable_audio import StableAudioConfig

ALL_CONFIGS = {
    "ace_step_env": AceStepV15Config(),
    "ace_step_v1_env": AceStepV1Config(),
    "heartmula_env": HeartMuLaConfig(),
    "diffrhythm_env": DiffRhythmConfig(),
    "yue_env": YuEConfig(),
    "musicgen_env": MusicGenConfig(),
    "riffusion_env": RiffusionConfig(),
    "stable_audio_env": StableAudioConfig(),
}
