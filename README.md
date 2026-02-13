# Music Module

![License: MIT](https://img.shields.io/github/license/rookiemann/portable-music-server) ![Platform: Windows](https://img.shields.io/badge/Platform-Windows%2010%2F11-blue) ![Python](https://img.shields.io/badge/Python-3.10%20Portable-green) ![Models](https://img.shields.io/badge/Music%20Models-8-orange)

**Portable multi-GPU music generation server for Windows -- 8 models, one-click install, zero dependencies.**

Send a prompt in, get mastered audio out. Lyrics-to-song, text-to-music, sound effects, spectrogram synthesis -- across any combination of NVIDIA GPUs. No system Python, no Git, no FFmpeg, no Docker, no admin rights. Everything downloads automatically into one portable folder.

---

## Highlights

- **8 music generation models** -- from 82M-parameter Riffusion to multi-billion-parameter YuE, covering lyrics-to-song, text-to-music, style transfer, and sound design
- **Multi-GPU inference** -- pin workers to any detected GPU, run the same model on multiple GPUs simultaneously
- **Zero-install portable app** -- embedded Python 3.10, portable Git, bundled FFmpeg + espeak-ng; copy the folder to any Windows machine
- **Production audio pipeline** -- 8-stage mastering (denoise, highpass, compress, stereo widen, EQ, trim, LUFS normalization, peak limiting), multi-format export
- **CLAP scoring** -- optional audio-text similarity scoring via LAION CLAP to objectively evaluate generation quality
- **Persistent output library** -- all generations auto-saved with metadata, searchable and exportable from the GUI
- **Auto-scaling workers** -- workers spawn on first request, fail over to siblings, and get health-checked every 10 seconds
- **Full REST API** -- 35 endpoints covering generation, worker management, install management, CLAP scoring, and output browsing
- **GUI environment manager** -- install venvs, download models, generate music, browse outputs, and manage workers from a Tkinter interface

---

## Architecture

```
                        Port 9150
                    +---------------+
  Client --------->|   GATEWAY     |
                    | (FastAPI)     |
                    |               |
                    | - Generation  |  Post-processing, CLAP scoring,
                    | - Output mgmt|  output persistence, format conversion,
                    | - Load balance|  worker management
                    | - Install mgmt|
                    +-------+-------+
                            |
            +---------------+---------------+------------------+
            |               |               |                  |
     Port 9151       Port 9152       Port 9153          Port 9154
   +-----------+   +-----------+   +-----------+   +-----------+
   | WORKER    |   | WORKER    |   | WORKER    |   | WORKER    |
   | musicgen  |   | stable_a  |   | diffrhythm|   | riffusion |
   | cuda:0    |   | cuda:0    |   | cuda:1    |   | cpu       |
   +-----------+   +-----------+   +-----------+   +-----------+
   musicgen_env    stable_audio_   diffrhythm_     riffusion_
                   env             env              env

                                  Port 9250
                              +-----------+
                              | CLAP      |
                              | Scorer    |
                              | (optional)|
                              +-----------+
                              stable_audio_env
```

- **Gateway** (port 9150) -- orchestrates generation, delegates inference to workers via HTTP
- **Workers** (ports 9151-9249) -- each runs one model on one device as an isolated subprocess
- **CLAP Scorer** (port 9250) -- optional micro-service for audio-text similarity scoring
- Each worker injects only its own venv's `site-packages` -- no cross-environment dependency conflicts
- Same model can run multiple instances across GPUs for concurrent inference
- Workers auto-spawn on first request if none exist for a model
- Auto-detects CUDA GPUs via nvidia-smi; falls back to CPU on GPU-less systems

---

## Quick Start

### First-Time Setup

```batch
install.bat
```

Downloads embedded Python 3.10, portable Git, portable FFmpeg, configures pip + tkinter, installs gateway dependencies, and launches the GUI. From there, install model environments and download weights.

### Start the API Server

```batch
launcher.bat api
```

Gateway starts on `http://127.0.0.1:9150`. Workers auto-spawn when you send your first request.

### Generate Music

```bash
curl -X POST http://127.0.0.1:9150/api/music/stable_audio \
  -H "Content-Type: application/json" \
  -d '{"prompt": "ambient electronic, warm pads, slow tempo", "duration": 30}'
```

### Lyrics-to-Song

```bash
curl -X POST http://127.0.0.1:9150/api/music/ace_step_v15 \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "upbeat pop rock, energetic",
    "params": {
      "caption": "upbeat pop rock, energetic, major key",
      "lyrics": "[verse]\nWalking down the road today\nSunshine melting clouds away",
      "duration": 120,
      "inference_steps": 60
    }
  }'
```

### Other Commands

```batch
launcher.bat gui                :: Launch environment manager GUI (default)
launcher.bat api --port 9200    :: Custom port
launcher.bat setup              :: Re-run full setup
launcher.bat help               :: Show all commands
```

---

## Music Models

Eight models spanning lyrics-to-song, text-to-music, style transfer, and spectrogram synthesis:

| Model | Key Capability | Output Rate | VRAM |
|-------|---------------|-------------|------|
| [**ACE-Step v1.5**](https://github.com/ace-step/ACE-Step) | Lyrics-to-song with DiT + 5Hz LLM, 51 languages, CoT | 48 kHz | <4 GB |
| [**ACE-Step v1**](https://github.com/ace-step/ACE-Step) | Original ACE-Step pipeline | 48 kHz | ~8 GB |
| [**HeartMuLa 3B**](https://huggingface.co/HeartMuLa) | Lyrics-to-music with RL optimization, HeartCodec | 44.1 kHz | ~16 GB |
| [**DiffRhythm**](https://github.com/ASLP-lab/DiffRhythm) | Diffusion-based full-song generation with lyrics | 44.1 kHz | 8 GB |
| [**YuE**](https://github.com/multimodal-art-projection/YuE) | Chain-of-thought lyrics-to-song, two-stage codec | 16 kHz | 24 GB+ |
| [**MusicGen**](https://huggingface.co/facebook/musicgen-large) | Meta's text-to-music (AudioCraft), melody conditioning | Model SR | 8-16 GB |
| [**Riffusion**](https://huggingface.co/riffusion/riffusion-model-v1) | Stable Diffusion fine-tuned for spectrograms | 44.1 kHz | 6-8 GB |
| [**Stable Audio Open**](https://huggingface.co/stabilityai/stable-audio-open-1.0) | Stability AI's latent diffusion for audio, max ~47s | 44.1 kHz | ~8 GB |

### Isolated Environments

Each model runs in its own virtual environment to avoid dependency conflicts:

| Environment | Models | PyTorch | CUDA |
|-------------|--------|---------|------|
| `ace_step_env` | ACE-Step v1.5 | 2.5.x+cu121 | 12.1 |
| `ace_step_v1_env` | ACE-Step v1 | 2.5.x+cu121 | 12.1 |
| `heartmula_env` | HeartMuLa 3B | 2.5.x+cu121 | 12.1 |
| `diffrhythm_env` | DiffRhythm | 2.5.x+cu121 | 12.1 |
| `yue_env` | YuE | 2.6.x+cu124 | 12.4 |
| `musicgen_env` | MusicGen | latest+cu124 | 12.4 |
| `riffusion_env` | Riffusion | latest+cu124 | 12.4 |
| `stable_audio_env` | Stable Audio Open, CLAP | latest+cu124 | 12.4 |

---

## API Reference

35 endpoints. Interactive Swagger docs available at `http://localhost:9150/docs` when the server is running.

### Music Generation

```
POST /api/music/{model}
```

#### Request Parameters

```jsonc
{
  "prompt": "ambient electronic pad",          // Required
  "negative_prompt": "drums, vocals",          // Optional
  "duration": 30,                              // Seconds
  "seed": -1,                                  // -1 = random
  "params": {},                                // Model-specific (see /api/models/{model}/params)
  "output_format": "wav",                      // wav, mp3, ogg, flac
  "denoise_strength": 0.2,                     // 0.0-1.0
  "stereo_width": 1.0,                         // 0.0-2.0
  "eq_preset": "balanced",                     // flat, balanced, warm, bright
  "target_lufs": -14.0,                        // -24 to -8
  "highpass_cutoff": 30,                        // Hz
  "compression_ratio": 2.0,                    // 1-8
  "peak_limit": 0.95,                          // 0.5-1.0
  "trim_db": -40.0,                            // Silence threshold
  "min_silence_ms": 500,                       // Minimum silence to trim
  "front_protect_ms": 0,                       // Don't trim first N ms
  "end_protect_ms": 0,                         // Don't trim last N ms
  "skip_post_process": false                   // Bypass mastering pipeline
}
```

#### Response

```jsonc
{
  "status": "completed",
  "model": "stable_audio",
  "audio_base64": "<base64-encoded audio>",
  "sample_rate": 44100,
  "duration_sec": 30.0,
  "format": "wav",
  "generation_time_sec": 12.3,
  "output_id": "a1b2c3"                       // Reference to persistent output
}
```

### Worker Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/workers` | List all workers with status, device, VRAM |
| `POST` | `/api/workers/spawn` | Spawn worker: `{"model":"musicgen","device":"cuda:0"}` |
| `DELETE` | `/api/workers/{worker_id}` | Kill a specific worker |
| `POST` | `/api/workers/kill-all` | Kill all active workers |
| `GET` | `/api/workers/{worker_id}/logs` | Get worker log output (last N lines) |
| `POST` | `/api/models/{model}/scale` | Scale to N instances: `{"count":2,"device":"cuda:0"}` |
| `POST` | `/api/models/{model}/load` | Load model (spawn a worker) |
| `POST` | `/api/models/{model}/unload` | Unload model (kill all its workers) |
| `GET` | `/api/devices` | GPU discovery with VRAM info |

#### Examples

```bash
# Spawn a MusicGen worker on GPU 0
curl -X POST http://127.0.0.1:9150/api/workers/spawn \
  -H "Content-Type: application/json" \
  -d '{"model": "musicgen", "device": "cuda:0"}'

# Run MusicGen on GPU 0 and Stable Audio on GPU 1 simultaneously
curl -X POST http://127.0.0.1:9150/api/workers/spawn \
  -d '{"model":"stable_audio","device":"cuda:1"}'

# Scale MusicGen to 2 instances for parallel inference
curl -X POST http://127.0.0.1:9150/api/models/musicgen/scale \
  -H "Content-Type: application/json" \
  -d '{"count": 2, "device": "cuda:0"}'

# Kill a specific worker
curl -X DELETE http://127.0.0.1:9150/api/workers/musicgen-1
```

### Install Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/install/status` | Install status for all 8 models |
| `GET` | `/api/install/status/{model_id}` | Status for a specific model |
| `POST` | `/api/install/{model_id}` | Start installing a model (venv + packages + weights) |
| `POST` | `/api/install/{model_id}/download` | Download model weights only |
| `POST` | `/api/install/{model_id}/cancel` | Cancel an active install job |
| `DELETE` | `/api/install/{model_id}` | Uninstall a model (remove venv + weights) |
| `GET` | `/api/install/jobs` | List all install/uninstall jobs |
| `GET` | `/api/install/jobs/{job_id}` | Job status with progress percentage |
| `GET` | `/api/install/jobs/{job_id}/logs` | Job log lines (incremental via offset) |

### CLAP Scoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/clap/status` | CLAP scorer status (running/stopped) |
| `POST` | `/api/clap/start` | Start CLAP scorer (requires stable_audio_env) |
| `POST` | `/api/clap/stop` | Stop CLAP scorer |
| `POST` | `/api/clap/score` | Score audio: `{"audio_base64":"...","prompt":"..."}` → `{"score":0.85}` |

### Output Library

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/outputs` | List all saved generations with metadata |
| `GET` | `/api/outputs/{id}` | Get metadata for a specific output |
| `GET` | `/api/outputs/{id}/audio` | Download the audio file |
| `DELETE` | `/api/outputs/{id}` | Delete a specific output |
| `DELETE` | `/api/outputs/batch` | Delete multiple outputs: `{"ids":["a1b2c3","d4e5f6"]}` |

### Discovery

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check with loaded models and worker count |
| `GET` | `/api/models` | List all available models with install status |
| `GET` | `/api/models/status` | Detailed status per model (workers, loaded) |
| `GET` | `/api/models/{model}/params` | Parameter definitions (control types, defaults, ranges) |
| `GET` | `/api/models/{model}/presets` | Available presets for a model |
| `GET` | `/api/models/{model}/display` | Display metadata (description, VRAM, weights size) |
| `POST` | `/api/shutdown` | Shutdown gateway, kill all workers, stop CLAP |

---

## Audio Pipeline

### 8-Stage Post-Processing (Mastering)

Applied per generation, fully configurable per request:

| Stage | Library | Description |
|-------|---------|-------------|
| 1. Denoise | [noisereduce](https://github.com/timsainb/noisereduce) | Spectral gating using first 0.5s as noise profile |
| 2. Highpass | [scipy](https://scipy.org/) | 4th-order Butterworth at 30 Hz (configurable) |
| 3. Compress | scipy | 3-band (low/mid/high) multiband compression, 2:1 ratio |
| 4. Stereo widen | numpy | Mid-side processing for wider stereo image |
| 5. EQ | scipy | Parametric EQ with presets (flat, balanced, warm, bright) |
| 6. Trim | [pydub](https://github.com/jiaaro/pydub) | Silence detection with front/end protection zones |
| 7. LUFS | [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) | Loudness normalization to -14 LUFS (configurable) |
| 8. Peak limit | numpy | Hard clamp to 0.95 threshold (configurable) |

Each stage degrades gracefully -- if a library is unavailable, that stage is skipped with a warning.

### Format Conversion

Export via [FFmpeg](https://ffmpeg.org/): **WAV**, **MP3**, **OGG**, **FLAC**.

---

## How It Works

### Request Flow

```
1. Client POSTs prompt + params to /api/music/{model}
2. Gateway checks for a ready worker (or auto-spawns one)
3. Forwards inference request to worker via HTTP
4. Worker loads model, generates audio, returns base64 numpy
5. Gateway applies 8-stage post-processing mastering pipeline
6. Converts to requested format via FFmpeg
7. Auto-saves to output library with metadata
8. Optionally scores with CLAP if running
9. Returns audio (base64 JSON) + output entry ID
```

### Worker Lifecycle

- Workers are Python subprocesses, each running its own [FastAPI](https://fastapi.tiangolo.com/) server
- Each injects only its model's venv `site-packages` into `sys.path`
- Health checks run every 10 seconds; 3 consecutive failures trigger cleanup
- Dead workers are terminated, their ports released, and they're unregistered
- Round-robin load balancing across multiple workers for the same model

---

## Configuration

Key settings in `config.py`:

```python
DEFAULT_API_PORT = 9150           # Gateway port

WORKER_PORT_MIN = 9151            # Worker port range
WORKER_PORT_MAX = 9249            # Up to 99 concurrent workers
WORKER_HEALTH_INTERVAL = 10      # Health check every 10 seconds
WORKER_STARTUP_TIMEOUT = 120     # Max wait for model loading (seconds)
WORKER_MAX_HEALTH_FAILURES = 3   # Failures before worker cleanup
WORKER_AUTO_SPAWN = True         # Spawn workers on first request
WORKER_DEFAULT_DEVICE = "cuda:0" # Auto-detected: cuda:0 if GPU present, else cpu

CLAP_PORT = 9250                 # CLAP scorer micro-service
```

### Path Resolution (Portable-First)

1. **Python**: `python_embedded/python.exe` > system Python
2. **Git**: `git_portable/cmd/git.exe` > system Git
3. **FFmpeg**: `ffmpeg/bin/ffmpeg.exe` > `ffmpeg/*/bin/ffmpeg.exe` > system FFmpeg

---

## Directory Structure

```
music_module/
├── install.bat                  One-click setup (downloads Python, Git, FFmpeg)
├── launcher.bat                 Command router (gui, api, setup, help)
├── requirements.txt             Gateway dependencies
├── requirements-dev.txt         Test dependencies (pytest, pytest-asyncio)
├── config.py                    Central configuration with dynamic path resolution
│
├── music_api_server.py          Gateway: generation pipeline, worker/install/output management
├── music_worker.py              Worker: single-model FastAPI inference server (8 models)
├── worker_registry.py           Worker tracking: port pool, round-robin balancing
├── worker_manager.py            Worker lifecycle: spawn, kill, scale, health checks
├── install_manager.py           Headless install/uninstall/status with job tracking
│
├── model_params.py              Shared parameter definitions, presets, display metadata
├── output_manager.py            Persistent output storage (audio + JSON metadata)
├── clap_scorer.py               CLAP audio-text similarity micro-service
│
├── audio_profiles.py            Per-model audio processing presets
├── audio_processing.py          8-stage mastering pipeline
├── audio_assembler.py           Format conversion via FFmpeg
│
├── music_manager.py             Tkinter GUI (5 tabs: Setup, API Server, Testing, Outputs, Log)
│
├── install_configs/             8 virtual environment configurations
│   ├── base.py                  Abstract base (pip install, git clone, verify)
│   ├── ace_step.py              ACE-Step v1.5 + v1
│   ├── heartmula.py             HeartMuLa 3B
│   ├── diffrhythm.py            DiffRhythm
│   ├── yue.py                   YuE
│   ├── musicgen.py              MusicGen (AudioCraft)
│   ├── riffusion.py             Riffusion
│   └── stable_audio.py          Stable Audio Open
│
├── tests/                       308 tests (all mocked, no GPU needed)
│   ├── conftest.py              Shared fixtures
│   ├── test_worker_registry.py  Port allocation, round-robin, thread safety
│   ├── test_worker_manager.py   Spawn, kill, scale, GPU detection
│   ├── test_install_manager.py  Job lifecycle, cancel, step tracking
│   ├── test_music_worker.py     Dispatch tables, all 4 endpoints
│   ├── test_audio.py            All 8 pipeline stages, format conversion
│   ├── test_api_server.py       All 35 gateway endpoints
│   ├── test_param_wiring.py     GUI → worker → API param consistency
│   ├── test_output_manager.py   Save, load, delete, CLAP score persistence
│   └── test_clap_scorer.py      Health, load, score, unload
│
├── python_embedded/             [Auto] Embedded Python 3.10.11
├── git_portable/                [Auto] Portable Git 2.47.1
├── ffmpeg/                      [Auto] Portable FFmpeg
├── espeak-ng/                   [Auto] espeak-ng for DiffRhythm phonemizer
├── venvs/                       [Auto] 8 isolated virtual environments
├── models/                      [Auto] HuggingFace model cache
└── output/                      [Auto] Generated audio + logs
    ├── generations/             Persistent output library (audio + JSON metadata)
    └── logs/                    Worker log files
```

Auto-downloaded directories are created by `install.bat` and excluded from git.

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **OS** | Windows 10 64-bit | Windows 11 |
| **GPU** | NVIDIA with CUDA (any), or CPU-only | 12 GB+ VRAM |
| **RAM** | 16 GB | 32 GB |
| **Disk** | ~2 GB (base install) | ~60 GB (all envs + models) |
| **Internet** | Required for setup | Required for setup |

CUDA versions used: **12.1** (ACE-Step, HeartMuLa, DiffRhythm), **12.4** (YuE, MusicGen, Riffusion, Stable Audio). CPU fallback is automatic on GPU-less systems.

### VRAM per Model (Approximate)

| Model | VRAM | Notes |
|-------|------|-------|
| ACE-Step v1.5 | <4 GB | Lightweight DiT + optional LLM (0.6B/1.7B/4B auto-selected) |
| Riffusion | 6-8 GB | Stable Diffusion backbone |
| Stable Audio Open | ~8 GB | Latent diffusion, max ~47s output |
| DiffRhythm | 8 GB | Requires espeak-ng (auto-installed) |
| MusicGen | 8-16 GB | Large variant (~3 GB weights download) |
| ACE-Step v1 | ~8 GB | Original pipeline |
| HeartMuLa 3B | ~16 GB | 3 weight repos (~6 GB download) |
| YuE | 24 GB+ | Two-stage codec, segment-by-segment generation |

### External Dependencies (Auto-Installed)

| Tool | Purpose | Source |
|------|---------|--------|
| [espeak-ng](https://github.com/espeak-ng/espeak-ng) | Phoneme synthesis for DiffRhythm | Auto-extracted from bin.zip |
| [FFmpeg](https://ffmpeg.org/) | Audio format conversion | Auto-downloaded by install.bat |

---

## GUI: Music Manager

The Tkinter GUI (`music_manager.py`) provides five tabs:

**Setup** -- Install, remove, and manage 8 virtual environments. Expandable cards with real-time progress bars, step tracking ("Step 3/7: Installing requirements (42%)"), inline log streaming, and cancel support. Install All button for sequential bulk installation.

**API Server** -- Start/stop the gateway, configure host/port, view worker status with right-click context menus (kill, view log, copy port).

**Testing** -- Select a model, choose a preset or configure all parameters manually, adjust post-processing settings (13 controls), and generate music. Model-specific parameter panels rebuild dynamically per model.

**Outputs** -- Browse all saved generations in a sortable table (model, prompt, duration, format, CLAP score, date). Start/stop CLAP scoring service. Right-click to play, save as, score with CLAP, open folder, copy prompt, or delete. Multi-select for batch download.

**Log** -- Real-time gateway log output.

---

## Model-Specific Notes

### ACE-Step v1.5

Uses a unified `inference.generate_music()` with DiT + 5Hz LLM (Qwen3-based). The LLM is auto-selected based on available VRAM: 0.6B (≤12 GB), 1.7B (12-24 GB), 4B (24 GB+). Supports 51 vocal languages, chain-of-thought generation, and `keyscale`/`timesignature` metadata. If `vllm` is unavailable (e.g., Windows embedded Python), falls back to PyTorch backend. 32 configurable parameters across 5 sections.

### DiffRhythm

Requires espeak-ng for phoneme processing (auto-installed during environment setup). Uses `os.chdir()` to the repo path for relative path resolution. Exposes `steps`, `cfg_strength`, and `sway_sampling_coef` parameters.

### YuE

Two-stage generation: Stage 1 generates coarse tokens, Stage 2 refines, then XCodec decodes to 16 kHz audio. Requires 24 GB+ VRAM. Uses custom `_MMSentencePieceTokenizer` and `CodecManipulator`. Worker uses `eval()` for codec class loading from the repo's own YAML config (mirrors upstream code).

### MusicGen

Uses Meta's AudioCraft library. The install config uses `--only-binary :all: av` before audiocraft to avoid Python.h build errors on embedded Python. Supports melody conditioning via base64 audio input.

### Riffusion

Generates spectrograms via Stable Diffusion, then converts to audio using librosa Griffin-Lim reconstruction. Long outputs are stitched with 100ms crossfade segments.

### Stable Audio Open

Stability AI's latent diffusion model. Maximum output duration is ~47 seconds (model limitation). Outputs at 44.1 kHz stereo. All 8 presets include negative prompts for better quality.

---

## Adding a New Model

1. Create an install config in `install_configs/new_model.py` extending `InstallConfig`
2. Register it in `install_configs/__init__.py`
3. Add model entry to `MODEL_SETUP` in `install_manager.py`
4. Add loader `_load_newmodel()` and inferencer `_infer_newmodel()` in `music_worker.py`
5. Add audio profile in `audio_profiles.py`
6. Add parameter definitions, presets, and display metadata in `model_params.py`
7. Add venv mapping in `worker_manager.py` (`MODEL_VENV_MAP`)

---

## References

### Music Models

| Model | Repository | HuggingFace / Paper |
|-------|-----------|---------------------|
| ACE-Step v1.5 | [ace-step/ACE-Step](https://github.com/ace-step/ACE-Step) | [ACE-Step](https://huggingface.co/ACE-Step) |
| ACE-Step v1 | [ace-step/ACE-Step](https://github.com/ace-step/ACE-Step) | [ACE-Step](https://huggingface.co/ACE-Step) |
| HeartMuLa 3B | [HeartMuLa](https://huggingface.co/HeartMuLa) | [HeartMuLa/HeartMuLaGen](https://huggingface.co/HeartMuLa/HeartMuLaGen) |
| DiffRhythm | [ASLP-lab/DiffRhythm](https://github.com/ASLP-lab/DiffRhythm) | [DiffRhythm](https://huggingface.co/ASLP-lab/DiffRhythm) |
| YuE | [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE) | [m-a-p/YuE-s1-7B-anneal](https://huggingface.co/m-a-p/YuE-s1-7B-anneal) |
| MusicGen | [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft) | [facebook/musicgen-large](https://huggingface.co/facebook/musicgen-large) |
| Riffusion | [riffusion/riffusion](https://github.com/riffusion/riffusion) | [riffusion/riffusion-model-v1](https://huggingface.co/riffusion/riffusion-model-v1) |
| Stable Audio Open | [Stability-AI/stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) | [stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) |

### Core Libraries

| Library | Purpose |
|---------|---------|
| [FastAPI](https://fastapi.tiangolo.com/) | Gateway and worker web framework |
| [LAION CLAP](https://github.com/LAION-AI/CLAP) | Audio-text similarity scoring |
| [HuggingFace Hub](https://github.com/huggingface/huggingface_hub) | Model downloads |
| [FFmpeg](https://ffmpeg.org/) | Audio format conversion |
| [espeak-ng](https://github.com/espeak-ng/espeak-ng) | Phoneme synthesis (DiffRhythm) |

### Audio Processing Libraries

| Library | Stage |
|---------|-------|
| [noisereduce](https://github.com/timsainb/noisereduce) | Denoise (spectral gating) |
| [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) | LUFS loudness normalization |
| [pyrubberband](https://github.com/bmcfee/pyrubberband) | Tempo adjustment (optional) |
| [pydub](https://github.com/jiaaro/pydub) | Silence detection and trimming |
| [scipy](https://scipy.org/) | Highpass filter, multiband compression, EQ |
| [soundfile](https://github.com/bastibe/python-soundfile) | WAV I/O |
| [numpy](https://numpy.org/) | Audio array processing |

---

## Troubleshooting

**"Worker failed to start" / model won't load**
- Ensure the model's virtual environment is installed (GUI > Setup tab)
- Check the worker log in `output/logs/` for the specific error
- Verify you have enough VRAM (see VRAM table above)

**DiffRhythm fails with phonemizer/espeak error**
- espeak-ng should be auto-installed during environment setup. If missing, re-run the DiffRhythm install from the GUI.

**CUDA out of memory**
- Kill workers for other models first: `curl -X DELETE http://127.0.0.1:9150/api/workers/{id}`
- Use a lighter model (ACE-Step v1.5 <4 GB vs YuE 24 GB+)
- Or pin models to different GPUs: `{"device": "cuda:1"}`

**Port already in use**
- Change the gateway port: `launcher.bat api --port 9200`
- Or kill whatever is using port 9150: `netstat -ano | findstr :9150`
- Note: port 9100 is reserved by Windows Print Spooler (HP JetDirect), which is why the default is 9150

**No GPU detected / running on CPU**
- The gateway auto-detects GPUs via nvidia-smi. If no NVIDIA GPU is found, workers default to CPU.
- CPU inference is significantly slower but works for all models except YuE (requires too much memory).

**install.bat fails at requirements**
- Check your internet connection. The installer verifies critical imports after installation and will report which packages failed.
- Try running `install.bat` again -- it skips already-completed steps.

---

## Version History

### v1.0.0 (2026-02-12)

- Gateway + Worker architecture with subprocess isolation
- 8 music generation models integrated and verified
- Multi-GPU support with round-robin load balancing
- Dynamic worker management (spawn, kill, scale, failover)
- Auto-spawn workers on first request
- Health checks with dead worker cleanup
- 8-stage audio mastering pipeline with 13 configurable parameters
- CLAP audio-text similarity scoring via dedicated micro-service
- Persistent output library with metadata and batch export
- Headless install manager with job tracking, cancel support, step progress
- 35-endpoint REST API for full programmatic control
- Tkinter GUI with 5 tabs (Setup, API Server, Testing, Outputs, Log)
- Right-click context menus on workers and outputs
- 308 unit tests (all mocked, no GPU required)
- Embedded Python 3.10, portable Git, FFmpeg, espeak-ng

---

## License

This project is a gateway/orchestrator. Each integrated music model has its own license:

| Model | License |
|-------|---------|
| ACE-Step v1.5, v1 | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| HeartMuLa 3B | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| DiffRhythm | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| YuE | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| MusicGen (AudioCraft) | [MIT](https://opensource.org/licenses/MIT) |
| Riffusion | [CreativeML Open RAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license) |
| Stable Audio Open | [Stability AI Community](https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/LICENSE) |

The gateway/orchestrator code is licensed under the [MIT License](LICENSE).

---

## Credits

Built with [Claude Opus 4.6](https://www.anthropic.com/claude) as a pair-programming partner.
