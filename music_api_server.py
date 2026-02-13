"""
Music API Server - Gateway + Worker Architecture

The gateway runs on the configured port (default 9150) and orchestrates:
- Audio post-processing (denoise, EQ, stereo, compress, normalize, peak limit)
- Format conversion (wav/mp3/ogg/flac/m4a)
- Worker lifecycle management (spawn/kill/scale/health)
- Load balancing across multiple worker instances
- Multi-GPU support

Workers are separate subprocesses, each running one music model instance.
The gateway delegates inference to workers via HTTP.

Run with: uvicorn music_api_server:app --host 0.0.0.0 --port 9150
"""

import os
import sys
from pathlib import Path

# Bootstrap: ensure project root is on sys.path for sibling imports
_BASE_DIR = Path(__file__).parent.resolve()
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

import uuid
import time
import base64
import logging
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import subprocess

from config import (
    BASE_DIR, MODELS_DIR, VENVS_DIR, OUTPUT_DIR, TEMP_DIR, FFMPEG_PATH,
    DEFAULT_API_HOST, DEFAULT_API_PORT,
    WORKER_AUTO_SPAWN, WORKER_DEFAULT_DEVICE,
    WORKER_LOG_DIR, GENERATIONS_DIR, CLAP_PORT,
    setup_environment,
)
from audio_profiles import PROFILES
from audio_processing import post_process
from audio_assembler import convert_format
from worker_registry import WorkerRegistry
from worker_manager import WorkerManager
from install_manager import InstallManager
from output_manager import OutputManager
from model_params import MODEL_DISPLAY, MODEL_PARAMS, MODEL_PRESETS

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup_environment()
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Worker infrastructure
# ---------------------------------------------------------------------------
registry = WorkerRegistry()
worker_manager = WorkerManager(registry)
install_manager = InstallManager(
    log_callback=lambda msg: logger.info("[Install] %s", msg)
)
output_manager = OutputManager(GENERATIONS_DIR)

# CLAP scorer subprocess
_clap_process: subprocess.Popen | None = None

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start health check loop on startup, kill all workers on shutdown."""
    worker_manager.start_health_checks()
    logger.info("Music API Gateway started - worker health checks active")
    yield
    logger.info("Gateway shutting down - killing all workers...")
    worker_manager.stop_health_checks()
    await worker_manager.kill_all_workers()
    # Stop CLAP scorer if running
    global _clap_process
    if _clap_process is not None and _clap_process.poll() is None:
        _clap_process.terminate()
        _clap_process = None
    logger.info("All workers stopped")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Music API Gateway",
    description="Gateway server orchestrating music generation workers",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
executor = ThreadPoolExecutor(max_workers=4)


def _ts():
    return time.strftime("%H:%M:%S")


# ============================================================
# Request models
# ============================================================
class MusicGenerationRequest(BaseModel):
    # Common
    prompt: str = ""
    lyrics: Optional[str] = None
    duration: float = 30.0
    output_format: str = "wav"
    seed: int = -1
    # Widely shared
    tags: Optional[str] = None
    instrumental: bool = False
    reference_audio: Optional[str] = None
    temperature: float = 1.0
    cfg_scale: float = 7.0
    # Post-processing
    denoise_strength: float = 0.2
    stereo_width: float = 1.2
    eq_preset: str = "balanced"
    target_lufs: float = -14.0
    highpass_cutoff: Optional[int] = None
    compression_ratio: Optional[float] = None
    clipping: Optional[float] = None
    trim_db: Optional[float] = None
    min_silence_ms: Optional[int] = None
    front_protect_ms: Optional[int] = None
    end_protect_ms: Optional[int] = None
    skip_post_process: bool = False
    # Model-specific (passed through to worker)
    model_params: Optional[dict] = None


class SpawnRequest(BaseModel):
    model: str
    device: Optional[str] = None


class ScaleRequest(BaseModel):
    count: int
    device: Optional[str] = None


# ============================================================
# Health / discovery
# ============================================================
@app.get("/")
@app.get("/health")
async def health():
    workers = registry.all_workers()
    loaded_models = list(set(w.model for w in workers if w.status in ("ready", "busy")))
    return {
        "status": "ok",
        "message": "Music API Gateway running",
        "loaded_models": loaded_models,
        "worker_count": len(workers),
    }


@app.get("/api/models")
async def list_models():
    statuses = install_manager.get_all_model_status()
    models = []
    for model_id, info in statuses.items():
        models.append({
            "id": model_id,
            "name": info["display_name"],
            "env": info["env_name"],
            "install_status": info["status"],
            "env_installed": info["env_installed"],
            "weights_installed": info["weights_installed"],
        })
    return {"models": models}


@app.get("/api/models/status")
async def models_status():
    workers = registry.all_workers()
    status = {}
    for w in workers:
        if w.model not in status:
            status[w.model] = {"workers": [], "loaded": False}
        status[w.model]["workers"].append({
            "worker_id": w.worker_id,
            "device": w.device,
            "status": w.status,
        })
        if w.status in ("ready", "busy"):
            status[w.model]["loaded"] = True
    return {"models": status}


# ============================================================
# GPU / Device discovery
# ============================================================
@app.get("/api/devices")
async def list_devices():
    devices = worker_manager.detect_devices()
    return {"devices": devices}


# ============================================================
# Worker management endpoints
# ============================================================
@app.get("/api/workers")
async def list_workers():
    return {"workers": registry.to_dict_list()}


@app.post("/api/workers/spawn")
async def spawn_worker(req: SpawnRequest):
    # Check model environment is installed before attempting spawn
    try:
        status = install_manager.get_model_status(req.model)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown model '{req.model}'")

    if not status.get("env_installed"):
        raise HTTPException(
            status_code=422,
            detail=f"Model '{req.model}' environment is not installed. "
                   f"Install it first via POST /api/install/{req.model}",
        )

    try:
        worker = await worker_manager.spawn_worker(req.model, req.device)
        return {
            "status": "spawned",
            "worker_id": worker.worker_id,
            "model": worker.model,
            "port": worker.port,
            "device": worker.device,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/workers/{worker_id}")
async def delete_worker(worker_id: str):
    killed = await worker_manager.kill_worker(worker_id)
    if not killed:
        raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' not found")
    return {"status": "killed", "worker_id": worker_id}


@app.post("/api/models/{model}/scale")
async def scale_model(model: str, req: ScaleRequest):
    try:
        workers = await worker_manager.scale_model(model, req.count, req.device)
        return {
            "status": "scaled",
            "model": model,
            "device": req.device or WORKER_DEFAULT_DEVICE,
            "count": len(workers),
            "workers": [w.worker_id for w in workers],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Model load/unload
# ============================================================
@app.post("/api/models/{model}/load")
async def load_model(model: str, device: Optional[str] = None):
    """Load a model by spawning a worker for it."""
    # Verify model environment is installed
    try:
        status = install_manager.get_model_status(model)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model}'")
    if not status.get("env_installed"):
        raise HTTPException(
            status_code=422,
            detail=f"Model '{model}' environment is not installed. "
                   f"Install it first via POST /api/install/{model}",
        )

    existing = registry.get_ready_workers(model)
    if existing:
        return {"status": "already_loaded", "model": model,
                "workers": [w.worker_id for w in existing]}
    try:
        worker = await worker_manager.spawn_worker(model, device)
        return {"status": "loaded", "model": model, "worker_id": worker.worker_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/{model}/unload")
async def unload_model(model: str):
    """Unload a model by killing all its workers."""
    workers = registry.workers_for_model(model)
    if not workers:
        return {"status": "not_loaded", "model": model}

    killed = 0
    for w in workers:
        if await worker_manager.kill_worker(w.worker_id):
            killed += 1
    return {"status": "unloaded", "model": model, "workers_killed": killed}


# ============================================================
# Music generation endpoints
# ============================================================
@app.post("/api/music/{model}")
async def generate_music(model: str, req: MusicGenerationRequest):
    """Generate music using the specified model."""
    return await _handle_generation_request(model, req)


async def _handle_generation_request(model: str, req: MusicGenerationRequest):
    """Shared handler for music generation requests."""
    # Atomically pick a worker and mark it busy (prevents TOCTOU race)
    job_id = str(uuid.uuid4())[:8]
    worker = registry.pick_and_mark_busy(model, job_id)
    if not worker:
        if WORKER_AUTO_SPAWN:
            try:
                spawned = await worker_manager.spawn_worker(model)
                worker = registry.pick_and_mark_busy(model, job_id)
                if not worker:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Worker spawned for '{model}' but not ready yet"
                    )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"No worker available for '{model}' and auto-spawn failed: {e}"
                )
        else:
            raise HTTPException(
                status_code=503,
                detail=f"No worker available for model '{model}'. Spawn one first."
            )

    # Build params for the worker
    infer_params = {
        "prompt": req.prompt,
        "duration": req.duration,
        "seed": req.seed,
        "temperature": req.temperature,
        "cfg_scale": req.cfg_scale,
    }
    if req.lyrics is not None:
        infer_params["lyrics"] = req.lyrics
    if req.tags is not None:
        infer_params["tags"] = req.tags
    if req.instrumental:
        infer_params["instrumental"] = True
    if req.reference_audio:
        infer_params["reference_audio"] = req.reference_audio
    # Merge model-specific params
    if req.model_params:
        infer_params.update(req.model_params)

    # Run inference via worker
    import asyncio
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        executor,
        _run_generation,
        model, worker, infer_params, req,
    )
    if result.get("status") == "error":
        return JSONResponse(result, status_code=502)
    return JSONResponse(result)


def _run_generation(model: str, worker, infer_params: dict,
                    req: MusicGenerationRequest) -> dict:
    """Execute generation pipeline in thread pool."""
    import httpx as sync_httpx

    profile = PROFILES.get(model, PROFILES.get("musicgen", {}))
    output_format = (req.output_format or "wav").lower()

    try:
        # Call worker /infer
        logger.info("[%s %s] Sending inference to worker %s ...",
                    _ts(), model.upper(), worker.worker_id)

        with sync_httpx.Client(timeout=3600.0) as client:
            resp = client.post(
                f"http://127.0.0.1:{worker.port}/infer",
                json={"params": infer_params},
            )

        if resp.status_code != 200:
            detail = resp.text[:300]
            logger.error("[%s %s] Worker inference failed: %s",
                        _ts(), model.upper(), detail)
            return {"status": "error", "detail": detail}

        worker_result = resp.json()
        audio_b64 = worker_result.get("audio_base64")
        sample_rate = worker_result.get("sample_rate", 44100)
        duration_sec = worker_result.get("duration_sec", 0)
        inference_time = worker_result.get("inference_time_sec", 0)

        if not audio_b64:
            return {"status": "error", "detail": "No audio in worker response"}

        # Decode audio
        audio_bytes = base64.b64decode(audio_b64)

        # Post-processing
        if not req.skip_post_process:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False,
                                              dir=str(TEMP_DIR))
            tmp.write(audio_bytes)
            tmp.close()

            try:
                post_process(
                    tmp.name, profile,
                    denoise_strength=req.denoise_strength,
                    stereo_width=req.stereo_width,
                    eq_preset=req.eq_preset,
                    target_lufs=req.target_lufs,
                    highpass_cutoff=req.highpass_cutoff,
                    compression_ratio=req.compression_ratio,
                    clipping=req.clipping,
                    trim_db=req.trim_db,
                    min_silence_ms=req.min_silence_ms,
                    front_protect_ms=req.front_protect_ms,
                    end_protect_ms=req.end_protect_ms,
                )
                with open(tmp.name, "rb") as f:
                    audio_bytes = f.read()
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

        # Format conversion
        if output_format != "wav":
            import tempfile
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False,
                                                  dir=str(TEMP_DIR))
            tmp_wav.write(audio_bytes)
            tmp_wav.close()
            tmp_out = tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False,
                                                  dir=str(TEMP_DIR))
            tmp_out.close()
            try:
                convert_format(tmp_wav.name, tmp_out.name, output_format, FFMPEG_PATH)
                with open(tmp_out.name, "rb") as f:
                    audio_bytes = f.read()
            finally:
                for p in [tmp_wav.name, tmp_out.name]:
                    try:
                        os.unlink(p)
                    except Exception:
                        pass

        # Encode final audio
        final_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Auto-save to persistent output storage
        entry_id = None
        try:
            pp_snapshot = {
                "denoise_strength": req.denoise_strength,
                "stereo_width": req.stereo_width,
                "eq_preset": req.eq_preset,
                "target_lufs": req.target_lufs,
                "skip_post_process": req.skip_post_process,
            }
            entry = output_manager.save_output(
                audio_bytes=audio_bytes,
                model=model,
                prompt=req.prompt,
                negative_prompt=infer_params.get("negative_prompt", ""),
                duration_sec=duration_sec,
                generation_time_sec=inference_time,
                sample_rate=sample_rate,
                fmt=output_format,
                seed=req.seed,
                params=infer_params,
                post_processing=pp_snapshot,
            )
            entry_id = entry.id
        except Exception as e:
            logger.warning("Failed to save output: %s", e)

        return {
            "status": "completed",
            "audio_base64": final_b64,
            "sample_rate": sample_rate,
            "duration_sec": duration_sec,
            "inference_time_sec": inference_time,
            "format": output_format,
            "model": model,
            "entry_id": entry_id,
        }

    except Exception as e:
        logger.error("[%s %s] Generation failed: %s", _ts(), model.upper(), e, exc_info=True)
        return {"status": "error", "detail": str(e)}

    finally:
        registry.mark_ready(worker.worker_id)


# ============================================================
# Installation management endpoints
# ============================================================
@app.get("/api/install/status")
async def all_install_status():
    """Get install status for all models."""
    return {"models": install_manager.get_all_model_status()}


@app.get("/api/install/status/{model_id}")
async def model_install_status(model_id: str):
    """Get install status for a specific model."""
    try:
        return install_manager.get_model_status(model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/install/{model_id}")
async def install_model(model_id: str):
    """Start installing a model (venv + packages + weights).
    Returns immediately with a job_id for progress polling."""
    try:
        job_id = install_manager.start_install(model_id)
        return {"job_id": job_id, "model_id": model_id, "status": "started"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/api/install/{model_id}/download")
async def download_model_weights(model_id: str):
    """Start downloading model weights only (no env install).
    Returns immediately with a job_id for progress polling."""
    try:
        job_id = install_manager.start_download(model_id)
        return {"job_id": job_id, "model_id": model_id, "status": "started"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.delete("/api/install/{model_id}")
async def uninstall_model(model_id: str):
    """Start uninstalling a model (remove venv + weights).
    Returns immediately with a job_id for progress polling."""
    try:
        job_id = install_manager.start_uninstall(model_id)
        return {"job_id": job_id, "model_id": model_id, "status": "started"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/api/install/jobs")
async def list_install_jobs(active_only: bool = False):
    """List all install/uninstall jobs."""
    return {"jobs": install_manager.list_jobs(active_only=active_only)}


@app.get("/api/install/jobs/{job_id}")
async def get_install_job(job_id: str):
    """Get status of a specific install/uninstall job."""
    result = install_manager.get_job(job_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return result


@app.get("/api/install/jobs/{job_id}/logs")
async def get_install_job_logs(job_id: str, offset: int = 0):
    """Get log lines for an install job. Use offset for incremental fetching.

    Polling pattern: client stores the returned total_lines, passes it as
    offset on next request to get only new lines."""
    result = install_manager.get_job_logs(job_id, offset=offset)
    if not result:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return result


# ============================================================
# CLAP scorer management
# ============================================================
@app.get("/api/clap/status")
async def clap_status():
    global _clap_process
    running = _clap_process is not None and _clap_process.poll() is None
    if _clap_process is not None and _clap_process.poll() is not None:
        _clap_process = None
    return {"running": running, "port": CLAP_PORT if running else None}


class ClapStartRequest(BaseModel):
    device: Optional[str] = None


@app.post("/api/clap/start")
async def clap_start(req: ClapStartRequest = ClapStartRequest()):
    global _clap_process
    if _clap_process is not None and _clap_process.poll() is None:
        return {"status": "already_running", "port": CLAP_PORT}

    # Use stable_audio_env Python (laion_clap is installed there)
    venv_python = VENVS_DIR / "stable_audio_env" / "Scripts" / "python.exe"
    if not venv_python.exists():
        raise HTTPException(
            status_code=422,
            detail="stable_audio_env not installed. Install Stable Audio first.",
        )

    # Default to same GPU as first running worker, or cuda:0
    device = req.device or WORKER_DEFAULT_DEVICE
    scorer_script = str(BASE_DIR / "clap_scorer.py")
    cmd = [str(venv_python), scorer_script, "--port", str(CLAP_PORT),
           "--device", device]

    try:
        _clap_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start CLAP: {e}")

    # Wait for health
    import asyncio
    import httpx as _httpx
    deadline = time.time() + 120
    async with _httpx.AsyncClient(timeout=5.0) as client:
        while time.time() < deadline:
            if _clap_process.poll() is not None:
                _clap_process = None
                raise HTTPException(
                    status_code=500, detail="CLAP process exited during startup"
                )
            try:
                resp = await client.get(f"http://127.0.0.1:{CLAP_PORT}/health")
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            await asyncio.sleep(1.0)
        else:
            _clap_process.terminate()
            _clap_process = None
            raise HTTPException(status_code=500, detail="CLAP startup timed out (120s)")

    # Auto-load model
    async with _httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(f"http://127.0.0.1:{CLAP_PORT}/load")
        if resp.status_code != 200:
            _clap_process.terminate()
            _clap_process = None
            raise HTTPException(status_code=500, detail="CLAP model load failed")

    logger.info("CLAP scorer started on port %d", CLAP_PORT)
    return {"status": "started", "port": CLAP_PORT}


@app.post("/api/clap/stop")
async def clap_stop():
    global _clap_process
    if _clap_process is None or _clap_process.poll() is not None:
        _clap_process = None
        return {"status": "not_running"}

    try:
        _clap_process.terminate()
        _clap_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _clap_process.kill()
    _clap_process = None
    logger.info("CLAP scorer stopped")
    return {"status": "stopped"}


class ClapScoreRequest(BaseModel):
    audio_base64: str
    prompt: str


@app.post("/api/clap/score")
async def clap_score(req: ClapScoreRequest):
    if _clap_process is None or _clap_process.poll() is not None:
        raise HTTPException(status_code=503, detail="CLAP scorer not running")

    import httpx as _httpx
    async with _httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(
                f"http://127.0.0.1:{CLAP_PORT}/score",
                json={"audio_base64": req.audio_base64, "prompt": req.prompt},
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return resp.json()
        except _httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="CLAP scoring timed out")


# ============================================================
# Output management endpoints
# ============================================================
class BatchDeleteRequest(BaseModel):
    ids: list[str]


@app.get("/api/outputs")
async def list_outputs():
    entries = output_manager.get_all_entries()
    return {"outputs": [e.to_dict() for e in entries]}


@app.delete("/api/outputs/batch")
async def batch_delete_outputs(req: BatchDeleteRequest):
    """Delete multiple outputs at once."""
    deleted = []
    not_found = []
    for entry_id in req.ids:
        if output_manager.delete_entry(entry_id):
            deleted.append(entry_id)
        else:
            not_found.append(entry_id)
    return {"deleted": deleted, "not_found": not_found}


@app.get("/api/outputs/{entry_id}")
async def get_output(entry_id: str):
    entry = output_manager.get_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Output '{entry_id}' not found")
    return entry.to_dict()


@app.get("/api/outputs/{entry_id}/audio")
async def get_output_audio(entry_id: str):
    """Serve the actual audio file for an output entry."""
    entry = output_manager.get_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Output '{entry_id}' not found")
    audio_path = output_manager.get_audio_path(entry_id)
    if not audio_path or not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found on disk")
    media_types = {
        "wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg",
        "flac": "audio/flac", "m4a": "audio/mp4",
    }
    media_type = media_types.get(entry.format, "application/octet-stream")
    return FileResponse(
        path=str(audio_path),
        media_type=media_type,
        filename=entry.filename,
    )


@app.delete("/api/outputs/{entry_id}")
async def delete_output(entry_id: str):
    deleted = output_manager.delete_entry(entry_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Output '{entry_id}' not found")
    return {"status": "deleted", "id": entry_id}


# ============================================================
# Model params / presets (shared data for GUI + API)
# ============================================================
@app.get("/api/models/{model}/params")
async def get_model_params(model: str):
    """Get parameter definitions for a model (control types, defaults, options)."""
    if model not in MODEL_PARAMS:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model}'")
    return MODEL_PARAMS[model]


@app.get("/api/models/{model}/presets")
async def get_model_presets(model: str):
    """Get available presets for a model."""
    if model not in MODEL_PRESETS:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model}'")
    return {"model": model, "presets": MODEL_PRESETS[model]}


@app.get("/api/models/{model}/display")
async def get_model_display(model: str):
    """Get display metadata for a model (description, VRAM, weights size)."""
    if model not in MODEL_DISPLAY:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model}'")
    return {"model": model, **MODEL_DISPLAY[model]}


# ============================================================
# Install cancel
# ============================================================
@app.post("/api/install/{model_id}/cancel")
async def cancel_install(model_id: str):
    """Cancel an active install/uninstall job for a model."""
    cancelled = install_manager.cancel_install(model_id)
    if not cancelled:
        raise HTTPException(
            status_code=404,
            detail=f"No active install job for '{model_id}'"
        )
    return {"status": "cancelled", "model_id": model_id}


# ============================================================
# Worker logs
# ============================================================
@app.get("/api/workers/{worker_id}/logs")
async def get_worker_logs(worker_id: str, tail: int = 100):
    """Get log output for a worker. Returns last `tail` lines."""
    worker = registry.get(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' not found")

    log_file = WORKER_LOG_DIR / f"worker_{worker.model}_{worker.port}.log"
    if not log_file.exists():
        return {"worker_id": worker_id, "lines": [], "total_lines": 0}

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        total = len(all_lines)
        lines = [line.rstrip("\n") for line in all_lines[-tail:]]
        return {"worker_id": worker_id, "lines": lines, "total_lines": total}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read log: {e}")


# ============================================================
# Kill all workers
# ============================================================
@app.post("/api/workers/kill-all")
async def kill_all_workers():
    """Kill all active workers."""
    count = await worker_manager.kill_all_workers()
    return {"status": "killed", "count": count}


# ============================================================
# Total shutdown
# ============================================================
@app.post("/api/shutdown")
async def shutdown_server():
    """Shutdown the gateway and all workers. The process will exit."""
    import signal

    # Kill all workers first
    await worker_manager.kill_all_workers()

    # Stop CLAP if running
    global _clap_process
    if _clap_process is not None and _clap_process.poll() is None:
        _clap_process.terminate()
        _clap_process = None

    logger.info("Shutdown requested via API - exiting")

    # Schedule the actual exit after responding
    import asyncio
    loop = asyncio.get_running_loop()
    loop.call_later(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM))

    return {"status": "shutting_down"}


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_API_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_API_PORT)
    cli_args = ap.parse_args()

    logger.info("Starting Music API Gateway on %s:%d", cli_args.host, cli_args.port)
    uvicorn.run(app, host=cli_args.host, port=cli_args.port, log_level="info")
