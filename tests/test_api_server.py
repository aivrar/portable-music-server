"""Tests for music_api_server.py — FastAPI endpoint tests."""

import base64
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from music_api_server import app, registry, worker_manager, install_manager, output_manager
import music_api_server
from worker_registry import WorkerInfo


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure registry is clean before each test."""
    # Remove all workers
    for w in registry.all_workers():
        registry.unregister(w.worker_id)
    yield
    for w in registry.all_workers():
        registry.unregister(w.worker_id)


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Health / discovery
# ---------------------------------------------------------------------------
class TestHealth:
    def test_root_health(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "loaded_models" in data

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_models_list(self, client):
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        model_ids = [m["id"] for m in data["models"]]
        assert "musicgen" in model_ids

    def test_models_status(self, client):
        resp = client.get("/api/models/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data


# ---------------------------------------------------------------------------
# Worker management
# ---------------------------------------------------------------------------
class TestWorkerEndpoints:
    def test_list_workers_empty(self, client):
        resp = client.get("/api/workers")
        assert resp.status_code == 200
        assert resp.json()["workers"] == []

    def test_list_workers_with_workers(self, client):
        w = WorkerInfo(worker_id="mg-1", model="musicgen", port=9101,
                       device="cuda:0", status="ready")
        registry.register(w)
        resp = client.get("/api/workers")
        data = resp.json()
        assert len(data["workers"]) == 1
        assert data["workers"][0]["worker_id"] == "mg-1"

    def test_spawn_worker(self, client):
        mock_worker = WorkerInfo(
            worker_id="mg-1", model="musicgen", port=9101,
            device="cuda:0", status="ready",
        )
        mock_status = {"env_installed": True, "weights_installed": True, "status": "ready"}
        with patch.object(install_manager, 'get_model_status', return_value=mock_status), \
             patch.object(worker_manager, 'spawn_worker',
                          new_callable=AsyncMock, return_value=mock_worker):
            resp = client.post("/api/workers/spawn",
                               json={"model": "musicgen"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "spawned"
        assert data["model"] == "musicgen"

    def test_spawn_worker_not_installed(self, client):
        mock_status = {"env_installed": False, "weights_installed": False, "status": "not_installed"}
        with patch.object(install_manager, 'get_model_status', return_value=mock_status):
            resp = client.post("/api/workers/spawn",
                               json={"model": "musicgen"})

        assert resp.status_code == 422
        assert "not installed" in resp.json()["detail"]

    def test_delete_worker_found(self, client):
        w = WorkerInfo(worker_id="mg-1", model="musicgen", port=9101,
                       device="cuda:0", status="ready")
        registry.register(w)

        with patch.object(worker_manager, 'kill_worker',
                          new_callable=AsyncMock, return_value=True):
            resp = client.delete("/api/workers/mg-1")

        assert resp.status_code == 200
        assert resp.json()["status"] == "killed"

    def test_delete_worker_not_found(self, client):
        with patch.object(worker_manager, 'kill_worker',
                          new_callable=AsyncMock, return_value=False):
            resp = client.delete("/api/workers/nonexistent")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Model load/unload
# ---------------------------------------------------------------------------
class TestModelLoadUnload:
    def test_load_model_already_loaded(self, client):
        w = WorkerInfo(worker_id="mg-1", model="musicgen", port=9101,
                       device="cuda:0", status="ready")
        registry.register(w)

        mock_status = {"env_installed": True, "weights_installed": True, "status": "ready"}
        with patch.object(install_manager, 'get_model_status', return_value=mock_status):
            resp = client.post("/api/models/musicgen/load")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "already_loaded"

    def test_load_model_spawns_worker(self, client):
        mock_worker = WorkerInfo(
            worker_id="mg-1", model="musicgen", port=9101,
            device="cuda:0", status="ready",
        )
        mock_status = {"env_installed": True, "weights_installed": True, "status": "ready"}
        with patch.object(install_manager, 'get_model_status', return_value=mock_status), \
             patch.object(worker_manager, 'spawn_worker',
                          new_callable=AsyncMock, return_value=mock_worker):
            resp = client.post("/api/models/musicgen/load")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "loaded"

    def test_load_model_not_installed(self, client):
        mock_status = {"env_installed": False, "weights_installed": False, "status": "not_installed"}
        with patch.object(install_manager, 'get_model_status', return_value=mock_status):
            resp = client.post("/api/models/musicgen/load")
        assert resp.status_code == 422
        assert "not installed" in resp.json()["detail"]

    def test_unload_model_not_loaded(self, client):
        resp = client.post("/api/models/musicgen/unload")
        data = resp.json()
        assert data["status"] == "not_loaded"

    def test_unload_model_kills_workers(self, client):
        w = WorkerInfo(worker_id="mg-1", model="musicgen", port=9101,
                       device="cuda:0", status="ready")
        registry.register(w)

        with patch.object(worker_manager, 'kill_worker',
                          new_callable=AsyncMock, return_value=True):
            resp = client.post("/api/models/musicgen/unload")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "unloaded"
        assert data["workers_killed"] == 1


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------
class TestDevices:
    def test_list_devices(self, client):
        with patch.object(worker_manager, 'detect_devices',
                          return_value=[{"id": "cpu", "name": "CPU",
                                         "vram_total_mb": 0, "vram_free_mb": 0,
                                         "workers": []}]):
            resp = client.get("/api/devices")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["devices"]) >= 1


# ---------------------------------------------------------------------------
# Install management endpoints
# ---------------------------------------------------------------------------
class TestInstallEndpoints:
    def test_all_install_status(self, client):
        resp = client.get("/api/install/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "musicgen" in data["models"]

    def test_single_model_install_status(self, client):
        resp = client.get("/api/install/status/musicgen")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "musicgen"

    def test_unknown_model_status_404(self, client):
        resp = client.get("/api/install/status/nonexistent")
        assert resp.status_code == 404

    def test_install_model(self, client):
        with patch.object(install_manager, 'start_install', return_value="abc123"):
            resp = client.post("/api/install/musicgen")

        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "abc123"
        assert data["status"] == "started"

    def test_install_unknown_model_404(self, client):
        with patch.object(install_manager, 'start_install',
                          side_effect=ValueError("Unknown model")):
            resp = client.post("/api/install/nonexistent")

        assert resp.status_code == 404

    def test_install_conflict_409(self, client):
        with patch.object(install_manager, 'start_install',
                          side_effect=RuntimeError("already active")):
            resp = client.post("/api/install/musicgen")

        assert resp.status_code == 409

    def test_uninstall_model(self, client):
        with patch.object(install_manager, 'start_uninstall', return_value="def456"):
            resp = client.delete("/api/install/musicgen")

        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "def456"

    def test_list_install_jobs(self, client):
        with patch.object(install_manager, 'list_jobs', return_value=[]):
            resp = client.get("/api/install/jobs")

        assert resp.status_code == 200
        assert resp.json()["jobs"] == []

    def test_get_install_job(self, client):
        mock_job = {
            "job_id": "abc123", "model_id": "musicgen",
            "operation": "install", "status": "running",
            "progress_pct": 50.0, "current_step": "Installing deps",
            "error": "", "created_at": 1000.0, "finished_at": None,
            "log_line_count": 10,
        }
        with patch.object(install_manager, 'get_job', return_value=mock_job):
            resp = client.get("/api/install/jobs/abc123")

        assert resp.status_code == 200
        data = resp.json()
        assert data["progress_pct"] == 50.0

    def test_get_install_job_not_found(self, client):
        with patch.object(install_manager, 'get_job', return_value=None):
            resp = client.get("/api/install/jobs/nonexistent")

        assert resp.status_code == 404

    def test_get_install_job_logs(self, client):
        mock_logs = {
            "job_id": "abc123", "offset": 0, "total_lines": 3,
            "lines": ["line1", "line2", "line3"], "status": "running",
        }
        with patch.object(install_manager, 'get_job_logs', return_value=mock_logs):
            resp = client.get("/api/install/jobs/abc123/logs")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["lines"]) == 3

    def test_get_install_job_logs_with_offset(self, client):
        mock_logs = {
            "job_id": "abc123", "offset": 2, "total_lines": 5,
            "lines": ["line3", "line4", "line5"], "status": "running",
        }
        with patch.object(install_manager, 'get_job_logs', return_value=mock_logs):
            resp = client.get("/api/install/jobs/abc123/logs?offset=2")

        assert resp.status_code == 200
        data = resp.json()
        assert data["offset"] == 2

    def test_get_install_job_logs_not_found(self, client):
        with patch.object(install_manager, 'get_job_logs', return_value=None):
            resp = client.get("/api/install/jobs/nonexistent/logs")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Download endpoint
# ---------------------------------------------------------------------------
class TestDownloadEndpoint:
    def test_download_unknown_model(self, client):
        with patch.object(install_manager, 'start_download',
                          side_effect=ValueError("Unknown model")):
            resp = client.post("/api/install/nonexistent/download")
        assert resp.status_code == 404

    def test_download_no_weights_model(self, client):
        with patch.object(install_manager, 'start_download',
                          side_effect=ValueError("no downloadable weights")):
            resp = client.post("/api/install/ace_step_v15/download")
        assert resp.status_code == 404

    def test_download_valid_model(self, client):
        with patch.object(install_manager, 'start_download', return_value="abc123"):
            resp = client.post("/api/install/musicgen/download")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "abc123"
        assert data["status"] == "started"

    def test_download_conflict(self, client):
        with patch.object(install_manager, 'start_download',
                          side_effect=RuntimeError("already active")):
            resp = client.post("/api/install/musicgen/download")
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Scale endpoint
# ---------------------------------------------------------------------------
class TestScaleEndpoint:
    def test_scale_model(self, client):
        mock_workers = [
            WorkerInfo(worker_id="mg-1", model="musicgen", port=9101,
                       device="cuda:0", status="ready"),
        ]
        with patch.object(worker_manager, 'scale_model',
                          new_callable=AsyncMock, return_value=mock_workers):
            resp = client.post("/api/models/musicgen/scale",
                               json={"count": 1})

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "scaled"
        assert data["count"] == 1


# ---------------------------------------------------------------------------
# CLAP endpoints
# ---------------------------------------------------------------------------
class TestClapEndpoints:
    def test_clap_status_not_running(self, client):
        resp = client.get("/api/clap/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is False
        assert data["port"] is None

    def test_clap_stop_when_not_running(self, client):
        resp = client.post("/api/clap/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "not_running"

    def test_clap_score_503_when_not_running(self, client):
        resp = client.post("/api/clap/score", json={
            "audio_base64": "dGVzdA==",
            "prompt": "test",
        })
        assert resp.status_code == 503

    def test_clap_start_no_venv(self, client):
        """CLAP start should fail if stable_audio_env isn't installed."""
        with patch("music_api_server.VENVS_DIR", Path("/nonexistent")):
            resp = client.post("/api/clap/start")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Output endpoints
# ---------------------------------------------------------------------------
class TestOutputEndpoints:
    def test_list_outputs_empty(self, client):
        resp = client.get("/api/outputs")
        assert resp.status_code == 200
        data = resp.json()
        assert "outputs" in data

    def test_delete_output_404(self, client):
        resp = client.delete("/api/outputs/nonexistent")
        assert resp.status_code == 404

    def test_get_output_404(self, client):
        resp = client.get("/api/outputs/nonexistent")
        assert resp.status_code == 404

    def test_get_output_audio_404(self, client):
        resp = client.get("/api/outputs/nonexistent/audio")
        assert resp.status_code == 404

    def test_get_output_audio_serves_file(self, client, tmp_path):
        """Audio endpoint should serve the actual file."""
        audio_data = b"RIFF" + b"\x00" * 100  # fake WAV
        entry = output_manager.save_output(
            audio_bytes=audio_data, model="musicgen", prompt="test",
            fmt="wav", sample_rate=44100,
        )
        resp = client.get(f"/api/outputs/{entry.id}/audio")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert len(resp.content) == len(audio_data)
        # Cleanup
        output_manager.delete_entry(entry.id)

    def test_batch_delete_outputs(self, client):
        """Batch delete should delete multiple outputs."""
        e1 = output_manager.save_output(
            audio_bytes=b"\x00" * 50, model="musicgen", prompt="a", fmt="wav",
        )
        e2 = output_manager.save_output(
            audio_bytes=b"\x00" * 50, model="musicgen", prompt="b", fmt="wav",
        )
        resp = client.request(
            "DELETE", "/api/outputs/batch",
            json={"ids": [e1.id, e2.id, "nonexistent"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert e1.id in data["deleted"]
        assert e2.id in data["deleted"]
        assert "nonexistent" in data["not_found"]


# ---------------------------------------------------------------------------
# Model params / presets / display
# ---------------------------------------------------------------------------
class TestModelParamsEndpoints:
    def test_get_model_params(self, client):
        resp = client.get("/api/models/musicgen/params")
        assert resp.status_code == 200
        data = resp.json()
        assert data["display"] == "MusicGen"
        assert "controls" in data
        ctrl_ids = [c["id"] for c in data["controls"]]
        assert "prompt" in ctrl_ids

    def test_get_model_params_unknown(self, client):
        resp = client.get("/api/models/nonexistent/params")
        assert resp.status_code == 404

    def test_get_model_presets(self, client):
        resp = client.get("/api/models/stable_audio/presets")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "stable_audio"
        assert len(data["presets"]) >= 1
        assert "name" in data["presets"][0]
        assert "params" in data["presets"][0]

    def test_get_model_presets_unknown(self, client):
        resp = client.get("/api/models/nonexistent/presets")
        assert resp.status_code == 404

    def test_get_model_display(self, client):
        resp = client.get("/api/models/riffusion/display")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "riffusion"
        assert "desc" in data
        assert "vram" in data

    def test_get_model_display_unknown(self, client):
        resp = client.get("/api/models/nonexistent/display")
        assert resp.status_code == 404

    def test_all_models_have_params_and_presets(self, client):
        """Every model in params should also be accessible via API."""
        from model_params import MODEL_PARAMS
        for model_id in MODEL_PARAMS:
            resp = client.get(f"/api/models/{model_id}/params")
            assert resp.status_code == 200
            resp = client.get(f"/api/models/{model_id}/presets")
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Install cancel
# ---------------------------------------------------------------------------
class TestInstallCancel:
    def test_cancel_install_success(self, client):
        with patch.object(install_manager, 'cancel_install', return_value=True):
            resp = client.post("/api/install/musicgen/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

    def test_cancel_install_no_active_job(self, client):
        with patch.object(install_manager, 'cancel_install', return_value=False):
            resp = client.post("/api/install/musicgen/cancel")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Worker logs
# ---------------------------------------------------------------------------
class TestWorkerLogs:
    def test_get_worker_logs_not_found(self, client):
        resp = client.get("/api/workers/nonexistent/logs")
        assert resp.status_code == 404

    def test_get_worker_logs_no_file(self, client, tmp_path):
        """Worker exists but no log file yet."""
        w = WorkerInfo(worker_id="mg-1", model="musicgen", port=9101,
                       device="cuda:0", status="ready")
        registry.register(w)
        with patch("music_api_server.WORKER_LOG_DIR", tmp_path):
            resp = client.get("/api/workers/mg-1/logs")
        assert resp.status_code == 200
        assert resp.json()["lines"] == []

    def test_get_worker_logs_with_content(self, client, tmp_path):
        """Worker log file exists with content."""
        w = WorkerInfo(worker_id="mg-1", model="musicgen", port=9101,
                       device="cuda:0", status="ready")
        registry.register(w)
        log_file = tmp_path / "worker_musicgen_9101.log"
        log_file.write_text("line1\nline2\nline3\n")
        with patch("music_api_server.WORKER_LOG_DIR", tmp_path):
            resp = client.get("/api/workers/mg-1/logs?tail=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_lines"] == 3
        assert len(data["lines"]) == 2
        assert data["lines"] == ["line2", "line3"]


# ---------------------------------------------------------------------------
# Kill all workers
# ---------------------------------------------------------------------------
class TestKillAllWorkers:
    def test_kill_all_workers(self, client):
        with patch.object(worker_manager, 'kill_all_workers',
                          new_callable=AsyncMock, return_value=3):
            resp = client.post("/api/workers/kill-all")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "killed"
        assert data["count"] == 3

    def test_kill_all_workers_none(self, client):
        with patch.object(worker_manager, 'kill_all_workers',
                          new_callable=AsyncMock, return_value=0):
            resp = client.post("/api/workers/kill-all")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


# ---------------------------------------------------------------------------
# Shutdown endpoint
# ---------------------------------------------------------------------------
class TestShutdown:
    def test_shutdown_returns_response(self, client):
        """Shutdown should return a response before exiting."""
        with patch.object(worker_manager, 'kill_all_workers',
                          new_callable=AsyncMock, return_value=0), \
             patch("music_api_server.os.kill"):
            resp = client.post("/api/shutdown")
        assert resp.status_code == 200
        assert resp.json()["status"] == "shutting_down"
