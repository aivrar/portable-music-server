"""Tests for worker_manager.py — WorkerManager."""

import asyncio
import subprocess
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from worker_registry import WorkerRegistry, WorkerInfo
from worker_manager import WorkerManager, MODEL_VENV_MAP


# ---------------------------------------------------------------------------
# MODEL_VENV_MAP
# ---------------------------------------------------------------------------
class TestModelVenvMap:
    def test_all_8_models_mapped(self):
        expected = {
            "ace_step_v15", "ace_step_v1", "heartmula", "diffrhythm",
            "yue", "musicgen", "riffusion", "stable_audio",
        }
        assert set(MODEL_VENV_MAP.keys()) == expected

    def test_env_names_match_install_configs(self):
        from install_manager import MODEL_SETUP
        for model_id, venv_name in MODEL_VENV_MAP.items():
            assert MODEL_SETUP[model_id]["env"] == venv_name


# ---------------------------------------------------------------------------
# WorkerManager init
# ---------------------------------------------------------------------------
class TestWorkerManagerInit:
    def test_creates_with_default_registry(self):
        with patch("worker_manager.WORKER_LOG_DIR", MagicMock()):
            mgr = WorkerManager()
            assert mgr.registry is not None

    def test_uses_provided_registry(self):
        reg = WorkerRegistry(port_min=9101, port_max=9110)
        with patch("worker_manager.WORKER_LOG_DIR", MagicMock()):
            mgr = WorkerManager(registry=reg)
            assert mgr.registry is reg


# ---------------------------------------------------------------------------
# spawn_worker
# ---------------------------------------------------------------------------
class TestSpawnWorker:
    @pytest.mark.asyncio
    async def test_spawn_allocates_port_and_registers(self):
        reg = WorkerRegistry(port_min=9101, port_max=9110)
        with patch("worker_manager.WORKER_LOG_DIR", MagicMock()):
            mgr = WorkerManager(registry=reg)

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None

        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("builtins.open", MagicMock()), \
             patch.object(mgr, '_wait_for_healthy', new_callable=AsyncMock), \
             patch.object(mgr, '_load_model', new_callable=AsyncMock):
            worker = await mgr.spawn_worker("musicgen", "cuda:0")

        assert worker.model == "musicgen"
        assert worker.device == "cuda:0"
        assert 9101 <= worker.port <= 9110
        assert reg.get(worker.worker_id) is worker

    @pytest.mark.asyncio
    async def test_spawn_uses_default_device(self):
        reg = WorkerRegistry(port_min=9101, port_max=9110)
        with patch("worker_manager.WORKER_LOG_DIR", MagicMock()):
            mgr = WorkerManager(registry=reg)

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None

        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("builtins.open", MagicMock()), \
             patch.object(mgr, '_wait_for_healthy', new_callable=AsyncMock), \
             patch.object(mgr, '_load_model', new_callable=AsyncMock):
            worker = await mgr.spawn_worker("musicgen")

        assert worker.device is not None  # Should use default

    @pytest.mark.asyncio
    async def test_spawn_failure_releases_port(self):
        reg = WorkerRegistry(port_min=9101, port_max=9101)
        with patch("worker_manager.WORKER_LOG_DIR", MagicMock()):
            mgr = WorkerManager(registry=reg)

        with patch("subprocess.Popen", side_effect=OSError("no such file")), \
             patch("builtins.open", MagicMock()):
            with pytest.raises(RuntimeError, match="Failed to launch"):
                await mgr.spawn_worker("musicgen")

        # Port should be released back
        port = reg.allocate_port()
        assert port == 9101

    @pytest.mark.asyncio
    async def test_spawn_health_failure_cleans_up(self):
        reg = WorkerRegistry(port_min=9101, port_max=9101)
        with patch("worker_manager.WORKER_LOG_DIR", MagicMock()):
            mgr = WorkerManager(registry=reg)

        mock_proc = MagicMock()
        mock_proc.pid = 12345

        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("builtins.open", MagicMock()), \
             patch.object(mgr, '_wait_for_healthy',
                          new_callable=AsyncMock,
                          side_effect=RuntimeError("timeout")), \
             patch.object(mgr, '_force_kill', new_callable=AsyncMock):
            with pytest.raises(RuntimeError, match="timeout"):
                await mgr.spawn_worker("musicgen")

        # Port and registry should be cleaned up
        port = reg.allocate_port()
        assert port == 9101
        assert reg.worker_count() == 0


# ---------------------------------------------------------------------------
# kill_worker
# ---------------------------------------------------------------------------
class TestKillWorker:
    @pytest.mark.asyncio
    async def test_kill_existing_worker(self):
        reg = WorkerRegistry(port_min=9101, port_max=9110)
        with patch("worker_manager.WORKER_LOG_DIR", MagicMock()):
            mgr = WorkerManager(registry=reg)

        mock_proc = MagicMock()
        w = WorkerInfo(worker_id="mg-1", model="musicgen", port=9101,
                       device="cuda:0", process=mock_proc, status="ready")
        reg.register(w)

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(post=AsyncMock())
            )
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await mgr.kill_worker("mg-1")

        assert result is True
        assert reg.get("mg-1") is None

    @pytest.mark.asyncio
    async def test_kill_nonexistent_worker(self):
        reg = WorkerRegistry()
        with patch("worker_manager.WORKER_LOG_DIR", MagicMock()):
            mgr = WorkerManager(registry=reg)
        result = await mgr.kill_worker("nonexistent")
        assert result is False


# ---------------------------------------------------------------------------
# scale_model
# ---------------------------------------------------------------------------
class TestScaleModel:
    @pytest.mark.asyncio
    async def test_scale_up(self):
        reg = WorkerRegistry(port_min=9101, port_max=9110)
        with patch("worker_manager.WORKER_LOG_DIR", MagicMock()):
            mgr = WorkerManager(registry=reg)

        spawn_count = 0

        async def mock_spawn(model, device=None):
            nonlocal spawn_count
            spawn_count += 1
            w = WorkerInfo(
                worker_id=f"mg-{spawn_count}", model=model,
                port=9150 + spawn_count, device=device or "cuda:0",
                status="ready",
            )
            reg.register(w)
            return w

        with patch.object(mgr, 'spawn_worker', side_effect=mock_spawn):
            workers = await mgr.scale_model("musicgen", 3, "cuda:0")

        assert spawn_count == 3
        assert len(workers) == 3

    @pytest.mark.asyncio
    async def test_scale_down(self):
        reg = WorkerRegistry(port_min=9101, port_max=9110)
        with patch("worker_manager.WORKER_LOG_DIR", MagicMock()):
            mgr = WorkerManager(registry=reg)

        # Register 3 workers
        for i in range(1, 4):
            w = WorkerInfo(
                worker_id=f"mg-{i}", model="musicgen",
                port=9150 + i, device="cuda:0", status="ready",
            )
            reg.register(w)

        killed = []

        async def mock_kill(worker_id):
            killed.append(worker_id)
            reg.unregister(worker_id)
            return True

        with patch.object(mgr, 'kill_worker', side_effect=mock_kill):
            workers = await mgr.scale_model("musicgen", 1, "cuda:0")

        assert len(killed) == 2


# ---------------------------------------------------------------------------
# detect_devices
# ---------------------------------------------------------------------------
class TestDetectDevices:
    def test_detect_with_nvidia_smi(self):
        reg = WorkerRegistry()
        with patch("worker_manager.WORKER_LOG_DIR", MagicMock()):
            mgr = WorkerManager(registry=reg)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "0, NVIDIA RTX 4090, 24576, 20000\n"

        with patch("subprocess.run", return_value=mock_result):
            devices = mgr.detect_devices()

        # Should have CUDA device + CPU
        assert len(devices) == 2
        assert devices[0]["id"] == "cuda:0"
        assert devices[0]["name"] == "NVIDIA RTX 4090"
        assert devices[1]["id"] == "cpu"

    def test_detect_without_nvidia_smi(self):
        reg = WorkerRegistry()
        with patch("worker_manager.WORKER_LOG_DIR", MagicMock()):
            mgr = WorkerManager(registry=reg)

        with patch("subprocess.run", side_effect=FileNotFoundError):
            devices = mgr.detect_devices()

        # Should still have CPU
        assert len(devices) == 1
        assert devices[0]["id"] == "cpu"
