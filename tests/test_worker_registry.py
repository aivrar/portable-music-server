"""Tests for worker_registry.py — WorkerRegistry and WorkerInfo."""

import threading
from unittest.mock import MagicMock

import pytest

from worker_registry import WorkerRegistry, WorkerInfo


# ---------------------------------------------------------------------------
# Port allocation
# ---------------------------------------------------------------------------
class TestPortAllocation:
    def test_allocate_returns_port_in_range(self):
        reg = WorkerRegistry(port_min=9101, port_max=9105)
        port = reg.allocate_port()
        assert 9101 <= port <= 9105

    def test_allocate_all_ports(self):
        reg = WorkerRegistry(port_min=9101, port_max=9103)
        ports = {reg.allocate_port() for _ in range(3)}
        assert ports == {9101, 9102, 9103}

    def test_allocate_exhausted_raises(self):
        reg = WorkerRegistry(port_min=9101, port_max=9101)
        reg.allocate_port()
        with pytest.raises(RuntimeError, match="No available ports"):
            reg.allocate_port()

    def test_release_port_makes_it_available_again(self):
        reg = WorkerRegistry(port_min=9101, port_max=9101)
        port = reg.allocate_port()
        reg.release_port(port)
        port2 = reg.allocate_port()
        assert port2 == port


# ---------------------------------------------------------------------------
# Worker ID generation
# ---------------------------------------------------------------------------
class TestWorkerIdGeneration:
    def test_sequential_ids(self):
        reg = WorkerRegistry()
        assert reg.next_worker_id("musicgen") == "musicgen-1"
        assert reg.next_worker_id("musicgen") == "musicgen-2"

    def test_independent_per_model(self):
        reg = WorkerRegistry()
        assert reg.next_worker_id("musicgen") == "musicgen-1"
        assert reg.next_worker_id("riffusion") == "riffusion-1"
        assert reg.next_worker_id("musicgen") == "musicgen-2"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
class TestRegistration:
    def _make_worker(self, worker_id="mg-1", model="musicgen", port=9101):
        return WorkerInfo(worker_id=worker_id, model=model, port=port, device="cuda:0")

    def test_register_and_get(self):
        reg = WorkerRegistry()
        w = self._make_worker()
        reg.register(w)
        assert reg.get("mg-1") is w

    def test_get_nonexistent(self):
        reg = WorkerRegistry()
        assert reg.get("nonexistent") is None

    def test_unregister(self):
        reg = WorkerRegistry()
        w = self._make_worker()
        reg.register(w)
        removed = reg.unregister("mg-1")
        assert removed is w
        assert reg.get("mg-1") is None

    def test_unregister_nonexistent(self):
        reg = WorkerRegistry()
        assert reg.unregister("nonexistent") is None

    def test_all_workers(self):
        reg = WorkerRegistry()
        w1 = self._make_worker("mg-1", "musicgen", 9101)
        w2 = self._make_worker("rf-1", "riffusion", 9102)
        reg.register(w1)
        reg.register(w2)
        assert len(reg.all_workers()) == 2

    def test_workers_for_model(self):
        reg = WorkerRegistry()
        w1 = self._make_worker("mg-1", "musicgen", 9101)
        w2 = self._make_worker("mg-2", "musicgen", 9102)
        w3 = self._make_worker("rf-1", "riffusion", 9103)
        for w in [w1, w2, w3]:
            reg.register(w)
        mg_workers = reg.workers_for_model("musicgen")
        assert len(mg_workers) == 2
        assert all(w.model == "musicgen" for w in mg_workers)

    def test_workers_on_device(self):
        reg = WorkerRegistry()
        w1 = WorkerInfo(worker_id="mg-1", model="musicgen", port=9101, device="cuda:0")
        w2 = WorkerInfo(worker_id="mg-2", model="musicgen", port=9102, device="cuda:1")
        reg.register(w1)
        reg.register(w2)
        on_0 = reg.workers_on_device("cuda:0")
        assert len(on_0) == 1
        assert on_0[0].worker_id == "mg-1"

    def test_worker_count(self):
        reg = WorkerRegistry()
        assert reg.worker_count() == 0
        reg.register(self._make_worker("mg-1", "musicgen", 9101))
        reg.register(self._make_worker("rf-1", "riffusion", 9102))
        assert reg.worker_count() == 2
        assert reg.worker_count("musicgen") == 1


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------
class TestStatusTransitions:
    def _ready_worker(self, reg):
        w = WorkerInfo(worker_id="mg-1", model="musicgen", port=9101,
                       device="cuda:0", status="ready")
        reg.register(w)
        return w

    def test_mark_busy(self):
        reg = WorkerRegistry()
        w = self._ready_worker(reg)
        reg.mark_busy("mg-1", "job-123")
        assert w.status == "busy"
        assert w.current_job == "job-123"

    def test_mark_ready_from_busy(self):
        reg = WorkerRegistry()
        w = self._ready_worker(reg)
        reg.mark_busy("mg-1", "job-123")
        reg.mark_ready("mg-1")
        assert w.status == "ready"
        assert w.current_job is None

    def test_mark_ready_ignores_non_busy(self):
        """mark_ready should only transition from busy — prevents resurrecting dead workers."""
        reg = WorkerRegistry()
        w = WorkerInfo(worker_id="mg-1", model="musicgen", port=9101,
                       device="cuda:0", status="dead")
        reg.register(w)
        reg.mark_ready("mg-1")
        assert w.status == "dead"  # unchanged

    def test_mark_dead(self):
        reg = WorkerRegistry()
        w = self._ready_worker(reg)
        reg.mark_dead("mg-1")
        assert w.status == "dead"
        assert w.current_job is None


# ---------------------------------------------------------------------------
# Pick and mark busy (atomic operation)
# ---------------------------------------------------------------------------
class TestPickAndMarkBusy:
    def test_picks_ready_worker(self):
        reg = WorkerRegistry()
        w = WorkerInfo(worker_id="mg-1", model="musicgen", port=9101,
                       device="cuda:0", status="ready")
        reg.register(w)
        picked = reg.pick_and_mark_busy("musicgen", "job-1")
        assert picked is w
        assert w.status == "busy"
        assert w.current_job == "job-1"

    def test_returns_none_when_no_ready(self):
        reg = WorkerRegistry()
        w = WorkerInfo(worker_id="mg-1", model="musicgen", port=9101,
                       device="cuda:0", status="busy")
        reg.register(w)
        assert reg.pick_and_mark_busy("musicgen", "job-2") is None

    def test_returns_none_for_unknown_model(self):
        reg = WorkerRegistry()
        assert reg.pick_and_mark_busy("unknown_model", "job-1") is None

    def test_round_robin_across_workers(self):
        reg = WorkerRegistry()
        w1 = WorkerInfo(worker_id="mg-1", model="musicgen", port=9101,
                        device="cuda:0", status="ready")
        w2 = WorkerInfo(worker_id="mg-2", model="musicgen", port=9102,
                        device="cuda:0", status="ready")
        reg.register(w1)
        reg.register(w2)

        picked1 = reg.pick_and_mark_busy("musicgen", "job-1")
        # First pick should get one of them
        assert picked1 in (w1, w2)

        # Mark it ready again so both are available
        reg.mark_ready(picked1.worker_id)

        picked2 = reg.pick_and_mark_busy("musicgen", "job-2")
        # Round-robin should pick the other one
        assert picked2 is not None

    def test_concurrent_picks_no_double_assign(self):
        """Two threads picking simultaneously should not get the same worker."""
        reg = WorkerRegistry()
        w = WorkerInfo(worker_id="mg-1", model="musicgen", port=9101,
                       device="cuda:0", status="ready")
        reg.register(w)

        results = []

        def pick(job_id):
            result = reg.pick_and_mark_busy("musicgen", job_id)
            results.append(result)

        t1 = threading.Thread(target=pick, args=("job-1",))
        t2 = threading.Thread(target=pick, args=("job-2",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Exactly one should have gotten the worker
        non_none = [r for r in results if r is not None]
        assert len(non_none) == 1


# ---------------------------------------------------------------------------
# Health tracking
# ---------------------------------------------------------------------------
class TestHealthTracking:
    def test_update_health(self):
        reg = WorkerRegistry()
        w = WorkerInfo(worker_id="mg-1", model="musicgen", port=9101,
                       device="cuda:0", status="ready")
        reg.register(w)
        reg.update_health("mg-1", vram_used_mb=4096, vram_total_mb=24576)
        assert w.vram_used_mb == 4096
        assert w.vram_total_mb == 24576
        assert w.last_health > 0
        assert w.health_failures == 0

    def test_record_health_failure(self):
        reg = WorkerRegistry()
        w = WorkerInfo(worker_id="mg-1", model="musicgen", port=9101,
                       device="cuda:0", status="ready")
        reg.register(w)
        count = reg.record_health_failure("mg-1")
        assert count == 1
        count = reg.record_health_failure("mg-1")
        assert count == 2

    def test_health_failure_nonexistent(self):
        reg = WorkerRegistry()
        assert reg.record_health_failure("nonexistent") == 0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
class TestSerialization:
    def test_to_dict_list(self):
        reg = WorkerRegistry()
        w = WorkerInfo(worker_id="mg-1", model="musicgen", port=9101,
                       device="cuda:0", status="ready", process=MagicMock(pid=12345))
        reg.register(w)
        dicts = reg.to_dict_list()
        assert len(dicts) == 1
        d = dicts[0]
        assert d["worker_id"] == "mg-1"
        assert d["model"] == "musicgen"
        assert d["port"] == 9101
        assert d["device"] == "cuda:0"
        assert d["status"] == "ready"
        assert d["pid"] == 12345

    def test_to_dict_list_no_process(self):
        reg = WorkerRegistry()
        w = WorkerInfo(worker_id="mg-1", model="musicgen", port=9101,
                       device="cuda:0", process=None)
        reg.register(w)
        dicts = reg.to_dict_list()
        assert dicts[0]["pid"] is None


# ---------------------------------------------------------------------------
# get_ready_workers and pick_worker
# ---------------------------------------------------------------------------
class TestGetReadyAndPick:
    def test_get_ready_workers_filters(self):
        reg = WorkerRegistry()
        w1 = WorkerInfo(worker_id="mg-1", model="musicgen", port=9101,
                        device="cuda:0", status="ready")
        w2 = WorkerInfo(worker_id="mg-2", model="musicgen", port=9102,
                        device="cuda:0", status="busy")
        w3 = WorkerInfo(worker_id="mg-3", model="musicgen", port=9103,
                        device="cuda:0", status="dead")
        for w in [w1, w2, w3]:
            reg.register(w)
        ready = reg.get_ready_workers("musicgen")
        assert len(ready) == 1
        assert ready[0].worker_id == "mg-1"

    def test_pick_worker_returns_none_when_empty(self):
        reg = WorkerRegistry()
        assert reg.pick_worker("musicgen") is None

    def test_pick_worker_round_robin(self):
        reg = WorkerRegistry()
        w1 = WorkerInfo(worker_id="mg-1", model="musicgen", port=9101,
                        device="cuda:0", status="ready")
        w2 = WorkerInfo(worker_id="mg-2", model="musicgen", port=9102,
                        device="cuda:0", status="ready")
        reg.register(w1)
        reg.register(w2)

        # Pick twice - should alternate (or at least pick different ones)
        picked_ids = set()
        for _ in range(4):
            p = reg.pick_worker("musicgen")
            if p:
                picked_ids.add(p.worker_id)
        # Both should have been picked at some point
        assert len(picked_ids) == 2
