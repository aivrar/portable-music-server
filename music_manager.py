"""
Music Environment Manager - Tkinter GUI Application

Manages multiple music generation model virtual environments, model downloads,
API server for music generation, and testing interface.
"""

import os
import sys
from pathlib import Path

# Bootstrap: ensure project root is on sys.path for sibling imports
_BASE_DIR = Path(__file__).parent.resolve()
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import subprocess
import threading
import queue
import json
import shutil
import base64
import winsound
import requests
from collections import deque

# Import configuration
from config import (
    BASE_DIR, VENVS_DIR, MODELS_DIR, OUTPUT_DIR, TEMP_DIR,
    PYTHON_PATH, GIT_PATH, FFMPEG_PATH, FFMPEG_BIN_DIR,
    PYTHON_EMBEDDED_DIR, USE_EMBEDDED,
    WINDOW_TITLE, WINDOW_SIZE, APP_VERSION,
    DEFAULT_API_HOST, DEFAULT_API_PORT,
    GENERATIONS_DIR, WORKER_LOG_DIR,
    setup_environment
)

# Set up environment (HuggingFace cache, FFmpeg PATH, etc.)
setup_environment()

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Add install_configs to path
sys.path.insert(0, str(BASE_DIR))

from install_manager import InstallManager, MusicEnvironment, MODEL_SETUP
from output_manager import OutputManager
from model_params import MODEL_DISPLAY, MODEL_PARAMS, MODEL_PRESETS


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
class MusicManagerApp:
    """Main application class."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.minsize(900, 700)

        # State
        self.log_queue = queue.Queue()
        self._cached_devices = []
        self._log_entries = deque(maxlen=1000)
        self._server_poll_id = None
        self._gateway_process = None
        self._gateway_port = DEFAULT_API_PORT
        self._unkillable_pids: set = set()
        self._restarting_gateway = False
        self._param_vars = {}
        self._current_test_model = None
        self._log_poll_id = None
        self._log_offsets = {}          # job_id -> int (log offset for incremental fetch)
        self._install_all_cancel = False
        self._install_all_running = False
        self._clap_running = False

        # Output manager (persistent generation storage)
        self._output_mgr = OutputManager(GENERATIONS_DIR)

        # Shared install manager (used by both GUI and API server)
        self._install_mgr = InstallManager(log_callback=self.log)
        self.environments = self._install_mgr.environments

        # Build GUI
        self._build_gui()

        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start log queue processor
        self._process_log_queue()

        # Detect GPUs locally
        self._detect_local_devices()

        # Auto-start gateway
        self._start_gateway()

        # Start polling for gateway connection
        self.root.after(2000, self._start_server_polling)

        # Initial status check
        self._refresh_setup_status()

    # ================================================================
    # GUI Building
    # ================================================================
    def _build_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(
            main_frame, text="Music Environment Manager",
            font=("Segoe UI", 16, "bold")
        )
        title_label.pack(pady=(0, 10))

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Setup
        self.setup_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.setup_tab, text="Setup")
        self._build_setup_tab()

        # Tab 2: API Server
        self.api_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.api_tab, text="API Server")
        self._build_api_tab()

        # Tab 3: Testing
        self.test_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.test_tab, text="Testing")
        self._build_test_tab()

        # Tab 4: Outputs
        self.outputs_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.outputs_tab, text="Outputs")
        self._build_outputs_tab()

        # Tab 5: Log
        self.log_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.log_tab, text="Log")
        self._build_log_tab()

    # ================================================================
    # Setup Tab
    # ================================================================
    def _build_setup_tab(self):
        python_mode = "Embedded" if USE_EMBEDDED else "System"
        info_label = ttk.Label(
            self.setup_tab,
            text=f"Python ({python_mode}): {PYTHON_PATH}  |  Venvs: {VENVS_DIR}  |  Models: {MODELS_DIR}",
            font=("Segoe UI", 8), foreground="gray"
        )
        info_label.pack(anchor=tk.W, pady=(0, 5))

        # Install All progress frame (hidden by default)
        self._install_all_frame = ttk.Frame(self.setup_tab)
        self._install_all_progress_label = ttk.Label(
            self._install_all_frame, text="", font=("Segoe UI", 10, "bold")
        )
        self._install_all_progress_label.pack(side=tk.LEFT, padx=(0, 10))
        self._install_all_cancel_btn = ttk.Button(
            self._install_all_frame, text="Cancel All",
            command=self._cancel_install_all
        )
        self._install_all_cancel_btn.pack(side=tk.LEFT)
        # Not packed yet — shown only during Install All

        # Model list with scrollbar
        canvas_frame = ttk.Frame(self.setup_tab)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self._setup_canvas = tk.Canvas(canvas_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self._setup_canvas.yview)
        self._models_frame = ttk.LabelFrame(self._setup_canvas, text="Models", padding="5")

        self._models_frame.bind(
            "<Configure>",
            lambda e: self._setup_canvas.configure(scrollregion=self._setup_canvas.bbox("all"))
        )
        self._setup_canvas_window_id = self._setup_canvas.create_window(
            (0, 0), window=self._models_frame, anchor="nw"
        )
        self._setup_canvas.configure(yscrollcommand=scrollbar.set)

        def _on_setup_canvas_configure(event):
            self._setup_canvas.itemconfigure(self._setup_canvas_window_id, width=event.width)
        self._setup_canvas.bind("<Configure>", _on_setup_canvas_configure)

        self._setup_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mouse wheel scrolling
        def _on_mousewheel(event):
            self._setup_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self._setup_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self.setup_widgets = {}
        for model_id, info in MODEL_SETUP.items():
            display = MODEL_DISPLAY.get(model_id, {})
            merged = {**info, **display}
            self._create_setup_row(self._models_frame, model_id, merged)

        # Actions frame
        actions_frame = ttk.Frame(self.setup_tab)
        actions_frame.pack(fill=tk.X, pady=(10, 0))

        self._install_all_btn = ttk.Button(
            actions_frame, text="Install All", command=self._install_all_models
        )
        self._install_all_btn.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(actions_frame, text="Refresh Status", command=self._refresh_setup_status).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(actions_frame, text="Open Venvs Folder", command=self._open_venvs_folder).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(actions_frame, text="Open Models Folder", command=lambda: os.startfile(MODELS_DIR)).pack(side=tk.LEFT)

    def _create_setup_row(self, parent: ttk.Frame, model_id: str, info: dict):
        # Outer card container
        card = ttk.Frame(parent)
        card.pack(fill=tk.X, pady=2)

        # ── Header row (always visible) ──
        header = ttk.Frame(card)
        header.pack(fill=tk.X)

        canvas = tk.Canvas(header, width=20, height=20, highlightthickness=0)
        canvas.pack(side=tk.LEFT, padx=(0, 5))
        status_circle = canvas.create_oval(4, 4, 16, 16, fill="gray", outline="")

        vram_str = f" [{info['vram']}]" if info.get("vram") else ""
        size_str = f" ({info['weights_size']})" if info.get("weights_size") else ""

        name_label = ttk.Label(
            header, text=info["display"],
            font=("Segoe UI", 10, "bold"), width=18, anchor=tk.W
        )
        name_label.pack(side=tk.LEFT, padx=(0, 5))

        desc_text = info.get("desc", "")
        desc_label = ttk.Label(
            header, text=f"{desc_text}{vram_str}{size_str}",
            width=50, anchor=tk.W
        )
        desc_label.pack(side=tk.LEFT, padx=(0, 5))

        status_label = ttk.Label(header, text="Checking...", width=30, anchor=tk.W)
        status_label.pack(side=tk.LEFT, padx=(0, 5))

        install_btn = ttk.Button(
            header, text="Install", width=8,
            command=lambda m=model_id: self._install_model(m)
        )
        install_btn.pack(side=tk.LEFT, padx=2)

        cancel_btn = ttk.Button(
            header, text="Cancel", width=8,
            command=lambda m=model_id: self._cancel_model_install(m)
        )
        # Cancel button starts hidden (not packed)

        download_btn = ttk.Button(
            header, text="Download", width=9,
            command=lambda m=model_id: self._download_model_weights(m)
        )
        # Download button starts hidden (not packed) — shown only when applicable

        remove_btn = ttk.Button(
            header, text="Remove", width=8,
            command=lambda m=model_id: self._remove_model(m)
        )
        remove_btn.pack(side=tk.LEFT, padx=2)

        # Click header to toggle detail panel
        for widget in (canvas, name_label, desc_label):
            widget.bind("<Button-1>", lambda e, m=model_id: self._toggle_card(m))

        # ── Detail panel (collapsible, hidden by default) ──
        detail = ttk.Frame(card, padding=(25, 5, 5, 5))
        # Not packed yet — expanded on demand

        progress = ttk.Progressbar(detail, mode="determinate", length=400)
        progress.pack(fill=tk.X, pady=(0, 3))

        step_label = ttk.Label(detail, text="", font=("Segoe UI", 9))
        step_label.pack(anchor=tk.W, pady=(0, 3))

        log_text = tk.Text(
            detail, height=6, font=("Consolas", 8), wrap=tk.WORD,
            bg="#1e1e1e", fg="#cccccc", state=tk.DISABLED,
            relief=tk.FLAT, borderwidth=1
        )
        log_text.pack(fill=tk.X, pady=(0, 3))

        error_label = ttk.Label(
            detail, text="", foreground="red", wraplength=600,
            font=("Segoe UI", 9)
        )
        # error_label not packed — shown only on failure

        self.setup_widgets[model_id] = {
            "card": card,
            "header": header,
            "canvas": canvas,
            "circle": status_circle,
            "status_label": status_label,
            "install_btn": install_btn,
            "cancel_btn": cancel_btn,
            "download_btn": download_btn,
            "remove_btn": remove_btn,
            "detail": detail,
            "progress": progress,
            "step_label": step_label,
            "log_text": log_text,
            "error_label": error_label,
            "expanded": False,
        }

    def _toggle_card(self, model_id: str):
        w = self.setup_widgets.get(model_id)
        if not w:
            return
        if w["expanded"]:
            self._collapse_card(model_id)
        else:
            self._expand_card(model_id)

    def _expand_card(self, model_id: str):
        w = self.setup_widgets.get(model_id)
        if not w or w["expanded"]:
            return
        w["detail"].pack(fill=tk.X, after=w["header"])
        w["expanded"] = True

    def _collapse_card(self, model_id: str):
        w = self.setup_widgets.get(model_id)
        if not w or not w["expanded"]:
            return
        w["detail"].pack_forget()
        w["expanded"] = False

    def _append_card_log(self, model_id: str, lines: list):
        w = self.setup_widgets.get(model_id)
        if not w:
            return
        log_text = w["log_text"]
        log_text.config(state=tk.NORMAL)
        for line in lines:
            log_text.insert(tk.END, line + "\n")
        # Keep only last ~100 lines
        line_count = int(log_text.index("end-1c").split(".")[0])
        if line_count > 100:
            log_text.delete("1.0", f"{line_count - 100}.0")
        log_text.see(tk.END)
        log_text.config(state=tk.DISABLED)

    def _cancel_model_install(self, model_id: str):
        cancelled = self._install_mgr.cancel_install(model_id)
        if cancelled:
            self.log(f"Cancelling {model_id} installation...")
        else:
            self.log(f"No active install to cancel for {model_id}.", tag="warning")

    def _download_model_weights(self, model_id: str):
        """Start downloading weights for a model (standalone, no env install)."""
        try:
            job_id = self._install_mgr.start_download(model_id)
        except (ValueError, RuntimeError) as e:
            messagebox.showwarning("Download", str(e))
            return

        # Reset card state for download
        w = self.setup_widgets.get(model_id)
        if w:
            w["progress"]["value"] = 0
            w["step_label"].config(text="Starting download...")
            w["error_label"].pack_forget()
            w["log_text"].config(state=tk.NORMAL)
            w["log_text"].delete("1.0", tk.END)
            w["log_text"].config(state=tk.DISABLED)

        self._log_offsets[job_id] = 0
        self._update_setup_row(model_id, "installing", "Downloading...")
        self._expand_card(model_id)
        self._poll_install_job(model_id, job_id)

    # ================================================================
    # Log Tab
    # ================================================================
    def _build_log_tab(self):
        filter_frame = ttk.Frame(self.log_tab)
        filter_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(filter_frame, text="Show:").pack(side=tk.LEFT, padx=(0, 5))
        self._log_filters = {}
        for tag, label, default in [
            ("info", "Info", True), ("success", "Success", True),
            ("error", "Errors", True), ("warning", "Warnings", True),
            ("gateway", "Gateway", True),
        ]:
            var = tk.BooleanVar(value=default)
            self._log_filters[tag] = var
            ttk.Checkbutton(
                filter_frame, text=label, variable=var,
                command=self._rerender_log
            ).pack(side=tk.LEFT, padx=(0, 8))

        self.log_text = scrolledtext.ScrolledText(
            self.log_tab, font=("Consolas", 9), state=tk.DISABLED, wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.log_text.tag_configure("info", foreground="black")
        self.log_text.tag_configure("success", foreground="green")
        self.log_text.tag_configure("error", foreground="red")
        self.log_text.tag_configure("warning", foreground="orange")
        self.log_text.tag_configure("gateway", foreground="#666666")

        bottom = ttk.Frame(self.log_tab)
        bottom.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(bottom, text="Last 1000 entries", foreground="gray", font=("Segoe UI", 8)).pack(side=tk.LEFT)
        ttk.Button(bottom, text="Clear Log", command=self._clear_log).pack(side=tk.RIGHT)

    # ================================================================
    # API Server Tab
    # ================================================================
    def _build_api_tab(self):
        # Gateway bar
        gw_frame = ttk.LabelFrame(self.api_tab, text="Gateway", padding="5")
        gw_frame.pack(fill=tk.X, pady=(0, 5))

        gw_row = ttk.Frame(gw_frame)
        gw_row.pack(fill=tk.X)

        self.server_status_canvas = tk.Canvas(gw_row, width=16, height=16, highlightthickness=0)
        self.server_status_canvas.pack(side=tk.LEFT, padx=(0, 5))
        self.server_status_circle = self.server_status_canvas.create_oval(2, 2, 14, 14, fill="gray", outline="")

        self.server_status_label = ttk.Label(gw_row, text="Starting...", font=("Segoe UI", 9))
        self.server_status_label.pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(gw_row, text="Port:").pack(side=tk.LEFT, padx=(0, 3))
        self.port_var = tk.StringVar(value=str(DEFAULT_API_PORT))
        self.port_entry = ttk.Entry(gw_row, textvariable=self.port_var, width=6)
        self.port_entry.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(gw_row, text="Restart Gateway", command=self._restart_gateway).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(gw_row, text="Shutdown All", command=self._shutdown_all).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(gw_row, text="Kill All Workers", command=self._kill_all_workers).pack(side=tk.RIGHT)

        # Workers section
        workers_frame = ttk.LabelFrame(self.api_tab, text="Workers", padding="5")
        workers_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        workers_btn_row = ttk.Frame(workers_frame)
        workers_btn_row.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(workers_btn_row, text="Kill Selected", command=self._kill_selected_worker).pack(side=tk.RIGHT)

        wrk_cols = ("worker_id", "model", "port", "device", "status")
        self.workers_tree = ttk.Treeview(workers_frame, columns=wrk_cols, show="headings", height=8)
        self.workers_tree.heading("worker_id", text="Worker ID")
        self.workers_tree.heading("model", text="Model")
        self.workers_tree.heading("port", text="Port")
        self.workers_tree.heading("device", text="Device")
        self.workers_tree.heading("status", text="Status")
        self.workers_tree.column("worker_id", width=140, minwidth=80)
        self.workers_tree.column("model", width=120, minwidth=70)
        self.workers_tree.column("port", width=70, minwidth=50)
        self.workers_tree.column("device", width=90, minwidth=60)
        self.workers_tree.column("status", width=80, minwidth=60)

        workers_scroll = ttk.Scrollbar(workers_frame, orient="vertical", command=self.workers_tree.yview)
        self.workers_tree.configure(yscrollcommand=workers_scroll.set)
        self.workers_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        workers_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.workers_tree.bind("<Button-3>", lambda e: self._show_worker_context_menu(e, self.workers_tree))

        # Spawn section
        spawn_frame = ttk.LabelFrame(self.api_tab, text="Spawn Worker", padding="5")
        spawn_frame.pack(fill=tk.X)

        spawn_row = ttk.Frame(spawn_frame)
        spawn_row.pack(fill=tk.X)

        ttk.Label(spawn_row, text="Model:").pack(side=tk.LEFT, padx=(0, 5))
        self.spawn_model_var = tk.StringVar(value="musicgen")
        model_ids = sorted(MODEL_SETUP.keys())
        self.spawn_model_combo = ttk.Combobox(
            spawn_row, textvariable=self.spawn_model_var, state="readonly", width=16,
            values=model_ids
        )
        self.spawn_model_combo.pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(spawn_row, text="Device:").pack(side=tk.LEFT, padx=(0, 5))
        self.spawn_device_var = tk.StringVar()
        self.spawn_device_combo = ttk.Combobox(
            spawn_row, textvariable=self.spawn_device_var, state="readonly", width=14
        )
        self.spawn_device_combo.pack(side=tk.LEFT, padx=(0, 15))

        ttk.Button(spawn_row, text="Spawn", command=self._spawn_worker).pack(side=tk.LEFT)

    # ================================================================
    # Testing Tab
    # ================================================================
    def _build_test_tab(self):
        # Top pane: worker selection + scrollable params
        top_pane = ttk.PanedWindow(self.test_tab, orient=tk.VERTICAL)
        top_pane.pack(fill=tk.BOTH, expand=True)

        # --- Loaded Workers ---
        workers_frame = ttk.LabelFrame(top_pane, text="Loaded Workers", padding="5")

        wrk_cols = ("worker_id", "model", "port", "device", "status")
        self.test_workers_tree = ttk.Treeview(
            workers_frame, columns=wrk_cols, show="headings", height=3
        )
        for col, label, w in [
            ("worker_id", "Worker ID", 140), ("model", "Model", 120),
            ("port", "Port", 70), ("device", "Device", 90), ("status", "Status", 80),
        ]:
            self.test_workers_tree.heading(col, text=label)
            self.test_workers_tree.column(col, width=w, minwidth=50)
        self.test_workers_tree.pack(fill=tk.X)
        self.test_workers_tree.bind(
            "<<TreeviewSelect>>", lambda e: self._on_test_worker_selected()
        )
        self.test_workers_tree.bind("<Button-3>", lambda e: self._show_worker_context_menu(e, self.test_workers_tree))

        top_pane.add(workers_frame, weight=0)

        # --- Model Parameters (dynamically rebuilt) ---
        params_outer = ttk.LabelFrame(top_pane, text="Model Parameters", padding="5")

        # Scrollable inner frame for model params
        params_canvas = tk.Canvas(params_outer, highlightthickness=0, height=200)
        params_scrollbar = ttk.Scrollbar(params_outer, orient="vertical", command=params_canvas.yview)
        self._test_params_inner = ttk.Frame(params_canvas)

        self._test_params_inner.bind(
            "<Configure>",
            lambda e: params_canvas.configure(scrollregion=params_canvas.bbox("all"))
        )
        params_canvas.create_window((0, 0), window=self._test_params_inner, anchor="nw")
        params_canvas.configure(yscrollcommand=params_scrollbar.set)

        params_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        params_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Placeholder
        ttk.Label(self._test_params_inner, text="Select a worker above to see model parameters", foreground="gray").pack(anchor=tk.W)

        top_pane.add(params_outer, weight=1)

        # --- Post-Processing Controls (always visible) ---
        pp_frame = ttk.LabelFrame(top_pane, text="Post-Processing", padding="5")
        pp_grid = ttk.Frame(pp_frame)
        pp_grid.pack(fill=tk.X)

        # Row 0: Core audio processing
        ttk.Label(pp_grid, text="Denoise:").grid(row=0, column=0, sticky=tk.W, padx=(0, 3))
        self.test_denoise_var = tk.StringVar(value="0.2")
        ttk.Spinbox(pp_grid, textvariable=self.test_denoise_var, from_=0.0, to=1.0,
                     increment=0.05, width=5, format="%.2f").grid(row=0, column=1, padx=(0, 8))

        ttk.Label(pp_grid, text="Stereo Width:").grid(row=0, column=2, sticky=tk.W, padx=(0, 3))
        self.test_stereo_var = tk.StringVar(value="1.2")
        ttk.Spinbox(pp_grid, textvariable=self.test_stereo_var, from_=0.0, to=2.0,
                     increment=0.1, width=5, format="%.1f").grid(row=0, column=3, padx=(0, 8))

        ttk.Label(pp_grid, text="EQ Preset:").grid(row=0, column=4, sticky=tk.W, padx=(0, 3))
        self.test_eq_var = tk.StringVar(value="balanced")
        ttk.Combobox(
            pp_grid, textvariable=self.test_eq_var, state="readonly",
            width=10, values=["flat", "balanced", "warm", "bright"]
        ).grid(row=0, column=5, padx=(0, 8))

        ttk.Label(pp_grid, text="Target LUFS:").grid(row=0, column=6, sticky=tk.W, padx=(0, 3))
        self.test_lufs_var = tk.StringVar(value="-14.0")
        ttk.Spinbox(pp_grid, textvariable=self.test_lufs_var, from_=-24.0, to=-8.0,
                     increment=0.5, width=6, format="%.1f").grid(row=0, column=7, padx=(0, 8))

        # Row 1: Filtering & dynamics
        ttk.Label(pp_grid, text="Highpass (Hz):").grid(row=1, column=0, sticky=tk.W, padx=(0, 3), pady=(4, 0))
        self.test_highpass_var = tk.StringVar(value="30")
        ttk.Spinbox(pp_grid, textvariable=self.test_highpass_var, from_=20, to=200,
                     increment=5, width=5).grid(row=1, column=1, padx=(0, 8), pady=(4, 0))

        ttk.Label(pp_grid, text="Compression:").grid(row=1, column=2, sticky=tk.W, padx=(0, 3), pady=(4, 0))
        self.test_compression_var = tk.StringVar(value="2.0")
        ttk.Spinbox(pp_grid, textvariable=self.test_compression_var, from_=1.0, to=8.0,
                     increment=0.5, width=5, format="%.1f").grid(row=1, column=3, padx=(0, 8), pady=(4, 0))

        ttk.Label(pp_grid, text="Peak Limit:").grid(row=1, column=4, sticky=tk.W, padx=(0, 3), pady=(4, 0))
        self.test_clipping_var = tk.StringVar(value="0.95")
        ttk.Spinbox(pp_grid, textvariable=self.test_clipping_var, from_=0.5, to=1.0,
                     increment=0.01, width=5, format="%.2f").grid(row=1, column=5, padx=(0, 8), pady=(4, 0))

        ttk.Label(pp_grid, text="Trim dB:").grid(row=1, column=6, sticky=tk.W, padx=(0, 3), pady=(4, 0))
        self.test_trim_db_var = tk.StringVar(value="-45")
        ttk.Spinbox(pp_grid, textvariable=self.test_trim_db_var, from_=-60, to=-20,
                     increment=1, width=5).grid(row=1, column=7, padx=(0, 8), pady=(4, 0))

        # Row 2: Silence trimming zones
        ttk.Label(pp_grid, text="Min Silence (ms):").grid(row=2, column=0, sticky=tk.W, padx=(0, 3), pady=(4, 0))
        self.test_min_silence_var = tk.StringVar(value="500")
        ttk.Spinbox(pp_grid, textvariable=self.test_min_silence_var, from_=100, to=2000,
                     increment=50, width=5).grid(row=2, column=1, padx=(0, 8), pady=(4, 0))

        ttk.Label(pp_grid, text="Front Protect (ms):").grid(row=2, column=2, sticky=tk.W, padx=(0, 3), pady=(4, 0))
        self.test_front_protect_var = tk.StringVar(value="200")
        ttk.Spinbox(pp_grid, textvariable=self.test_front_protect_var, from_=0, to=5000,
                     increment=50, width=5).grid(row=2, column=3, padx=(0, 8), pady=(4, 0))

        ttk.Label(pp_grid, text="End Protect (ms):").grid(row=2, column=4, sticky=tk.W, padx=(0, 3), pady=(4, 0))
        self.test_end_protect_var = tk.StringVar(value="1000")
        ttk.Spinbox(pp_grid, textvariable=self.test_end_protect_var, from_=0, to=5000,
                     increment=50, width=5).grid(row=2, column=5, padx=(0, 8), pady=(4, 0))

        # Row 3: Output & skip
        ttk.Label(pp_grid, text="Output Format:").grid(row=3, column=0, sticky=tk.W, padx=(0, 3), pady=(4, 0))
        self.test_format_var = tk.StringVar(value="wav")
        ttk.Combobox(
            pp_grid, textvariable=self.test_format_var, state="readonly",
            width=5, values=["wav", "mp3", "ogg", "flac"]
        ).grid(row=3, column=1, padx=(0, 8), pady=(4, 0))

        self.test_skip_pp_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            pp_grid, text="Skip Post-Processing", variable=self.test_skip_pp_var
        ).grid(row=3, column=2, columnspan=2, sticky=tk.W, pady=(4, 0))

        top_pane.add(pp_frame, weight=0)

        # --- Send button + status ---
        btn_frame = ttk.Frame(self.test_tab)
        btn_frame.pack(fill=tk.X, pady=(5, 5))
        self.test_send_btn = ttk.Button(
            btn_frame, text="Generate Music", command=self._send_test_request
        )
        self.test_send_btn.pack(side=tk.LEFT)

        self.test_status_label = ttk.Label(
            btn_frame, text="", font=("Segoe UI", 9), foreground="gray"
        )
        self.test_status_label.pack(side=tk.LEFT, padx=(15, 0))

    # ================================================================
    # Outputs Tab
    # ================================================================
    def _build_outputs_tab(self):
        # Toolbar
        toolbar = ttk.Frame(self.outputs_tab)
        toolbar.pack(fill=tk.X, pady=(0, 5))

        # CLAP section
        clap_frame = ttk.LabelFrame(toolbar, text="CLAP Scorer", padding="3")
        clap_frame.pack(side=tk.LEFT, padx=(0, 10))
        self._clap_auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(clap_frame, text="Auto", variable=self._clap_auto_var,
                         command=self._on_clap_auto_toggle).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(clap_frame, text="Device:").pack(side=tk.LEFT, padx=(0, 3))
        self._clap_device_var = tk.StringVar(value="cuda:0")
        self._clap_device_combo = ttk.Combobox(
            clap_frame, textvariable=self._clap_device_var, state="readonly",
            width=8, values=["cuda:0", "cuda:1", "cpu"]
        )
        self._clap_device_combo.pack(side=tk.LEFT, padx=(0, 5))
        self._clap_status_label = ttk.Label(clap_frame, text="Stopped", foreground="gray")
        self._clap_status_label.pack(side=tk.LEFT, padx=(0, 5))
        self._clap_start_btn = ttk.Button(clap_frame, text="Start", command=self._start_clap,
                                           state=tk.DISABLED)
        self._clap_start_btn.pack(side=tk.LEFT, padx=(0, 3))
        self._clap_stop_btn = ttk.Button(clap_frame, text="Stop", command=self._stop_clap,
                                          state=tk.DISABLED)
        self._clap_stop_btn.pack(side=tk.LEFT)

        # Batch section
        batch_frame = ttk.LabelFrame(toolbar, text="Batch", padding="3")
        batch_frame.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(batch_frame, text="Select All", command=self._select_all_outputs).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(batch_frame, text="Deselect All", command=self._deselect_all_outputs).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(batch_frame, text="Download Selected", command=self._download_selected).pack(side=tk.LEFT)

        # Playback section (right-aligned)
        play_frame = ttk.Frame(toolbar)
        play_frame.pack(side=tk.RIGHT)
        self._out_play_btn = ttk.Button(play_frame, text="Play", command=self._play_selected_output, state=tk.DISABLED)
        self._out_play_btn.pack(side=tk.LEFT, padx=(0, 3))
        self._out_stop_btn = ttk.Button(play_frame, text="Stop", command=self._stop_audio, state=tk.DISABLED)
        self._out_stop_btn.pack(side=tk.LEFT)

        # Outputs treeview
        tree_frame = ttk.Frame(self.outputs_tab)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        out_cols = ("num", "model", "prompt", "duration", "format", "clap", "date")
        self.outputs_tree = ttk.Treeview(
            tree_frame, columns=out_cols, show="headings", height=15,
            selectmode="extended"
        )
        for col, label, w in [
            ("num", "#", 40), ("model", "Model", 100),
            ("prompt", "Prompt", 250), ("duration", "Duration", 70),
            ("format", "Format", 50), ("clap", "CLAP", 60),
            ("date", "Date", 130),
        ]:
            self.outputs_tree.heading(col, text=label)
            self.outputs_tree.column(col, width=w, minwidth=30)

        out_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.outputs_tree.yview)
        self.outputs_tree.configure(yscrollcommand=out_scroll.set)
        self.outputs_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        out_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.outputs_tree.bind("<Button-3>", self._show_output_context_menu)
        self.outputs_tree.bind("<<TreeviewSelect>>", lambda e: self._on_output_selected())

        # Populate from existing saved outputs
        self._populate_outputs_tree()

    def _populate_outputs_tree(self):
        """Load all saved outputs into the treeview."""
        for item in self.outputs_tree.get_children():
            self.outputs_tree.delete(item)
        entries = self._output_mgr.get_all_entries()
        for i, entry in enumerate(entries, 1):
            prompt_short = entry.prompt[:50] + "..." if len(entry.prompt) > 50 else entry.prompt
            clap_str = f"{entry.clap_score:.3f}" if entry.clap_score is not None else "--"
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(entry.timestamp)
                date_str = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                date_str = entry.timestamp[:16]
            self.outputs_tree.insert("", tk.END, iid=entry.id, values=(
                i, entry.model, prompt_short,
                f"{entry.duration_sec:.1f}s", entry.format,
                clap_str, date_str,
            ))

    def _on_output_selected(self):
        sel = self.outputs_tree.selection()
        has_sel = len(sel) > 0
        if has_sel and len(sel) == 1:
            entry = self._output_mgr.get_entry(sel[0])
            is_wav = entry and entry.format == "wav"
            self._out_play_btn.config(state=tk.NORMAL if is_wav else tk.DISABLED)
            self._out_stop_btn.config(state=tk.NORMAL if is_wav else tk.DISABLED)
        else:
            self._out_play_btn.config(state=tk.DISABLED)
            self._out_stop_btn.config(state=tk.DISABLED)

    def _play_selected_output(self):
        sel = self.outputs_tree.selection()
        if not sel:
            return
        audio_path = self._output_mgr.get_audio_path(sel[0])
        if audio_path:
            try:
                winsound.PlaySound(str(audio_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception as e:
                self.log(f"Playback failed: {e}", tag="warning")

    # ================================================================
    # Right-click context menus
    # ================================================================
    def _show_worker_context_menu(self, event, tree: ttk.Treeview):
        """Right-click menu for worker treeviews."""
        item = tree.identify_row(event.y)
        if not item:
            return
        tree.selection_set(item)
        values = tree.item(item, "values")
        if not values:
            return

        worker_id = values[0]
        model = values[1]
        port = values[2]

        menu = tk.Menu(tree, tearoff=0)
        menu.add_command(label="Kill Worker", command=lambda: self._ctx_kill_worker(worker_id))
        menu.add_command(label="View Worker Log", command=lambda: self._ctx_view_worker_log(model, port))
        menu.add_command(label="Copy Port", command=lambda: self._ctx_copy_to_clipboard(str(port)))
        menu.tk_popup(event.x_root, event.y_root)

    def _ctx_kill_worker(self, worker_id: str):
        if not messagebox.askyesno("Confirm Kill", f"Kill worker '{worker_id}'?"):
            return

        def do_kill():
            result = self._api_delete(f"/api/workers/{worker_id}")
            if result:
                self.root.after(0, lambda: self.log(f"Worker '{worker_id}' killed.", tag="success"))
                self.root.after(500, self._refresh_workers)
            else:
                self.root.after(0, lambda: self.log(f"Failed to kill worker '{worker_id}'.", tag="error"))

        threading.Thread(target=do_kill, daemon=True).start()

    def _ctx_view_worker_log(self, model: str, port: str):
        log_file = WORKER_LOG_DIR / f"worker_{model}_{port}.log"
        if log_file.exists():
            os.startfile(str(log_file))
        else:
            messagebox.showinfo("No Log", f"Log file not found:\n{log_file}")

    def _ctx_copy_to_clipboard(self, text: str):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _show_output_context_menu(self, event):
        """Right-click menu for outputs treeview."""
        item = self.outputs_tree.identify_row(event.y)
        if not item:
            return
        # Add to selection if not already selected
        if item not in self.outputs_tree.selection():
            self.outputs_tree.selection_set(item)

        entry = self._output_mgr.get_entry(item)
        if not entry:
            return

        menu = tk.Menu(self.outputs_tree, tearoff=0)

        # Play (WAV only)
        is_wav = entry.format == "wav"
        menu.add_command(
            label="Play",
            command=lambda: self._play_selected_output(),
            state=tk.NORMAL if is_wav else tk.DISABLED,
        )
        menu.add_command(label="Save As...", command=lambda: self._ctx_save_output(entry.id))
        menu.add_separator()

        # CLAP scoring
        menu.add_command(
            label="Score with CLAP",
            command=lambda: self._ctx_score_with_clap(entry.id),
            state=tk.NORMAL if self._clap_running else tk.DISABLED,
        )
        menu.add_separator()

        menu.add_command(label="Open Containing Folder", command=lambda: os.startfile(str(GENERATIONS_DIR)))
        menu.add_command(label="Copy Prompt", command=lambda: self._ctx_copy_to_clipboard(entry.prompt))
        menu.add_separator()
        menu.add_command(label="Delete", command=lambda: self._ctx_delete_output(entry.id))

        menu.tk_popup(event.x_root, event.y_root)

    def _ctx_save_output(self, entry_id: str):
        audio_path = self._output_mgr.get_audio_path(entry_id)
        if not audio_path:
            return
        entry = self._output_mgr.get_entry(entry_id)
        ext = f".{entry.format}" if entry else ".wav"
        dest = filedialog.asksaveasfilename(
            defaultextension=ext,
            filetypes=[("Audio files", f"*{ext}"), ("All files", "*.*")],
            title="Save Audio"
        )
        if dest:
            try:
                shutil.copy2(str(audio_path), dest)
                self.log(f"Audio saved to {dest}", tag="success")
            except Exception as e:
                self.log(f"Failed to save: {e}", tag="error")

    def _ctx_delete_output(self, entry_id: str):
        if not messagebox.askyesno("Confirm Delete", f"Delete output '{entry_id}'?"):
            return
        self._output_mgr.delete_entry(entry_id)
        if self.outputs_tree.exists(entry_id):
            self.outputs_tree.delete(entry_id)
        self.log(f"Output {entry_id} deleted.", tag="success")

    def _ctx_score_with_clap(self, entry_id: str):
        """Score a single output with CLAP in background."""
        entry = self._output_mgr.get_entry(entry_id)
        audio_path = self._output_mgr.get_audio_path(entry_id)
        if not entry or not audio_path:
            return

        def do_score():
            try:
                with open(audio_path, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode("utf-8")
                result = self._api_post(
                    "/api/clap/score",
                    json_body={"audio_base64": audio_b64, "prompt": entry.prompt},
                    timeout=120,
                )
                if result and "score" in result:
                    score = result["score"]
                    self._output_mgr.update_clap_score(entry_id, score)
                    self.root.after(0, lambda: self._update_output_clap(entry_id, score))
                    self.root.after(0, lambda: self.log(
                        f"CLAP score for {entry_id}: {score:.4f}", tag="success"
                    ))
                else:
                    self.root.after(0, lambda: self.log(
                        f"CLAP scoring failed for {entry_id}", tag="error"
                    ))
            except Exception as e:
                self.root.after(0, lambda: self.log(f"CLAP error: {e}", tag="error"))

        threading.Thread(target=do_score, daemon=True).start()

    def _update_output_clap(self, entry_id: str, score: float):
        """Update the CLAP column in the outputs treeview."""
        if self.outputs_tree.exists(entry_id):
            values = list(self.outputs_tree.item(entry_id, "values"))
            values[5] = f"{score:.3f}"
            self.outputs_tree.item(entry_id, values=values)

    # ================================================================
    # CLAP Management
    # ================================================================
    def _on_clap_auto_toggle(self):
        """Toggle manual Start/Stop buttons based on Auto checkbox."""
        auto = self._clap_auto_var.get()
        if auto:
            # In auto mode, disable manual buttons — CLAP starts/stops with workers
            self._clap_start_btn.config(state=tk.DISABLED)
            if not self._clap_running:
                self._clap_stop_btn.config(state=tk.DISABLED)
        else:
            # Manual mode — enable Start if not running
            if not self._clap_running:
                self._clap_start_btn.config(state=tk.NORMAL)
            else:
                self._clap_stop_btn.config(state=tk.NORMAL)

    def _auto_start_clap_if_needed(self, worker_device: str = None):
        """Auto-start CLAP if Auto mode is on and CLAP isn't running.
        Uses the worker's device so CLAP shares the same GPU."""
        if not self._clap_auto_var.get() or self._clap_running:
            return
        # Use the worker's device, fall back to the combo selection
        if worker_device:
            self._clap_device_var.set(worker_device)
        self._start_clap()

    def _start_clap(self, device: str = None):
        self._clap_start_btn.config(state=tk.DISABLED)
        self._clap_device_combo.config(state=tk.DISABLED)
        self._clap_status_label.config(text="Starting...", foreground="orange")
        device = device or self._clap_device_var.get()
        self.log(f"Starting CLAP scorer on {device}...")

        def do_start():
            result = self._api_post("/api/clap/start",
                                    json_body={"device": device}, timeout=600)
            if result and result.get("status") in ("started", "already_running"):
                self.root.after(0, lambda: self._on_clap_started())
            else:
                self.root.after(0, lambda: self._on_clap_failed())

        threading.Thread(target=do_start, daemon=True).start()

    def _on_clap_started(self):
        self._clap_running = True
        self._clap_status_label.config(text="Running", foreground="green")
        self._clap_start_btn.config(state=tk.DISABLED)
        self._clap_stop_btn.config(state=tk.NORMAL if not self._clap_auto_var.get() else tk.DISABLED)
        self.log("CLAP scorer started.", tag="success")

    def _on_clap_failed(self):
        self._clap_running = False
        self._clap_status_label.config(text="Failed", foreground="red")
        auto = self._clap_auto_var.get()
        self._clap_start_btn.config(state=tk.DISABLED if auto else tk.NORMAL)
        self._clap_stop_btn.config(state=tk.DISABLED)
        self._clap_device_combo.config(state="readonly")
        self.log("CLAP scorer failed to start.", tag="error")

    def _stop_clap(self):
        self._clap_stop_btn.config(state=tk.DISABLED)

        def do_stop():
            self._api_post("/api/clap/stop", timeout=30)
            self.root.after(0, lambda: self._on_clap_stopped())

        threading.Thread(target=do_stop, daemon=True).start()

    def _on_clap_stopped(self):
        self._clap_running = False
        self._clap_status_label.config(text="Stopped", foreground="gray")
        auto = self._clap_auto_var.get()
        self._clap_start_btn.config(state=tk.DISABLED if auto else tk.NORMAL)
        self._clap_stop_btn.config(state=tk.DISABLED)
        self._clap_device_combo.config(state="readonly")
        self.log("CLAP scorer stopped.")

    # ================================================================
    # Batch Download
    # ================================================================
    def _select_all_outputs(self):
        self.outputs_tree.selection_set(self.outputs_tree.get_children())

    def _deselect_all_outputs(self):
        self.outputs_tree.selection_remove(self.outputs_tree.get_children())

    def _download_selected(self):
        sel = self.outputs_tree.selection()
        if not sel:
            messagebox.showinfo("No Selection", "Select outputs to download.")
            return

        dest_dir = filedialog.askdirectory(title="Select Download Folder")
        if not dest_dir:
            return

        copied = 0
        for entry_id in sel:
            audio_path = self._output_mgr.get_audio_path(entry_id)
            if audio_path:
                try:
                    shutil.copy2(str(audio_path), dest_dir)
                    # Also copy JSON metadata
                    json_path = audio_path.with_suffix(".json")
                    if json_path.exists():
                        shutil.copy2(str(json_path), dest_dir)
                    copied += 1
                except Exception as e:
                    self.log(f"Failed to copy {entry_id}: {e}", tag="warning")

        self.log(f"Downloaded {copied}/{len(sel)} outputs to {dest_dir}", tag="success")
        messagebox.showinfo("Download Complete", f"Copied {copied} of {len(sel)} outputs.")

    # ================================================================
    # Dynamic Model Parameter UI
    # ================================================================
    def _on_test_worker_selected(self):
        """Handle worker selection - rebuild model-specific parameter controls."""
        sel = self.test_workers_tree.selection()
        if not sel:
            return
        values = self.test_workers_tree.item(sel[0], "values")
        if not values:
            return
        model = values[1]
        self._current_test_model = model
        self._build_model_params(model)

    def _build_model_params(self, model: str):
        """Rebuild parameter controls when user selects a different worker/model.

        Layout: 5-column grid.  Wide controls (text, file) span the full row.
        Narrow controls (entry, combo, check) pair two-per-row, left then right.
        """
        # Clear existing controls
        for w in self._test_params_inner.winfo_children():
            w.destroy()
        self._param_vars = {}

        params_def = MODEL_PARAMS.get(model, {})
        controls = params_def.get("controls", [])

        if not controls:
            ttk.Label(self._test_params_inner, text=f"No configurable parameters for {model}", foreground="gray").pack(anchor=tk.W)
            return

        # Column weights: labels fixed, widgets expand, spacer between halves
        self._test_params_inner.columnconfigure(0, weight=0)
        self._test_params_inner.columnconfigure(1, weight=1)
        self._test_params_inner.columnconfigure(2, weight=0, minsize=20)
        self._test_params_inner.columnconfigure(3, weight=0)
        self._test_params_inner.columnconfigure(4, weight=1)

        row = 0
        side = "left"  # alternates "left" / "right" for narrow controls

        # Preset dropdown (only shown if model has presets)
        presets = MODEL_PRESETS.get(model, [])
        if presets:
            ttk.Label(self._test_params_inner, text="Preset:").grid(
                row=row, column=0, sticky=tk.W, padx=(0, 3), pady=(0, 6))
            preset_names = ["(custom)"] + [p["name"] for p in presets]
            self._preset_var = tk.StringVar(value="(custom)")
            preset_combo = ttk.Combobox(
                self._test_params_inner, textvariable=self._preset_var,
                state="readonly", values=preset_names, width=35)
            preset_combo.grid(row=row, column=1, columnspan=4,
                              sticky=tk.W, pady=(0, 6))
            preset_combo.bind("<<ComboboxSelected>>",
                              lambda e, m=model: self._apply_preset(m))
            row += 1

        for ctrl in controls:
            ctrl_id = ctrl["id"]
            ctrl_type = ctrl["type"]
            label = ctrl.get("label", ctrl_id)

            # --- Section heading: full-width separator + bold label ---
            if ctrl_type == "heading":
                if side == "right":
                    row += 1
                    side = "left"
                sep = ttk.Separator(self._test_params_inner, orient=tk.HORIZONTAL)
                sep.grid(row=row, column=0, columnspan=5, sticky=tk.EW,
                         pady=(8, 2))
                row += 1
                ttk.Label(
                    self._test_params_inner, text=label,
                    font=("Segoe UI", 9, "bold"),
                ).grid(row=row, column=0, columnspan=5, sticky=tk.W,
                       pady=(0, 4))
                row += 1
                continue  # headings are not stored in _param_vars

            # --- Wide controls: own full-width row ---
            if ctrl_type == "text":
                if side == "right":
                    row += 1
                    side = "left"
                ttk.Label(self._test_params_inner, text=f"{label}:").grid(
                    row=row, column=0, sticky=tk.NW, padx=(0, 5), pady=2
                )
                text_widget = scrolledtext.ScrolledText(
                    self._test_params_inner, height=ctrl.get("rows", 3),
                    width=60, font=("Consolas", 9), wrap=tk.WORD
                )
                text_widget.grid(row=row, column=1, columnspan=4, sticky=tk.EW, pady=2)
                if ctrl.get("default"):
                    text_widget.insert("1.0", ctrl["default"])
                self._param_vars[ctrl_id] = text_widget
                row += 1
                side = "left"

            elif ctrl_type == "file":
                if side == "right":
                    row += 1
                    side = "left"
                ttk.Label(self._test_params_inner, text=f"{label}:").grid(
                    row=row, column=0, sticky=tk.W, padx=(0, 5), pady=2
                )
                var = tk.StringVar()
                ttk.Entry(
                    self._test_params_inner, textvariable=var
                ).grid(row=row, column=1, columnspan=3, sticky=tk.EW, pady=2)
                ttk.Button(
                    self._test_params_inner, text="Browse...",
                    command=lambda v=var: self._browse_file(v)
                ).grid(row=row, column=4, sticky=tk.W, padx=(5, 0), pady=2)
                self._param_vars[ctrl_id] = var
                row += 1
                side = "left"

            # --- Narrow controls: pair two-per-row ---
            elif ctrl_type == "combo":
                lbl_col = 0 if side == "left" else 3
                wgt_col = 1 if side == "left" else 4
                ttk.Label(self._test_params_inner, text=f"{label}:").grid(
                    row=row, column=lbl_col, sticky=tk.W, padx=(0, 5), pady=2
                )
                var = tk.StringVar(value=ctrl.get("default", ""))
                combo = ttk.Combobox(
                    self._test_params_inner, textvariable=var, state="readonly",
                    values=ctrl.get("options", [])
                )
                combo.grid(row=row, column=wgt_col, sticky=tk.EW, pady=2)
                self._param_vars[ctrl_id] = var
                if side == "left":
                    side = "right"
                else:
                    row += 1
                    side = "left"

            elif ctrl_type == "check":
                lbl_col = 0 if side == "left" else 3
                wgt_col = 1 if side == "left" else 4
                var = tk.BooleanVar(value=ctrl.get("default", False))
                ttk.Checkbutton(
                    self._test_params_inner, text=label, variable=var
                ).grid(row=row, column=lbl_col, columnspan=2 if side == "left" else 2,
                       sticky=tk.W, pady=2)
                self._param_vars[ctrl_id] = var
                if side == "left":
                    side = "right"
                else:
                    row += 1
                    side = "left"

            elif ctrl_type == "entry":
                lbl_col = 0 if side == "left" else 3
                wgt_col = 1 if side == "left" else 4
                ttk.Label(self._test_params_inner, text=f"{label}:").grid(
                    row=row, column=lbl_col, sticky=tk.W, padx=(0, 5), pady=2
                )
                var = tk.StringVar(value=ctrl.get("default", ""))
                ttk.Entry(
                    self._test_params_inner, textvariable=var,
                    width=ctrl.get("width", 10)
                ).grid(row=row, column=wgt_col, sticky=tk.EW, pady=2)
                self._param_vars[ctrl_id] = var
                if side == "left":
                    side = "right"
                else:
                    row += 1
                    side = "left"

            elif ctrl_type == "spin":
                lbl_col = 0 if side == "left" else 3
                wgt_col = 1 if side == "left" else 4
                ttk.Label(self._test_params_inner, text=f"{label}:").grid(
                    row=row, column=lbl_col, sticky=tk.W, padx=(0, 5), pady=2
                )
                var = tk.StringVar(value=ctrl.get("default", "0"))
                spin_kw = dict(
                    textvariable=var,
                    from_=ctrl.get("from_", 0),
                    to=ctrl.get("to", 100),
                    increment=ctrl.get("increment", 1),
                    width=ctrl.get("width", 8),
                )
                fmt = ctrl.get("format", "")
                if fmt:
                    spin_kw["format"] = fmt
                ttk.Spinbox(
                    self._test_params_inner, **spin_kw
                ).grid(row=row, column=wgt_col, sticky=tk.EW, pady=2)
                self._param_vars[ctrl_id] = var
                if side == "left":
                    side = "right"
                else:
                    row += 1
                    side = "left"

        # Flush final row if last narrow control was on the left
        if side == "right":
            row += 1

    def _browse_file(self, var: tk.StringVar):
        path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg"), ("All files", "*.*")]
        )
        if path:
            var.set(path)

    def _apply_preset(self, model_id: str):
        """Fill model parameter controls from the selected preset."""
        preset_name = self._preset_var.get()
        if preset_name == "(custom)":
            return

        presets = MODEL_PRESETS.get(model_id, [])
        preset = next((p for p in presets if p["name"] == preset_name), None)
        if not preset:
            return

        # First reset all controls to MODEL_PARAMS defaults
        config = MODEL_PARAMS.get(model_id, {})
        defaults = {c["id"]: c.get("default", "")
                    for c in config.get("controls", [])
                    if c["type"] != "heading"}
        for ctrl_id, default in defaults.items():
            self._set_param_value(ctrl_id, default)

        # Then apply preset overrides
        for ctrl_id, value in preset["params"].items():
            self._set_param_value(ctrl_id, value)

    def _set_param_value(self, ctrl_id: str, value):
        """Set a single model parameter control to a given value."""
        var = self._param_vars.get(ctrl_id)
        if var is None:
            return
        if isinstance(var, scrolledtext.ScrolledText):
            var.delete("1.0", tk.END)
            var.insert("1.0", str(value))
        elif isinstance(var, tk.BooleanVar):
            var.set(bool(value))
        elif isinstance(var, tk.StringVar):
            var.set(str(value))

    def _collect_model_params(self) -> dict:
        """Collect all current model parameter values."""
        params = {}
        for ctrl_id, var in self._param_vars.items():
            if isinstance(var, scrolledtext.ScrolledText):
                params[ctrl_id] = var.get("1.0", tk.END).strip()
            elif isinstance(var, tk.BooleanVar):
                params[ctrl_id] = var.get()
            elif isinstance(var, tk.StringVar):
                params[ctrl_id] = var.get()
        return params

    # ================================================================
    # Send Request
    # ================================================================
    def _send_test_request(self):
        sel = self.test_workers_tree.selection()
        if not sel:
            messagebox.showwarning("No Worker", "Select a loaded worker from the list.")
            return

        values = self.test_workers_tree.item(sel[0], "values")
        model = values[1]

        model_params = self._collect_model_params()

        # Build the request
        prompt = model_params.pop("prompt", model_params.pop("caption", ""))
        lyrics = model_params.pop("lyrics", None)

        try:
            duration = float(model_params.pop("duration", "30"))
            seed = int(float(model_params.pop("seed", "-1")))
            request_data = {
                "prompt": prompt,
                "duration": duration,
                "seed": seed,
                "output_format": self.test_format_var.get(),
                "denoise_strength": float(self.test_denoise_var.get()),
                "stereo_width": float(self.test_stereo_var.get()),
                "eq_preset": self.test_eq_var.get(),
                "target_lufs": float(self.test_lufs_var.get()),
                "highpass_cutoff": int(float(self.test_highpass_var.get())),
                "compression_ratio": float(self.test_compression_var.get()),
                "clipping": float(self.test_clipping_var.get()),
                "trim_db": float(self.test_trim_db_var.get()),
                "min_silence_ms": int(float(self.test_min_silence_var.get())),
                "front_protect_ms": int(float(self.test_front_protect_var.get())),
                "end_protect_ms": int(float(self.test_end_protect_var.get())),
                "skip_post_process": self.test_skip_pp_var.get(),
                "model_params": model_params,
            }
        except ValueError as e:
            messagebox.showerror("Invalid Parameter", f"Check numeric parameters: {e}")
            return

        if lyrics:
            request_data["lyrics"] = lyrics

        ref_audio = model_params.pop("reference_audio", None)
        if ref_audio:
            request_data["reference_audio"] = ref_audio

        tags = model_params.pop("tags", None)
        if tags:
            request_data["tags"] = tags

        instrumental = model_params.pop("instrumental", False)
        if instrumental:
            request_data["instrumental"] = True

        self.test_send_btn.config(state=tk.DISABLED, text="Generating...")
        self.log(f"Sending music request: model={model}, prompt='{prompt[:50]}...'")

        def do_request():
            import time as _time
            start = _time.time()
            try:
                resp = requests.post(
                    self._api_url(f"/api/music/{model}"),
                    json=request_data, timeout=900
                )
                elapsed = _time.time() - start
                if resp.status_code == 200:
                    data = resp.json()
                    self.root.after(0, lambda: self._handle_test_response(data, elapsed, model))
                else:
                    detail = self._extract_error_detail(resp)
                    self.root.after(0, lambda: self._handle_test_error(
                        f"HTTP {resp.status_code}: {detail}", model
                    ))
            except requests.Timeout:
                self.root.after(0, lambda: self._handle_test_error("Request timed out (900s)", model))
            except requests.ConnectionError:
                self.root.after(0, lambda: self._handle_test_error("Connection failed. Is the server running?", model))
            except Exception as e:
                self.root.after(0, lambda: self._handle_test_error(str(e), model))

        threading.Thread(target=do_request, daemon=True).start()

    # ================================================================
    # Response Handling
    # ================================================================
    def _handle_test_response(self, data: dict, elapsed: float, model: str):
        self.test_send_btn.config(state=tk.NORMAL, text="Generate Music")

        duration = data.get("duration_sec", 0)
        sr = data.get("sample_rate", 0)
        fmt = data.get("format", "wav")
        entry_id = data.get("entry_id")

        self.test_status_label.config(
            text=f"Last: {model} - {duration:.1f}s in {elapsed:.1f}s",
            foreground="green",
        )
        self.log(f"Music: {model} - {duration:.2f}s audio in {elapsed:.2f}s", tag="success")

        # Detect worker device for CLAP auto-start
        worker_device = None
        sel = self.test_workers_tree.selection()
        if sel:
            values = self.test_workers_tree.item(sel[0], "values")
            if values and len(values) >= 4:
                worker_device = values[3]  # device column

        # Auto-start CLAP on same device if Auto mode + first generation
        if self._clap_auto_var.get() and not self._clap_running:
            self._auto_start_clap_if_needed(worker_device)

        # Add to outputs treeview (entry was auto-saved by gateway)
        if entry_id:
            entry = self._output_mgr.get_entry(entry_id)
            if not entry:
                # Gateway saved it but our local manager doesn't know yet — reload
                self._output_mgr._load_existing()
                entry = self._output_mgr.get_entry(entry_id)
            if entry:
                self._insert_output_row(entry)

                # Auto-score with CLAP if running
                if self._clap_running:
                    self._ctx_score_with_clap(entry_id)

    def _insert_output_row(self, entry):
        """Insert a single output entry at the top of the outputs treeview."""
        prompt_short = entry.prompt[:50] + "..." if len(entry.prompt) > 50 else entry.prompt
        clap_str = f"{entry.clap_score:.3f}" if entry.clap_score is not None else "--"
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(entry.timestamp)
            date_str = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            date_str = entry.timestamp[:16]
        # Insert at top (index 0)
        num = len(self.outputs_tree.get_children()) + 1
        if self.outputs_tree.exists(entry.id):
            return  # already present
        self.outputs_tree.insert("", 0, iid=entry.id, values=(
            num, entry.model, prompt_short,
            f"{entry.duration_sec:.1f}s", entry.format,
            clap_str, date_str,
        ))

    def _handle_test_error(self, msg: str, model: str = ""):
        self.test_send_btn.config(state=tk.NORMAL, text="Generate Music")
        self.test_status_label.config(
            text=f"Error: {msg[:60]}", foreground="red",
        )
        self.log(f"Music request failed: {msg}", tag="error")

    # ================================================================
    # Audio Playback
    # ================================================================
    def _stop_audio(self):
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

    # ================================================================
    # API Server Tab Actions
    # ================================================================
    def _update_device_combos(self, devices: list):
        self._cached_devices = devices
        device_ids = [d["id"] for d in devices]
        if not device_ids:
            device_ids = ["cuda:0", "cpu"]
        self.spawn_device_combo.config(values=device_ids)
        if not self.spawn_device_var.get() and device_ids:
            self.spawn_device_var.set(device_ids[0])

    def _refresh_workers(self):
        def do_fetch():
            data = self._api_get("/api/workers")
            if data and "workers" in data:
                self.root.after(0, lambda: self._update_workers_display(data["workers"]))
            else:
                self.root.after(0, lambda: self.log("Could not fetch workers.", tag="warning"))

        threading.Thread(target=do_fetch, daemon=True).start()

    def _update_workers_display(self, workers: list):
        self._update_tree_inplace(self.workers_tree, workers)

    def _update_tree_inplace(self, tree: ttk.Treeview, workers: list):
        incoming_ids = set()
        for w in workers:
            wid = w["worker_id"]
            incoming_ids.add(wid)
            values = (wid, w["model"], w["port"], w["device"], w["status"])
            if tree.exists(wid):
                tree.item(wid, values=values)
            else:
                tree.insert("", tk.END, iid=wid, values=values)
        for existing_id in tree.get_children():
            if existing_id not in incoming_ids:
                tree.delete(existing_id)

    def _kill_selected_worker(self):
        sel = self.workers_tree.selection()
        if not sel:
            messagebox.showinfo("No Selection", "Select a worker to kill.")
            return
        worker_id = sel[0]
        if not messagebox.askyesno("Confirm Kill", f"Kill worker '{worker_id}'?"):
            return

        def do_kill():
            result = self._api_delete(f"/api/workers/{worker_id}")
            if result:
                self.root.after(0, lambda: self.log(f"Worker '{worker_id}' killed.", tag="success"))
                self.root.after(500, self._refresh_workers)
            else:
                self.root.after(0, lambda: self.log(f"Failed to kill worker '{worker_id}'.", tag="error"))

        threading.Thread(target=do_kill, daemon=True).start()

    def _kill_all_workers(self):
        children = self.workers_tree.get_children()
        if not children:
            messagebox.showinfo("No Workers", "No workers to kill.")
            return
        if not messagebox.askyesno("Confirm", f"Kill all {len(children)} workers?"):
            return

        self.log("Killing all workers...")

        def do_kill_all():
            try:
                resp = requests.get(self._api_url("/api/workers"), timeout=3)
                if resp.status_code == 200:
                    for w in resp.json().get("workers", []):
                        try:
                            requests.delete(
                                self._api_url(f"/api/workers/{w['worker_id']}"),
                                timeout=10
                            )
                        except Exception:
                            pass
            except Exception:
                pass
            # Auto-stop CLAP when all workers killed
            if self._clap_auto_var.get() and self._clap_running:
                self._api_post("/api/clap/stop", timeout=30)
                self.root.after(0, lambda: self._on_clap_stopped())
            self.root.after(0, lambda: self.log("All workers killed.", tag="success"))

        threading.Thread(target=do_kill_all, daemon=True).start()

    def _spawn_worker(self):
        model = self.spawn_model_var.get()
        device = self.spawn_device_var.get()
        if not model:
            messagebox.showwarning("No Model", "Select a model to spawn.")
            return

        # Check if model environment is installed before trying to spawn
        try:
            status = self._install_mgr.get_model_status(model)
            if not status.get("env_installed"):
                messagebox.showwarning(
                    "Not Installed",
                    f"Model '{model}' is not installed.\n\n"
                    "Install the environment first from the Setup tab.",
                )
                return
        except Exception:
            pass

        self.log(f"Spawning {model} worker on {device}...")

        def do_spawn():
            result = self._api_post("/api/workers/spawn", {"model": model, "device": device}, timeout=960)
            if result:
                wid = result.get("worker_id", "unknown")
                self.root.after(0, lambda: self.log(f"Worker '{wid}' spawned on {device}.", tag="success"))
                self.root.after(1000, self._refresh_workers)
            else:
                self.root.after(0, lambda: self.log(f"Failed to spawn {model} worker.", tag="error"))

        threading.Thread(target=do_spawn, daemon=True).start()

    # ================================================================
    # Model Status & Installation
    # ================================================================
    def _check_model_status(self, model_id: str) -> tuple:
        info = self._install_mgr.get_model_status(model_id)
        status = info["status"]
        label_map = {
            "ready": "Ready",
            "env_installed": "Weights needed",
            "weights_only": "Env needed",
            "not_installed": "Not installed",
        }
        # Pass specific status through so button logic can distinguish states
        return (status, label_map.get(status, status))

    def _update_setup_row(self, model_id: str, status: str, label: str):
        w = self.setup_widgets.get(model_id)
        if not w:
            return

        colors = {
            "ready": "#32CD32",
            "partial": "#FFA500",
            "env_installed": "#FFA500",
            "weights_only": "#FFA500",
            "not_installed": "gray",
            "installing": "#FFD700",
            "error": "#FF4444",
            "cancelled": "#FFA500",
        }
        color = colors.get(status, "gray")
        w["canvas"].itemconfig(w["circle"], fill=color)
        w["status_label"].config(text=label)

        has_weights_repo = bool(MODEL_SETUP.get(model_id, {}).get("weights_repo"))

        # Button visibility: install_btn, download_btn, cancel_btn swap places
        if status == "installing":
            w["install_btn"].pack_forget()
            w["download_btn"].pack_forget()
            w["cancel_btn"].pack(side=tk.LEFT, padx=2, before=w["remove_btn"])
            w["remove_btn"].config(state=tk.DISABLED)
        else:
            w["cancel_btn"].pack_forget()
            # Show Install or Retry depending on status
            btn_text = "Retry" if status in ("error", "cancelled") else "Install"
            w["install_btn"].config(text=btn_text, state=tk.NORMAL)
            w["install_btn"].pack(side=tk.LEFT, padx=2, before=w["remove_btn"])

            # Download button: show when env installed but weights missing
            if has_weights_repo and status == "env_installed":
                w["download_btn"].pack(side=tk.LEFT, padx=2, before=w["remove_btn"])
            else:
                w["download_btn"].pack_forget()

            # Enable Remove if anything exists on disk (venv or weights),
            # even if logical status is "not_installed" (partial/failed install)
            info = MODEL_SETUP.get(model_id, {})
            env = self.environments.get(info.get("env", ""))
            has_files = (env and env.venv_path.exists())
            if not has_files and info.get("weights_dir"):
                has_files = (MODELS_DIR / info["weights_dir"]).exists()
            w["remove_btn"].config(
                state=tk.NORMAL if has_files else tk.DISABLED
            )

    def _refresh_setup_status(self):
        self.log("Checking model status...")

        def do_check():
            results = {}
            for model_id in MODEL_SETUP:
                results[model_id] = self._check_model_status(model_id)
            self.root.after(0, lambda: self._apply_setup_status(results))

        threading.Thread(target=do_check, daemon=True).start()

    def _apply_setup_status(self, results: dict):
        for model_id, (status, label) in results.items():
            self._update_setup_row(model_id, status, label)
        self.log("Status check complete.")

    def _install_model(self, model_id: str):
        info = MODEL_SETUP[model_id]

        if info.get("gated"):
            if not messagebox.askyesno(
                "Gated Model",
                f"'{info['display']}' requires HuggingFace authentication.\n\n"
                "1. Accept the terms at the model page\n"
                "2. Run 'huggingface-cli login' in a terminal\n\n"
                f"Repo: {info['weights_repo']}\n\nHave you completed these steps?"
            ):
                return

        try:
            job_id = self._install_mgr.start_install(model_id)
        except RuntimeError as e:
            messagebox.showwarning("Installation in Progress", str(e))
            return

        # Reset card state
        w = self.setup_widgets.get(model_id)
        if w:
            w["progress"]["value"] = 0
            w["step_label"].config(text="Starting...")
            w["error_label"].pack_forget()
            w["log_text"].config(state=tk.NORMAL)
            w["log_text"].delete("1.0", tk.END)
            w["log_text"].config(state=tk.DISABLED)

        self._log_offsets[job_id] = 0
        self._update_setup_row(model_id, "installing", "Installing...")
        self._expand_card(model_id)
        self._poll_install_job(model_id, job_id)

    def _poll_install_job(self, model_id: str, job_id: str):
        """Poll InstallManager for job status, update GUI accordingly."""
        job = self._install_mgr.get_job(job_id)
        if not job:
            return

        w = self.setup_widgets.get(model_id)

        if job["status"] in ("pending", "running"):
            step = job["current_step"] or "Starting..."
            pct = job["progress_pct"]
            step_idx = job.get("step_index", 0)
            step_total = job.get("step_total", 0)

            # Update progress bar and step label
            if w:
                w["progress"]["value"] = pct
                if step_total > 0:
                    w["step_label"].config(
                        text=f"Step {step_idx}/{step_total}: {step}"
                    )
                else:
                    w["step_label"].config(text=step)

            # Fetch new log lines incrementally
            offset = self._log_offsets.get(job_id, 0)
            logs = self._install_mgr.get_job_logs(job_id, offset=offset)
            if logs and logs["lines"]:
                self._log_offsets[job_id] = logs["total_lines"]
                self._append_card_log(model_id, logs["lines"])

            # Update status label
            if step_total > 0:
                self._update_setup_row(
                    model_id, "installing",
                    f"Step {step_idx}/{step_total} ({pct:.0f}%)"
                )
            else:
                self._update_setup_row(model_id, "installing", f"{step} ({pct:.0f}%)")

            self.root.after(1000, lambda: self._poll_install_job(model_id, job_id))

        elif job["status"] == "completed":
            if w:
                w["progress"]["value"] = 100
                w["step_label"].config(text="Complete")
            # Fetch any remaining log lines
            offset = self._log_offsets.get(job_id, 0)
            logs = self._install_mgr.get_job_logs(job_id, offset=offset)
            if logs and logs["lines"]:
                self._append_card_log(model_id, logs["lines"])
            self._collapse_card(model_id)
            status, label = self._check_model_status(model_id)
            self._update_setup_row(model_id, status, label)

        elif job["status"] == "cancelled":
            if w:
                w["step_label"].config(text="Cancelled by user")
            self._update_setup_row(model_id, "cancelled", "Cancelled")

        else:  # failed
            error = job.get("error", "Unknown error")
            if w:
                w["error_label"].config(text=f"ERROR: {error}")
                w["error_label"].pack(anchor=tk.W)
                w["step_label"].config(text="Failed")
            # Fetch remaining log lines
            offset = self._log_offsets.get(job_id, 0)
            logs = self._install_mgr.get_job_logs(job_id, offset=offset)
            if logs and logs["lines"]:
                self._append_card_log(model_id, logs["lines"])
            self._expand_card(model_id)
            self._update_setup_row(model_id, "error", "Error")

    def _remove_model(self, model_id: str):
        info = MODEL_SETUP[model_id]
        status_info = self._install_mgr.get_model_status(model_id)

        if status_info["status"] == "not_installed":
            messagebox.showinfo("Nothing to Remove", f"{info['display']} is not installed.")
            return

        env = self.environments.get(info["env"])
        has_env = env and env.venv_path.exists()
        has_weights = status_info["weights_installed"] and info.get("weights_dir")

        shared_models = [m for m, i in MODEL_SETUP.items()
                         if i["env"] == info["env"] and m != model_id]
        env_note = ""
        if shared_models and has_env:
            env_note = (f"\n\nNote: env '{info['env']}' is shared with: "
                        f"{', '.join(shared_models)}.\nRemoving it will affect those models too.")

        parts = []
        if has_env:
            parts.append(f"Environment: {env.venv_path}")
        if has_weights:
            parts.append(f"Weights: {MODELS_DIR / info['weights_dir']}")

        if not messagebox.askyesno(
            "Confirm Remove",
            f"Remove {info['display']}?\n\n" + "\n".join(parts) + env_note
        ):
            return

        try:
            job_id = self._install_mgr.start_uninstall(model_id)
        except RuntimeError as e:
            messagebox.showwarning("Operation in Progress", str(e))
            return

        self._update_setup_row(model_id, "installing", "Removing...")
        self._poll_install_job(model_id, job_id)

    def _install_all_models(self):
        if self._install_all_running:
            messagebox.showinfo("In Progress", "Install All is already running.")
            return

        statuses = self._install_mgr.get_all_model_status()
        to_install = [mid for mid, info in statuses.items() if info["status"] != "ready"]

        if not to_install:
            messagebox.showinfo("All Ready", "All models are already installed!")
            return

        names = "\n".join(f"- {MODEL_SETUP[m]['display']}" for m in to_install)
        if not messagebox.askyesno("Install All", f"Install {len(to_install)} models?\n\n{names}"):
            return

        self._install_all_cancel = False
        self._install_all_running = True
        self._install_all_btn.config(state=tk.DISABLED)
        self._install_all_frame.pack(fill=tk.X, pady=(5, 5), before=self._setup_canvas.master)

        def do_install_all():
            import time as _time
            succeeded, failed = [], []

            for i, model_id in enumerate(to_install):
                if self._install_all_cancel:
                    break

                display = MODEL_SETUP[model_id]["display"]
                # Update overall progress label on main thread
                self.root.after(0, lambda idx=i, name=display, total=len(to_install):
                    self._install_all_progress_label.config(
                        text=f"Installing {idx + 1}/{total}: {name}..."
                    ))

                try:
                    job_id = self._install_mgr.start_install(model_id)
                except RuntimeError:
                    continue  # already has an active job

                # Reset card and start polling on main thread
                self.root.after(0, lambda m=model_id, jid=job_id: self._start_install_card(m, jid))

                # Wait for job to finish in background
                while True:
                    if self._install_all_cancel:
                        self._install_mgr.cancel_install(model_id)
                        break
                    job = self._install_mgr.get_job(job_id)
                    if job and job["status"] in ("completed", "failed", "cancelled"):
                        break
                    _time.sleep(2)

                job = self._install_mgr.get_job(job_id)
                if job and job["status"] == "completed":
                    succeeded.append(model_id)
                else:
                    error = job.get("error", "") if job else "cancelled"
                    failed.append((model_id, error))

                # Update status on main thread
                self.root.after(0, lambda m=model_id: self._refresh_single_status(m))

            # Show summary on main thread
            self.root.after(0, lambda: self._install_all_complete(succeeded, failed))

        threading.Thread(target=do_install_all, daemon=True).start()

    def _start_install_card(self, model_id: str, job_id: str):
        """Initialize card UI for an install and start polling. Called from main thread."""
        w = self.setup_widgets.get(model_id)
        if w:
            w["progress"]["value"] = 0
            w["step_label"].config(text="Starting...")
            w["error_label"].pack_forget()
            w["log_text"].config(state=tk.NORMAL)
            w["log_text"].delete("1.0", tk.END)
            w["log_text"].config(state=tk.DISABLED)

        self._log_offsets[job_id] = 0
        self._update_setup_row(model_id, "installing", "Installing...")
        self._expand_card(model_id)
        self._poll_install_job(model_id, job_id)

    def _cancel_install_all(self):
        self._install_all_cancel = True
        self.log("Cancelling Install All...")

    def _install_all_complete(self, succeeded: list, failed: list):
        self._install_all_running = False
        self._install_all_frame.pack_forget()
        self._install_all_btn.config(state=tk.NORMAL)

        # Build summary
        total = len(succeeded) + len(failed)
        if not failed:
            msg = f"All {total} models installed successfully!"
        elif not succeeded:
            names = "\n".join(f"- {MODEL_SETUP[m]['display']}: {err or 'cancelled'}"
                              for m, err in failed)
            msg = f"All {total} installs failed:\n\n{names}"
        else:
            ok_names = ", ".join(MODEL_SETUP[m]["display"] for m in succeeded)
            fail_names = "\n".join(f"- {MODEL_SETUP[m]['display']}: {err or 'cancelled'}"
                                   for m, err in failed)
            msg = (f"{len(succeeded)}/{total} installed successfully.\n\n"
                   f"Succeeded: {ok_names}\n\nFailed:\n{fail_names}")

        messagebox.showinfo("Install All Complete", msg)
        self.log(f"Install All finished: {len(succeeded)} succeeded, {len(failed)} failed.")

    def _refresh_single_status(self, model_id: str):
        """Refresh status for a single model in the GUI."""
        status, label = self._check_model_status(model_id)
        self._update_setup_row(model_id, status, label)

    # ================================================================
    # Gateway Management
    # ================================================================
    def _kill_port_occupant(self, port: int):
        """Kill any of OUR processes currently listening on the given port.

        Skips PIDs that previously failed to be killed (system services, etc.).
        """
        try:
            result = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return

            my_pid = os.getpid()
            target = f":{port} "
            killed = set()
            for line in result.stdout.splitlines():
                if target in line and "LISTENING" in line:
                    parts = line.split()
                    try:
                        pid = int(parts[-1])
                    except (ValueError, IndexError):
                        continue
                    if pid in killed or pid == my_pid or pid == 0:
                        continue
                    if pid in self._unkillable_pids:
                        continue
                    self.log(f"Killing process on port {port} (PID {pid})...")
                    try:
                        kill_result = subprocess.run(
                            ["taskkill", "/PID", str(pid), "/T", "/F"],
                            capture_output=True, text=True, timeout=10,
                        )
                        if kill_result.returncode != 0:
                            msg = (kill_result.stderr or kill_result.stdout).strip()
                            self.log(f"  Could not kill PID {pid} (skipping in future): {msg}", tag="warning")
                            self._unkillable_pids.add(pid)
                    except Exception as e:
                        self.log(f"  Could not kill PID {pid}: {e}", tag="warning")
                        self._unkillable_pids.add(pid)
                    killed.add(pid)
        except Exception:
            pass

    def _start_gateway(self):
        port = self.port_var.get()
        try:
            port = int(port)
        except ValueError:
            port = DEFAULT_API_PORT

        self._gateway_port = port
        server_python = str(PYTHON_PATH)

        if not os.path.exists(server_python):
            self.log(f"Embedded Python not found at: {server_python}", tag="error")
            return

        # Kill any orphaned gateway on the port (skips known system services)
        self._kill_port_occupant(port)

        self.log(f"Starting gateway on port {port}...")

        try:
            self._gateway_process = subprocess.Popen(
                [server_python, "-m", "uvicorn", "music_api_server:app",
                 "--host", "0.0.0.0", "--port", str(port)],
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )

            thread = threading.Thread(target=self._monitor_gateway, daemon=True)
            thread.start()

            self.log(f"Gateway starting (PID {self._gateway_process.pid})...")

            # Verify process is still alive after a short delay
            self.root.after(3000, self._verify_gateway_started)
        except Exception as e:
            self.log(f"Failed to start gateway: {e}", tag="error")

    def _verify_gateway_started(self):
        """Check that the gateway subprocess didn't crash on startup."""
        proc = self._gateway_process
        if proc is None:
            return
        retcode = proc.poll()
        if retcode is not None:
            self.log(
                f"Gateway process exited immediately (code {retcode}). "
                "Check the Log tab for [Gateway] error messages.",
                tag="error",
            )
            self._gateway_process = None

    def _stop_gateway(self):
        self._stop_server_polling()

        proc = self._gateway_process
        if proc:
            pid = proc.pid
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True, timeout=10,
                )
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            # Wait for the process to actually terminate so the port is freed
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
            self._gateway_process = None

        # Also kill anything still on the port (orphaned children)
        self._kill_port_occupant(self._gateway_port)

        self.server_status_canvas.itemconfig(self.server_status_circle, fill="gray")
        self.server_status_label.config(text="Gateway: stopped")

    def _restart_gateway(self):
        if self._restarting_gateway:
            self.log("Restart already in progress...")
            return
        self._restarting_gateway = True
        self.log("Restarting gateway...")
        self._stop_gateway()
        self.root.after(1500, self._finish_restart_gateway)

    def _finish_restart_gateway(self):
        self._restarting_gateway = False
        self._start_gateway()
        self._start_server_polling()

    def _monitor_gateway(self):
        proc = self._gateway_process
        if not proc or not proc.stdout:
            return
        try:
            for line in proc.stdout:
                if self._gateway_process is not proc:
                    break
                line = line.strip()
                if line:
                    self.log(f"[Gateway] {line}", tag="gateway")
        except Exception:
            pass

        # Process exited — log the exit code
        if self._gateway_process is proc:
            retcode = proc.poll()
            if retcode is not None and retcode != 0:
                self.log(
                    f"[Gateway] Process exited with code {retcode}",
                    tag="error",
                )

    # ================================================================
    # Worker Cleanup
    # ================================================================
    def _kill_all_workers_sync(self):
        try:
            resp = requests.get(self._api_url("/api/workers"), timeout=3)
            if resp.status_code == 200:
                workers = resp.json().get("workers", [])
                for w in workers:
                    try:
                        requests.delete(
                            self._api_url(f"/api/workers/{w['worker_id']}"),
                            timeout=5
                        )
                    except Exception:
                        pass
        except Exception:
            pass

    # ================================================================
    # Local Device Detection
    # ================================================================
    def _detect_local_devices(self):
        devices = []
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        device_id = f"cuda:{parts[0]}"
                        devices.append({
                            "id": device_id,
                            "name": parts[1],
                            "vram_total_mb": int(float(parts[2])),
                            "vram_free_mb": int(float(parts[3])),
                            "workers": [],
                        })
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        devices.append({
            "id": "cpu", "name": "CPU",
            "vram_total_mb": 0, "vram_free_mb": 0, "workers": [],
        })

        self._update_device_combos(devices)

    # ================================================================
    # Logging
    # ================================================================
    def log(self, message: str, tag: str = "info"):
        self.log_queue.put((message, tag))

    def _process_log_queue(self):
        try:
            while True:
                item = self.log_queue.get_nowait()
                if isinstance(item, tuple):
                    message, tag = item
                else:
                    message, tag = item, "info"

                self._log_entries.append((message, tag))

                filter_var = self._log_filters.get(tag, self._log_filters.get("info"))
                if filter_var and filter_var.get():
                    self._append_log_line(message, tag)
        except queue.Empty:
            pass

        self._log_poll_id = self.root.after(100, self._process_log_queue)

    def _append_log_line(self, message: str, tag: str):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n", tag)
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > 1000:
            self.log_text.delete("1.0", f"{line_count - 1000}.0")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _rerender_log(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        for message, tag in self._log_entries:
            filter_var = self._log_filters.get(tag, self._log_filters.get("info"))
            if filter_var and filter_var.get():
                self.log_text.insert(tk.END, message + "\n", tag)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _clear_log(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)
        self._log_entries.clear()

    # ================================================================
    # API Helpers
    # ================================================================
    def _api_url(self, path: str) -> str:
        return f"http://127.0.0.1:{self._gateway_port}{path}"

    def _api_get(self, path: str, timeout: int = 10):
        try:
            resp = requests.get(self._api_url(path), timeout=timeout)
            if resp.status_code >= 400:
                detail = self._extract_error_detail(resp)
                self.log(f"API GET {path} -> {resp.status_code}: {detail}", tag="error")
                return None
            return resp.json()
        except Exception as e:
            self.log(f"API GET {path} failed: {e}", tag="error")
            return None

    def _api_post(self, path: str, json_body=None, timeout: int = 300):
        try:
            resp = requests.post(self._api_url(path), json=json_body, timeout=timeout)
            if resp.status_code >= 400:
                detail = self._extract_error_detail(resp)
                self.log(f"API POST {path} -> {resp.status_code}: {detail}", tag="error")
                return None
            return resp.json()
        except Exception as e:
            self.log(f"API POST {path} failed: {e}", tag="error")
            return None

    def _api_delete(self, path: str, timeout: int = 30):
        try:
            resp = requests.delete(self._api_url(path), timeout=timeout)
            if resp.status_code >= 400:
                detail = self._extract_error_detail(resp)
                self.log(f"API DELETE {path} -> {resp.status_code}: {detail}", tag="error")
                return None
            return resp.json()
        except Exception as e:
            self.log(f"API DELETE {path} failed: {e}", tag="error")
            return None

    @staticmethod
    def _extract_error_detail(resp) -> str:
        try:
            data = resp.json()
            return data.get("detail", resp.text[:300])
        except Exception:
            return resp.text[:300]

    # ================================================================
    # Server Polling
    # ================================================================
    def _start_server_polling(self):
        self._stop_server_polling()
        self._poll_server_status()

    def _stop_server_polling(self):
        if self._server_poll_id is not None:
            self.root.after_cancel(self._server_poll_id)
            self._server_poll_id = None

    def _poll_server_status(self):
        def do_poll():
            results = {}
            try:
                resp = requests.get(self._api_url("/health"), timeout=3)
                results["healthy"] = resp.status_code == 200
            except Exception:
                results["healthy"] = False

            if results["healthy"]:
                try:
                    resp = requests.get(self._api_url("/api/workers"), timeout=5)
                    if resp.status_code == 200:
                        results["workers"] = resp.json()
                except Exception:
                    pass

            try:
                self.root.after(0, lambda: self._handle_poll_results(results))
            except RuntimeError:
                pass

        threading.Thread(target=do_poll, daemon=True).start()
        self._server_poll_id = self.root.after(10000, self._poll_server_status)

    def _handle_poll_results(self, results: dict):
        if results.get("healthy"):
            self.server_status_canvas.itemconfig(self.server_status_circle, fill="#32CD32")
            self.server_status_label.config(text=f"Gateway: connected (port {self._gateway_port})")
        else:
            self.server_status_canvas.itemconfig(self.server_status_circle, fill="gray")
            self.server_status_label.config(text="Gateway: not connected")

        if "workers" in results:
            workers_data = results["workers"]
            worker_list = workers_data.get("workers", []) if isinstance(workers_data, dict) else workers_data
            self._update_workers_display(worker_list)
            self._update_tree_inplace(self.test_workers_tree, worker_list)

    # ================================================================
    # Utilities
    # ================================================================
    def _open_venvs_folder(self):
        VENVS_DIR.mkdir(exist_ok=True)
        os.startfile(VENVS_DIR)

    def _shutdown_all(self):
        """Aggressively kill all app processes (workers, CLAP, gateway) and exit."""
        if not messagebox.askyesno(
            "Shutdown All",
            "This will forcefully kill all workers, CLAP scorer, "
            "gateway, and close the application.\n\nContinue?"
        ):
            return

        self.log("SHUTDOWN ALL — killing everything...")

        # 1. Cancel timers
        self._stop_server_polling()
        if self._log_poll_id is not None:
            self.root.after_cancel(self._log_poll_id)
            self._log_poll_id = None

        # 2. Stop audio playback
        self._stop_audio()

        # 3. Try graceful API-level kills first (best effort, quick timeouts)
        try:
            requests.post(self._api_url("/api/clap/stop"), timeout=2)
        except Exception:
            pass
        self._kill_all_workers_sync()

        # 4. Kill gateway process tree (taskkill /T /F kills children too)
        self._stop_gateway()

        # 5. Aggressively kill anything still on our ports
        from config import WORKER_PORT_MIN, WORKER_PORT_MAX, CLAP_PORT
        ports_to_clean = [self._gateway_port, CLAP_PORT]
        # Scan all worker ports that could be in use
        try:
            result = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                my_pid = os.getpid()
                pids_to_kill = set()
                for line in result.stdout.splitlines():
                    if "LISTENING" not in line:
                        continue
                    for port in range(WORKER_PORT_MIN, WORKER_PORT_MAX + 1):
                        if f":{port} " in line:
                            parts = line.split()
                            try:
                                pid = int(parts[-1])
                                if pid != my_pid and pid != 0:
                                    pids_to_kill.add(pid)
                            except (ValueError, IndexError):
                                pass
                    for port in ports_to_clean:
                        if f":{port} " in line:
                            parts = line.split()
                            try:
                                pid = int(parts[-1])
                                if pid != my_pid and pid != 0:
                                    pids_to_kill.add(pid)
                            except (ValueError, IndexError):
                                pass

                for pid in pids_to_kill:
                    try:
                        subprocess.run(
                            ["taskkill", "/PID", str(pid), "/T", "/F"],
                            capture_output=True, timeout=5,
                        )
                    except Exception:
                        pass
        except Exception:
            pass

        # 6. Destroy GUI and exit
        self.root.destroy()
        os._exit(0)

    def _on_close(self):
        # Cancel recurring timers first
        self._stop_server_polling()
        if self._log_poll_id is not None:
            self.root.after_cancel(self._log_poll_id)
            self._log_poll_id = None
        self._stop_audio()
        # Stop CLAP scorer if running
        if self._clap_running:
            try:
                requests.post(self._api_url("/api/clap/stop"), timeout=5)
            except Exception:
                pass
        self._kill_all_workers_sync()
        self._stop_gateway()
        self.root.destroy()


# ================================================================
# Main
# ================================================================
def main():
    if not os.path.exists(PYTHON_PATH):
        if USE_EMBEDDED:
            msg = (
                f"Embedded Python not found at:\n{PYTHON_PATH}\n\n"
                "Run install.bat to set up the embedded Python environment."
            )
        else:
            msg = (
                f"Python not found at:\n{PYTHON_PATH}\n\n"
                "Run install.bat to set up the embedded Python environment,\n"
                "or install Python 3.10+ on your system."
            )
        messagebox.showerror("Python Not Found", msg)
        return

    root = tk.Tk()

    try:
        root.iconbitmap(default="")
    except Exception:
        pass

    style = ttk.Style()
    style.theme_use("clam")

    app = MusicManagerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
