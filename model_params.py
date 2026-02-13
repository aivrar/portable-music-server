"""Shared model parameter definitions, presets, and display metadata.

Used by both the GUI (music_manager.py) and the API server (music_api_server.py).
No tkinter dependency — pure data only.
"""

# ---------------------------------------------------------------------------
# Display metadata (supplements MODEL_SETUP from install_manager)
# ---------------------------------------------------------------------------
MODEL_DISPLAY = {
    "ace_step_v15": {"desc": "High-quality music generation with lyrics",     "weights_size": None,    "vram": "<4GB"},
    "ace_step_v1":  {"desc": "Original ACE-Step music generation",            "weights_size": None,    "vram": "~8GB"},
    "heartmula":    {"desc": "Lyrics-to-music with RL optimization",         "weights_size": "~6GB",  "vram": "~16GB"},
    "diffrhythm":   {"desc": "Diffusion-based full-song generation",         "weights_size": None,    "vram": "8GB"},
    "yue":          {"desc": "Lyrics-to-song with chain-of-thought",         "weights_size": None,    "vram": "24GB+"},
    "musicgen":     {"desc": "Meta's text-to-music (AudioCraft)",            "weights_size": "~3GB",  "vram": "8-16GB"},
    "riffusion":    {"desc": "Stable Diffusion fine-tuned for music",        "weights_size": "~2GB",  "vram": "6-8GB"},
    "stable_audio": {"desc": "Stability AI's open-source audio generation",  "weights_size": "~2GB",  "vram": "~8GB"},
}


# ---------------------------------------------------------------------------
# Per-model parameter definitions for dynamic Testing tab / API
# ---------------------------------------------------------------------------
MODEL_PARAMS = {
    "ace_step_v15": {
        "display": "ACE-Step v1.5",
        "controls": [
            # --- Content ---
            {"id": "_h_content", "type": "heading", "label": "Content"},
            {"id": "caption", "type": "text", "label": "Style Description", "default": "", "rows": 2},
            {"id": "lyrics", "type": "text", "label": "Lyrics", "default": "", "rows": 5},

            # --- Music Metadata ---
            {"id": "_h_metadata", "type": "heading", "label": "Music Metadata"},
            {"id": "task_type", "type": "combo", "label": "Task",
             "options": ["text2music", "cover", "repaint"], "default": "text2music"},
            {"id": "vocal_language", "type": "combo", "label": "Language",
             "options": [
                 "unknown", "ar", "az", "bg", "bn", "ca", "cs", "da", "de",
                 "el", "en", "es", "fa", "fi", "fr", "he", "hi", "hr", "ht",
                 "hu", "id", "is", "it", "ja", "ko", "la", "lt", "ms", "ne",
                 "nl", "no", "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr",
                 "sv", "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "vi",
                 "yue", "zh",
             ], "default": "unknown"},
            {"id": "instrumental", "type": "check", "label": "Instrumental", "default": False},
            {"id": "keyscale", "type": "combo", "label": "Key",
             "options": [
                 "",
                 "C major", "C minor", "C# major", "C# minor",
                 "D major", "D minor", "D# major", "D# minor",
                 "Db major", "Db minor",
                 "E major", "E minor", "Eb major", "Eb minor",
                 "F major", "F minor", "F# major", "F# minor",
                 "G major", "G minor", "G# major", "G# minor",
                 "Gb major", "Gb minor",
                 "A major", "A minor", "A# major", "A# minor",
                 "Ab major", "Ab minor",
                 "B major", "B minor", "Bb major", "Bb minor",
             ], "default": ""},
            {"id": "timesignature", "type": "combo", "label": "Time Sig",
             "options": ["", "2", "3", "4", "6"], "default": ""},
            {"id": "bpm", "type": "spin", "label": "BPM (0=auto)",
             "default": "0", "from_": 0, "to": 300, "increment": 1, "width": 8},
            {"id": "duration", "type": "spin", "label": "Duration (s)",
             "default": "120", "from_": 10, "to": 600, "increment": 10, "width": 8},

            # --- Generation ---
            {"id": "_h_generation", "type": "heading", "label": "Generation"},
            {"id": "inference_steps", "type": "spin", "label": "Steps",
             "default": "8", "from_": 1, "to": 200, "increment": 1, "width": 8},
            {"id": "guidance_scale", "type": "spin", "label": "CFG Scale",
             "default": "7.0", "from_": 0.0, "to": 30.0, "increment": 0.5,
             "format": "%.1f", "width": 8},
            {"id": "shift", "type": "spin", "label": "Shift",
             "default": "3.0", "from_": 0.0, "to": 20.0, "increment": 0.1,
             "format": "%.1f", "width": 8},
            {"id": "infer_method", "type": "combo", "label": "Solver",
             "options": ["ode", "sde"], "default": "ode"},
            {"id": "seed", "type": "entry", "label": "Seed (-1=random)",
             "default": "-1", "width": 10},
            {"id": "batch_size", "type": "spin", "label": "Batch Size",
             "default": "1", "from_": 1, "to": 8, "increment": 1, "width": 6},

            # --- LLM (Chain-of-Thought) ---
            {"id": "_h_lm", "type": "heading", "label": "LLM (Chain-of-Thought)"},
            {"id": "thinking", "type": "check", "label": "Enable CoT Thinking",
             "default": True},
            {"id": "use_cot_metas", "type": "check", "label": "CoT Metadata",
             "default": True},
            {"id": "use_cot_caption", "type": "check", "label": "CoT Caption",
             "default": True},
            {"id": "use_cot_language", "type": "check", "label": "CoT Language",
             "default": True},
            {"id": "lm_temperature", "type": "spin", "label": "LM Temperature",
             "default": "0.85", "from_": 0.0, "to": 2.0, "increment": 0.05,
             "format": "%.2f", "width": 8},
            {"id": "lm_cfg_scale", "type": "spin", "label": "LM CFG Scale",
             "default": "2.0", "from_": 1.0, "to": 5.0, "increment": 0.1,
             "format": "%.1f", "width": 8},
            {"id": "lm_top_k", "type": "spin", "label": "LM Top-K (0=off)",
             "default": "0", "from_": 0, "to": 100, "increment": 1, "width": 8},
            {"id": "lm_top_p", "type": "spin", "label": "LM Top-P",
             "default": "0.9", "from_": 0.0, "to": 1.0, "increment": 0.05,
             "format": "%.2f", "width": 8},
            {"id": "lm_negative_prompt", "type": "text",
             "label": "LM Negative Prompt", "default": "NO USER INPUT", "rows": 2},

            # --- Advanced / Task-Specific ---
            {"id": "_h_advanced", "type": "heading", "label": "Advanced / Task-Specific"},
            {"id": "use_adg", "type": "check",
             "label": "Adaptive Dual Guidance", "default": False},
            {"id": "cfg_interval_start", "type": "spin", "label": "CFG Start",
             "default": "0.0", "from_": 0.0, "to": 1.0, "increment": 0.05,
             "format": "%.2f", "width": 8},
            {"id": "cfg_interval_end", "type": "spin", "label": "CFG End",
             "default": "1.0", "from_": 0.0, "to": 1.0, "increment": 0.05,
             "format": "%.2f", "width": 8},
            {"id": "audio_cover_strength", "type": "spin",
             "label": "Cover Strength", "default": "1.0", "from_": 0.0,
             "to": 1.0, "increment": 0.05, "format": "%.2f", "width": 8},
            {"id": "repainting_start", "type": "spin",
             "label": "Repaint Start (s)", "default": "0.0", "from_": 0.0,
             "to": 600.0, "increment": 1.0, "format": "%.1f", "width": 8},
            {"id": "repainting_end", "type": "spin",
             "label": "Repaint End (s, -1=end)", "default": "-1",
             "from_": -1, "to": 600.0, "increment": 1.0, "format": "%.1f",
             "width": 8},
            {"id": "reference_audio", "type": "file", "label": "Reference Audio"},
            {"id": "src_audio", "type": "file",
             "label": "Source Audio (cover/repaint)"},
        ],
    },
    "ace_step_v1": {
        "display": "ACE-Step v1",
        "controls": [
            # --- Content ---
            {"id": "_h_content", "type": "heading", "label": "Content"},
            {"id": "caption", "type": "text", "label": "Style Description", "default": "", "rows": 2},
            {"id": "lyrics", "type": "text", "label": "Lyrics", "default": "", "rows": 5},
            {"id": "instrumental", "type": "check", "label": "Instrumental", "default": False},

            # --- Generation ---
            {"id": "_h_generation", "type": "heading", "label": "Generation"},
            {"id": "duration", "type": "spin", "label": "Duration (s)",
             "default": "60", "from_": 1, "to": 240, "increment": 10, "width": 8},
            {"id": "inference_steps", "type": "spin", "label": "Diffusion Steps",
             "default": "60", "from_": 1, "to": 200, "increment": 1, "width": 8},
            {"id": "guidance_scale", "type": "spin", "label": "CFG Scale",
             "default": "15.0", "from_": 0.0, "to": 30.0, "increment": 0.5,
             "format": "%.1f", "width": 8},
            {"id": "scheduler_type", "type": "combo", "label": "Scheduler",
             "options": ["euler", "heun", "pingpong"], "default": "euler"},
            {"id": "cfg_type", "type": "combo", "label": "CFG Type",
             "options": ["apg", "cfg", "cfg_star"], "default": "apg"},
            {"id": "omega_scale", "type": "spin", "label": "Omega Scale",
             "default": "10.0", "from_": 0.0, "to": 30.0, "increment": 1.0,
             "format": "%.1f", "width": 8},
            {"id": "seed", "type": "entry", "label": "Seed (-1=random)",
             "default": "-1", "width": 10},
            {"id": "batch_size", "type": "spin", "label": "Batch Size",
             "default": "1", "from_": 1, "to": 8, "increment": 1, "width": 6},

            # --- Guidance ---
            {"id": "_h_guidance", "type": "heading", "label": "Guidance"},
            {"id": "guidance_interval", "type": "spin", "label": "Guidance Interval",
             "default": "0.5", "from_": 0.0, "to": 1.0, "increment": 0.05,
             "format": "%.2f", "width": 8},
            {"id": "guidance_interval_decay", "type": "spin", "label": "Interval Decay",
             "default": "0.0", "from_": 0.0, "to": 2.0, "increment": 0.05,
             "format": "%.2f", "width": 8},
            {"id": "min_guidance_scale", "type": "spin", "label": "Min CFG Scale",
             "default": "3.0", "from_": 0.0, "to": 20.0, "increment": 0.5,
             "format": "%.1f", "width": 8},
            {"id": "guidance_scale_text", "type": "spin", "label": "Text Guidance",
             "default": "0.0", "from_": 0.0, "to": 20.0, "increment": 0.5,
             "format": "%.1f", "width": 8},
            {"id": "guidance_scale_lyric", "type": "spin", "label": "Lyric Guidance",
             "default": "0.0", "from_": 0.0, "to": 20.0, "increment": 0.5,
             "format": "%.1f", "width": 8},

            # --- ERG (Energy Regularization) ---
            {"id": "_h_erg", "type": "heading", "label": "ERG (Energy Regularization)"},
            {"id": "use_erg_tag", "type": "check", "label": "ERG on Tags", "default": True},
            {"id": "use_erg_lyric", "type": "check", "label": "ERG on Lyrics", "default": True},
            {"id": "use_erg_diffusion", "type": "check", "label": "ERG on Diffusion", "default": True},

            # --- Retake / Variation ---
            {"id": "_h_retake", "type": "heading", "label": "Retake / Variation"},
            {"id": "retake_variance", "type": "spin", "label": "Retake Variance",
             "default": "0.5", "from_": 0.0, "to": 1.0, "increment": 0.05,
             "format": "%.2f", "width": 8},
        ],
    },
    "heartmula": {
        "display": "HeartMuLa 3B",
        "controls": [
            # --- Content ---
            {"id": "_h_content", "type": "heading", "label": "Content"},
            {"id": "lyrics", "type": "text", "label": "Lyrics (LRC format)", "default": "", "rows": 5},
            {"id": "tags", "type": "text", "label": "Tags (comma-separated)", "default": "", "rows": 2},

            # --- Generation ---
            {"id": "_h_generation", "type": "heading", "label": "Generation"},
            {"id": "duration", "type": "spin", "label": "Duration (s)",
             "default": "120", "from_": 1, "to": 300, "increment": 10, "width": 8},
            {"id": "temperature", "type": "spin", "label": "Temperature",
             "default": "1.0", "from_": 0.1, "to": 3.0, "increment": 0.05,
             "format": "%.2f", "width": 8},
            {"id": "top_k", "type": "spin", "label": "Top-K",
             "default": "50", "from_": 1, "to": 500, "increment": 5, "width": 8},
            {"id": "cfg_scale", "type": "spin", "label": "CFG Scale",
             "default": "1.5", "from_": 1.0, "to": 10.0, "increment": 0.1,
             "format": "%.1f", "width": 8},
            {"id": "seed", "type": "entry", "label": "Seed (-1=random)",
             "default": "-1", "width": 10},
        ],
    },
    "diffrhythm": {
        "display": "DiffRhythm",
        "controls": [
            # --- Content ---
            {"id": "_h_content", "type": "heading", "label": "Content"},
            {"id": "prompt", "type": "text", "label": "Style Description", "default": "", "rows": 2},
            {"id": "lyrics", "type": "text", "label": "Lyrics (LRC format)", "default": "", "rows": 5},

            # --- Generation ---
            {"id": "_h_generation", "type": "heading", "label": "Generation"},
            {"id": "duration", "type": "combo", "label": "Duration (s)",
             "options": ["95", "120", "180", "240", "285"], "default": "285"},
            {"id": "steps", "type": "spin", "label": "ODE Steps",
             "default": "32", "from_": 10, "to": 100, "increment": 1, "width": 8},
            {"id": "cfg_strength", "type": "spin", "label": "CFG Strength",
             "default": "4.0", "from_": 1.0, "to": 15.0, "increment": 0.5,
             "format": "%.1f", "width": 8},
            {"id": "sway_sampling_coef", "type": "entry",
             "label": "Sway Coef (empty=off)", "default": "", "width": 8},
            {"id": "seed", "type": "entry", "label": "Seed (-1=random)",
             "default": "-1", "width": 10},

            # --- Memory ---
            {"id": "_h_memory", "type": "heading", "label": "Memory"},
            {"id": "chunked", "type": "check", "label": "Chunked VAE Decoding",
             "default": False},
        ],
    },
    "yue": {
        "display": "YuE",
        "controls": [
            # --- Content ---
            {"id": "_h_content", "type": "heading", "label": "Content"},
            {"id": "tags", "type": "text", "label": "Genre Tags", "default": "", "rows": 2},
            {"id": "lyrics", "type": "text", "label": "Lyrics ([verse]/[chorus]/...)",
             "default": "", "rows": 5},

            # --- Model ---
            {"id": "_h_model", "type": "heading", "label": "Model"},
            {"id": "language", "type": "combo", "label": "Language",
             "options": ["en", "zh", "ja", "ko"], "default": "en"},
            {"id": "mode", "type": "combo", "label": "Generation Mode",
             "options": ["cot", "icl"], "default": "cot"},

            # --- Generation ---
            {"id": "_h_generation", "type": "heading", "label": "Generation"},
            {"id": "duration", "type": "spin", "label": "Max Duration (s)",
             "default": "120", "from_": 30, "to": 600, "increment": 30, "width": 8},
            {"id": "num_segments", "type": "spin", "label": "Segments (0=all)",
             "default": "0", "from_": 0, "to": 10, "increment": 1, "width": 8},
            {"id": "max_new_tokens", "type": "spin", "label": "Max Tokens/Segment",
             "default": "3000", "from_": 500, "to": 6000, "increment": 500, "width": 8},
            {"id": "temperature", "type": "spin", "label": "Temperature",
             "default": "1.0", "from_": 0.1, "to": 2.0, "increment": 0.1,
             "format": "%.1f", "width": 8},
            {"id": "top_p", "type": "spin", "label": "Top-P",
             "default": "0.93", "from_": 0.1, "to": 1.0, "increment": 0.01,
             "format": "%.2f", "width": 8},
            {"id": "repetition_penalty", "type": "spin", "label": "Repetition Penalty",
             "default": "1.1", "from_": 1.0, "to": 2.0, "increment": 0.1,
             "format": "%.1f", "width": 8},
            {"id": "guidance_scale", "type": "spin", "label": "Guidance Scale",
             "default": "1.5", "from_": 1.0, "to": 3.0, "increment": 0.1,
             "format": "%.1f", "width": 8},
            {"id": "seed", "type": "entry", "label": "Seed (-1=random)",
             "default": "-1", "width": 10},

            # --- Performance ---
            {"id": "_h_performance", "type": "heading", "label": "Performance"},
            {"id": "stage2_batch_size", "type": "spin", "label": "Stage 2 Batch Size",
             "default": "4", "from_": 1, "to": 16, "increment": 1, "width": 8},
        ],
    },
    "musicgen": {
        "display": "MusicGen",
        "controls": [
            # --- Content ---
            {"id": "_h_content", "type": "heading", "label": "Content"},
            {"id": "prompt", "type": "text", "label": "Style Description", "default": "", "rows": 3},
            {"id": "reference_audio", "type": "file", "label": "Melody Audio (melody model only)"},

            # --- Generation ---
            {"id": "_h_generation", "type": "heading", "label": "Generation"},
            {"id": "duration", "type": "spin", "label": "Duration (s)",
             "default": "10", "from_": 1, "to": 300, "increment": 5, "width": 8},
            {"id": "temperature", "type": "spin", "label": "Temperature",
             "default": "1.0", "from_": 0.0, "to": 2.0, "increment": 0.05,
             "format": "%.2f", "width": 8},
            {"id": "top_k", "type": "spin", "label": "Top-K (0=off)",
             "default": "250", "from_": 0, "to": 1000, "increment": 10, "width": 8},
            {"id": "top_p", "type": "spin", "label": "Top-P (0=off)",
             "default": "0.0", "from_": 0.0, "to": 1.0, "increment": 0.05,
             "format": "%.2f", "width": 8},
            {"id": "cfg_coeff", "type": "spin", "label": "CFG Coeff",
             "default": "3.0", "from_": 0.0, "to": 10.0, "increment": 0.5,
             "format": "%.1f", "width": 8},
            {"id": "use_sampling", "type": "check", "label": "Use Sampling",
             "default": True},
            {"id": "two_step_cfg", "type": "check", "label": "Two-Step CFG",
             "default": False},

            # --- Output ---
            {"id": "_h_output", "type": "heading", "label": "Output"},
            {"id": "extend_stride", "type": "spin", "label": "Extend Stride (s, 0=off)",
             "default": "0", "from_": 0, "to": 30, "increment": 1, "width": 8},
            {"id": "seed", "type": "entry", "label": "Seed (-1=random)",
             "default": "-1", "width": 10},
        ],
    },
    "riffusion": {
        "display": "Riffusion",
        "controls": [
            # --- Content ---
            {"id": "_h_content", "type": "heading", "label": "Content"},
            {"id": "prompt", "type": "text", "label": "Prompt", "default": "", "rows": 2},
            {"id": "negative_prompt", "type": "text", "label": "Negative Prompt", "default": "", "rows": 2},

            # --- Generation ---
            {"id": "_h_generation", "type": "heading", "label": "Generation"},
            {"id": "num_inference_steps", "type": "spin", "label": "Steps",
             "default": "50", "from_": 1, "to": 150, "increment": 5, "width": 8},
            {"id": "guidance_scale", "type": "spin", "label": "Guidance Scale",
             "default": "7.0", "from_": 1.0, "to": 20.0, "increment": 0.5,
             "format": "%.1f", "width": 8},
            {"id": "duration", "type": "spin", "label": "Duration (s)",
             "default": "10", "from_": 1, "to": 30, "increment": 1, "width": 8},
            {"id": "seed", "type": "entry", "label": "Seed (-1=random)",
             "default": "-1", "width": 10},
        ],
    },
    "stable_audio": {
        "display": "Stable Audio Open",
        "controls": [
            # --- Content ---
            {"id": "_h_content", "type": "heading", "label": "Content"},
            {"id": "prompt", "type": "text", "label": "Prompt", "default": "", "rows": 3},
            {"id": "negative_prompt", "type": "text", "label": "Negative Prompt", "default": "", "rows": 2},

            # --- Generation ---
            {"id": "_h_generation", "type": "heading", "label": "Generation"},
            {"id": "duration", "type": "spin", "label": "Duration (s, max 47)",
             "default": "30", "from_": 1, "to": 47, "increment": 1, "width": 8},
            {"id": "steps", "type": "spin", "label": "Steps",
             "default": "100", "from_": 10, "to": 250, "increment": 10, "width": 8},
            {"id": "cfg_scale", "type": "spin", "label": "CFG Scale",
             "default": "7.0", "from_": 1.0, "to": 15.0, "increment": 0.5,
             "format": "%.1f", "width": 8},
            {"id": "sampler_type", "type": "combo", "label": "Sampler",
             "options": ["dpmpp-3m-sde", "dpmpp-2m-sde", "k-dpm-2", "k-dpm-fast"],
             "default": "dpmpp-3m-sde"},

            # --- Advanced ---
            {"id": "_h_advanced", "type": "heading", "label": "Advanced"},
            {"id": "sigma_min", "type": "spin", "label": "Sigma Min",
             "default": "0.3", "from_": 0.01, "to": 1.0, "increment": 0.01,
             "format": "%.2f", "width": 8},
            {"id": "sigma_max", "type": "spin", "label": "Sigma Max",
             "default": "500", "from_": 100, "to": 1000, "increment": 50, "width": 8},
            {"id": "seed", "type": "entry", "label": "Seed (-1=random)",
             "default": "-1", "width": 10},
        ],
    },
}

# ---------------------------------------------------------------------------
# Per-model testing presets — each preset overrides specific MODEL_PARAMS defaults
# ---------------------------------------------------------------------------
MODEL_PRESETS = {
    "ace_step_v15": [
        {
            "name": "Quick Test (8-step turbo)",
            "params": {
                "caption": "upbeat pop song with catchy melody",
                "duration": "30", "inference_steps": "8", "shift": "3.0",
                "guidance_scale": "7.0", "thinking": True,
            },
        },
        {
            "name": "Vocal Pop",
            "params": {
                "caption": "catchy pop song with bright vocals and modern production",
                "lyrics": "[Verse]\nWalking down the street today\nSunshine melting clouds away\n\n[Chorus]\nFeel the rhythm feel the beat\nMoving to the sound so sweet",
                "vocal_language": "en", "keyscale": "C major",
                "timesignature": "4", "bpm": "120", "duration": "120",
                "inference_steps": "8", "thinking": True,
            },
        },
        {
            "name": "Instrumental Lo-Fi",
            "params": {
                "caption": "chill lo-fi hip hop beat, relaxing study music, vinyl crackle, mellow piano",
                "instrumental": True, "keyscale": "Eb major",
                "bpm": "85", "duration": "120", "timesignature": "4",
                "inference_steps": "8", "thinking": True,
            },
        },
        {
            "name": "Epic Cinematic",
            "params": {
                "caption": "epic cinematic orchestral score, dramatic strings, powerful brass, war drums",
                "instrumental": True, "keyscale": "D minor",
                "bpm": "130", "duration": "180", "timesignature": "4",
                "inference_steps": "8", "thinking": True,
            },
        },
        {
            "name": "Rock / Metal",
            "params": {
                "caption": "heavy rock song with distorted electric guitar, powerful drums, bass groove",
                "lyrics": "[Verse]\nBurning through the night\nChasing down the light\n\n[Chorus]\nWe won't stop we won't break\nEvery step is ours to take",
                "vocal_language": "en", "keyscale": "E minor",
                "bpm": "140", "duration": "120", "timesignature": "4",
                "inference_steps": "8", "thinking": True,
            },
        },
        {
            "name": "Jazz / Blues",
            "params": {
                "caption": "smooth jazz with walking bass, brushed drums, saxophone solo, warm piano chords",
                "instrumental": True, "keyscale": "Bb major",
                "bpm": "110", "duration": "120", "timesignature": "4",
                "inference_steps": "8", "thinking": True,
            },
        },
        {
            "name": "Electronic / EDM",
            "params": {
                "caption": "electronic dance music with driving bass, synth leads, four-on-the-floor kick",
                "instrumental": True, "keyscale": "A minor",
                "bpm": "128", "duration": "120", "timesignature": "4",
                "inference_steps": "8", "thinking": True,
            },
        },
        {
            "name": "R&B / Soul",
            "params": {
                "caption": "smooth r&b with silky vocals, warm keys, mellow groove, emotional ballad",
                "lyrics": "[Verse]\nLate night whispers in the dark\nYour voice still echoes in my heart\n\n[Chorus]\nStay with me until the dawn\nHold me close and carry on",
                "vocal_language": "en", "keyscale": "Ab major",
                "bpm": "90", "duration": "120", "timesignature": "4",
                "inference_steps": "8", "thinking": True,
            },
        },
        {
            "name": "Chinese Pop (C-Pop)",
            "params": {
                "caption": "Chinese pop ballad with emotional vocals, piano accompaniment, string arrangement",
                "lyrics": "[Verse]\n\u6708\u5149\u6d12\u5728\u7a97\u53f0\u4e0a\n\u601d\u5ff5\u968f\u98ce\u8f7b\u8f7b\u8361\n\n[Chorus]\n\u4f60\u7684\u7b11\u5bb9\u5728\u5fc3\u4e2d\n\u6e29\u6696\u4e86\u6574\u4e2a\u51ac",
                "vocal_language": "zh", "keyscale": "G major",
                "bpm": "80", "duration": "120", "timesignature": "4",
                "inference_steps": "8", "thinking": True,
            },
        },
        {
            "name": "Japanese Pop (J-Pop)",
            "params": {
                "caption": "upbeat J-Pop with bright vocals, catchy synth melody, energetic drums",
                "lyrics": "[Verse]\n\u5922\u3092\u8ffd\u3044\u304b\u3051\u3066\u8d70\u308b\n\u98a8\u304c\u80cc\u4e2d\u3092\u62bc\u3059\n\n[Chorus]\n\u8f1d\u304f\u672a\u6765\u3078\n\u4e00\u6b69\u305a\u3064\u9032\u3082\u3046",
                "vocal_language": "ja", "keyscale": "D major",
                "bpm": "135", "duration": "120", "timesignature": "4",
                "inference_steps": "8", "thinking": True,
            },
        },
    ],
    "ace_step_v1": [
        {
            "name": "Quick Test (27-step)",
            "params": {
                "caption": "upbeat pop song with catchy melody",
                "duration": "30", "inference_steps": "27",
                "guidance_scale": "15.0",
            },
        },
        {
            "name": "Vocal Pop",
            "params": {
                "caption": "catchy pop song with bright vocals and modern production",
                "lyrics": "[Verse]\nWalking down the street today\n"
                          "Sunshine melting clouds away\n\n"
                          "[Chorus]\nFeel the rhythm feel the beat\n"
                          "Moving to the sound so sweet",
                "duration": "60", "inference_steps": "60",
                "guidance_scale": "15.0",
            },
        },
        {
            "name": "Instrumental Lo-Fi",
            "params": {
                "caption": "chill lo-fi hip hop beat, relaxing study music, vinyl crackle, mellow piano",
                "instrumental": True,
                "duration": "60", "inference_steps": "60",
                "guidance_scale": "15.0",
            },
        },
        {
            "name": "Epic Cinematic",
            "params": {
                "caption": "epic cinematic orchestral score, dramatic strings, powerful brass, war drums",
                "instrumental": True,
                "duration": "120", "inference_steps": "60",
                "guidance_scale": "18.0",
            },
        },
        {
            "name": "Rock / Metal",
            "params": {
                "caption": "heavy rock song with distorted electric guitar, powerful drums, bass groove",
                "lyrics": "[Verse]\nBurning through the night\n"
                          "Chasing down the light\n\n"
                          "[Chorus]\nWe won't stop we won't break\n"
                          "Every step is ours to take",
                "duration": "60", "inference_steps": "60",
                "guidance_scale": "15.0",
            },
        },
        {
            "name": "Jazz / Blues",
            "params": {
                "caption": "smooth jazz with walking bass, brushed drums, saxophone solo, warm piano chords",
                "instrumental": True,
                "duration": "60", "inference_steps": "60",
                "guidance_scale": "15.0",
            },
        },
        {
            "name": "Electronic / EDM",
            "params": {
                "caption": "electronic dance music with driving bass, synth leads, four-on-the-floor kick",
                "instrumental": True,
                "duration": "60", "inference_steps": "60",
                "guidance_scale": "15.0",
            },
        },
        {
            "name": "High Quality (60-step Heun)",
            "params": {
                "caption": "beautiful acoustic ballad with fingerpicked guitar and warm vocals",
                "lyrics": "[Verse]\nSitting by the window pane\n"
                          "Watching as it starts to rain\n\n"
                          "[Chorus]\nEvery drop a memory\n"
                          "Of the love you gave to me",
                "duration": "60", "inference_steps": "60",
                "guidance_scale": "15.0", "scheduler_type": "heun",
            },
        },
    ],
    "heartmula": [
        {
            "name": "Quick Test",
            "params": {
                "tags": "pop,upbeat,energetic",
                "lyrics": "[Verse]\nHello world\nThis is a test\n\n"
                          "[Chorus]\nLa la la",
                "duration": "30", "temperature": "1.0", "top_k": "50",
            },
        },
        {
            "name": "Pop Ballad",
            "params": {
                "tags": "pop,ballad,emotional,piano,strings",
                "lyrics": "[Verse]\nWalking down the street today\n"
                          "Sunshine melting clouds away\n"
                          "Every step a little lighter\n"
                          "Every breath a little brighter\n\n"
                          "[Chorus]\nFeel the rhythm feel the beat\n"
                          "Moving to the sound so sweet\n"
                          "Dancing through the golden light\n"
                          "Everything will be alright",
                "duration": "120", "temperature": "1.0", "cfg_scale": "1.5",
            },
        },
        {
            "name": "Lo-Fi Chill",
            "params": {
                "tags": "lofi,chill,piano,mellow,relaxing,study",
                "lyrics": "[Intro]\n\n"
                          "[Verse]\nQuiet nights and city lights\n"
                          "Raindrops on the window pane\n\n"
                          "[Chorus]\nJust relax and let it go\n"
                          "Let the music take you slow",
                "duration": "120", "temperature": "1.0", "cfg_scale": "1.5",
            },
        },
        {
            "name": "Rock",
            "params": {
                "tags": "rock,electric,guitar,drums,powerful,energetic",
                "lyrics": "[Verse]\nBurning through the night\n"
                          "Chasing down the light\n"
                          "Nothing gonna stop us now\n\n"
                          "[Chorus]\nWe won't stop we won't break\n"
                          "Every step is ours to take\n"
                          "Raise your voice and shout it loud",
                "duration": "120", "temperature": "1.0", "cfg_scale": "1.5",
            },
        },
        {
            "name": "Jazz",
            "params": {
                "tags": "jazz,saxophone,piano,smooth,brushed,drums,bass",
                "lyrics": "[Intro]\n\n"
                          "[Verse]\nSmoke and mirrors in the night\n"
                          "Neon glow and candlelight\n\n"
                          "[Chorus]\nSwing along with me tonight\n"
                          "Everything is gonna be alright",
                "duration": "120", "temperature": "1.0", "cfg_scale": "1.5",
            },
        },
        {
            "name": "Electronic / EDM",
            "params": {
                "tags": "electronic,edm,synthesizer,bass,dance,energetic",
                "lyrics": "[Intro]\n\n"
                          "[Verse]\nPulse of the city beating strong\n"
                          "Lights are flashing all night long\n\n"
                          "[Chorus]\nDrop the bass and feel the sound\n"
                          "Feet are lifting off the ground",
                "duration": "120", "temperature": "1.0", "cfg_scale": "1.5",
            },
        },
        {
            "name": "Acoustic Folk",
            "params": {
                "tags": "acoustic,folk,guitar,warm,gentle,singer-songwriter",
                "lyrics": "[Verse]\nSitting by the window pane\n"
                          "Watching as it starts to rain\n"
                          "Memories of distant days\n"
                          "Drifting through the morning haze\n\n"
                          "[Chorus]\nEvery drop a melody\n"
                          "Of the love you gave to me\n"
                          "Simple words and honest heart\n"
                          "That's the place where songs can start",
                "duration": "120", "temperature": "1.0", "cfg_scale": "1.5",
            },
        },
        {
            "name": "High Creativity",
            "params": {
                "tags": "experimental,ambient,atmospheric,dreamy",
                "lyrics": "[Intro]\n\n"
                          "[Verse]\nColors blend in open sky\n"
                          "Whispers float as time drifts by\n\n"
                          "[Chorus]\nLet the sound paint what it will\n"
                          "In the silence something still",
                "duration": "120", "temperature": "1.3", "top_k": "100",
                "cfg_scale": "1.2",
            },
        },
    ],
    "diffrhythm": [
        {
            "name": "Quick Test (95s)",
            "params": {
                "prompt": "upbeat pop song with catchy melody",
                "lyrics": "[00:05.00] Hello world testing one two three",
                "duration": "95", "steps": "32", "cfg_strength": "4.0",
            },
        },
        {
            "name": "Pop (LRC)",
            "params": {
                "prompt": "catchy pop song with bright vocals and modern production",
                "lyrics": "[00:10.00] Walking down the street today\n"
                          "[00:15.00] Sunshine melting clouds away\n"
                          "[00:25.00] Feel the rhythm feel the beat\n"
                          "[00:30.00] Moving to the sound so sweet",
                "duration": "285", "steps": "32", "cfg_strength": "4.0",
            },
        },
        {
            "name": "Lo-Fi Instrumental",
            "params": {
                "prompt": "chill lo-fi hip hop beat, relaxing study music, "
                          "vinyl crackle, mellow piano",
                "lyrics": "",
                "duration": "285", "steps": "32", "cfg_strength": "4.0",
            },
        },
        {
            "name": "Cinematic Orchestral",
            "params": {
                "prompt": "epic cinematic orchestral score, dramatic strings, "
                          "powerful brass, war drums",
                "lyrics": "",
                "duration": "285", "steps": "50", "cfg_strength": "5.0",
            },
        },
        {
            "name": "Rock",
            "params": {
                "prompt": "heavy rock song with distorted electric guitar, "
                          "powerful drums, bass groove",
                "lyrics": "[00:10.00] Burning through the night\n"
                          "[00:15.00] Chasing down the light\n"
                          "[00:25.00] We won't stop we won't break\n"
                          "[00:30.00] Every step is ours to take",
                "duration": "285", "steps": "32", "cfg_strength": "4.0",
            },
        },
        {
            "name": "Jazz",
            "params": {
                "prompt": "smooth jazz with walking bass, brushed drums, "
                          "saxophone solo, warm piano chords",
                "lyrics": "",
                "duration": "285", "steps": "32", "cfg_strength": "4.0",
            },
        },
        {
            "name": "Electronic / EDM",
            "params": {
                "prompt": "electronic dance music with driving bass, "
                          "synth leads, four-on-the-floor kick",
                "lyrics": "",
                "duration": "285", "steps": "32", "cfg_strength": "4.0",
            },
        },
        {
            "name": "High Quality (50-step)",
            "params": {
                "prompt": "beautiful acoustic ballad with fingerpicked guitar "
                          "and warm harmonies",
                "lyrics": "[00:10.00] Sitting by the window pane\n"
                          "[00:15.00] Watching as it starts to rain\n"
                          "[00:25.00] Every drop a memory\n"
                          "[00:30.00] Of the love you gave to me",
                "duration": "285", "steps": "50", "cfg_strength": "5.0",
            },
        },
    ],
    "yue": [
        {
            "name": "Quick Test (1 segment)",
            "params": {
                "tags": "pop, upbeat, female vocals",
                "lyrics": "[verse]\nHello world, this is a test\n"
                          "Singing loud, feeling blessed",
                "num_segments": "1", "duration": "60",
            },
        },
        {
            "name": "Pop Ballad (EN)",
            "params": {
                "tags": "pop ballad, emotional, piano, strings, female vocals",
                "lyrics": "[verse]\nStanding in the pouring rain\n"
                          "Trying to wash away the pain\n"
                          "Every word you said to me\n"
                          "Echoes through my memory\n\n"
                          "[chorus]\nBut I will rise above it all\n"
                          "I won't stumble, I won't fall\n"
                          "Through the darkness I will shine\n"
                          "This broken heart is mine",
                "duration": "120", "guidance_scale": "1.5",
            },
        },
        {
            "name": "Chinese Pop",
            "params": {
                "tags": "mandopop, emotional, piano, strings, male vocals",
                "lyrics": "[verse]\n\u6211\u7ad9\u5728\u5927\u96e8\u4e2d\n"
                          "\u8bd5\u7740\u6d17\u53bb\u4f24\u75db\n"
                          "\u4f60\u8bf4\u7684\u6bcf\u4e00\u53e5\u8bdd\n"
                          "\u56de\u8361\u5728\u6211\u8bb0\u5fc6\u4e2d\n\n"
                          "[chorus]\n\u4f46\u6211\u4f1a\u8d85\u8d8a\u8fd9\u4e00\u5207\n"
                          "\u4e0d\u4f1a\u8dcc\u5012 \u4e0d\u4f1a\u5931\u8d25\n"
                          "\u7a7f\u8d8a\u9ed1\u6697\u6211\u4f1a\u53d1\u5149\n"
                          "\u8fd9\u7834\u788e\u7684\u5fc3\u662f\u6211\u7684",
                "language": "zh", "duration": "120",
            },
        },
        {
            "name": "Rock",
            "params": {
                "tags": "rock, electric guitar, drums, energetic, male vocals",
                "lyrics": "[verse]\nDriving down the highway fast\n"
                          "Living like it's gonna last\n"
                          "Turn the amplifier loud\n"
                          "Screaming to the restless crowd\n\n"
                          "[chorus]\nWe are the fire, we are the flame\n"
                          "Nothing in this world will be the same\n"
                          "Raise your hands up to the sky\n"
                          "Tonight we're never gonna die",
                "duration": "120", "temperature": "1.0",
                "guidance_scale": "1.5",
            },
        },
        {
            "name": "Lo-Fi Chill",
            "params": {
                "tags": "lo-fi, chill, mellow, ambient, soft beats, female vocals",
                "lyrics": "[verse]\nSunlight through the windowpane\n"
                          "Coffee steam and gentle rain\n"
                          "Pages turning, quiet room\n"
                          "Afternoon will end too soon\n\n"
                          "[chorus]\nLet it go, let it flow\n"
                          "Drifting soft and drifting slow",
                "duration": "120", "temperature": "0.8",
                "guidance_scale": "1.3",
            },
        },
        {
            "name": "Jazz",
            "params": {
                "tags": "jazz, saxophone, piano, double bass, smooth, instrumental",
                "lyrics": "[instrumental]\nSmooth jazz with saxophone melody\n"
                          "Piano comping underneath\n"
                          "Walking bass line",
                "duration": "120", "temperature": "1.0",
                "top_p": "0.95",
            },
        },
        {
            "name": "Electronic / EDM",
            "params": {
                "tags": "electronic, EDM, synth, bass drop, energetic, dance",
                "lyrics": "[verse]\nFeel the bass begin to drop\n"
                          "Moving bodies never stop\n"
                          "Laser lights across the floor\n"
                          "Give me give me give me more\n\n"
                          "[chorus]\nWe are electric, we are alive\n"
                          "Dancing until the morning light",
                "duration": "120", "temperature": "1.0",
                "guidance_scale": "1.5",
            },
        },
        {
            "name": "High Quality",
            "params": {
                "tags": "acoustic, folk, warm, fingerpicked guitar, female vocals",
                "lyrics": "[verse]\nSitting by the riverside\n"
                          "Watching as the world goes by\n"
                          "Every ripple tells a tale\n"
                          "Of the wind and morning hail\n\n"
                          "[chorus]\nOh the river knows my name\n"
                          "Whispers it and starts again\n"
                          "Carry me to where I'll find\n"
                          "Peace of body, peace of mind",
                "duration": "120", "temperature": "0.8",
                "repetition_penalty": "1.2", "max_new_tokens": "4000",
            },
        },
    ],
    "musicgen": [
        {
            "name": "Quick Test",
            "params": {
                "prompt": "upbeat electronic music with bright synths",
                "duration": "10", "temperature": "1.0", "cfg_coeff": "3.0",
            },
        },
        {
            "name": "Cinematic Orchestra",
            "params": {
                "prompt": "epic orchestral score with dramatic strings, powerful brass, and timpani",
                "duration": "30", "temperature": "0.9", "top_k": "250",
                "cfg_coeff": "4.0",
            },
        },
        {
            "name": "Lo-Fi Chill",
            "params": {
                "prompt": "lo-fi hip hop beat with mellow piano, vinyl crackle, relaxing study music",
                "duration": "30", "temperature": "1.1", "top_k": "300",
                "cfg_coeff": "3.0",
            },
        },
        {
            "name": "Ambient Pad",
            "params": {
                "prompt": "ambient atmospheric pad, ethereal drones, slow evolving textures",
                "duration": "30", "temperature": "1.2", "top_p": "0.9",
                "top_k": "0", "cfg_coeff": "2.5",
            },
        },
        {
            "name": "Funky Groove",
            "params": {
                "prompt": "funky bass groove with slap bass, wah guitar, and tight drums",
                "duration": "20", "temperature": "1.0", "top_k": "250",
                "cfg_coeff": "3.5",
            },
        },
        {
            "name": "Rock Energy",
            "params": {
                "prompt": "energetic rock with driving drums, distorted electric guitar, powerful bass",
                "duration": "20", "temperature": "0.95", "top_k": "250",
                "cfg_coeff": "3.5",
            },
        },
        {
            "name": "Jazz Smooth",
            "params": {
                "prompt": "smooth jazz with walking bass, brushed drums, mellow saxophone solo",
                "duration": "30", "temperature": "1.1", "top_k": "300",
                "cfg_coeff": "3.0",
            },
        },
        {
            "name": "Electronic Dance",
            "params": {
                "prompt": "EDM dance beat with driving four-on-the-floor kick, synth leads, big drop",
                "duration": "25", "temperature": "1.0", "top_k": "200",
                "cfg_coeff": "3.5",
            },
        },
    ],
    "riffusion": [
        {
            "name": "Quick Test",
            "params": {
                "prompt": "funky synth lead, electronic",
                "duration": "5", "num_inference_steps": "30", "guidance_scale": "7.0",
            },
        },
        {
            "name": "Acoustic Guitar",
            "params": {
                "prompt": "acoustic guitar fingerpicking, folk, warm, intimate",
                "negative_prompt": "distorted, electric, heavy",
                "duration": "10", "num_inference_steps": "50", "guidance_scale": "7.5",
            },
        },
        {
            "name": "Synthwave",
            "params": {
                "prompt": "80s synthwave, retro electronic, neon, pulsing bass, arpeggiated synth",
                "negative_prompt": "acoustic, vocals, lo-fi",
                "duration": "10", "num_inference_steps": "50", "guidance_scale": "8.0",
            },
        },
        {
            "name": "Piano Melody",
            "params": {
                "prompt": "soft piano melody, classical, gentle, emotional",
                "negative_prompt": "drums, bass, distortion",
                "duration": "10", "num_inference_steps": "50", "guidance_scale": "7.0",
            },
        },
        {
            "name": "Heavy Riff",
            "params": {
                "prompt": "heavy metal guitar riff, distorted, aggressive, powerful",
                "negative_prompt": "soft, acoustic, gentle",
                "duration": "5", "num_inference_steps": "50", "guidance_scale": "8.5",
            },
        },
        {
            "name": "Drum Beat",
            "params": {
                "prompt": "drum beat, hip hop, boom bap, crisp snare, deep kick",
                "negative_prompt": "melody, vocals, synth",
                "duration": "10", "num_inference_steps": "40", "guidance_scale": "7.0",
            },
        },
        {
            "name": "Ambient Texture",
            "params": {
                "prompt": "ambient drone texture, atmospheric, ethereal, reverb, slow",
                "negative_prompt": "drums, fast, loud",
                "duration": "10", "num_inference_steps": "60", "guidance_scale": "6.0",
            },
        },
        {
            "name": "Chiptune",
            "params": {
                "prompt": "chiptune 8-bit game music, retro, square wave, upbeat, pixel",
                "negative_prompt": "realistic, orchestral, vocals",
                "duration": "5", "num_inference_steps": "50", "guidance_scale": "7.5",
            },
        },
    ],
    "stable_audio": [
        {
            "name": "Quick Test",
            "params": {
                "prompt": "ambient electronic music, pad, atmospheric",
                "negative_prompt": "vocals, speech, distortion, noise, clipping",
                "duration": "10", "steps": "50", "cfg_scale": "7.0",
            },
        },
        {
            "name": "Ambient Soundscape",
            "params": {
                "prompt": "ethereal ambient soundscape, slowly evolving drones, deep reverb, nature sounds",
                "negative_prompt": "drums, percussion, vocals, speech, harsh, distorted",
                "duration": "47", "steps": "100", "cfg_scale": "6.0",
                "sigma_min": "0.3", "sigma_max": "500",
            },
        },
        {
            "name": "Drum Loop",
            "params": {
                "prompt": "drum loop, 120 BPM, tight kick and snare, hi-hats, percussive",
                "negative_prompt": "melody, harmony, vocals, speech, pad, drone",
                "duration": "30", "steps": "100", "cfg_scale": "7.0",
            },
        },
        {
            "name": "Cinematic Score",
            "params": {
                "prompt": "cinematic film score, orchestral tension, suspenseful strings, brass swells",
                "negative_prompt": "electronic, beats, lo-fi, noise, distortion, vocals",
                "duration": "47", "steps": "150", "cfg_scale": "8.0",
            },
        },
        {
            "name": "Electronic Bass",
            "params": {
                "prompt": "deep electronic bassline, sub bass, dark, pulsing, minimal techno",
                "negative_prompt": "acoustic, vocals, bright, treble, strings, piano",
                "duration": "30", "steps": "100", "cfg_scale": "7.5",
            },
        },
        {
            "name": "Sound Effects",
            "params": {
                "prompt": "thunder storm, heavy rain, distant lightning cracks, wind howling",
                "negative_prompt": "music, melody, vocals, speech, instruments",
                "duration": "20", "steps": "100", "cfg_scale": "7.0",
            },
        },
        {
            "name": "Warm Pad",
            "params": {
                "prompt": "warm analog synth pad, lush chords, slow filter sweep, dreamy",
                "negative_prompt": "drums, percussion, harsh, noise, staccato, sharp attack",
                "duration": "47", "steps": "120", "cfg_scale": "6.5",
            },
        },
        {
            "name": "Guitar Riff",
            "params": {
                "prompt": "clean electric guitar riff, funk, rhythmic strumming, bright tone",
                "negative_prompt": "vocals, synthesizer, electronic, distortion, noise",
                "duration": "30", "steps": "100", "cfg_scale": "7.5",
            },
        },
    ],
}
