"""
Music Worker - Generic worker subprocess for music generation models.

Each worker is a FastAPI micro-service that loads one model and serves inference.
Endpoints:
    GET  /health  - readiness check
    POST /load    - load model into GPU memory
    POST /infer   - run generation, return audio as base64
    POST /unload  - free GPU memory

Usage:
    python music_worker.py --model musicgen --port 9101 --device cuda:0
"""

import os
import sys
import argparse
import logging
import base64
import time
import io
from pathlib import Path

# Bootstrap: ensure project root is on sys.path
_BASE_DIR = Path(__file__).parent.resolve()
_TEMP_DIR = _BASE_DIR / "temp"
_TEMP_DIR.mkdir(parents=True, exist_ok=True)
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Music Worker")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

MODEL_ID = args.model
PORT = args.port
DEVICE = args.device

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_model = None
_model_loaded = False
_loading = False


# ---------------------------------------------------------------------------
# Model loading/inference dispatch
# ---------------------------------------------------------------------------
def _load_musicgen(device: str):
    """Load MusicGen model. Uses MODEL_SIZE env var or defaults to 'large'.

    Set MUSICGEN_MODEL_SIZE=small|medium|large|melody before spawning.
    """
    # Patch xformers import: audiocraft has a hard 'from xformers import ops'
    # but xformers DLLs are incompatible on Windows embedded Python.
    # Audiocraft defaults to torch attention backend, so xformers is not needed.
    try:
        import xformers  # noqa: F401
    except (ImportError, OSError):
        import types
        xf = types.ModuleType("xformers")
        xf.ops = None
        sys.modules.setdefault("xformers", xf)

    from audiocraft.models import MusicGen
    model_size = os.environ.get("MUSICGEN_MODEL_SIZE", "large").lower()
    valid_sizes = ("small", "medium", "large", "melody")
    if model_size not in valid_sizes:
        logger.warning("Invalid MUSICGEN_MODEL_SIZE '%s', using 'large'", model_size)
        model_size = "large"
    model_name = f"facebook/musicgen-{model_size}"
    logger.info("Loading MusicGen variant: %s", model_name)
    model = MusicGen.get_pretrained(model_name, device=device)
    return {"model": model, "model_size": model_size}


def _infer_musicgen(model_dict, params: dict, device: str) -> tuple:
    """Run MusicGen inference. Returns (audio_numpy, sample_rate)."""
    import torch
    model = model_dict["model"]
    model_size = model_dict["model_size"]

    prompt = params.get("prompt", "epic orchestral music")
    duration = float(params.get("duration", 10))
    temperature = float(params.get("temperature", 1.0))
    top_k = int(params.get("top_k", 250))
    top_p = float(params.get("top_p", 0.0))
    cfg_coeff = float(params.get("cfg_coeff", 3.0))
    use_sampling = _str_to_bool(params.get("use_sampling", True))
    two_step_cfg = _str_to_bool(params.get("two_step_cfg", False))
    extend_stride = int(params.get("extend_stride", 0))
    seed = int(params.get("seed", -1))

    if seed >= 0:
        torch.manual_seed(seed)

    gen_kwargs = dict(
        duration=duration,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        cfg_coef=cfg_coeff,
        use_sampling=use_sampling,
        two_step_cfg=two_step_cfg,
    )
    if extend_stride > 0:
        gen_kwargs["extend_stride"] = extend_stride
    model.set_generation_params(**gen_kwargs)

    # Melody conditioning only works with the melody model variant
    ref_audio = params.get("reference_audio")
    if ref_audio and os.path.exists(ref_audio) and model_size == "melody":
        import torchaudio
        melody, melody_sr = torchaudio.load(ref_audio)
        melody = melody.to(device)
        wav = model.generate_with_chroma([prompt], melody[None], melody_sr)
    else:
        if ref_audio and model_size != "melody":
            logger.warning(
                "Reference audio provided but model is '%s', not 'melody'. "
                "Ignoring reference audio.", model_size
            )
        wav = model.generate([prompt])

    audio = wav[0, 0].cpu().numpy()
    return audio, model.sample_rate


def _load_yue(device: str):
    """Load YuE two-stage model (7B LLM + 1B codec).

    YuE generates music via a two-stage LLM pipeline:
      Stage 1 (7B): lyrics+genre → interleaved vocal/instrumental codec tokens (1 codebook)
      Stage 2 (1B): 1-codebook tokens → 8-codebook tokens (teacher-forcing)
    Then XCodec decodes tokens → 16kHz audio.

    Both models are loaded to the device. Stage 1 will be offloaded to CPU
    during stage 2 inference to save VRAM.
    """
    import torch
    from transformers import AutoModelForCausalLM
    from omegaconf import OmegaConf

    # Add YuE repo directories to sys.path for helper utilities
    # mmtokenizer.py & codecmanipulator.py live in inference/
    repo_path = _BASE_DIR / "venvs" / "yue_env" / "repos" / "YuE"
    inference_path = repo_path / "inference"
    for p in (str(repo_path), str(inference_path)):
        if Path(p).exists() and p not in sys.path:
            sys.path.insert(0, p)

    # Determine attn_implementation: flash_attention_2 > sdpa > eager
    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            logger.info(
                "Flash Attention 2 not available, using PyTorch SDPA. "
                "Install flash-attn for lower VRAM usage."
            )
            attn_impl = "sdpa"
        else:
            logger.warning(
                "Flash Attention 2 and SDPA not available, falling back to eager. "
                "This will use more VRAM and be slower."
            )
            attn_impl = "eager"

    # Custom multimodal tokenizer (not HuggingFace AutoTokenizer)
    from mmtokenizer import _MMSentencePieceTokenizer
    tokenizer_path = inference_path / "mm_tokenizer_v0.2_hf" / "tokenizer.model"
    mmtokenizer = _MMSentencePieceTokenizer(str(tokenizer_path))

    # Codec manipulators for token ID offset management
    from codecmanipulator import CodecManipulator
    codectool = CodecManipulator("xcodec", 0, 1)       # stage1: 1 codebook
    codectool_stage2 = CodecManipulator("xcodec", 0, 8) # stage2: 8 codebooks

    # XCodec SoundStream model for decoding tokens → audio at 16kHz
    xcodec_path = inference_path / "xcodec_mini_infer"
    if str(xcodec_path) not in sys.path:
        sys.path.insert(0, str(xcodec_path))

    # XCodec's SoundStream uses relative path "./xcodec_mini_infer/..." internally,
    # so CWD must be the inference directory during model construction.
    os.chdir(str(inference_path))
    from models.soundstream_hubert_new import SoundStream
    codec_config = OmegaConf.load(str(xcodec_path / "final_ckpt" / "config.yaml"))
    # SAFETY: eval() here loads a class name from the repo's own YAML config file
    # (e.g. "SoundStream"), not from user input. This mirrors upstream YuE code.
    codec_model = eval(codec_config.generator.name)(**codec_config.generator.config)
    codec_weights = torch.load(
        str(xcodec_path / "final_ckpt" / "ckpt_00360000.pth"),
        map_location="cpu", weights_only=False,
    )
    codec_model.load_state_dict(codec_weights["codec_model"])
    # Codec model stays on CPU until decode phase — saves VRAM during generation
    codec_model.eval()

    stage1_model_id = "m-a-p/YuE-s1-7B-anneal-en-cot"
    logger.info("Loading YuE stage1 model (7B) with attn=%s ...", attn_impl)
    stage1_model = AutoModelForCausalLM.from_pretrained(
        stage1_model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    ).to(device)
    stage1_model.eval()

    # Stage2 stays on CPU until needed — saves ~2GB VRAM during stage1 inference
    logger.info("Loading YuE stage2 model (1B) to CPU with attn=%s ...", attn_impl)
    stage2_model = AutoModelForCausalLM.from_pretrained(
        "m-a-p/YuE-s2-1B-general",
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    stage2_model.eval()

    return {
        "stage1_model": stage1_model,
        "stage1_model_id": stage1_model_id,
        "attn_impl": attn_impl,
        "stage2_model": stage2_model,
        "mmtokenizer": mmtokenizer,
        "codectool": codectool,
        "codectool_stage2": codectool_stage2,
        "codec_model": codec_model,
        "repo_path": str(repo_path) if repo_path.exists() else None,
    }


def _infer_yue(model_dict, params: dict, device: str) -> tuple:
    """Run YuE two-stage inference. Returns (audio_numpy, sample_rate).

    Flow:
      1. Hot-swap stage1 model if language/mode changed
      2. Format lyrics into [tag]\\ntext segments, build prompt with genre tags
      3. Stage 1: Generate interleaved vocal+instrumental codec tokens segment-by-segment
      4. De-interleave into separate vocal and instrumental tracks
      5. Move stage1 to CPU, stage2 to GPU
      6. Stage 2: Teacher-forcing — expand 1-codebook → 8-codebook tokens per track
      7. XCodec decode each track → 16kHz audio
      8. Mix vocal + instrumental
    """
    import torch
    import re
    import copy
    from collections import Counter
    from einops import rearrange
    from transformers import LogitsProcessorList, LogitsProcessor

    seed = int(params.get("seed", -1))
    if seed >= 0:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    lyrics = params.get("lyrics", "[verse]\nLa la la\n[chorus]\nOh oh oh")
    tags = params.get("tags", params.get("prompt", "pop rock, upbeat, male vocals"))
    duration = float(params.get("duration", 120))
    temperature = float(params.get("temperature", 1.0))
    top_p = float(params.get("top_p", 0.93))
    repetition_penalty = float(params.get("repetition_penalty", 1.1))
    max_new_tokens = int(params.get("max_new_tokens", 3000))
    guidance_scale = float(params.get("guidance_scale", 1.5))
    num_segments = int(params.get("num_segments", 0))
    stage2_batch_size = int(params.get("stage2_batch_size", 4))
    language = params.get("language", "en")
    mode = params.get("mode", "cot")

    repo_path = model_dict.get("repo_path")
    if repo_path and repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    # --- Hot-swap stage1 model if language/mode changed ---
    lang_map = {"en": "en", "zh": "zh", "ja": "jp-kr", "ko": "jp-kr"}
    lang_key = lang_map.get(language, "en")
    target_model_id = f"m-a-p/YuE-s1-7B-anneal-{lang_key}-{mode}"

    stage1_model = model_dict["stage1_model"]
    if target_model_id != model_dict.get("stage1_model_id"):
        logger.warning(
            "YuE: switching stage1 model %s → %s (this takes 1-2 minutes) ...",
            model_dict.get("stage1_model_id"), target_model_id,
        )
        from transformers import AutoModelForCausalLM
        stage1_model.to("cpu")
        del stage1_model
        import gc
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        attn_impl = model_dict.get("attn_impl", "eager")
        stage1_model = AutoModelForCausalLM.from_pretrained(
            target_model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        ).to(device)
        stage1_model.eval()
        model_dict["stage1_model"] = stage1_model
        model_dict["stage1_model_id"] = target_model_id
        logger.info("YuE: stage1 model swapped to %s", target_model_id)

    stage2_model = model_dict["stage2_model"]
    mmtokenizer = model_dict["mmtokenizer"]
    codectool = model_dict["codectool"]
    codectool_stage2 = model_dict["codectool_stage2"]
    codec_model = model_dict["codec_model"]

    # --- Token blockers for constrained generation ---
    class BlockTokenRangeProcessor(LogitsProcessor):
        def __init__(self, start_id, end_id):
            self.blocked = list(range(start_id, end_id))
        def __call__(self, input_ids, scores):
            scores[:, self.blocked] = -float("inf")
            return scores

    # --- Parse lyrics into segments ---
    def split_lyrics(text):
        pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
        segments = re.findall(pattern, text, re.DOTALL)
        return [f"[{tag}]\n{body.strip()}\n\n" for tag, body in segments]

    lyric_segments = split_lyrics(lyrics)
    if not lyric_segments:
        lyric_segments = [f"[verse]\n{lyrics}\n\n"]

    # Cap segments if num_segments > 0
    if num_segments > 0:
        lyric_segments = lyric_segments[:num_segments]

    full_lyrics = "\n".join(lyric_segments)
    header = (
        f"Generate music from the given lyrics segment by segment.\n"
        f"[Genre] {tags}\n{full_lyrics}"
    )
    prompt_texts = [header] + lyric_segments

    # --- Stage 1: Segment-by-segment generation ---
    logger.info("YuE Stage 1: generating coarse tokens for %d segments ...", len(lyric_segments))

    stage1_logit_blockers = LogitsProcessorList([
        BlockTokenRangeProcessor(0, 32002),
        BlockTokenRangeProcessor(32016, 32016),
    ])

    start_seg = mmtokenizer.tokenize("[start_of_segment]")
    end_seg = mmtokenizer.tokenize("[end_of_segment]")
    raw_output = None

    for i in range(1, len(prompt_texts)):
        section_text = prompt_texts[i]
        # First segment gets full guidance, later segments get reduced guidance
        seg_guidance = guidance_scale if i <= 1 else max(1.0, guidance_scale - 0.3)

        if i == 1:
            head_ids = mmtokenizer.tokenize(prompt_texts[0])
            prompt_ids = head_ids + start_seg + mmtokenizer.tokenize(section_text) + \
                         [mmtokenizer.soa] + codectool.sep_ids
        else:
            prompt_ids = end_seg + start_seg + mmtokenizer.tokenize(section_text) + \
                         [mmtokenizer.soa] + codectool.sep_ids

        prompt_ids = torch.as_tensor([prompt_ids], dtype=torch.long).to(device)

        if raw_output is not None:
            input_ids = torch.cat([raw_output, prompt_ids], dim=1)
        else:
            input_ids = prompt_ids

        # Truncate to fit context window
        max_context = 16384 - max_new_tokens - 1
        if input_ids.shape[-1] > max_context:
            input_ids = input_ids[:, -max_context:]

        with torch.no_grad():
            output_seq = stage1_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=100,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=mmtokenizer.eoa,
                logits_processor=stage1_logit_blockers,
                guidance_scale=seg_guidance,
            )

        # Ensure EOA at end
        if output_seq[0][-1].item() != mmtokenizer.eoa:
            eoa_tensor = torch.as_tensor([[mmtokenizer.eoa]]).to(device)
            output_seq = torch.cat([output_seq, eoa_tensor], dim=1)

        if i > 1:
            raw_output = torch.cat([raw_output, prompt_ids, output_seq[:, input_ids.shape[-1]:]], dim=1)
        else:
            raw_output = output_seq

    # --- De-interleave vocal/instrumental from stage 1 output ---
    ids = raw_output[0].cpu().numpy()
    soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
    eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()

    vocals, instrumentals = [], []
    for seg_i in range(len(soa_idx)):
        codec_ids = ids[soa_idx[seg_i] + 1:eoa_idx[seg_i]]
        if len(codec_ids) > 0 and codec_ids[0] == 32016:  # skip separator
            codec_ids = codec_ids[1:]
        codec_ids = codec_ids[:2 * (len(codec_ids) // 2)]  # ensure even length
        if len(codec_ids) == 0:
            continue
        deinterleaved = rearrange(codec_ids, "(n b) -> b n", b=2)
        vocals.append(codectool.ids2npy(deinterleaved[0]))
        instrumentals.append(codectool.ids2npy(deinterleaved[1]))

    if not vocals:
        raise RuntimeError("YuE Stage 1 produced no audio tokens")

    vocals_npy = np.concatenate(vocals, axis=1)            # (1, T)
    instrumentals_npy = np.concatenate(instrumentals, axis=1)  # (1, T)

    # --- Offload stage1 to CPU, keep stage2 on GPU ---
    logger.info("YuE: offloading stage1 to CPU for stage2 ...")
    stage1_model.to("cpu")
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    stage2_model.to(device)

    # --- Inlined stage2 generate (cannot import from infer.py — argparse at module level) ---
    def _stage2_gen(model, prompt, batch_sz):
        """Teacher-forcing: expand 1-codebook → 8-codebook tokens."""
        cids = codectool.unflatten(prompt, n_quantizer=1)
        cids = codectool.offset_tok_ids(
            cids, global_offset=codectool.global_offset,
            codebook_size=codectool.codebook_size,
            num_codebooks=codectool.num_codebooks,
        ).astype(np.int32)

        if batch_sz > 1:
            chunks = [cids[:, i * 300:(i + 1) * 300] for i in range(batch_sz)]
            cids = np.concatenate(chunks, axis=0)
            pids = np.concatenate([
                np.tile([mmtokenizer.soa, mmtokenizer.stage_1], (batch_sz, 1)),
                cids,
                np.tile([mmtokenizer.stage_2], (batch_sz, 1)),
            ], axis=1)
        else:
            pids = np.concatenate([
                np.array([mmtokenizer.soa, mmtokenizer.stage_1]),
                cids.flatten(),
                np.array([mmtokenizer.stage_2]),
            ]).astype(np.int32)[np.newaxis, ...]

        cids_t = torch.as_tensor(cids).to(device)
        pids_t = torch.as_tensor(pids).to(device)
        len_p = pids_t.shape[-1]

        s2_blockers = LogitsProcessorList([
            BlockTokenRangeProcessor(0, 46358),
            BlockTokenRangeProcessor(53526, mmtokenizer.vocab_size),
        ])

        for f_idx in range(cids_t.shape[1]):
            cb0 = cids_t[:, f_idx:f_idx + 1]
            pids_t = torch.cat([pids_t, cb0], dim=1)
            with torch.no_grad():
                s2out = model.generate(
                    input_ids=pids_t,
                    min_new_tokens=7, max_new_tokens=7,
                    eos_token_id=mmtokenizer.eoa,
                    pad_token_id=mmtokenizer.eoa,
                    logits_processor=s2_blockers,
                )
            pids_t = s2out

        if batch_sz > 1:
            out = pids_t.cpu().numpy()[:, len_p:]
            out = np.concatenate([out[j] for j in range(batch_sz)], axis=0)
        else:
            out = pids_t[0].cpu().numpy()[len_p:]
        return out

    def _stage2_expand(model, stage1_npy, batch_sz):
        """Run stage2 with chunking for long sequences."""
        prompt = stage1_npy.astype(np.int32)
        output_duration = prompt.shape[-1] // 50 // 6 * 6  # snap to 6s multiples
        num_batch = output_duration // 6

        if num_batch <= batch_sz:
            output = _stage2_gen(model, prompt[:, :output_duration * 50], batch_sz=num_batch)
        else:
            segs = []
            n_segs = (num_batch + batch_sz - 1) // batch_sz
            for s in range(n_segs):
                s_start = s * batch_sz * 300
                s_end = min((s + 1) * batch_sz * 300, output_duration * 50)
                cur_bs = batch_sz if s < n_segs - 1 or num_batch % batch_sz == 0 else num_batch % batch_sz
                segs.append(_stage2_gen(model, prompt[:, s_start:s_end], batch_sz=cur_bs))
            output = np.concatenate(segs, axis=0)

        # Handle remainder past the 6s-aligned boundary
        if output_duration * 50 != prompt.shape[-1]:
            ending = _stage2_gen(model, prompt[:, output_duration * 50:], batch_sz=1)
            output = np.concatenate([output, ending], axis=0)

        codec_8cb = codectool_stage2.ids2npy(output)

        # Fix invalid codes (clamp to 0-1023)
        fixed = copy.deepcopy(codec_8cb)
        for row_i, row in enumerate(codec_8cb):
            for col_j, val in enumerate(row):
                if val < 0 or val > 1023:
                    counter = Counter(row)
                    fixed[row_i, col_j] = counter.most_common(1)[0][0]
        return fixed

    # --- Stage 2: Teacher-forcing expansion (1 → 8 codebooks) per track ---
    logger.info("YuE Stage 2: expanding to 8 codebooks (batch_size=%d) ...", stage2_batch_size)
    tracks = {"vocal": vocals_npy, "instrumental": instrumentals_npy}
    decoded_tracks = {}

    # Stage 2 expansion for all tracks
    expanded_tracks = {}
    for track_name, stage1_npy in tracks.items():
        expanded_tracks[track_name] = _stage2_expand(
            stage2_model, stage1_npy, stage2_batch_size)

    # Offload stage2, bring codec model to GPU for decode
    stage2_model.to("cpu")
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    codec_model.to(device)

    for track_name, codec_8cb in expanded_tracks.items():
        # Decode via XCodec → 16kHz
        decoded = codec_model.decode(
            torch.as_tensor(codec_8cb.astype(np.int16), dtype=torch.long)
            .unsqueeze(0).permute(1, 0, 2).to(device)
        )
        decoded_tracks[track_name] = decoded.cpu().squeeze(0)  # (1, samples)

    # Move codec to CPU, stage1 back to GPU for next inference
    codec_model.to("cpu")
    stage1_model.to(device)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # --- Mix vocal + instrumental ---
    vocal_audio = decoded_tracks["vocal"]
    inst_audio = decoded_tracks["instrumental"]
    # Pad shorter track to match length
    max_len = max(vocal_audio.shape[-1], inst_audio.shape[-1])
    if vocal_audio.shape[-1] < max_len:
        vocal_audio = torch.nn.functional.pad(vocal_audio, (0, max_len - vocal_audio.shape[-1]))
    if inst_audio.shape[-1] < max_len:
        inst_audio = torch.nn.functional.pad(inst_audio, (0, max_len - inst_audio.shape[-1]))

    mix = (vocal_audio + inst_audio).numpy()
    if mix.ndim == 2:
        mix = mix[0]  # (samples,) mono at 16kHz

    # Normalize
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix / peak * 0.95

    # Trim to requested duration
    sample_rate = 16000
    max_samples = int(sample_rate * duration)
    if len(mix) > max_samples:
        mix = mix[:max_samples]

    return mix.astype(np.float32), sample_rate


def _load_diffrhythm(device: str):
    """Load DiffRhythm model components via prepare_model().

    The repo is cloned to venvs/diffrhythm_env/repos/DiffRhythm.
    prepare_model() returns (cfm, tokenizer, muq, vae).
    We load at max_frames=6144 (285s) — shorter songs use fewer frames at inference.
    """
    import torch

    # Ensure espeak-ng env vars are set (phonemizer needs them before import)
    espeak_dir = _BASE_DIR / "espeak-ng"
    espeak_dll = espeak_dir / "libespeak-ng.dll"
    if espeak_dll.exists():
        os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", str(espeak_dll))
        os.environ.setdefault("PHONEMIZER_ESPEAK_PATH", str(espeak_dir))

    # Add DiffRhythm repo to sys.path
    repo_path = _BASE_DIR / "venvs" / "diffrhythm_env" / "repos" / "DiffRhythm"
    if repo_path.exists() and str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    from infer.infer_utils import prepare_model

    # DiffRhythm uses relative paths (./config/, infer/example/) — chdir to repo
    prev_cwd = os.getcwd()
    os.chdir(str(repo_path))
    try:
        max_frames = 6144  # supports up to 285s; 2048 would limit to 95s
        cfm, tokenizer, muq, vae = prepare_model(max_frames, device)
    finally:
        os.chdir(prev_cwd)

    return {
        "cfm": cfm,
        "tokenizer": tokenizer,
        "muq": muq,
        "vae": vae,
        "max_frames": max_frames,
        "repo_path": str(repo_path),
    }


def _infer_diffrhythm(model_dict, params: dict, device: str) -> tuple:
    """Run DiffRhythm inference via cfm.sample() + VAE decode.

    Returns (audio_numpy, sample_rate).
    """
    import torch
    from einops import rearrange

    seed = int(params.get("seed", -1))
    if seed >= 0:
        torch.manual_seed(seed)

    # Ensure repo is on path
    repo_path = model_dict.get("repo_path")
    if repo_path and repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from infer.infer_utils import (
        get_lrc_token,
        get_style_prompt,
        get_negative_style_prompt,
        get_reference_latent,
        decode_audio,
    )

    prompt = params.get("prompt", "upbeat electronic dance music")
    lyrics = params.get("lyrics", "")
    duration = float(params.get("duration", 95))
    chunked = bool(params.get("chunked", False))
    steps = int(params.get("steps", 32))
    cfg_strength = float(params.get("cfg_strength", 4.0))
    sway_raw = params.get("sway_sampling_coef", "")
    sway_coef = float(sway_raw) if sway_raw not in ("", "none", "None") else None

    # Snap duration: <=95 uses base model frames, >95 uses full model frames
    if duration <= 95:
        duration = 95.0
        max_frames = 2048
    else:
        max_frames = 6144

    if lyrics and not lyrics.strip().startswith("["):
        logger.warning(
            "DiffRhythm works best with LRC-formatted lyrics (with timing tags). "
            "Plain lyrics will be passed through but may produce suboptimal results."
        )

    cfm = model_dict["cfm"]
    tokenizer = model_dict["tokenizer"]
    muq = model_dict["muq"]
    vae = model_dict["vae"]

    # DiffRhythm utils use relative paths (infer/example/vocal.npy) — chdir to repo
    prev_cwd = os.getcwd()
    if repo_path:
        os.chdir(repo_path)
    try:
        # Preprocess lyrics → token embedding
        lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(
            max_frames, lyrics, tokenizer, duration, device,
        )

        # Text → style embedding via MuQ-MuLan
        style_prompt = get_style_prompt(muq, prompt=prompt)

        # Negative style embedding (pre-computed from infer/example/vocal.npy)
        negative_style_prompt = get_negative_style_prompt(device)

        # Latent conditioning (zeros for text-to-music)
        cond, pred_frames = get_reference_latent(
            device, max_frames, edit=False, pred_segments=None,
            ref_song=None, vae_model=vae,
        )
    finally:
        os.chdir(prev_cwd)

    # Call cfm.sample() directly (the inference() wrapper hardcodes steps/cfg)
    with torch.inference_mode():
        latents, _ = cfm.sample(
            cond=cond,
            text=lrc_prompt,
            duration=end_frame,
            style_prompt=style_prompt,
            negative_style_prompt=negative_style_prompt,
            max_duration=end_frame,
            song_duration=song_duration,
            steps=steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_coef,
            start_time=start_time,
            latent_pred_segments=pred_frames,
            batch_infer_num=1,
        )

        # Decode latent → audio via VAE
        latent = latents[0].to(torch.float32).transpose(1, 2)  # [b, d, t]

        # Try full decode first; fall back to chunked on OOM
        if not chunked:
            try:
                output = decode_audio(latent, vae, chunked=False)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    logger.warning("VAE decode OOM — retrying with chunked decoding")
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    output = decode_audio(latent, vae, chunked=True)
                else:
                    raise
        else:
            output = decode_audio(latent, vae, chunked=chunked)

        output = rearrange(output, "b d n -> d (b n)")

        # Peak normalize to [-1, 1]
        peak = torch.max(torch.abs(output)).clamp(min=1e-8)
        audio = output.to(torch.float32).div(peak).clamp(-1, 1).cpu().numpy()

    return audio.T.astype(np.float32), 44100  # [samples, channels], 44.1kHz


def _load_heartmula(device: str):
    """Load HeartMuLa generation pipeline."""
    import torch
    from heartlib import HeartMuLaGenPipeline
    from config import MODELS_DIR

    weights_path = str(MODELS_DIR / "heartmula")
    torch_device = torch.device(device)

    pipeline = HeartMuLaGenPipeline.from_pretrained(
        pretrained_path=weights_path,
        device=torch_device,
        dtype=torch.bfloat16,
        version="3B",
        lazy_load=True,
    )
    return {"pipeline": pipeline, "device": device}


def _infer_heartmula(model_dict, params: dict, device: str) -> tuple:
    """Run HeartMuLa inference. Returns (audio_numpy, sample_rate)."""
    import torch
    import soundfile as sf
    import tempfile

    seed = int(params.get("seed", -1))
    if seed >= 0:
        torch.manual_seed(seed)

    pipeline = model_dict["pipeline"]

    lyrics = params.get("lyrics", "")
    tags = params.get("tags", params.get("prompt", "pop"))
    duration = float(params.get("duration", 30))
    temperature = float(params.get("temperature", 1.0))
    top_k = int(params.get("top_k", 50))
    cfg_scale = float(params.get("cfg_scale", 1.5))

    # Pipeline saves output to disk, so use a temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False,
                                      dir=str(_TEMP_DIR))
    tmp_path = tmp.name
    tmp.close()

    try:
        with torch.no_grad():
            pipeline(
                {"lyrics": lyrics, "tags": tags},
                max_audio_length_ms=int(duration * 1000),
                temperature=temperature,
                topk=top_k,
                cfg_scale=cfg_scale,
                save_path=tmp_path,
            )

        # Read back the generated audio
        audio_np, sr = sf.read(tmp_path, dtype="float32")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return audio_np.astype(np.float32), sr


def _patch_pack_sequences_bool_sort(repo_path):
    """Monkey-patch pack_sequences to cast bool masks to long before argsort.

    The ACE-Step model's pack_sequences() calls mask_cat.argsort() on boolean
    attention masks, which fails on CUDA ('Sort currently does not support bool
    dtype on CUDA').  This patches the function in-place so the fix survives
    repo updates.
    """
    import importlib
    model_file = repo_path / "checkpoints" / "acestep-v15-turbo" / "modeling_acestep_v15_turbo"
    try:
        # The model module is loaded by transformers AutoModel; find it in sys.modules
        mod = None
        for name, m in sys.modules.items():
            if "modeling_acestep" in name and hasattr(m, "pack_sequences"):
                mod = m
                break
        if mod is None:
            return
        original_fn = mod.pack_sequences
        # Already patched?
        if getattr(original_fn, "_bool_sort_patched", False):
            return
        import torch as _torch

        def patched_pack_sequences(hidden1, hidden2, mask1, mask2):
            # Cast bool masks to long so argsort works on CUDA
            if mask1.dtype == _torch.bool:
                mask1 = mask1.long()
            if mask2.dtype == _torch.bool:
                mask2 = mask2.long()
            return original_fn(hidden1, hidden2, mask1, mask2)

        patched_pack_sequences._bool_sort_patched = True
        mod.pack_sequences = patched_pack_sequences
        logger.info("Patched pack_sequences for CUDA bool sort compatibility")
    except Exception as e:
        logger.warning("Failed to patch pack_sequences: %s", e)


def _load_ace_step_v15(device: str):
    """Load ACE-Step v1.5 DiT handler + optional 5Hz LLM handler.

    The v1.5 repo requires Python 3.11 so it can't be pip-installed on 3.10.
    We add its repo path to sys.path and import directly.

    Returns dict with keys: handler (DiT), llm_handler (LLM or None).
    """
    # Add v1.5 repo to sys.path so 'acestep' package is importable
    repo_path = _BASE_DIR / "venvs" / "ace_step_env" / "repos" / "ACE-Step-1.5"
    if repo_path.exists() and str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    from acestep.handler import AceStepHandler

    handler = AceStepHandler()

    ace_device = "cuda" if device.startswith("cuda") else device
    use_cpu_offload = not device.startswith("cuda")

    status_msg, enable_gen = handler.initialize_service(
        project_root=str(repo_path) if repo_path.exists() else None,
        config_path="acestep-v15-turbo",
        device=ace_device,
        use_flash_attention=True,
        compile_model=False,
        offload_to_cpu=use_cpu_offload,
        offload_dit_to_cpu=False,
    )

    if not enable_gen:
        raise RuntimeError(f"ACE-Step v1.5 init failed: {status_msg}")

    logger.info("ACE-Step v1.5 DiT initialized: %s", status_msg)

    # Patch: CUDA doesn't support sort/argsort on bool dtype — cast to long first.
    # The model's pack_sequences() calls mask_cat.argsort() on boolean attention
    # masks which fails on CUDA.  This survives repo updates.
    _patch_pack_sequences_bool_sort(repo_path)

    # --- Load 5Hz LLM handler (optional, VRAM-aware) ---
    llm_handler = None
    try:
        from acestep.gpu_config import get_gpu_config
        gpu_config = get_gpu_config()  # auto-detects VRAM

        if gpu_config.init_lm_default and gpu_config.recommended_lm_model:
            from acestep.llm_inference import LLMHandler

            # Determine checkpoint directory
            checkpoint_dir = str(repo_path / "checkpoints")
            if not (repo_path / "checkpoints").exists():
                checkpoint_dir = str(repo_path)

            lm_model = gpu_config.recommended_lm_model
            backend = gpu_config.recommended_backend

            # vllm may not be available on Windows embedded Python
            if backend == "vllm":
                try:
                    import vllm  # noqa: F401
                except ImportError:
                    backend = "pt"
                    logger.info("vllm unavailable, falling back to pt backend")

            llm_handler = LLMHandler()
            lm_status, lm_ok = llm_handler.initialize(
                checkpoint_dir=checkpoint_dir,
                lm_model_path=lm_model,
                backend=backend,
                device=ace_device,
                offload_to_cpu=gpu_config.offload_to_cpu_default,
            )
            if not lm_ok:
                logger.warning("LLM init failed: %s — continuing DiT-only",
                               lm_status)
                llm_handler = None
            else:
                logger.info("5Hz LLM initialized (%s, %s): %s",
                            lm_model, backend, lm_status)
        else:
            logger.info("GPU tier %s: LLM not recommended (%.1fGB VRAM)",
                        gpu_config.tier, gpu_config.gpu_memory_gb)
    except Exception as exc:
        logger.warning("LLM init exception: %s — continuing DiT-only", exc)
        llm_handler = None

    return {"handler": handler, "llm_handler": llm_handler}


def _str_to_bool(val, default=False):
    """Convert string/bool param to Python bool."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    return default


def _infer_ace_step_v15(model_dict, params: dict, device: str) -> tuple:
    """Run ACE-Step v1.5 inference via unified generate_music().

    Uses both DiT handler and optional LLM handler for Chain-of-Thought
    reasoning (metadata completion, caption enhancement, audio code gen).
    Returns (audio_numpy, sample_rate).
    """
    handler = model_dict["handler"]
    llm_handler = model_dict.get("llm_handler")

    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    # --- Extract params with type coercion ---
    caption = params.get("caption", params.get("prompt", "pop music"))
    lyrics = params.get("lyrics", "")
    instrumental = _str_to_bool(params.get("instrumental", False))
    if instrumental and not lyrics:
        lyrics = "[Instrumental]"

    bpm_raw = int(params.get("bpm", 0))
    bpm = bpm_raw if bpm_raw > 0 else None
    seed = int(params.get("seed", -1))

    ref_audio = params.get("reference_audio") or None
    if ref_audio and not os.path.exists(ref_audio):
        ref_audio = None
    src_audio = params.get("src_audio") or None
    if src_audio and not os.path.exists(src_audio):
        src_audio = None

    gen_params = GenerationParams(
        caption=caption,
        lyrics=lyrics,
        task_type=params.get("task_type", "text2music"),
        vocal_language=params.get("vocal_language", "unknown"),
        instrumental=instrumental,
        duration=float(params.get("duration", -1)),
        bpm=bpm,
        keyscale=params.get("keyscale", ""),
        timesignature=params.get("timesignature", ""),
        inference_steps=int(params.get("inference_steps", 8)),
        guidance_scale=float(params.get("guidance_scale", 7.0)),
        shift=float(params.get("shift", 3.0)),
        seed=seed,
        infer_method=params.get("infer_method", "ode"),
        # LLM params
        thinking=_str_to_bool(params.get("thinking", True), default=True),
        lm_temperature=float(params.get("lm_temperature", 0.85)),
        lm_cfg_scale=float(params.get("lm_cfg_scale", 2.0)),
        lm_top_k=int(params.get("lm_top_k", 0)),
        lm_top_p=float(params.get("lm_top_p", 0.9)),
        lm_negative_prompt=params.get("lm_negative_prompt", "NO USER INPUT"),
        use_cot_metas=_str_to_bool(params.get("use_cot_metas", True),
                                   default=True),
        use_cot_caption=_str_to_bool(params.get("use_cot_caption", True),
                                     default=True),
        use_cot_language=_str_to_bool(params.get("use_cot_language", True),
                                      default=True),
        # Advanced DiT
        use_adg=_str_to_bool(params.get("use_adg", False)),
        cfg_interval_start=float(params.get("cfg_interval_start", 0.0)),
        cfg_interval_end=float(params.get("cfg_interval_end", 1.0)),
        audio_cover_strength=float(params.get("audio_cover_strength", 1.0)),
        repainting_start=float(params.get("repainting_start", 0.0)),
        repainting_end=float(params.get("repainting_end", -1)),
        reference_audio=ref_audio,
        src_audio=src_audio,
    )

    gen_config = GenerationConfig(
        batch_size=int(params.get("batch_size", 1)),
        use_random_seed=(seed < 0),
        seeds=[seed] if seed >= 0 else None,
        audio_format="wav",
    )

    result = generate_music(
        dit_handler=handler,
        llm_handler=llm_handler,
        params=gen_params,
        config=gen_config,
        save_dir=None,  # no file I/O; extract tensors directly
    )

    if not result.success:
        raise RuntimeError(
            f"ACE-Step generation failed: {result.error}")

    if not result.audios:
        raise RuntimeError("ACE-Step returned no audio")

    # Audio tensor is [channels, samples] at 48kHz
    audio_tensor = result.audios[0]["tensor"]
    sr = result.audios[0].get("sample_rate", 48000)

    if audio_tensor is None:
        raise RuntimeError("ACE-Step returned None audio tensor")

    # Convert to numpy: (channels, samples) -> (samples, channels)
    audio = audio_tensor.cpu().numpy()
    if audio.ndim == 2:
        audio = audio.T  # (samples, channels)

    return audio.astype(np.float32), sr


def _load_ace_step_v1(device: str):
    """Load ACE-Step v1 via ACEStepPipeline."""
    from acestep.pipeline_ace_step import ACEStepPipeline

    device_id = int(device.split(":")[1]) if "cuda:" in device else 0
    cpu_offload = not device.startswith("cuda")

    pipeline = ACEStepPipeline(
        checkpoint_dir=None,
        device_id=device_id,
        cpu_offload=cpu_offload,
    )
    return {"pipeline": pipeline}


def _infer_ace_step_v1(model_dict, params: dict, device: str) -> tuple:
    """Run ACE-Step v1 inference. Returns (audio_numpy, sample_rate)."""
    import soundfile as sf
    import tempfile

    pipeline = model_dict["pipeline"]

    # Core params
    prompt = params.get("caption", params.get("prompt", "pop music"))
    lyrics = params.get("lyrics", "")
    instrumental = bool(params.get("instrumental", False))
    duration = float(params.get("duration", 60))
    infer_step = int(params.get("inference_steps", 60))
    guidance_scale = float(params.get("guidance_scale", 15.0))
    seed = int(params.get("seed", -1))

    # Scheduler / CFG
    scheduler_type = params.get("scheduler_type", "euler")
    cfg_type = params.get("cfg_type", "apg")
    omega_scale = float(params.get("omega_scale", 10.0))

    # Guidance interval
    guidance_interval = float(params.get("guidance_interval", 0.5))
    guidance_interval_decay = float(params.get("guidance_interval_decay", 0.0))
    min_guidance_scale = float(params.get("min_guidance_scale", 3.0))
    guidance_scale_text = float(params.get("guidance_scale_text", 0.0))
    guidance_scale_lyric = float(params.get("guidance_scale_lyric", 0.0))

    # ERG flags
    use_erg_tag = _str_to_bool(params.get("use_erg_tag", True))
    use_erg_lyric = _str_to_bool(params.get("use_erg_lyric", True))
    use_erg_diffusion = _str_to_bool(params.get("use_erg_diffusion", True))

    # Retake / variation
    retake_variance = float(params.get("retake_variance", 0.5))

    # Batch
    batch_size = int(params.get("batch_size", 1))

    if instrumental and not lyrics:
        lyrics = "[Instrumental]"

    tmp_dir = tempfile.mkdtemp(prefix="ace_v1_", dir=str(_TEMP_DIR))
    manual_seeds = [seed] if seed >= 0 else None

    result = pipeline(
        prompt=prompt,
        lyrics=lyrics,
        audio_duration=duration,
        infer_step=infer_step,
        guidance_scale=guidance_scale,
        scheduler_type=scheduler_type,
        cfg_type=cfg_type,
        omega_scale=omega_scale,
        manual_seeds=manual_seeds,
        guidance_interval=guidance_interval,
        guidance_interval_decay=guidance_interval_decay,
        min_guidance_scale=min_guidance_scale,
        guidance_scale_text=guidance_scale_text,
        guidance_scale_lyric=guidance_scale_lyric,
        use_erg_tag=use_erg_tag,
        use_erg_lyric=use_erg_lyric,
        use_erg_diffusion=use_erg_diffusion,
        retake_variance=retake_variance,
        save_path=tmp_dir,
        batch_size=batch_size,
    )

    # Result is list of file paths + a dict at the end
    audio_paths = [r for r in result if isinstance(r, str) and os.path.isfile(r)]
    if not audio_paths:
        raise RuntimeError("ACE-Step v1 returned no audio files")

    audio_path = audio_paths[0]
    try:
        audio, sr = sf.read(audio_path, dtype="float32")
    finally:
        # Clean up temp files
        import shutil
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    return audio.astype(np.float32), sr


def _load_riffusion(device: str):
    """Load Riffusion model (Stable Diffusion fine-tuned for spectrograms)."""
    import torch
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "riffusion/riffusion-model-v1",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    return {"pipe": pipe}


def _infer_riffusion(model_dict, params: dict, device: str) -> tuple:
    """Run Riffusion inference. Returns (audio_numpy, sample_rate)."""
    import torch
    from PIL import Image

    prompt = params.get("prompt", "funky synth lead")
    negative_prompt = params.get("negative_prompt", "")
    duration = float(params.get("duration", 5))
    num_inference_steps = int(params.get("num_inference_steps", 50))
    guidance_scale = float(params.get("guidance_scale", 7.0))
    seed = int(params.get("seed", -1))

    pipe = model_dict["pipe"]
    sample_rate = 44100
    segment_duration = 5.12  # each spectrogram covers ~5.12s

    generator = None
    if seed >= 0:
        generator = torch.Generator(device=device).manual_seed(seed)

    num_segments = max(1, int(np.ceil(duration / segment_duration)))
    segments = []

    for i in range(num_segments):
        # Vary seed slightly for each segment to get variation
        seg_generator = generator
        if seed >= 0 and i > 0:
            seg_generator = torch.Generator(device=device).manual_seed(seed + i)

        result = pipe(
            prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=512,
            height=512,
            generator=seg_generator,
        )
        spectrogram_image = result.images[0]

        # Convert spectrogram image to audio using inverse STFT
        audio_segment = _spectrogram_image_to_audio(spectrogram_image, sample_rate)
        segments.append(audio_segment)

    # Concatenate segments with crossfade
    if len(segments) == 1:
        audio = segments[0]
    else:
        crossfade_samples = int(sample_rate * 0.1)  # 100ms crossfade
        audio = segments[0]
        for seg in segments[1:]:
            if len(audio) >= crossfade_samples and len(seg) >= crossfade_samples:
                fade_out = np.linspace(1.0, 0.0, crossfade_samples, dtype=np.float32)
                fade_in = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)
                audio[-crossfade_samples:] *= fade_out
                seg[:crossfade_samples] *= fade_in
                audio[-crossfade_samples:] += seg[:crossfade_samples]
                audio = np.concatenate([audio, seg[crossfade_samples:]])
            else:
                audio = np.concatenate([audio, seg])

    # Trim to requested duration
    max_samples = int(sample_rate * duration)
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    return audio.astype(np.float32), sample_rate


def _spectrogram_image_to_audio(image, sample_rate: int) -> np.ndarray:
    """Convert a Riffusion spectrogram image to audio using the riffusion library.

    Uses riffusion's own SpectrogramParams, image_util, and torchaudio transforms
    to correctly invert the spectrogram back to a waveform.
    """
    import torch
    import torchaudio
    from riffusion.spectrogram_params import SpectrogramParams
    from riffusion.util import image_util

    params = SpectrogramParams()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Image → mel amplitude array using riffusion's exact conversion
    spectrogram = image_util.spectrogram_from_image(
        image, power=params.power_for_image, stereo=params.stereo, max_value=30e6,
    )

    # Build torchaudio transforms matching riffusion's parameters
    inverse_mel_scaler = torchaudio.transforms.InverseMelScale(
        n_stft=params.n_fft // 2 + 1,
        n_mels=params.num_frequencies,
        sample_rate=params.sample_rate,
        f_min=params.min_frequency,
        f_max=params.max_frequency,
        norm=params.mel_scale_norm,
        mel_scale=params.mel_scale_type,
    ).to(device)

    inverse_spectrogram_func = torchaudio.transforms.GriffinLim(
        n_fft=params.n_fft,
        n_iter=params.num_griffin_lim_iters,
        win_length=params.win_length,
        hop_length=params.hop_length,
        window_fn=torch.hann_window,
        power=1.0,
        momentum=0.99,
        rand_init=True,
    ).to(device)

    # Reconstruct: mel amplitudes → linear → waveform
    amplitudes_mel = torch.from_numpy(spectrogram).to(device)
    amplitudes_linear = inverse_mel_scaler(amplitudes_mel)
    waveform = inverse_spectrogram_func(amplitudes_linear)

    audio = waveform[0].cpu().numpy()  # take first channel

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    return audio.astype(np.float32)


def _load_stable_audio(device: str):
    """Load Stable Audio Open model."""
    from stable_audio_tools import get_pretrained_model
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    model = model.to(device)
    return {"model": model, "model_config": model_config}


def _infer_stable_audio(model_dict, params: dict, device: str) -> tuple:
    """Run Stable Audio Open inference. Returns (audio_numpy, sample_rate)."""
    import torch
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    prompt = params.get("prompt", "ambient electronic music")
    negative_prompt = params.get("negative_prompt", "")
    duration = min(float(params.get("duration", 30)), 47.0)
    steps = int(params.get("steps", 100))
    cfg_scale = float(params.get("cfg_scale", 7.0))
    sampler_type = params.get("sampler_type", "dpmpp-3m-sde")
    sigma_min = float(params.get("sigma_min", 0.3))
    sigma_max = float(params.get("sigma_max", 500))
    seed = int(params.get("seed", -1))
    if seed < 0:
        seed = np.random.randint(0, 2**31 - 1)  # library's 2**32-1 overflows int32 on Windows

    model = model_dict["model"]
    model_config = model_dict["model_config"]
    sample_rate = model_config["sample_rate"]
    sample_size = int(sample_rate * duration)

    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": duration,
    }]

    negative_conditioning = None
    if negative_prompt:
        negative_conditioning = [{
            "prompt": negative_prompt,
            "seconds_start": 0,
            "seconds_total": duration,
        }]

    output = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        negative_conditioning=negative_conditioning,
        sample_size=sample_size,
        seed=seed,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sampler_type=sampler_type,
        device=device,
    )

    # output shape: [batch, channels, samples] -> take first batch item
    audio = output[0].cpu().numpy()
    # If stereo (2, samples), transpose to (samples, 2) for soundfile
    if audio.ndim == 2 and audio.shape[0] in (1, 2):
        if audio.shape[0] == 1:
            audio = audio[0]  # mono: flatten to 1D
        else:
            audio = audio.T  # stereo: (samples, 2)

    return audio.astype(np.float32), sample_rate


def _load_generic(model_id: str, device: str):
    """Placeholder loader for models that need model-specific code."""
    logger.info("Loading model '%s' on %s (generic loader)", model_id, device)
    return {"model_id": model_id, "device": device, "loaded": True}


def _infer_generic(model, params: dict, device: str) -> tuple:
    """Placeholder inference — returns silence. Replace with real model code."""
    duration = float(params.get("duration", 10))
    sr = 44100
    samples = int(sr * duration)
    audio = np.zeros(samples, dtype=np.float32)
    logger.warning("Generic inference: returning %ds silence (model not fully integrated)", duration)
    return audio, sr


# Model dispatch tables
LOADERS = {
    "musicgen": _load_musicgen,
    "ace_step_v15": _load_ace_step_v15,
    "ace_step_v1": _load_ace_step_v1,
    "heartmula": _load_heartmula,
    "diffrhythm": _load_diffrhythm,
    "yue": _load_yue,
    "riffusion": _load_riffusion,
    "stable_audio": _load_stable_audio,
}

INFERENCERS = {
    "musicgen": _infer_musicgen,
    "ace_step_v15": _infer_ace_step_v15,
    "ace_step_v1": _infer_ace_step_v1,
    "heartmula": _infer_heartmula,
    "diffrhythm": _infer_diffrhythm,
    "yue": _infer_yue,
    "riffusion": _infer_riffusion,
    "stable_audio": _infer_stable_audio,
}


# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------
def _get_vram_info(device: str) -> dict:
    """Get VRAM usage for the device."""
    try:
        import torch
        if device.startswith("cuda") and torch.cuda.is_available():
            idx = int(device.split(":")[1]) if ":" in device else 0
            used = torch.cuda.memory_allocated(idx) // (1024 * 1024)
            total = torch.cuda.get_device_properties(idx).total_memory // (1024 * 1024)
            return {"vram_used_mb": used, "vram_total_mb": total}
    except Exception:
        pass
    return {"vram_used_mb": 0, "vram_total_mb": 0}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title=f"Music Worker ({MODEL_ID})")


class InferRequest(BaseModel):
    params: dict = {}


@app.get("/health")
async def health():
    vram = _get_vram_info(DEVICE)
    return {
        "status": "ready" if _model_loaded else ("loading" if _loading else "idle"),
        "model": MODEL_ID,
        "device": DEVICE,
        "port": PORT,
        **vram,
    }


@app.post("/load")
async def load():
    global _model, _model_loaded, _loading

    if _model_loaded:
        return {"status": "already_loaded", "model": MODEL_ID}

    _loading = True
    try:
        loader = LOADERS.get(MODEL_ID)
        if not loader:
            raise HTTPException(status_code=400, detail=f"Unknown model: {MODEL_ID}")

        logger.info("Loading model '%s' on %s ...", MODEL_ID, DEVICE)
        start = time.time()
        _model = loader(DEVICE)
        elapsed = time.time() - start
        _model_loaded = True
        logger.info("Model '%s' loaded in %.1fs", MODEL_ID, elapsed)
        return {"status": "loaded", "model": MODEL_ID, "load_time_sec": round(elapsed, 1)}
    except Exception as e:
        logger.error("Failed to load model '%s': %s", MODEL_ID, e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _loading = False


@app.post("/infer")
async def infer(req: InferRequest):
    global _model

    if not _model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Call /load first.")

    inferencer = INFERENCERS.get(MODEL_ID)
    if not inferencer:
        raise HTTPException(status_code=400, detail=f"No inferencer for model: {MODEL_ID}")

    try:
        logger.info("Starting inference for '%s' ...", MODEL_ID)
        start = time.time()
        audio_np, sample_rate = inferencer(_model, req.params, DEVICE)
        elapsed = time.time() - start
        logger.info("Inference complete in %.1fs (%.1fs audio at %dHz)",
                    elapsed, len(audio_np) / sample_rate, sample_rate)

        # Encode audio as base64 WAV
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio_np, sample_rate, format="WAV", subtype="PCM_16")
        audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "status": "ok",
            "audio_base64": audio_b64,
            "sample_rate": sample_rate,
            "duration_sec": round(len(audio_np) / sample_rate, 2),
            "inference_time_sec": round(elapsed, 2),
            "model": MODEL_ID,
        }

    except Exception as e:
        logger.error("Inference failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload")
async def unload():
    global _model, _model_loaded

    if not _model_loaded:
        return {"status": "not_loaded"}

    logger.info("Unloading model '%s' ...", MODEL_ID)
    _model = None
    _model_loaded = False

    # Free memory: gc.collect() first to release Python refs to GPU tensors,
    # then empty_cache() to reclaim the freed CUDA memory
    import gc
    gc.collect()

    try:
        import torch
        if DEVICE.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return {"status": "unloaded", "model": MODEL_ID}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting Music Worker: model=%s port=%d device=%s", MODEL_ID, PORT, DEVICE)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
