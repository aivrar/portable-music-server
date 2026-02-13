"""CLAP Scorer - FastAPI micro-service for audio-text similarity scoring.

Uses laion_clap (already installed in stable_audio_env) to compute
cosine similarity between audio and text embeddings.

Run: python clap_scorer.py --port 9250
"""

import argparse
import base64
import gc
import io
import logging

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
app = FastAPI(title="CLAP Scorer")

_model = None
_device = "cuda:0"


class ScoreRequest(BaseModel):
    audio_base64: str
    prompt: str


@app.get("/health")
def health():
    status = "ready" if _model is not None else "idle"
    return {"status": status}


@app.post("/load")
def load_model():
    global _model
    if _model is not None:
        return {"status": "already_loaded"}

    try:
        import laion_clap
        model = laion_clap.CLAP_Module(enable_fusion=False, device=_device)
        model.load_ckpt()  # 630k-audioset-best.pt (default)
        _model = model
        logger.info("CLAP model loaded on %s", _device)
        return {"status": "loaded", "device": _device}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load CLAP: {e}")


@app.post("/score")
def score(req: ScoreRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. POST /load first.")

    try:
        import soundfile as sf

        # Decode base64 audio to numpy
        audio_bytes = base64.b64decode(req.audio_base64)
        audio_data, sr = sf.read(io.BytesIO(audio_bytes))

        # Convert to mono if stereo
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample to 48kHz if needed (CLAP requirement)
        if sr != 48000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=48000)

        audio_data = audio_data.astype(np.float32)

        # Get embeddings
        audio_embed = _model.get_audio_embedding_from_data(
            [audio_data], use_tensor=False
        )
        text_embed = _model.get_text_embedding(
            [req.prompt], use_tensor=False
        )

        # Cosine similarity
        audio_embed = audio_embed / np.linalg.norm(audio_embed)
        text_embed = text_embed / np.linalg.norm(text_embed)
        similarity = float(np.dot(audio_embed.flatten(), text_embed.flatten()))
        score = max(0.0, min(1.0, similarity))

        return {"score": round(score, 4)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")


@app.post("/unload")
def unload_model():
    global _model
    if _model is None:
        return {"status": "already_unloaded"}

    _model = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    logger.info("CLAP model unloaded")
    return {"status": "unloaded"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLAP Scorer Service")
    parser.add_argument("--port", type=int, default=9250)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for CLAP model (cuda:0, cuda:1, cpu)")
    args = parser.parse_args()
    _device = args.device

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run(app, host=args.host, port=args.port)
