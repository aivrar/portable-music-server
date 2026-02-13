# audio_assembler.py
"""Audio assembly and format conversion for music generation.

Supports crossfade-based concatenation (music-specific) and format conversion via FFmpeg.
"""

import subprocess
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def assemble_chunks(chunk_files: list[str | Path], output_path: str | Path,
                    sr: int, crossfade_ms: int = 100,
                    front_pad: float = 0.0, end_pad: float = 0.0) -> str:
    """Concatenate audio chunks with optional crossfade.

    Args:
        chunk_files: Ordered list of audio file paths.
        output_path: Where to write the assembled audio.
        sr: Sample rate.
        crossfade_ms: Crossfade overlap in milliseconds (0 = no crossfade).
        front_pad: Seconds of silence at the beginning.
        end_pad: Seconds of silence at the end.

    Returns:
        The output_path as a string.
    """
    parts = []
    for f in chunk_files:
        try:
            data, file_sr = sf.read(str(f))
            parts.append(data)
        except Exception as e:
            logger.warning("Failed to read chunk %s: %s", f, e)

    if not parts:
        logger.error("No audio chunks to assemble")
        silence = np.zeros(int(sr * 0.1), dtype=np.float32)
        sf.write(str(output_path), silence, sr, subtype="PCM_16")
        return str(output_path)

    crossfade_samples = int(sr * crossfade_ms / 1000)

    if crossfade_samples > 0 and len(parts) > 1:
        # Crossfade assembly
        result = parts[0]
        for i in range(1, len(parts)):
            chunk = parts[i]
            overlap = min(crossfade_samples, len(result), len(chunk))
            if overlap > 0:
                # Create crossfade
                fade_out = np.linspace(1.0, 0.0, overlap)
                fade_in = np.linspace(0.0, 1.0, overlap)

                if result.ndim == 2:
                    fade_out = fade_out[:, np.newaxis]
                    fade_in = fade_in[:, np.newaxis]

                # Apply crossfade
                crossfaded = result[-overlap:] * fade_out + chunk[:overlap] * fade_in
                result = np.concatenate([result[:-overlap], crossfaded, chunk[overlap:]])
            else:
                result = np.concatenate([result, chunk])

        final_wav = result
    else:
        # Simple concatenation (no crossfade)
        final_wav = np.concatenate(parts)

    # Add padding
    if front_pad > 0:
        if final_wav.ndim == 2:
            front_silence = np.zeros((int(sr * front_pad), final_wav.shape[1]), dtype=np.float32)
        else:
            front_silence = np.zeros(int(sr * front_pad), dtype=np.float32)
        final_wav = np.concatenate([front_silence, final_wav])

    if end_pad > 0:
        if final_wav.ndim == 2:
            end_silence = np.zeros((int(sr * end_pad), final_wav.shape[1]), dtype=np.float32)
        else:
            end_silence = np.zeros(int(sr * end_pad), dtype=np.float32)
        final_wav = np.concatenate([final_wav, end_silence])

    sf.write(str(output_path), final_wav, sr, subtype="PCM_16")
    logger.info("Assembled %d chunks -> %s (%.1fs)", len(parts), output_path,
                len(final_wav) / sr)
    return str(output_path)


def _ffmpeg_args(fmt: str) -> list[str]:
    """Return ffmpeg encoding arguments for common output formats."""
    return {
        "mp3": ["-c:a", "libmp3lame", "-q:a", "0"],
        "ogg": ["-c:a", "libvorbis", "-q:a", "6"],
        "flac": ["-c:a", "flac", "-compression_level", "12"],
        "m4a": ["-c:a", "aac", "-b:a", "320k"],
    }.get(fmt, [])


def convert_format(input_path: str | Path, output_path: str | Path,
                   fmt: str = "wav", ffmpeg_path: str = "ffmpeg") -> str:
    """Convert audio file to the requested format via FFmpeg.

    Args:
        input_path: Source WAV file.
        output_path: Destination file path (should have correct extension).
        fmt: Target format (wav, mp3, ogg, flac, m4a).
        ffmpeg_path: Path to the ffmpeg executable.

    Returns:
        The output_path as a string.
    """
    if fmt == "wav":
        if str(input_path) != str(output_path):
            import shutil
            shutil.copy2(str(input_path), str(output_path))
        return str(output_path)

    cmd = [
        ffmpeg_path, "-y", "-i", str(input_path),
        *_ffmpeg_args(fmt),
        str(output_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Converted %s -> %s", input_path, output_path)
    except FileNotFoundError:
        logger.error("FFmpeg not found at '%s', returning WAV", ffmpeg_path)
        import shutil
        shutil.copy2(str(input_path), str(output_path))
    except subprocess.CalledProcessError as e:
        logger.error("FFmpeg conversion failed: %s", e.stderr)
        import shutil
        shutil.copy2(str(input_path), str(output_path))

    return str(output_path)
