# audio_processing.py
"""Unified 8-stage music post-processing (mastering) pipeline with graceful degradation.

Stages (each accepts/returns numpy array or operates in-place on file):
1. Noise reduction    - noisereduce with first 0.5s as noise profile (light)
2. 30Hz high-pass     - 4th-order Butterworth, preserves bass
3. Multi-band compress - 3-band (low/mid/high) with gentle 2:1 ratio
4. Stereo enhancement - Mid-side processing for wider image
5. EQ tonal balance   - Parametric EQ with presets (flat/balanced/warm/bright)
6. Silence trimming   - pydub detect_silence with protection zones
7. LUFS normalization  - pyloudnorm to target LUFS (-14 for streaming)
8. Peak limiting       - Hard clamp to 0.95 threshold
"""

import os
import time
import logging

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def _ts():
    return time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Stage 1: Noise reduction (light, for music)
# ---------------------------------------------------------------------------
def _apply_denoise(data: np.ndarray, rate: int, strength: float = 0.2) -> np.ndarray:
    """Light noise reduction using first 0.5s as noise profile."""
    if strength <= 0.0 or len(data) <= int(rate * 0.5):
        return data
    try:
        import noisereduce as nr
        noise_samples = int(rate * 0.5)
        if data.ndim == 2:
            # Process each channel separately for stereo audio
            channels = []
            for ch in range(data.shape[1]):
                ch_data = data[:, ch]
                ch_noise = ch_data[:noise_samples]
                channels.append(nr.reduce_noise(y=ch_data, sr=rate, y_noise=ch_noise, prop_decrease=strength))
            return np.column_stack(channels)
        noise_clip = data[:noise_samples]
        return nr.reduce_noise(y=data, sr=rate, y_noise=noise_clip, prop_decrease=strength)
    except ImportError:
        logger.warning("[%s POST] noisereduce not available, skipping denoise", _ts())
        return data


# ---------------------------------------------------------------------------
# Stage 2: High-pass filter (30Hz for music — preserve bass)
# ---------------------------------------------------------------------------
def _apply_highpass(data: np.ndarray, rate: int, cutoff: int = 30) -> np.ndarray:
    """4th-order Butterworth high-pass filter to remove subsonic rumble."""
    try:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(4, cutoff, 'high', fs=rate, output='sos')
        if data.ndim == 2:
            # Process each channel
            return np.column_stack([sosfiltfilt(sos, data[:, ch]) for ch in range(data.shape[1])])
        return sosfiltfilt(sos, data)
    except ImportError:
        logger.warning("[%s POST] scipy not available, skipping high-pass", _ts())
        return data


# ---------------------------------------------------------------------------
# Stage 3: Multi-band compression
# ---------------------------------------------------------------------------
def _apply_multiband_compression(data: np.ndarray, rate: int,
                                  ratio: float = 2.0) -> np.ndarray:
    """3-band compression (low <200Hz, mid 200Hz-4kHz, high >4kHz).
    Gentle 2:1 ratio to even out dynamics across frequency spectrum."""
    if ratio <= 1.0:
        return data
    try:
        from scipy.signal import butter, sosfiltfilt
    except ImportError:
        logger.warning("[%s POST] scipy not available, skipping compression", _ts())
        return data

    def compress_band(band: np.ndarray, ratio: float, threshold_db: float = -20.0) -> np.ndarray:
        """Apply simple compression to a frequency band."""
        rms = np.sqrt(np.mean(band ** 2)) + 1e-10
        rms_db = 20 * np.log10(rms)
        if rms_db > threshold_db:
            gain_reduction_db = (rms_db - threshold_db) * (1 - 1 / ratio)
            gain = 10 ** (-gain_reduction_db / 20)
            return band * gain
        return band

    def filter_band(data_1d, low, high, rate):
        if low is None:
            sos = butter(4, high, 'low', fs=rate, output='sos')
        elif high is None:
            sos = butter(4, low, 'high', fs=rate, output='sos')
        else:
            sos = butter(4, [low, high], 'band', fs=rate, output='sos')
        return sosfiltfilt(sos, data_1d)

    def process_1d(d):
        low_band = filter_band(d, None, 200, rate)
        mid_band = filter_band(d, 200, 4000, rate)
        high_band = filter_band(d, 4000, None, rate)
        return compress_band(low_band, ratio) + compress_band(mid_band, ratio) + compress_band(high_band, ratio)

    if data.ndim == 2:
        return np.column_stack([process_1d(data[:, ch]) for ch in range(data.shape[1])])
    return process_1d(data)


# ---------------------------------------------------------------------------
# Stage 4: Stereo enhancement (mid-side processing)
# ---------------------------------------------------------------------------
def _apply_stereo_enhancement(data: np.ndarray, width: float = 1.2) -> np.ndarray:
    """Enhance stereo width via mid-side processing.
    width: 0.0=mono, 1.0=normal, >1.0=wider."""
    if data.ndim != 2 or data.shape[1] != 2:
        return data  # Only works on stereo
    if abs(width - 1.0) < 1e-6:
        return data

    left = data[:, 0]
    right = data[:, 1]

    mid = (left + right) / 2
    side = (left - right) / 2

    # Adjust side channel relative to mid
    side = side * width

    new_left = mid + side
    new_right = mid - side

    return np.column_stack([new_left, new_right])


# ---------------------------------------------------------------------------
# Stage 5: EQ tonal balance
# ---------------------------------------------------------------------------
def _apply_eq(data: np.ndarray, rate: int, preset: str = "balanced") -> np.ndarray:
    """Gentle parametric EQ with presets.
    balanced: +1.5dB low shelf 100Hz, -1dB notch 3kHz, +1dB high shelf 10kHz
    """
    try:
        from audio_profiles import EQ_PRESETS
    except ImportError:
        logger.warning("[%s POST] audio_profiles not available, skipping EQ", _ts())
        return data
    gains = EQ_PRESETS.get(preset, EQ_PRESETS["flat"])
    low_db, mid_db, high_db = gains

    if abs(low_db) < 0.01 and abs(mid_db) < 0.01 and abs(high_db) < 0.01:
        return data

    try:
        from scipy.signal import butter, sosfiltfilt
    except ImportError:
        logger.warning("[%s POST] scipy not available, skipping EQ", _ts())
        return data

    def apply_shelf(d, freq, gain_db, shelf_type, rate):
        if abs(gain_db) < 0.01:
            return d
        gain_linear = 10 ** (gain_db / 20)
        if shelf_type == 'low':
            sos = butter(2, freq, 'low', fs=rate, output='sos')
        else:
            sos = butter(2, freq, 'high', fs=rate, output='sos')
        filtered = sosfiltfilt(sos, d)
        return d + (filtered * (gain_linear - 1))

    def process_1d(d):
        d = apply_shelf(d, 100, low_db, 'low', rate)
        # Mid notch at 3kHz: extract and reduce
        if abs(mid_db) > 0.01:
            sos_bp = butter(2, [2500, 3500], 'band', fs=rate, output='sos')
            mid_band = sosfiltfilt(sos_bp, d)
            gain = 10 ** (mid_db / 20)
            d = d + mid_band * (gain - 1)
        d = apply_shelf(d, 10000, high_db, 'high', rate)
        return d

    if data.ndim == 2:
        return np.column_stack([process_1d(data[:, ch]) for ch in range(data.shape[1])])
    return process_1d(data)


# ---------------------------------------------------------------------------
# Stage 6: Silence trimming
# ---------------------------------------------------------------------------
def _trim_silence(wav_path: str, profile: dict) -> None:
    """Trim leading/trailing silence with configurable protection zones."""
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_silence
    except ImportError:
        logger.warning("[%s POST] pydub not available, skipping silence trim", _ts())
        return

    trim_db = profile.get("trim_db", -45)
    min_silence = profile.get("min_silence_ms", 500)
    front_protect = profile.get("front_protect_ms", 200)
    end_protect = profile.get("end_protect_ms", 1000)

    try:
        audio = AudioSegment.from_wav(wav_path)
    except Exception as e:
        logger.warning("[%s POST] Failed to load audio for trimming: %s", _ts(), e)
        return

    sil = detect_silence(audio, min_silence_len=min_silence, silence_thresh=trim_db)

    start_trim = 0
    if sil and sil[0][0] == 0:
        front_ms = sil[0][1]
        start_trim = max(0, front_ms - front_protect)

    end_trim = 0
    if sil and sil[-1][1] == len(audio):
        tail_ms = len(audio) - sil[-1][0]
        end_trim = max(0, tail_ms - end_protect)

    if start_trim or end_trim:
        result_end = len(audio) - end_trim
        if result_end <= start_trim:
            logger.warning("[%s POST] Trim would produce empty audio, skipping", _ts())
            return
        trimmed = audio[start_trim:result_end]
        trimmed.export(wav_path, format="wav")
        logger.debug("[%s POST] Trimmed start=%dms end=%dms -> %dms",
                     _ts(), start_trim, end_trim, len(trimmed))


# ---------------------------------------------------------------------------
# Stage 7: LUFS normalization
# ---------------------------------------------------------------------------
def _normalize_loudness(wav_path: str, target_lufs: float = -14.0) -> None:
    """Normalize integrated loudness to target LUFS (-14 for streaming)."""
    try:
        import pyloudnorm as pyln
    except ImportError:
        logger.warning("[%s POST] pyloudnorm not available, skipping LUFS normalization", _ts())
        return

    try:
        data, rate = sf.read(wav_path)
    except Exception as e:
        logger.warning("[%s POST] Failed to load for normalization: %s", _ts(), e)
        return

    meter = pyln.Meter(rate)
    min_samples = int(0.4 * rate) + 1
    if len(data) < min_samples:
        logger.warning("[%s POST] Audio too short for LUFS normalization, skipping", _ts())
        return
    loudness = meter.integrated_loudness(data)
    if np.isinf(loudness):
        logger.warning("[%s POST] Measured loudness is -inf, skipping normalization", _ts())
        return
    normalized = pyln.normalize.loudness(data, loudness, target_lufs)
    sf.write(wav_path, normalized, rate, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Stage 8: Peak limiting
# ---------------------------------------------------------------------------
def _apply_peak_limit(wav_path: str, threshold: float = 0.95) -> None:
    """Hard clamp peaks to threshold."""
    try:
        data, rate = sf.read(wav_path)
    except Exception as e:
        logger.warning("[%s POST] Failed to load for peak limiting: %s", _ts(), e)
        return
    peak = np.max(np.abs(data))
    if peak > threshold:
        data = data * (threshold / peak)
        sf.write(wav_path, data, rate, subtype="PCM_16")
        logger.debug("[%s POST] Peak limited %.6f -> %.3f", _ts(), peak, threshold)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def post_process(wav_path: str, profile: dict,
                 denoise_strength: float | None = None,
                 stereo_width: float | None = None,
                 eq_preset: str | None = None,
                 target_lufs: float | None = None,
                 highpass_cutoff: int | None = None,
                 compression_ratio: float | None = None,
                 clipping: float | None = None,
                 trim_db: float | None = None,
                 min_silence_ms: int | None = None,
                 front_protect_ms: int | None = None,
                 end_protect_ms: int | None = None) -> str:
    """Run the full 8-stage music mastering pipeline on a WAV file.

    Args:
        wav_path: Path to the raw WAV file (modified in-place).
        profile: Audio profile dict from audio_profiles.PROFILES.
        denoise_strength: Override profile denoise (0.0-1.0).
        stereo_width: Override profile stereo width (0.0-2.0).
        eq_preset: Override profile EQ preset (flat/balanced/warm/bright).
        target_lufs: Override profile LUFS target.
        highpass_cutoff: Override profile highpass cutoff in Hz (20-200).
        compression_ratio: Override profile compression ratio (1.0-8.0).
        clipping: Override profile peak limit threshold (0.5-1.0).
        trim_db: Override profile silence detection threshold in dB.
        min_silence_ms: Override profile minimum silence duration in ms.
        front_protect_ms: Override profile front protection zone in ms.
        end_protect_ms: Override profile end protection zone in ms.

    Returns:
        The same wav_path (file is modified in-place).
    """
    if not os.path.exists(wav_path):
        logger.warning("[%s POST] File not found: %s", _ts(), wav_path)
        return wav_path

    try:
        data, rate = sf.read(wav_path)
    except Exception as e:
        logger.warning("[%s POST] Failed to load %s: %s", _ts(), wav_path, e)
        return wav_path

    # Resolve parameters (request overrides > profile > defaults)
    _denoise = denoise_strength if denoise_strength is not None else profile.get("denoise_strength", 0.2)
    _width = stereo_width if stereo_width is not None else profile.get("stereo_width", 1.2)
    _eq = eq_preset if eq_preset is not None else profile.get("eq_preset", "balanced")
    _lufs = target_lufs if target_lufs is not None else profile.get("lufs", -14.0)
    _cutoff = highpass_cutoff if highpass_cutoff is not None else profile.get("highpass_cutoff", 30)
    _ratio = compression_ratio if compression_ratio is not None else profile.get("compression_ratio", 2.0)
    _clipping = clipping if clipping is not None else profile.get("clipping", 0.95)

    logger.info("[%s POST] Starting 8-stage pipeline: denoise=%.2f width=%.2f eq=%s lufs=%.1f",
                _ts(), _denoise, _width, _eq, _lufs)

    # Stages 1-5 operate on numpy arrays — wrap each so one failure doesn't kill the pipeline
    for stage_name, stage_fn in [
        ("Noise reduction", lambda d: _apply_denoise(d, rate, _denoise)),
        ("High-pass", lambda d: _apply_highpass(d, rate, _cutoff)),
        ("Multi-band compression", lambda d: _apply_multiband_compression(d, rate, _ratio)),
        ("Stereo enhancement", lambda d: _apply_stereo_enhancement(d, _width)),
        ("EQ tonal balance", lambda d: _apply_eq(d, rate, _eq)),
    ]:
        try:
            data = stage_fn(data)
        except Exception as e:
            logger.warning("[%s POST] Stage '%s' failed, skipping: %s", _ts(), stage_name, e)

    # Save intermediate for pydub-based trimming
    sf.write(wav_path, data, rate, subtype="PCM_16")

    # Stage 6: Silence trimming (build override profile for trim params)
    trim_profile = dict(profile)
    if trim_db is not None:
        trim_profile["trim_db"] = trim_db
    if min_silence_ms is not None:
        trim_profile["min_silence_ms"] = min_silence_ms
    if front_protect_ms is not None:
        trim_profile["front_protect_ms"] = front_protect_ms
    if end_protect_ms is not None:
        trim_profile["end_protect_ms"] = end_protect_ms
    _trim_silence(wav_path, trim_profile)

    # Stage 7: LUFS normalization
    _normalize_loudness(wav_path, _lufs)

    # Stage 8: Peak limiting
    _apply_peak_limit(wav_path, _clipping)

    logger.info("[%s POST] Pipeline complete: %s", _ts(), wav_path)
    return wav_path
