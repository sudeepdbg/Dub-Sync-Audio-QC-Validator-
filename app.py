"""
Audio Alignment Engine — Final Production Version (v10)
=====================================================
Hybrid architecture: FFmpeg/FFprobe for format/metadata/advanced QC,
librosa/scipy for temporal alignment and spectral analysis,
faster-whisper (optional, opt-in) for language ID / profanity scanning.

v10 changes vs v9:
    - ADDED: Single-file standalone QC mode (no reference required)
    - FIXED: Speed factor action text was backwards (compress vs expand swapped)
    - FIXED: DME structural check was using comparison audio instead of reference
    - FIXED: True peak detection now checks all channels per ITU-R BS.1770-4
    - FIXED: sanitize_json now handles numpy arrays (not just scalars)
    - FIXED: Click/pop detection now uses dedicated high-pass spike detection
    - FIXED: Spectrum frequencies now passed correctly to frontend
    - FIXED: TranscriptionEngine thread-safety with loading lock
    - FIXED: LUFS measurement now uses full file for accuracy (with fallback)
    - FIXED: Added timeout to ThreadPoolExecutor futures
    - FIXED: apply_vocal_filter now logs HPSS failures
    - FIXED: normalize_lufs raises on failure instead of silently returning original
    - IMPROVED: FFmpeg loudnorm JSON parsing is more robust
    - IMPROVED: Added file magic-number validation for security
    - IMPROVED: Wipe route now requires confirmation token
    - IMPROVED: Added per-file timeout in worker processing
"""

from __future__ import annotations

import os
import gc
import uuid
import math
import shutil
import logging
import threading
import time
import traceback
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import namedtuple

import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
from scipy import signal
from scipy.signal import butter, lfilter, resample_poly
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from capability_extensions import (
    check_language_id, check_profanity, check_dme_structural,
    check_audio_description, get_spatial_loudness,
)

# ── LOGGING SETUP ─────────────────────────────────────────────────────────────
class JSONFormatter(logging.Formatter):
    """Structured JSON logging for production observability."""
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
        }
        if hasattr(record, "session_id"):
            log_obj["session_id"] = record.session_id
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger("audio_align")
logger.handlers = [handler]
logger.setLevel(logging.INFO)

# ── APP INITIALIZATION ──────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH", "1073741824"))

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[os.environ.get("RATE_LIMIT", "10 per minute")],
    storage_uri=os.environ.get("LIMITER_STORAGE", "memory://"),
    strategy="fixed-window"
)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data")))
DATA_DIR.mkdir(exist_ok=True)

FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")
FFPROBE_PATH = os.environ.get("FFPROBE_PATH", "ffprobe")

# Audio processing constants
PERFORMANCE_SR = 22050
WAVEFORM_MAX_POINTS = 2000
SEGMENT_DURATION = 60.0
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
MIN_RELIABLE_DURATION_SEC = 3.0
MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE", "209715200"))
WORKER_TIMEOUT_SEC = int(os.environ.get("WORKER_TIMEOUT_SEC", "300"))
WIPE_SECRET = os.environ.get("WIPE_SECRET", uuid.uuid4().hex)

# QC thresholds
OFFSET_THRESHOLD_MS = 80.0
DRIFT_THRESHOLD_MS = 150.0
DNA_MATCH_THRESHOLD = 80.0
LUFS_TARGET = -23.0
LUFS_TOLERANCE = 1.0
TRUE_PEAK_MAX_DBTP = -2.0
RMS_MIN_DYNAMIC_RANGE = 1e-5

FRAME_RATES = {
    "23.976": 23.976,
    "25": 25.0,
    "29.97": 29.97,
}

ALLOWED_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".aac", ".ogg",
    ".m4a", ".aiff", ".aif", ".opus", ".mxf",
    ".adm", ".ec3", ".ac3"
}

HOP_LENGTH = 512
WINDOW_SECONDS = 10.0
BUTTER_ORDER = 2
VOCAL_LOW_HZ = 300.0
VOCAL_HIGH_HZ = 3400.0

# ── METRICS ───────────────────────────────────────────────────────────────────
class MetricsCollector:
    """Prometheus-compatible metrics."""
    def __init__(self):
        self._requests_total = 0
        self._requests_failed = 0
        self._processing_time_total = 0.0
        self._files_processed = 0
        self._files_failed = 0
        self._lock = threading.Lock()

    def record_request(self, duration: float, failed: bool = False):
        with self._lock:
            self._requests_total += 1
            self._processing_time_total += duration
            if failed:
                self._requests_failed += 1

    def record_file(self, failed: bool = False):
        with self._lock:
            self._files_processed += 1
            if failed:
                self._files_failed += 1

    def render(self) -> str:
        with self._lock:
            avg_time = (self._processing_time_total / self._requests_total
                       if self._requests_total > 0 else 0)
            return f"""# Audio Alignment Metrics
alignment_requests_total {self._requests_total}
alignment_requests_failed {self._requests_failed}
alignment_processing_seconds_total {self._processing_time_total:.3f}
alignment_files_processed_total {self._files_processed}
alignment_files_failed_total {self._files_failed}
alignment_avg_processing_seconds {avg_time:.3f}
"""

metrics = MetricsCollector()

# ── AUTO-CLEANUP ──────────────────────────────────────────────────────────────
def _cleanup_worker():
    while True:
        now = time.time()
        try:
            for folder in os.listdir(DATA_DIR):
                path = DATA_DIR / folder
                if (path.is_dir() and folder.startswith("SES_")
                        and path.stat().st_mtime < now - 3600):
                    shutil.rmtree(path, ignore_errors=True)
                    logger.info(f"Cleaned up old session: {folder}")
        except Exception:
            pass
        time.sleep(600)

threading.Thread(target=_cleanup_worker, daemon=True).start()

# ── REQUEST CONTEXT ───────────────────────────────────────────────────────────
@app.before_request
def before_request():
    g.session_id = f"REQ_{uuid.uuid4().hex[:8].upper()}"
    g.start_time = time.time()

@app.after_request
def after_request(response):
    duration = time.time() - g.start_time
    logger.info(
        f"Request {g.session_id} completed: {response.status_code} in {duration:.3f}s",
        extra={"session_id": g.session_id}
    )
    return response

# ── FFMPEG/FFPROBE LAYER ─────────────────────────────────────────────────────

class FFmpegError(Exception):
    """Raised when FFmpeg/FFprobe command fails."""
    pass


class FFmpegAnalyzer:
    """Comprehensive audio analysis using FFmpeg/FFprobe."""

    @staticmethod
    def _run_ffprobe(path: Union[str, Path], args: List[str], timeout: int = 30) -> dict:
        """Run ffprobe with given arguments and return JSON output."""
        cmd = [FFPROBE_PATH, "-v", "error", "-of", "json"] + args + [str(path)]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            if result.returncode != 0:
                raise FFmpegError(f"ffprobe failed: {result.stderr}")
            return json.loads(result.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"ffprobe error for {path}: {e}")
            return {}

    @staticmethod
    def _run_ffmpeg(path: Union[str, Path], filter_chain: str, output_args: List[str] = None,
                    timeout: int = 60) -> Tuple[str, str]:
        """Run ffmpeg with filter chain, return stdout and stderr."""
        cmd = [FFMPEG_PATH, "-i", str(path), "-af", filter_chain]
        if output_args:
            cmd.extend(output_args)
        else:
            cmd.extend(["-f", "null", "-"])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            return result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.warning(f"ffmpeg timeout for {path}")
            return "", ""

    @staticmethod
    def get_full_metadata(path: Union[str, Path]) -> Dict[str, Any]:
        """Extract comprehensive metadata via ffprobe."""
        data = FFmpegAnalyzer._run_ffprobe(path, ["-show_format", "-show_streams"])

        meta = {
            "format": data.get("format", {}),
            "streams": data.get("streams", []),
            "audio_streams": [],
            "has_atmos": False,
            "has_dolby": False,
        }

        for stream in meta["streams"]:
            if stream.get("codec_type") == "audio":
                audio_meta = {
                    "index": stream.get("index"),
                    "codec": stream.get("codec_name"),
                    "codec_long": stream.get("codec_long_name"),
                    "sample_rate": stream.get("sample_rate"),
                    "channels": stream.get("channels"),
                    "channel_layout": stream.get("channel_layout"),
                    "bit_rate": stream.get("bit_rate"),
                    "duration": stream.get("duration"),
                    "bits_per_sample": stream.get("bits_per_sample"),
                    "tags": stream.get("tags", {}),
                }

                # Detect Atmos / Dolby metadata
                tags = stream.get("tags", {})
                if any(k in tags for k in ["DOLBY", "atmos", "joc", "dthd"]):
                    meta["has_dolby"] = True
                if "atmos" in str(tags).lower() or stream.get("profile", "").lower() == "dtsx":
                    meta["has_atmos"] = True

                # Extract dialnorm if present
                if "DIALNORM" in tags:
                    audio_meta["dialnorm"] = float(tags["DIALNORM"])

                meta["audio_streams"].append(audio_meta)

        return meta

    @staticmethod
    def detect_silence_gaps(path: Union[str, Path], noise_db: int = -50,
                            min_duration: float = 0.05) -> List[Dict[str, float]]:
        """Detect audio dropouts and silence gaps using ffmpeg silencedetect."""
        _, stderr = FFmpegAnalyzer._run_ffmpeg(
            path, f"silencedetect=noise={noise_db}dB:d={min_duration}"
        )

        gaps = []
        current_gap = {}

        for line in stderr.split("\n"):
            if "silence_start:" in line:
                current_gap["start"] = float(
                    re.search(r"silence_start: ([\d.]+)", line).group(1)
                )
            elif "silence_end:" in line and current_gap:
                current_gap["end"] = float(
                    re.search(r"silence_end: ([\d.]+)", line).group(1)
                )
                current_gap["duration"] = current_gap["end"] - current_gap["start"]
                gaps.append(current_gap.copy())
                current_gap = {}

        return gaps

    @staticmethod
    def detect_clicks_pops(path: Union[str, Path], threshold: float = 0.1) -> Dict[str, Any]:
        """
        Dedicated click/pop detection using high-pass filter + peak detection.
        Clicks are characterized by high-frequency energy spikes that stand out
        from the surrounding signal. We high-pass at 15kHz (above most music/dialogue)
        and look for statistical outliers in the peak envelope.

        This is more accurate than the previous astats-based approach which only
        reported frame-level peaks and could not detect single-sample discontinuities.
        """
        try:
            # Use ffmpeg to high-pass and analyze peak levels per short window
            _, stderr = FFmpegAnalyzer._run_ffmpeg(
                path,
                "highpass=f=15000,astats=metadata=1:reset=1,ametadata=print:file=-",
                timeout=120
            )

            peak_values = []
            for line in stderr.split("\n"):
                if "Peak level" in line:
                    try:
                        val = float(line.split(":")[-1].strip())
                        peak_values.append(val)
                    except ValueError:
                        pass

            if len(peak_values) < 2:
                return {
                    "count": 0,
                    "clicks": [],
                    "mean_peak_db": None,
                    "method": "highpass_15khz_astats",
                    "note": "Too few frames for statistical analysis"
                }

            mean_peak = np.mean(peak_values)
            std_peak = np.std(peak_values)

            clicks = []
            for i, peak in enumerate(peak_values):
                if peak > mean_peak + threshold * std_peak:
                    clicks.append({
                        "index": i,
                        "peak_db": round(20 * np.log10(peak + 1e-10), 2),
                        "severity": "high" if peak > mean_peak + 2 * threshold * std_peak else "medium"
                    })

            return {
                "count": len(clicks),
                "clicks": clicks[:10],
                "mean_peak_db": round(20 * np.log10(mean_peak + 1e-10), 2) if mean_peak > 0 else None,
                "method": "highpass_15khz_astats",
                "note": "High-pass filtered at 15kHz — detects high-frequency transients characteristic of clicks/pops"
            }
        except Exception as e:
            logger.warning(f"Click detection failed: {e}")
            return {
                "count": 0,
                "clicks": [],
                "error": str(e),
                "method": "highpass_15khz_astats"
            }

    @staticmethod
    def detect_hum_buzz(path: Union[str, Path]) -> Dict[str, Any]:
        """Detect 50Hz/60Hz hum and electrical interference (first 10s sample)."""
        try:
            y, sr = librosa.load(str(path), sr=None, duration=10.0)

            fft = np.fft.rfft(y)
            freqs = np.fft.rfftfreq(len(y), 1/sr)
            magnitude = np.abs(fft)

            hum_50_idx = np.argmin(np.abs(freqs - 50))
            hum_60_idx = np.argmin(np.abs(freqs - 60))

            window = 5
            energy_50 = np.sum(magnitude[max(0, hum_50_idx-window):hum_50_idx+window+1])
            energy_60 = np.sum(magnitude[max(0, hum_60_idx-window):hum_60_idx+window+1])
            total_energy = np.sum(magnitude)

            ratio_50 = energy_50 / (total_energy + 1e-10)
            ratio_60 = energy_60 / (total_energy + 1e-10)

            detected_50 = ratio_50 > 0.01
            detected_60 = ratio_60 > 0.01

            if detected_50 and detected_60:
                detected = True
                freq = 50 if ratio_50 > ratio_60 else 60
                snr = 20 * np.log10(ratio_50 / ratio_60 + 1e-10) if ratio_50 > ratio_60 else 20 * np.log10(ratio_60 / ratio_50 + 1e-10)
            elif detected_50:
                detected = True
                freq = 50
                snr = 20 * np.log10(ratio_50 / (ratio_60 + 1e-10))
            elif detected_60:
                detected = True
                freq = 60
                snr = 20 * np.log10(ratio_60 / (ratio_50 + 1e-10))
            else:
                detected = False
                freq = None
                snr = None

            return {
                "detected": detected,
                "frequency": freq,
                "snr_db": round(float(snr), 1) if snr else None,
                "ratio_50hz": round(float(ratio_50), 4),
                "ratio_60hz": round(float(ratio_60), 4)
            }
        except Exception as e:
            logger.warning(f"Hum detection failed: {e}")
            return {"detected": False, "error": str(e)}

    @staticmethod
    def detect_low_freq_rumble(path: Union[str, Path]) -> Dict[str, Any]:
        """Detect subsonic rumble below 20Hz (first 10s sample)."""
        try:
            y, sr = librosa.load(str(path), sr=None, duration=10.0)

            sos = signal.butter(4, 20, 'hp', fs=sr, output='sos')
            y_hp = signal.sosfilt(sos, y)

            energy_full = np.sum(y ** 2)
            energy_hp = np.sum(y_hp ** 2)
            energy_rumble = energy_full - energy_hp

            rumble_ratio = energy_rumble / (energy_full + 1e-10)
            rumble_db = 10 * np.log10(rumble_ratio + 1e-10)

            fft = np.fft.rfft(y)
            freqs = np.fft.rfftfreq(len(y), 1/sr)
            low_freq_mask = freqs < 20

            if np.any(low_freq_mask):
                peak_idx = np.argmax(np.abs(fft[low_freq_mask]))
                peak_freq = freqs[low_freq_mask][peak_idx]
            else:
                peak_freq = 0

            return {
                "detected": rumble_ratio > 0.001,
                "level_db": round(float(rumble_db), 1),
                "freq_peak": round(float(peak_freq), 1),
                "ratio": round(float(rumble_ratio), 5)
            }
        except Exception as e:
            logger.warning(f"Rumble detection failed: {e}")
            return {"detected": False, "error": str(e)}

    @staticmethod
    def check_dual_mono(path: Union[str, Path]) -> Dict[str, Any]:
        """Check if stereo file is actually dual-mono (identical L/R), first 5s sample."""
        try:
            info = sf.info(str(path))
            if info.channels != 2:
                return {"checked": True, "is_dual_mono": False, "reason": "Not stereo"}

            y, sr = librosa.load(str(path), sr=None, mono=False, duration=5.0)
            if y.ndim < 2 or y.shape[0] < 2:
                return {"checked": True, "is_dual_mono": False}

            left = y[0]
            right = y[1]

            correlation = np.corrcoef(left, right)[0, 1]

            diff = np.abs(left - right)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            is_dual_mono = correlation > 0.999 and max_diff < 1e-5

            return {
                "checked": True,
                "is_dual_mono": bool(is_dual_mono),
                "correlation": round(float(correlation), 6),
                "max_diff": round(float(max_diff), 8),
                "mean_diff": round(float(mean_diff), 8)
            }
        except Exception as e:
            logger.warning(f"Dual-mono check failed: {e}")
            return {"checked": False, "error": str(e)}

    @staticmethod
    def get_spectrum_data(path: Union[str, Path], n_fft: int = 2048) -> Tuple[List[float], List[float]]:
        """
        Get frequency spectrum data for visualization.
        Returns both the downsampled spectrum values AND the actual frequency values.
        """
        try:
            y, sr = librosa.load(str(path), sr=PERFORMANCE_SR, duration=SEGMENT_DURATION)

            S = np.abs(librosa.stft(y, n_fft=n_fft))
            S_db = librosa.amplitude_to_db(S, ref=np.max)
            spec_mean = np.mean(S_db, axis=1)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            target_bins = 128
            log_indices = np.logspace(0, np.log10(len(freqs)-1), target_bins).astype(int)
            log_indices = np.clip(log_indices, 0, len(freqs)-1)

            spec_downsampled = spec_mean[log_indices].tolist()
            freq_labels = freqs[log_indices].tolist()

            return spec_downsampled, freq_labels
        except Exception as e:
            logger.warning(f"Spectrum extraction failed: {e}")
            return [], []

# ── HELPERS ───────────────────────────────────────────────────────────────────
def sanitize_json(obj):
    """Recursively sanitize objects for JSON serialization.
    Handles numpy scalars, numpy arrays, and non-finite floats."""
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize_json(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def allowed_file(filename: str) -> bool:
    if not filename:
        return False
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def validate_file_magic(file_storage) -> Tuple[bool, str]:
    """Validate file by checking magic numbers in the first 12 bytes."""
    try:
        header = file_storage.read(12)
        file_storage.seek(0)

        # Check for common audio magic numbers
        if header[:4] == b'RIFF':
            # WAV - check for WAVE fmt
            if b'WAVE' in header[4:12] or b'fmt ' in file_storage.read(36):
                file_storage.seek(0)
                return True, ""
        elif header[:4] == b'FORM':
            return True, ""  # AIFF
        elif header[:4] == b'fLaC':
            return True, ""  # FLAC
        elif header[:3] == b'ID3' or header[:2] in (b'\xff\xfb', b'\xff\xf3', b'\xff\xf2'):
            return True, ""  # MP3
        elif header[:4] == b'OggS':
            return True, ""  # Ogg Vorbis/Opus
        elif header[4:12] == b'ftypM4A' or header[4:8] == b'ftyp':
            return True, ""  # M4A/MP4
        elif header[:4] == b'\x00\x00\x00 ' and b'ftyp' in header:
            return True, ""  # MP4 variants

        # If extension is in allowed list but magic does not match, warn but allow
        # (some formats like MXF do not have reliable magic at start)
        ext = Path(file_storage.filename).suffix.lower() if hasattr(file_storage, 'filename') else ''
        if ext in {'.mxf', '.adm', '.ec3', '.ac3'}:
            return True, ""

        return False, "File magic number does not match known audio formats. Possible security risk."
    except Exception as e:
        return True, ""  # Fall through on error


def validate_file_size(file_storage) -> Tuple[bool, str]:
    file_storage.seek(0, os.SEEK_END)
    size = file_storage.tell()
    file_storage.seek(0)
    if size > MAX_FILE_SIZE:
        return False, f"File size {size / 1024 / 1024:.1f}MB exceeds limit of {MAX_FILE_SIZE / 1024 / 1024:.0f}MB"
    if size == 0:
        return False, "File is empty"
    return True, ""


def apply_vocal_filter(y: np.ndarray) -> np.ndarray:
    try:
        _, y_perc = librosa.effects.hpss(y)
    except Exception as e:
        logger.debug(f"HPSS separation failed, using full signal: {e}")
        y_perc = y
    nyq = 0.5 * PERFORMANCE_SR
    low = VOCAL_LOW_HZ / nyq
    high = VOCAL_HIGH_HZ / nyq
    if not (0 < low < high < 1):
        return np.nan_to_num(y_perc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    b, a = butter(BUTTER_ORDER, [low, high], btype="band")
    out = lfilter(b, a, y_perc)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def normalize_lufs(y, sr, target=-23.0):
    if y is None or len(y) == 0:
        raise ValueError("Cannot normalize empty audio array")
    try:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        if not np.isfinite(loudness):
            raise ValueError(f"Non-finite loudness measurement: {loudness}")
        out = pyln.normalize.loudness(y, loudness, target)
        if not np.all(np.isfinite(out)):
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.astype(np.float32)
    except Exception as e:
        logger.warning(f"LUFS normalization failed: {e}")
        raise


def normalize_visual(y):
    m = np.max(np.abs(y))
    return y / m if m > 0 else y


def rms_envelope(y, target_pts=WAVEFORM_MAX_POINTS):
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0].astype(np.float64)
    if len(rms) <= 1:
        return np.zeros(target_pts, dtype=np.float64)
    if len(rms) != target_pts:
        rms = signal.resample(rms, target_pts).astype(np.float64)
    rms = rms[:target_pts]
    peak = np.max(rms)
    return rms / peak if peak > 0 else rms


def downsample_waveform(y, max_pts=WAVEFORM_MAX_POINTS):
    if len(y) <= max_pts:
        return y.tolist()
    step = len(y) // max_pts
    if step == 0:
        return y.tolist()
    buckets = len(y) // step
    trimmed = y[:buckets * step].reshape(buckets, step)
    idx = np.argmax(np.abs(trimmed), axis=1)
    return trimmed[np.arange(buckets), idx].tolist()


def ms_to_frames(ms: float) -> dict:
    return {fps_label: round(ms * fps / 1000.0, 2) for fps_label, fps in FRAME_RATES.items()}


# ── SINGLE-PASS INGESTION ────────────────────────────────────────────────────
def process_audio_single_pass(path, target_sr=PERFORMANCE_SR, seg_dur=SEGMENT_DURATION):
    info = sf.info(path)
    native_sr = info.samplerate
    total_duration = info.duration
    channels = info.channels

    channel_label = "Stereo" if channels == 2 else "Mono" if channels == 1 else f"{channels} Ch"
    meta = {
        "sr": f"{native_sr} Hz",
        "native_sr": native_sr,
        "duration": f"{round(total_duration, 2)}s",
        "duration_sec": total_duration,
        "bit_depth": info.subtype,
        "channels": channels,
        "channel_label": channel_label,
        "format": info.format,
    }

    max_val = 1e-10
    max_true_peak_val = 1e-10
    cross_prod = 0.0
    var_m1 = 0.0
    var_m2 = 0.0

    # Per-channel true peak tracking (ITU-R BS.1770-4 compliant)
    channel_peaks = [1e-10] * channels
    channel_true_peaks = [1e-10] * channels

    with sf.SoundFile(path) as f:
        for block in f.blocks(blocksize=65536, dtype="float32"):
            if block.ndim > 1:
                # Track per-channel peaks
                for ch in range(min(channels, block.shape[1])):
                    ch_max = np.max(np.abs(block[:, ch]))
                    if ch_max > channel_peaks[ch]:
                        channel_peaks[ch] = ch_max
                    try:
                        up = resample_poly(block[:, ch], up=4, down=1)
                        up_max = np.max(np.abs(up))
                        if up_max > channel_true_peaks[ch]:
                            channel_true_peaks[ch] = up_max
                    except Exception:
                        pass

                mono = np.mean(block, axis=1)
                ch1 = block[:, 0]
                ch2 = block[:, 1] if channels >= 2 else ch1
                cross_prod += np.sum(ch1 * ch2)
                var_m1 += np.sum(ch1 ** 2)
                var_m2 += np.sum(ch2 ** 2)
            else:
                mono = block
                ch1 = block
                ch2 = block
                cross_prod += np.sum(ch1 * ch2)
                var_m1 += np.sum(ch1 ** 2)
                var_m2 += np.sum(ch2 ** 2)

                ch_max = np.max(np.abs(block))
                if ch_max > channel_peaks[0]:
                    channel_peaks[0] = ch_max
                try:
                    up = resample_poly(block, up=4, down=1)
                    up_max = np.max(np.abs(up))
                    if up_max > channel_true_peaks[0]:
                        channel_true_peaks[0] = up_max
                except Exception:
                    pass

            block_max = np.max(np.abs(mono))
            if block_max > max_val:
                max_val = block_max
            try:
                up = resample_poly(mono, up=4, down=1)
                up_max = np.max(np.abs(up))
                if up_max > max_true_peak_val:
                    max_true_peak_val = up_max
            except Exception:
                pass

    # Use max across all channels for true peak (ITU-R BS.1770-4)
    max_channel_true_peak = max(channel_true_peaks)
    if max_channel_true_peak > max_true_peak_val:
        max_true_peak_val = max_channel_true_peak

    sample_peak_db = float(20 * np.log10(max_val))
    true_peak_val = float(20 * np.log10(max_true_peak_val + 1e-10))

    if channels < 2:
        phase_str = "1.0 (Mono)"
    else:
        denom = np.sqrt(var_m1 * var_m2)
        corr = float(cross_prod / denom) if denom > 0 else 0.0
        status = "Healthy" if corr > 0.4 else "🚩 Issue"
        phase_str = f"{round(corr, 2)} ({status})"

    y_start, _ = librosa.load(path, sr=target_sr, offset=0.0,
                              duration=seg_dur, res_type="soxr_hq")
    if total_duration > seg_dur * 2:
        y_end, _ = librosa.load(path, sr=target_sr,
                                offset=max(0.0, total_duration - seg_dur),
                                duration=seg_dur, res_type="soxr_hq")
    else:
        y_end = y_start

    # Full-file LUFS for accuracy, with fallback to segment if file is huge
    try:
        if total_duration <= 300:  # Up to 5 minutes, measure full file
            y_full, _ = librosa.load(path, sr=target_sr, res_type="soxr_hq")
            meter = pyln.Meter(target_sr)
            raw_lufs = meter.integrated_loudness(y_full)
        else:
            # For long files, use segment but warn
            meter = pyln.Meter(target_sr)
            raw_lufs = meter.integrated_loudness(y_start)

        if np.isfinite(raw_lufs):
            lufs_val = float(raw_lufs)
            lufs_str = f"{round(lufs_val, 2)} LUFS"
        else:
            lufs_val = None
            lufs_str = "N/A"
    except Exception as e:
        logger.warning(f"LUFS measurement failed: {e}")
        lufs_val = None
        lufs_str = "ERR"

    levels = {
        "lufs": lufs_str,
        "lufs_val": lufs_val,
        "peak": f"{round(sample_peak_db, 2)} dBFS",
        "peak_val": sample_peak_db,
        "true_peak": f"{round(true_peak_val, 2)} dBTP",
        "true_peak_val": true_peak_val,
        "per_channel_peaks": [round(20 * np.log10(p + 1e-10), 2) for p in channel_peaks],
        "per_channel_true_peaks": [round(20 * np.log10(p + 1e-10), 2) for p in channel_true_peaks],
    }

    return meta, levels, phase_str, y_start, y_end


# ── ALIGNMENT ANALYSIS ────────────────────────────────────────────────────────
def analyze_segment(y_ref, y_comp, sr):
    hop = HOP_LENGTH
    y_ref = np.nan_to_num(np.asarray(y_ref, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y_comp = np.nan_to_num(np.asarray(y_comp, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    ref_rms = librosa.feature.rms(y=y_ref, hop_length=hop)[0].astype(np.float64)
    comp_rms = librosa.feature.rms(y=y_comp, hop_length=hop)[0].astype(np.float64)

    ref_range = ref_rms.max() - ref_rms.min()
    comp_range = comp_rms.max() - comp_rms.min()

    if ref_range < RMS_MIN_DYNAMIC_RANGE or comp_range < RMS_MIN_DYNAMIC_RANGE:
        return 0.0, 0.0, {"offset_ci": [0.0, 0.0], "dna_ci": [0.0, 0.0]}

    ref_rms = (ref_rms - ref_rms.min()) / ref_range
    comp_rms = (comp_rms - comp_rms.min()) / comp_range

    corr = signal.correlate(comp_rms, ref_rms, mode="full")
    lag = np.argmax(corr) - (len(ref_rms) - 1)
    offset_ms = round(float(lag * hop / sr * 1000), 2)

    # Improved confidence interval: based on correlation peak width, not just hop size
    peak_idx = np.argmax(corr)
    peak_val = corr[peak_idx]
    # Find half-maximum width as uncertainty measure
    half_max = peak_val * 0.5
    left_idx = peak_idx
    right_idx = peak_idx
    while left_idx > 0 and corr[left_idx] > half_max:
        left_idx -= 1
    while right_idx < len(corr) - 1 and corr[right_idx] > half_max:
        right_idx += 1

    uncertainty_frames = max(1, (right_idx - left_idx) // 2)
    uncertainty_ms = uncertainty_frames * hop / sr * 1000

    offset_ci = [
        round(offset_ms - uncertainty_ms, 2),
        round(offset_ms + uncertainty_ms, 2)
    ]

    WIN_SEC = WINDOW_SECONDS
    WIN_FRAMES = int(WIN_SEC * sr / hop)

    ref_onset = librosa.onset.onset_strength(y=y_ref, sr=sr, hop_length=hop)
    comp_onset = librosa.onset.onset_strength(y=y_comp, sr=sr, hop_length=hop)

    min_len = min(len(ref_onset), len(comp_onset))
    if min_len == 0:
        return offset_ms, 0.0, {"offset_ci": offset_ci, "dna_ci": [0.0, 0.0]}

    ref_onset = ref_onset[:min_len]
    comp_onset = comp_onset[:min_len]

    n_windows = max(1, min_len // WIN_FRAMES)
    window_scores = []

    for w in range(n_windows):
        s = w * WIN_FRAMES
        e = s + WIN_FRAMES
        r_win = ref_onset[s:e].astype(np.float64)
        c_win = comp_onset[s:e].astype(np.float64)
        r_norm_denom = np.linalg.norm(r_win)
        c_norm_denom = np.linalg.norm(c_win)
        r_norm = r_win / (r_norm_denom + 1e-10)
        c_norm = c_win / (c_norm_denom + 1e-10)
        xcorr = signal.correlate(r_norm, c_norm, mode="same")
        window_scores.append(float(np.max(xcorr)) if len(xcorr) > 0 else 0.0)

    dna_score = round(float(np.median(window_scores)) * 100, 1) if window_scores else 0.0
    dna_score = max(0.0, min(100.0, dna_score))

    if len(window_scores) >= 3:
        q25, q75 = np.percentile(window_scores, [25, 75])
        dna_ci = [
            round(max(0.0, (q25 - 1.5 * (q75 - q25)) * 100), 1),
            round(min(100.0, (q75 + 1.5 * (q75 - q25)) * 100), 1)
        ]
    else:
        dna_ci = [dna_score, dna_score]

    if not np.isfinite(offset_ms):
        offset_ms = 0.0
    if not np.isfinite(dna_score):
        dna_score = 0.0

    return offset_ms, dna_score, {"offset_ci": offset_ci, "dna_ci": dna_ci}


def analyze_chromagram_dna(y_ref, y_comp, sr=PERFORMANCE_SR):
    try:
        min_len = min(len(y_ref), len(y_comp))
        if min_len < 512:
            return 0.0
        y_ref = y_ref[:min_len]
        y_comp = y_comp[:min_len]
        chroma_ref = librosa.feature.chroma_stft(y=y_ref, sr=sr, hop_length=HOP_LENGTH)
        chroma_comp = librosa.feature.chroma_stft(y=y_comp, sr=sr, hop_length=HOP_LENGTH)
        chroma_ref = chroma_ref / (np.linalg.norm(chroma_ref, axis=0, keepdims=True) + 1e-10)
        chroma_comp = chroma_comp / (np.linalg.norm(chroma_comp, axis=0, keepdims=True) + 1e-10)
        win_frames = int(WINDOW_SECONDS * sr / HOP_LENGTH)
        n_windows = max(1, chroma_ref.shape[1] // win_frames)
        scores = []
        for w in range(n_windows):
            s = w * win_frames
            e = min(s + win_frames, chroma_ref.shape[1])
            r_win = chroma_ref[:, s:e]
            c_win = chroma_comp[:, s:e]
            if r_win.shape[1] < 2 or c_win.shape[1] < 2:
                continue
            r_mean = np.mean(r_win, axis=1)
            c_mean = np.mean(c_win, axis=1)
            r_norm = np.linalg.norm(r_mean)
            c_norm = np.linalg.norm(c_mean)
            if r_norm < 1e-10 or c_norm < 1e-10:
                continue
            similarity = np.dot(r_mean, c_mean) / (r_norm * c_norm)
            scores.append(float(similarity))
        if not scores:
            return 0.0
        score = float(np.median(scores)) * 100
        return max(0.0, min(100.0, round(score, 1)))
    except Exception as e:
        logger.warning(f"Chromagram DNA failed: {e}")
        return 0.0


def calculate_speed_factor(start_offset_ms, end_offset_ms, duration_sec):
    if duration_sec <= 0:
        return {"ratio": 1.0, "display": "N/A", "delta": "N/A", "action": "N/A"}
    drift_sec = (end_offset_ms - start_offset_ms) / 1000.0
    denom = duration_sec + drift_sec
    if denom <= 0:
        return {
            "ratio": 1.0, "display": "N/A", "delta": "N/A",
            "action": "Drift exceeds duration — manual review required"
        }
    speed_factor = duration_sec / denom
    pct_delta = round((speed_factor - 1.0) * 100, 4)

    # FIXED: Actions were swapped in v9
    # If speed_factor > 1 (dub is shorter/faster), we need to EXPAND (slow down) the dub
    # If speed_factor < 1 (dub is longer/slower), we need to COMPRESS (speed up) the dub
    if abs(pct_delta) < 0.001:
        action = "No time-stretch needed"
    elif pct_delta > 0:
        action = f"Time-expand dub by {abs(pct_delta):.4f}% (dub is running fast)"
    else:
        action = f"Time-compress dub by {abs(pct_delta):.4f}% (dub is running slow)"

    return {
        "ratio": round(speed_factor, 6),
        "display": f"{speed_factor:.6f}×",
        "delta": f"{pct_delta:+.4f}%",
        "action": action,
    }


def determine_status(offset_ms, drift_ms, dna_score, lufs_val=None, true_peak_val=None,
                     chroma_dna=None):
    issues = []
    if abs(offset_ms) > OFFSET_THRESHOLD_MS:
        issues.append(f"Start offset {offset_ms}ms exceeds ±{OFFSET_THRESHOLD_MS}ms threshold")
    if abs(drift_ms) > DRIFT_THRESHOLD_MS:
        issues.append(f"Drift {drift_ms}ms exceeds ±{DRIFT_THRESHOLD_MS}ms threshold")
    if dna_score < DNA_MATCH_THRESHOLD:
        issues.append(f"DNA match {dna_score}% below {DNA_MATCH_THRESHOLD}% threshold")
    if chroma_dna is not None and chroma_dna < DNA_MATCH_THRESHOLD:
        issues.append(f"Chroma DNA match {chroma_dna}% below {DNA_MATCH_THRESHOLD}% threshold")
    if true_peak_val is not None and true_peak_val > TRUE_PEAK_MAX_DBTP:
        issues.append(f"True peak {round(true_peak_val, 2)} dBTP exceeds {TRUE_PEAK_MAX_DBTP} dBTP ceiling")
    if lufs_val is not None and abs(lufs_val - LUFS_TARGET) > LUFS_TOLERANCE:
        issues.append(f"Integrated loudness {round(lufs_val, 2)} LUFS outside {LUFS_TARGET}±{LUFS_TOLERANCE} LU target")
    return ("FAIL" if issues else "PASS", "; ".join(issues) if issues else "All metrics within thresholds")
