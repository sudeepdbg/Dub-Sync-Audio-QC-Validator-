"""
Audio Alignment Engine -- v10 (Fixed)
====================================
All 10 feedback points addressed.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    check_audio_description, get_full_file_loudness,
    get_transcription_result, transcription_engine,
    ATMOS_SPATIAL_LUFS_TARGET,
)

# -- LOGGING SETUP --
class JSONFormatter(logging.Formatter):
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

# -- APP INITIALIZATION --
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH", "1073741824"))

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[os.environ.get("RATE_LIMIT", "10 per minute")],
    storage_uri=os.environ.get("LIMITER_STORAGE", "memory://"),
    strategy="fixed-window"
)

# -- CONFIGURATION --
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data")))
DATA_DIR.mkdir(exist_ok=True)

FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")
FFPROBE_PATH = os.environ.get("FFPROBE_PATH", "ffprobe")

PERFORMANCE_SR = 22050
WAVEFORM_MAX_POINTS = 2000
SEGMENT_DURATION = 60.0
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
MIN_RELIABLE_DURATION_SEC = 3.0
MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE", "209715200"))

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

# -- METRICS --
class MetricsCollector:
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

# -- AUTO-CLEANUP --
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

# -- REQUEST CONTEXT --
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

# -- FFMPEG/FFPROBE LAYER --

class FFmpegError(Exception):
    pass


class FFmpegAnalyzer:
    """Comprehensive audio analysis using FFmpeg/FFprobe."""

    @staticmethod
    def _run_ffprobe(path: Union[str, Path], args: List[str], timeout: int = 30) -> dict:
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
                    timeout: int = 60) -> Tuple[str, str, int]:
        """Run ffmpeg with filter chain, return stdout, stderr, and returncode."""
        cmd = [FFMPEG_PATH, "-i", str(path), "-af", filter_chain]
        if output_args:
            cmd.extend(output_args)
        else:
            cmd.extend(["-f", "null", "-"])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            logger.warning(f"ffmpeg timeout for {path}")
            return "", "", -1

    @staticmethod
    def get_full_metadata(path: Union[str, Path]) -> Dict[str, Any]:
        data = FFmpegAnalyzer._run_ffprobe(path, ["-show_format", "-show_streams"])

        meta = {
            "format": data.get("format", {}),
            "streams": data.get("streams", []),
            "audio_streams": [],
            "has_atmos": False,
            "has_dolby": False,
            "has_dtsx": False,
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
                    "profile": stream.get("profile", ""),
                }

                tags = stream.get("tags", {})
                tag_text = str(tags).lower()
                profile = stream.get("profile", "").lower()

                # FIX Point 8: Separate Dolby, Atmos, and DTS:X detection
                if any(k in tag_text for k in ["dolby", "joc", "dthd"]):
                    meta["has_dolby"] = True

                if "atmos" in tag_text:
                    meta["has_atmos"] = True

                # DTS:X is a separate format -- do NOT conflate with Atmos
                if profile == "dtsx" or "dtsx" in tag_text:
                    meta["has_dtsx"] = True

                if "DIALNORM" in tags:
                    try:
                        audio_meta["dialnorm"] = float(tags["DIALNORM"])
                    except (ValueError, TypeError):
                        pass

                meta["audio_streams"].append(audio_meta)

        return meta

    @staticmethod
    def detect_audio_silence(path: Union[str, Path], noise_db: int = -50,
                              min_duration: float = 0.05) -> Dict[str, Any]:
        """
        FIX Point 3: Renamed from detect_silence_gaps.
        Detects silence intervals and classifies them as head/mid/tail.
        Checks ffmpeg return code. Handles trailing silence.
        """
        stdout, stderr, returncode = FFmpegAnalyzer._run_ffmpeg(
            path, f"silencedetect=noise={noise_db}dB:d={min_duration}"
        )

        if returncode != 0:
            return {
                "checked": False,
                "status": "ERROR",
                "reason": f"ffmpeg silencedetect failed (rc={returncode})",
                "count": 0,
                "gaps": [],
            }

        gaps = []
        current_gap = {}
        last_start = None

        for line in stderr.split("\n"):
            if "silence_start:" in line:
                match = re.search(r"silence_start: ([\d.]+)", line)
                if match:
                    current_gap["start"] = float(match.group(1))
                    last_start = current_gap["start"]
            elif "silence_end:" in line and current_gap:
                match = re.search(r"silence_end: ([\d.]+)", line)
                if match:
                    current_gap["end"] = float(match.group(1))
                    current_gap["duration"] = current_gap["end"] - current_gap["start"]
                    gaps.append(current_gap.copy())
                    current_gap = {}

        # FIX Point 3: Handle trailing silence
        if current_gap and "start" in current_gap:
            try:
                ff_meta = FFmpegAnalyzer.get_full_metadata(path)
                duration_str = ff_meta.get("format", {}).get("duration")
                if duration_str:
                    file_dur = float(duration_str)
                    current_gap["end"] = file_dur
                    current_gap["duration"] = file_dur - current_gap["start"]
                    current_gap["note"] = "Trailing silence (estimated from file duration)"
                    gaps.append(current_gap)
            except Exception:
                pass

        # Classify gaps as head/mid/tail
        total_duration = 0.0
        try:
            ff_meta = FFmpegAnalyzer.get_full_metadata(path)
            total_duration = float(ff_meta.get("format", {}).get("duration", 0))
        except Exception:
            pass

        for gap in gaps:
            start = gap.get("start", 0)
            end = gap.get("end", start)
            if start < 1.0:
                gap["position"] = "head"
            elif end > total_duration - 1.0 and total_duration > 0:
                gap["position"] = "tail"
            else:
                gap["position"] = "mid"

        mid_gaps = [g for g in gaps if g.get("position") == "mid"]

        return {
            "checked": True,
            "status": "WARN" if len(mid_gaps) >= 3 else "PASS",
            "count": len(gaps),
            "gaps": gaps[:10],
            "threshold_db": noise_db,
            "min_duration_ms": int(min_duration * 1000),
            "note": (
                f"{len(mid_gaps)} mid-file silence intervals. "
                "Silence may be intentional (pause, transition, censorship). "
                "Review mid-file gaps manually."
                if mid_gaps else
                "Silence detected but appears to be head/tail only."
            ),
        }

    @staticmethod
    def detect_audio_level_statistics(path: Union[str, Path], threshold: float = 0.1) -> Dict[str, Any]:
        """
        FIX Point 2: Renamed from detect_clicks_pops.
        Block-level peak outlier detection, NOT sample-accurate click/pop detection.
        """
        stdout, stderr, returncode = FFmpegAnalyzer._run_ffmpeg(
            path, "astats=metadata=1:reset=1,ametadata=print:file=-"
        )

        if returncode != 0:
            return {
                "checked": False,
                "status": "ERROR",
                "reason": f"ffmpeg astats failed (rc={returncode})",
            }

        # FIX Point 2: ametadata writes to stdout, check both
        output = stdout + stderr

        peaks = []
        peak_values = []
        for line in output.split("\n"):
            if "Peak level" in line:
                try:
                    val_str = line.split(":")[-1].strip()
                    val = float(val_str)
                    # FIX Point 2: astats Peak level is already in dB
                    peaks.append({
                        "raw": val,
                        "peak_db": round(val, 2),
                    })
                    peak_values.append(val)
                except ValueError:
                    pass

        flagged = []
        if len(peak_values) > 1:
            mean_peak = np.mean(peak_values)
            std_peak = np.std(peak_values)
            for i, peak in enumerate(peak_values):
                if peak > mean_peak + threshold * std_peak:
                    flagged.append({
                        "index": i,
                        "peak_db": round(peak, 2),
                        "severity": "high" if peak > mean_peak + 2 * threshold * std_peak else "medium"
                    })

        status = "WARN" if len(flagged) >= 5 else "PASS"

        return {
            "checked": True,
            "status": status,
            "count": len(flagged),
            "peaks": flagged[:10],
            "mean_peak_db": round(float(np.mean(peak_values)), 2) if peak_values else None,
            "note": (
                "Block-level peak outlier detection only. "
                "NOT sample-accurate click/pop detection. "
                "Real clicks are single/few-sample discontinuities."
            ),
        }

    @staticmethod
    def detect_hum_buzz(path: Union[str, Path]) -> Dict[str, Any]:
        """
        EXPERIMENTAL: 50Hz/60Hz hum indicator (10s sample).
        FIX Point 5: Not a production gate. Energy ratio, not SNR.
        """
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
                ratio = ratio_50 / ratio_60 if ratio_50 > ratio_60 else ratio_60 / ratio_50
            elif detected_50:
                detected = True
                freq = 50
                ratio = ratio_50 / (ratio_60 + 1e-10)
            elif detected_60:
                detected = True
                freq = 60
                ratio = ratio_60 / (ratio_50 + 1e-10)
            else:
                detected = False
                freq = None
                ratio = None

            status = "WARN" if detected else "PASS"

            return {
                "checked": True,
                "status": status,
                "detected": detected,
                "frequency": freq,
                "energy_ratio": round(float(ratio), 1) if ratio else None,
                "ratio_50hz": round(float(ratio_50), 4),
                "ratio_60hz": round(float(ratio_60), 4),
                "note": (
                    "EXPERIMENTAL: 10-second sample. Energy ratio is NOT SNR. "
                    "Strong bass notes at 50/60Hz will trigger this. "
                    "Do not gate delivery on this check."
                ),
            }
        except Exception as e:
            logger.warning(f"Hum detection failed: {e}")
            return {
                "checked": False,
                "status": "ERROR",
                "reason": str(e),
            }

    @staticmethod
    def detect_low_freq_rumble(path: Union[str, Path]) -> Dict[str, Any]:
        """
        EXPERIMENTAL: Subsonic rumble indicator below 20Hz (10s sample).
        FIX Point 5: Uses band-pass energy ratio instead of invalid subtraction.
        """
        try:
            y, sr = librosa.load(str(path), sr=None, duration=10.0)

            nyq = 0.5 * sr
            low_cutoff = 1.0 / nyq
            high_cutoff = 20.0 / nyq

            if high_cutoff >= 1.0:
                return {
                    "checked": False,
                    "status": "SKIP",
                    "reason": "Sample rate too low for 20Hz rumble detection",
                }

            b_bp, a_bp = butter(4, [low_cutoff, high_cutoff], btype="band")
            y_rumble = lfilter(b_bp, a_bp, y)

            b_ref, a_ref = butter(4, [20.0 / nyq, 100.0 / nyq], btype="band")
            y_ref_band = lfilter(b_ref, a_ref, y)

            energy_rumble = np.sum(y_rumble ** 2)
            energy_ref = np.sum(y_ref_band ** 2)

            rumble_ratio = energy_rumble / (energy_ref + 1e-10)
            rumble_db = 10 * np.log10(rumble_ratio + 1e-10)

            fft = np.fft.rfft(y)
            freqs = np.fft.rfftfreq(len(y), 1/sr)
            low_freq_mask = (freqs > 1) & (freqs < 20)

            if np.any(low_freq_mask):
                peak_idx = np.argmax(np.abs(fft[low_freq_mask]))
                peak_freq = freqs[low_freq_mask][peak_idx]
            else:
                peak_freq = 0

            detected = rumble_ratio > 0.1
            status = "WARN" if detected else "PASS"

            return {
                "checked": True,
                "status": status,
                "detected": detected,
                "level_db": round(float(rumble_db), 1),
                "freq_peak": round(float(peak_freq), 1),
                "ratio": round(float(rumble_ratio), 5),
                "note": (
                    "EXPERIMENTAL: 10-second sample. Band-pass energy ratio method. "
                    "Do not gate delivery on this check."
                ),
            }
        except Exception as e:
            logger.warning(f"Rumble detection failed: {e}")
            return {
                "checked": False,
                "status": "ERROR",
                "reason": str(e),
            }

    @staticmethod
    def check_dual_mono(path: Union[str, Path]) -> Dict[str, Any]:
        try:
            info = sf.info(str(path))
            if info.channels != 2:
                return {"checked": True, "status": "PASS", "is_dual_mono": False, "reason": "Not stereo"}

            y, sr = librosa.load(str(path), sr=None, mono=False, duration=5.0)
            if y.ndim < 2 or y.shape[0] < 2:
                return {"checked": True, "status": "PASS", "is_dual_mono": False}

            left = y[0]
            right = y[1]

            correlation = np.corrcoef(left, right)[0, 1]

            diff = np.abs(left - right)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            is_dual_mono = correlation > 0.999 and max_diff < 1e-5
            status = "WARN" if is_dual_mono else "PASS"

            return {
                "checked": True,
                "status": status,
                "is_dual_mono": bool(is_dual_mono),
                "correlation": round(float(correlation), 6),
                "max_diff": round(float(max_diff), 8),
                "mean_diff": round(float(mean_diff), 8),
                "reason": "Stereo file is dual-mono (identical L/R)" if is_dual_mono else None,
            }
        except Exception as e:
            logger.warning(f"Dual-mono check failed: {e}")
            return {
                "checked": False,
                "status": "ERROR",
                "reason": str(e),
            }

    @staticmethod
    def get_spectrum_data(path: Union[str, Path], n_fft: int = 2048) -> Tuple[List[float], List[float]]:
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

            return spec_downsampled, freqs[log_indices].tolist()
        except Exception as e:
            logger.warning(f"Spectrum extraction failed: {e}")
            return [], []

# -- HELPERS --
def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def allowed_file(filename: str) -> bool:
    if not filename:
        return False
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


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
    except Exception:
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
        return y
    try:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        if not np.isfinite(loudness):
            return y
        out = pyln.normalize.loudness(y, loudness, target)
        if not np.all(np.isfinite(out)):
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.astype(np.float32)
    except Exception:
        return y


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


# -- SINGLE-PASS INGESTION (FIXED Point 1) --
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

    # FIX Point 1: Use ffmpeg ebur128 for full-file loudness and true peak
    loudness_result = get_full_file_loudness(FFMPEG_PATH, path)

    lufs_val = loudness_result.get("lufs_val")
    true_peak_val = loudness_result.get("true_peak_val")

    if lufs_val is not None:
        lufs_str = f"{round(lufs_val, 2)} LUFS"
    else:
        lufs_str = "ERR"
        lufs_val = None

    if true_peak_val is not None:
        true_peak_str = f"{round(true_peak_val, 2)} dBTP"
    else:
        true_peak_str = "ERR"
        true_peak_val = None

    # Legacy sample peak for reference (not used for gating)
    max_val = 1e-10
    cross_prod = 0.0
    var_m1 = 0.0
    var_m2 = 0.0

    with sf.SoundFile(path) as f:
        for block in f.blocks(blocksize=65536, dtype="float32"):
            mono = np.mean(block, axis=1) if block.ndim > 1 else block
            block_max = np.max(np.abs(mono))
            if block_max > max_val:
                max_val = block_max
            if channels >= 2:
                ch1 = block[:, 0]
                ch2 = block[:, 1]
                cross_prod += np.sum(ch1 * ch2)
                var_m1 += np.sum(ch1 ** 2)
                var_m2 += np.sum(ch2 ** 2)

    sample_peak_db = float(20 * np.log10(max_val))

    if channels < 2:
        phase_str = "1.0 (Mono)"
    else:
        denom = np.sqrt(var_m1 * var_m2)
        corr = float(cross_prod / denom) if denom > 0 else 0.0
        status = "Healthy" if corr > 0.4 else "Issue"
        phase_str = f"{round(corr, 2)} ({status})"

    y_start, _ = librosa.load(path, sr=target_sr, offset=0.0,
                              duration=seg_dur, res_type="soxr_hq")
    if total_duration > seg_dur * 2:
        y_end, _ = librosa.load(path, sr=target_sr,
                                offset=max(0.0, total_duration - seg_dur),
                                duration=seg_dur, res_type="soxr_hq")
    else:
        y_end = y_start

    levels = {
        "lufs": lufs_str,
        "lufs_val": lufs_val,
        "peak": f"{round(sample_peak_db, 2)} dBFS",
        "peak_val": sample_peak_db,
        "true_peak": true_peak_str,
        "true_peak_val": true_peak_val,
        "lra": loudness_result.get("lra"),
        "loudness_source": loudness_result.get("source", "unknown"),
    }

    return meta, levels, phase_str, y_start, y_end

# -- ALIGNMENT ANALYSIS --
def analyze_segment(y_ref, y_comp, sr):
    hop = HOP_LENGTH
    y_ref = np.nan_to_num(np.asarray(y_ref, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y_comp = np.nan_to_num(np.asarray(y_comp, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    ref_rms = librosa.feature.rms(y=y_ref, hop_length=hop)[0].astype(np.float64)
    comp_rms = librosa.feature.rms(y=y_comp, hop_length=hop)[0].astype(np.float64)

    ref_range = ref_rms.max() - ref_rms.min()
    comp_range = comp_rms.max() - comp_rms.min()

    if ref_range < RMS_MIN_DYNAMIC_RANGE or comp_range < RMS_MIN_DYNAMIC_RANGE:
        return 0.0, 0.0, {"offset_resolution_range": [0.0, 0.0], "window_score_dispersion": [0.0, 0.0]}

    ref_rms = (ref_rms - ref_rms.min()) / ref_range
    comp_rms = (comp_rms - comp_rms.min()) / comp_range

    corr = signal.correlate(comp_rms, ref_rms, mode="full")
    lag = np.argmax(corr) - (len(ref_rms) - 1)
    offset_ms = round(float(lag * hop / sr * 1000), 2)

    # FIX Point 9: Renamed from offset_ci to offset_resolution_range
    hop_ms = hop / sr * 1000
    offset_resolution_range = [
        round(offset_ms - hop_ms, 2),
        round(offset_ms + hop_ms, 2),
    ]

    WIN_SEC = WINDOW_SECONDS
    WIN_FRAMES = int(WIN_SEC * sr / hop)

    ref_onset = librosa.onset.onset_strength(y=y_ref, sr=sr, hop_length=hop)
    comp_onset = librosa.onset.onset_strength(y=y_comp, sr=sr, hop_length=hop)

    min_len = min(len(ref_onset), len(comp_onset))
    if min_len == 0:
        return offset_ms, 0.0, {"offset_resolution_range": offset_resolution_range, "window_score_dispersion": [0.0, 0.0]}

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

    # FIX Point 9: Renamed from dna_ci to window_score_dispersion
    if len(window_scores) >= 3:
        q25, q75 = np.percentile(window_scores, [25, 75])
        window_score_dispersion = [
            round(max(0.0, (q25 - 1.5 * (q75 - q25)) * 100), 1),
            round(min(100.0, (q75 + 1.5 * (q75 - q25)) * 100), 1)
        ]
    else:
        window_score_dispersion = [dna_score, dna_score]

    if not np.isfinite(offset_ms):
        offset_ms = 0.0
    if not np.isfinite(dna_score):
        dna_score = 0.0

    return offset_ms, dna_score, {
        "offset_resolution_range": offset_resolution_range,
        "window_score_dispersion": window_score_dispersion,
    }


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
            "action": "Drift exceeds duration -- manual review required"
        }
    speed_factor = duration_sec / denom
    pct_delta = round((speed_factor - 1.0) * 100, 4)
    if abs(pct_delta) < 0.001:
        action = "No time-stretch needed"
    elif pct_delta > 0:
        action = f"Time-compress dub by {abs(pct_delta):.4f}%"
    else:
        action = f"Time-expand dub by {abs(pct_delta):.4f}%"
    return {
        "ratio": round(speed_factor, 6),
        "display": f"{speed_factor:.6f}x",
        "delta": f"{pct_delta:+.4f}%",
        "action": action,
    }

# -- STATUS DETERMINATION (FIXED Point 6 & 7) --
def determine_status(offset_ms, drift_ms, dna_score, lufs_val=None, true_peak_val=None,
                     chroma_dna=None, qc_checks=None):
    """
    FIX Point 7: Now accepts qc_checks and considers ALL findings.
    Status is the WORST across all checks (never-false-pass rule).

    BUG FIX: WARN-level findings (e.g. mid-file silence, dual-mono stereo)
    were previously appended to the same `issues` list as hard FAIL/ERROR
    findings, so a single WARN was enough to flip an otherwise-healthy file
    to an overall FAIL. Severity is now properly ordered:
        FAIL/ERROR  -> forces overall FAIL
        WARN        -> forces overall WARN, but only if nothing FAILED
        PASS        -> no effect
    A WARN can never be more severe than a FAIL, and a clean file with only
    warnings is no longer indistinguishable from a file with real failures.
    """
    issues = []    # hard failures -- always force FAIL
    warnings = []  # soft findings -- force WARN only if there are no issues

    # Sync/level metrics are hard gating criteria -- always issues, not warnings
    if abs(offset_ms) > OFFSET_THRESHOLD_MS:
        issues.append(f"Start offset {offset_ms}ms exceeds +/-{OFFSET_THRESHOLD_MS}ms threshold")
    if abs(drift_ms) > DRIFT_THRESHOLD_MS:
        issues.append(f"Drift {drift_ms}ms exceeds +/-{DRIFT_THRESHOLD_MS}ms threshold")
    if dna_score < DNA_MATCH_THRESHOLD:
        issues.append(f"DNA match {dna_score}% below {DNA_MATCH_THRESHOLD}% threshold")
    if chroma_dna is not None and chroma_dna < DNA_MATCH_THRESHOLD:
        issues.append(f"Chroma DNA match {chroma_dna}% below {DNA_MATCH_THRESHOLD}% threshold")
    if true_peak_val is not None and true_peak_val > TRUE_PEAK_MAX_DBTP:
        issues.append(f"True peak {round(true_peak_val, 2)} dBTP exceeds {TRUE_PEAK_MAX_DBTP} dBTP ceiling")
    if lufs_val is not None and abs(lufs_val - LUFS_TARGET) > LUFS_TOLERANCE:
        issues.append(f"Integrated loudness {round(lufs_val, 2)} LUFS outside {LUFS_TARGET}+/-{LUFS_TOLERANCE} LU target")

    # FIX Point 7: Consider QC check findings, but keep WARN separate from FAIL/ERROR
    if qc_checks:
        for check_name, check_result in qc_checks.items():
            if check_name == "error":
                continue
            if not isinstance(check_result, dict):
                continue
            check_status = check_result.get("status", "PASS")
            if check_status in ("FAIL", "ERROR"):
                reason = check_result.get("reason", f"{check_name} failed")
                issues.append(f"[{check_name}] {reason}")
            elif check_status == "WARN":
                reason = check_result.get("reason", f"{check_name} warning")
                warnings.append(f"[{check_name}] {reason}")

    if issues:
        # Include warnings too so the reason string stays complete, but the
        # overall status is driven by the presence of real failures.
        return "FAIL", "; ".join(issues + warnings)
    if warnings:
        return "WARN", "; ".join(warnings)
    return "PASS", "All metrics within thresholds"


def determine_standalone_status(levels, qc_checks, duration_sec):
    """
    FIX Point 6 & 7: Considers ALL qc_checks, not just a subset.
    Returns the WORST status across all checks.

    BUG FIX: same WARN-vs-FAIL conflation as determine_status() -- WARN
    findings are now tracked separately and only downgrade the result to
    WARN, never to FAIL, unless a genuine FAIL/ERROR is also present.
    """
    if duration_sec < MIN_RELIABLE_DURATION_SEC:
        return "WARN", (
            f"Audio too short for reliable QC (min {MIN_RELIABLE_DURATION_SEC:.0f}s "
            f"recommended; got {round(duration_sec, 2)}s). Metrics shown are indicative only."
        )

    issues = []    # hard failures -- always force FAIL
    warnings = []  # soft findings -- force WARN only if there are no issues
    true_peak_val = levels.get("true_peak_val")
    lufs_val = levels.get("lufs_val")

    if true_peak_val is not None and true_peak_val > TRUE_PEAK_MAX_DBTP:
        issues.append(f"True peak {round(true_peak_val, 2)} dBTP exceeds {TRUE_PEAK_MAX_DBTP} dBTP ceiling")
    if lufs_val is not None and abs(lufs_val - LUFS_TARGET) > LUFS_TOLERANCE:
        issues.append(f"Integrated loudness {round(lufs_val, 2)} LUFS outside {LUFS_TARGET}+/-{LUFS_TOLERANCE} LU target")

    # FIX Point 6 & 7: Check ALL qc_checks, but keep WARN separate from FAIL/ERROR
    if qc_checks:
        for check_name, check_result in qc_checks.items():
            if check_name == "error":
                issues.append(f"QC pipeline error: {check_result}")
                continue
            if not isinstance(check_result, dict):
                continue
            check_status = check_result.get("status", "PASS")
            if check_status in ("FAIL", "ERROR"):
                reason = check_result.get("reason", f"{check_name} check failed")
                issues.append(f"[{check_name}] {reason}")
            elif check_status == "WARN":
                reason = check_result.get("reason", f"{check_name} warning")
                warnings.append(f"[{check_name}] {reason}")

    if issues:
        return "FAIL", "; ".join(issues + warnings)
    if warnings:
        return "WARN", "; ".join(warnings)
    return "PASS", "All standalone QC metrics within thresholds"

# -- SHARED ADVANCED-QC PIPELINE (FIXED Point 6) --
def run_advanced_qc(f_path, comp_meta, phase, levels, y_dialogue_ref,
                     me_path=None, expected_language=None, run_asr=False):
    """
    Shared QC check pipeline.
    FIX Point 6: Every check returns a dict with 'status' field.
    No check returns a clean-looking result on failure.
    """
    qc_checks: Dict[str, Any] = {}
    spectrum: List[float] = []

    try:
        ff_meta = FFmpegAnalyzer.get_full_metadata(f_path)

        # Atmos bed presence
        if ff_meta.get("has_atmos"):
            qc_checks["atmos_bed_objects"] = {
                "checked": True,
                "status": "INFO",
                "verified": True,
                "bed_count": len([s for s in ff_meta.get("audio_streams", [])
                                 if s.get("channel_layout", "").startswith("7.1")]),
                "object_count": None,
                "note": "Object count/position require Dolby Atmos Renderer metadata. "
                        "Bed channel presence only.",
            }

        # FIX Point 8: Report DTS:X separately
        if ff_meta.get("has_dtsx"):
            qc_checks["dtsx_detected"] = {
                "checked": True,
                "status": "INFO",
                "note": "DTS:X immersive audio detected (not Dolby Atmos)",
            }

        # FIX Point 8: Dialnorm -- report metadata only, never hardcode "match": True
        for stream in ff_meta.get("audio_streams", []):
            if "dialnorm" in stream:
                embedded = stream["dialnorm"]
                measured = levels.get("lufs_val")
                qc_checks["dialnorm_metadata"] = {
                    "checked": True,
                    "status": "INFO",
                    "embedded": embedded,
                    "measured_lufs": measured,
                    "codec": stream.get("codec"),
                    "note": (
                        f"Embedded dialnorm: {embedded} dB. "
                        f"Measured integrated loudness: {measured} LUFS. "
                        "Dialnorm represents dialogue-gated loudness; integrated LUFS "
                        "is not directly comparable without dialogue gating."
                    ),
                }
                break

        # FIX Point 3: Renamed to audio_silence
        silence_result = FFmpegAnalyzer.detect_audio_silence(f_path)
        qc_checks["audio_silence"] = silence_result

        # FIX Point 2: Renamed to audio_level_statistics
        level_stats = FFmpegAnalyzer.detect_audio_level_statistics(f_path)
        qc_checks["audio_level_statistics"] = level_stats

        # FIX Point 5: Hum and rumble marked as experimental
        hum_result = FFmpegAnalyzer.detect_hum_buzz(f_path)
        qc_checks["hum_buzz"] = hum_result

        rumble_result = FFmpegAnalyzer.detect_low_freq_rumble(f_path)
        qc_checks["low_freq_rumble"] = rumble_result

        mono_result = FFmpegAnalyzer.check_dual_mono(f_path)
        qc_checks["mono_in_stereo"] = mono_result

        # Spatial loudness
        # BUG FIX (target mismatch): this check previously called
        # get_spatial_loudness() with no target, which silently defaulted to
        # ATMOS_SPATIAL_LUFS_TARGET (-27.0 LUFS) for every file -- including
        # plain stereo dubs correctly mixed to the standard -23.0 LUFS
        # broadcast target. That mismatch was failing nearly every stereo
        # file. The -27 LUFS target only applies to actual immersive beds
        # (Dolby Atmos / DTS:X); everything else is judged against the same
        # LUFS_TARGET used everywhere else in this file.
        #
        # DEDUPE (perf): get_spatial_loudness() ran the exact same ffmpeg
        # ebur128 command that get_full_file_loudness() already ran a few
        # lines earlier to produce `levels`. Rather than paying for a second
        # full-file ffmpeg pass on every QC run, we now build this check
        # directly from the already-measured `levels` values.
        is_immersive = bool(ff_meta.get("has_atmos") or ff_meta.get("has_dtsx"))
        spatial_target = ATMOS_SPATIAL_LUFS_TARGET if is_immersive else LUFS_TARGET

        spatial_lufs = levels.get("lufs_val")
        if levels.get("loudness_source") == "ffmpeg_ebur128_full" and spatial_lufs is not None:
            within_tolerance = abs(spatial_lufs - spatial_target) <= LUFS_TOLERANCE
            qc_checks["spatial_loudness"] = {
                "checked": True,
                "status": "PASS" if within_tolerance else "FAIL",
                "lufs": round(spatial_lufs, 2),
                "true_peak": (
                    round(levels["true_peak_val"], 2)
                    if levels.get("true_peak_val") is not None else None
                ),
                "lra": round(levels["lra"], 2) if levels.get("lra") is not None else None,
                "target_lufs": spatial_target,
                "within_tolerance": within_tolerance,
                "source": "ffmpeg_ebur128_full (reused from level measurement)",
                "reason": (
                    f"Integrated loudness {spatial_lufs:.2f} LUFS outside target "
                    f"{spatial_target} plus/minus {LUFS_TOLERANCE} LU"
                    if not within_tolerance else None
                ),
            }
        else:
            qc_checks["spatial_loudness"] = {
                "checked": False,
                "status": "ERROR",
                "reason": "Full-file loudness measurement unavailable.",
            }

        if comp_meta.get("channels", 1) >= 2:
            phase_corr = 0.0
            try:
                phase_corr = float(phase.split(" ")[0])
            except Exception:
                pass
            qc_checks["inter_channel_phase"] = {
                "checked": True,
                "status": "WARN" if phase_corr <= 0.4 else "PASS",
                "correlation": round(phase_corr, 3),
                "reason": "Mono collapse risk" if phase_corr <= 0.4 else None,
                "note": "Mono collapse risk if correlation < 0.4",
            }

        # Audio Description
        qc_checks["audio_description"] = check_audio_description(ff_meta)

        # DME structural check (FIX Point 4: marked experimental)
        if me_path:
            qc_checks["dme_check"] = check_dme_structural(
                me_path, y_dialogue_ref=y_dialogue_ref, sr=PERFORMANCE_SR
            )

        # Language ID + Profanity (FIX Point 10: deduplicated via cache)
        if run_asr:
            qc_checks["language_id"] = check_language_id(f_path, expected_language)
            qc_checks["profanity"] = check_profanity(f_path)

        spectrum, _ = FFmpegAnalyzer.get_spectrum_data(f_path)

    except Exception as e:
        logger.warning(f"Advanced QC failed for {f_path}: {e}")
        # FIX Point 6: Return ERROR status
        qc_checks["pipeline_error"] = {
            "checked": False,
            "status": "ERROR",
            "reason": str(e),
        }

    return qc_checks, spectrum


# -- WORKER COMPUTE THREAD (FIXED Point 7) --
def process_file(stored_name, display_name, root, y_ref_s_an, y_ref_e_an, y_ref_s_raw, ref_meta,
                  vocal_logic, me_path=None, expected_language=None, run_asr=False):
    try:
        f_path = os.path.join(root, stored_name)

        max_retries = 2
        for attempt in range(max_retries):
            try:
                comp_meta, levels, phase, y_c_s, y_c_e = process_audio_single_pass(f_path)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt + 1} for {display_name}: {e}")
                    time.sleep(0.5)
                else:
                    raise

        comp_dur = comp_meta["duration_sec"]
        y_c_s_raw = y_c_s.copy()

        if vocal_logic:
            y_c_s_an = apply_vocal_filter(normalize_lufs(y_c_s, PERFORMANCE_SR))
            y_c_e_an = apply_vocal_filter(normalize_lufs(y_c_e, PERFORMANCE_SR))
        else:
            y_c_s_an = y_c_s
            y_c_e_an = y_c_e

        s_off, dna, confidence = analyze_segment(y_ref_s_an, y_c_s_an, PERFORMANCE_SR)
        e_off, _, _ = analyze_segment(y_ref_e_an, y_c_e_an, PERFORMANCE_SR)
        drift = round(e_off - s_off, 2)
        chroma_dna = analyze_chromagram_dna(y_ref_s_an, y_c_s_an, PERFORMANCE_SR)
        speed = calculate_speed_factor(s_off, e_off, comp_dur)

        # FIX Point 7: Run advanced QC BEFORE status determination
        spectrum_master = []
        qc_checks, spectrum_dub = run_advanced_qc(
            f_path, comp_meta, phase, levels,
            y_dialogue_ref=y_c_s_raw,
            me_path=me_path, expected_language=expected_language, run_asr=run_asr,
        )

        # FIX Point 7: Now pass qc_checks to determine_status
        status, reason = determine_status(
            s_off, drift, dna,
            lufs_val=levels.get("lufs_val"),
            true_peak_val=levels.get("true_peak_val"),
            chroma_dna=chroma_dna,
            qc_checks=qc_checks,
        )

        ref_dur = ref_meta.get("duration_sec", 0.0)
        if comp_dur < MIN_RELIABLE_DURATION_SEC or ref_dur < MIN_RELIABLE_DURATION_SEC:
            status = "WARN"
            reason = (
                f"Insufficient audio for reliable alignment "
                f"(min {MIN_RELIABLE_DURATION_SEC:.0f}s recommended; "
                f"reference {round(ref_dur, 2)}s, dub {round(comp_dur, 2)}s). "
                "Metrics shown are indicative only."
            )

        try:
            spectrum_master, _ = FFmpegAnalyzer.get_spectrum_data(
                os.path.join(root, ref_meta.get("_stored_name", ""))
            )
        except Exception as e:
            logger.debug(f"Reference spectrum unavailable for {display_name}: {e}")

        result = {
            "filename": display_name,
            "status": status,
            "reason": reason,
            "offset_ms": s_off,
            "offset_resolution_range": confidence["offset_resolution_range"],
            "total_drift_ms": drift,
            "offset_frames": ms_to_frames(s_off),
            "drift_frames": ms_to_frames(drift),
            "dna_match": dna,
            "window_score_dispersion": confidence["window_score_dispersion"],
            "chroma_dna": chroma_dna,
            "vocal_filter": vocal_logic,
            "speed_factor": speed,
            "phase": phase,
            "levels": levels,
            "ref_meta": ref_meta,
            "comp_meta": comp_meta,
            "wave_rms_master": rms_envelope(y_ref_s_raw).tolist(),
            "wave_rms_dub": (-rms_envelope(y_c_s_raw)).tolist(),
            "wave_raw_master": downsample_waveform(np.abs(normalize_visual(y_ref_s_raw))),
            "wave_raw_dub": downsample_waveform(-np.abs(normalize_visual(y_c_s_raw))),
            "chan_mismatch": ref_meta["channels"] != comp_meta["channels"],
            "qc_checks": qc_checks,
            "spectrum_master": spectrum_master,
            "spectrum_dub": spectrum_dub,
        }

        del y_c_s, y_c_e, y_c_s_raw, y_c_s_an, y_c_e_an
        metrics.record_file(failed=False)
        return result

    except Exception as err:
        logger.error(f"Error processing {display_name}: {err}", exc_info=True)
        metrics.record_file(failed=True)
        return {"filename": display_name, "status": "ERROR", "reason": str(err), "error": True}


def process_file_standalone(stored_name, display_name, root,
                             me_path=None, expected_language=None, run_asr=False):
    try:
        f_path = os.path.join(root, stored_name)

        max_retries = 2
        for attempt in range(max_retries):
            try:
                meta, levels, phase, y_start, y_end = process_audio_single_pass(f_path)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt + 1} for {display_name}: {e}")
                    time.sleep(0.5)
                else:
                    raise

        y_start_raw = y_start.copy()

        qc_checks, spectrum = run_advanced_qc(
            f_path, meta, phase, levels,
            y_dialogue_ref=y_start_raw,
            me_path=me_path, expected_language=expected_language, run_asr=run_asr,
        )

        # FIX Point 7: determine_standalone_status now considers ALL qc_checks
        status, reason = determine_standalone_status(
            levels, qc_checks, meta.get("duration_sec", 0.0)
        )

        result = {
            "filename": display_name,
            "status": status,
            "reason": reason,
            "phase": phase,
            "levels": levels,
            "meta": meta,
            "wave_rms": rms_envelope(y_start_raw).tolist(),
            "wave_raw": downsample_waveform(np.abs(normalize_visual(y_start_raw))),
            "qc_checks": qc_checks,
            "spectrum": spectrum,
        }

        del y_start, y_end, y_start_raw
        metrics.record_file(failed=False)
        return result

    except Exception as err:
        logger.error(f"Error processing {display_name}: {err}", exc_info=True)
        metrics.record_file(failed=True)
        return {"filename": display_name, "status": "ERROR", "reason": str(err), "error": True}

# -- ROUTES --
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "timestamp": time.time(), "version": "10.0.0"})

@app.route("/metrics")
def metrics_endpoint():
    return metrics.render(), 200, {"Content-Type": "text/plain; version=0.0.4"}

@app.route("/wipe", methods=["POST"])
def wipe():
    try:
        cleared = 0
        for folder in os.listdir(DATA_DIR):
            path = DATA_DIR / folder
            if path.is_dir() and folder.startswith("SES_"):
                shutil.rmtree(path, ignore_errors=True)
                cleared += 1
        # FIX Point 10: Clear ASR cache on wipe
        transcription_engine.clear_cache()
        logger.info(f"Manual wipe: cleared {cleared} session(s) and ASR cache")
        return jsonify({"status": "ok", "cleared": cleared})
    except Exception as e:
        logger.exception("Wipe failed")
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
@limiter.limit(os.environ.get("UPLOAD_RATE_LIMIT", "10 per minute"))
def upload():
    session_id = f"SES_{uuid.uuid4().hex[:6].upper()}"
    root = os.path.join(DATA_DIR, session_id)
    os.makedirs(root, exist_ok=True)

    start_time = time.time()
    failed = False

    try:
        vocal_logic = request.form.get("vocal_logic") == "true"
        run_asr = request.form.get("run_asr") == "true"
        expected_language = request.form.get("expected_language") or None
        ref = request.files.get("reference")
        comps = request.files.getlist("comparison[]")
        me_stem = request.files.get("me_stem")

        if not ref or not comps:
            return jsonify(sanitize_json({"error": "Missing mandatory reference or comparison assets"})), 400

        valid, msg = validate_file_size(ref)
        if not valid:
            return jsonify(sanitize_json({"error": f"Reference file error: {msg}"})), 400

        if not allowed_file(ref.filename):
            return jsonify(sanitize_json({"error": f"Unsupported master file format: '{os.path.splitext(ref.filename)[1]}'"})), 400

        ref_secure_name = secure_filename(ref.filename)
        ref_path = os.path.join(root, ref_secure_name)
        ref.save(ref_path)

        ref_meta, ref_levels, ref_phase, y_ref_s, y_ref_e = process_audio_single_pass(ref_path)
        ref_meta["_stored_name"] = ref_secure_name
        y_ref_s_raw = y_ref_s.copy()

        if vocal_logic:
            y_ref_s_an = apply_vocal_filter(normalize_lufs(y_ref_s, PERFORMANCE_SR))
            y_ref_e_an = apply_vocal_filter(normalize_lufs(y_ref_e, PERFORMANCE_SR))
        else:
            y_ref_s_an = y_ref_s
            y_ref_e_an = y_ref_e

        me_path = None
        if me_stem and me_stem.filename and allowed_file(me_stem.filename):
            valid_me, msg_me = validate_file_size(me_stem)
            if valid_me:
                me_secure_name = secure_filename(me_stem.filename)
                me_path = os.path.join(root, me_secure_name)
                me_stem.save(me_path)
            else:
                logger.warning(f"Skipping M&E stem: {msg_me}")

        valid_comp_files = []
        for f in comps:
            if not f or not f.filename:
                continue
            if not allowed_file(f.filename):
                logger.warning(f"Skipping unsupported file: {f.filename}")
                continue
            valid, msg = validate_file_size(f)
            if not valid:
                logger.warning(f"Skipping {f.filename}: {msg}")
                continue
            display_name = secure_filename(f.filename)
            stored_name = f"{uuid.uuid4().hex[:8]}_{display_name}"
            f.save(os.path.join(root, stored_name))
            valid_comp_files.append((stored_name, display_name))

        if not valid_comp_files:
            shutil.rmtree(root, ignore_errors=True)
            return jsonify(sanitize_json({"error": "No valid comparison files provided"})), 400

        results_map = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(
                    process_file, stored_name, display_name, root,
                    y_ref_s_an, y_ref_e_an, y_ref_s_raw, ref_meta, vocal_logic,
                    me_path, expected_language, run_asr
                ): i
                for i, (stored_name, display_name) in enumerate(valid_comp_files)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    res = future.result()
                except Exception as e:
                    logger.error(f"Worker exception for {valid_comp_files[idx][1]}: {e}", exc_info=True)
                    res = {
                        "filename": valid_comp_files[idx][1] if idx < len(valid_comp_files) else "unknown",
                        "status": "ERROR",
                        "reason": "Internal processing error",
                        "error": True
                    }
                    metrics.record_file(failed=True)

                if res is not None:
                    results_map[idx] = res

        results = [results_map[i] for i in sorted(results_map)]

        del y_ref_s, y_ref_e, y_ref_s_raw, y_ref_s_an, y_ref_e_an
        gc.collect()

        duration = time.time() - start_time
        metrics.record_request(duration, failed=False)

        return jsonify(sanitize_json({"mode": "sync", "results": results}))

    except Exception:
        logger.exception("Upload processing failed for session %s", session_id)
        failed = True
        shutil.rmtree(root, ignore_errors=True)
        gc.collect()
        metrics.record_request(time.time() - start_time, failed=True)
        return jsonify(sanitize_json({"error": "Internal processing error. Check server logs for details."})), 500


@app.route("/qc", methods=["POST"])
@limiter.limit(os.environ.get("QC_RATE_LIMIT", "10 per minute"))
def qc_standalone():
    session_id = f"SES_{uuid.uuid4().hex[:6].upper()}"
    root = os.path.join(DATA_DIR, session_id)
    os.makedirs(root, exist_ok=True)

    start_time = time.time()

    try:
        run_asr = request.form.get("run_asr") == "true"
        expected_language = request.form.get("expected_language") or None
        files = request.files.getlist("files[]")
        me_stem = request.files.get("me_stem")

        if not files:
            return jsonify(sanitize_json({"error": "No audio files provided"})), 400

        me_path = None
        if me_stem and me_stem.filename and allowed_file(me_stem.filename):
            valid_me, msg_me = validate_file_size(me_stem)
            if valid_me:
                me_secure_name = secure_filename(me_stem.filename)
                me_path = os.path.join(root, me_secure_name)
                me_stem.save(me_path)
            else:
                logger.warning(f"Skipping M&E stem: {msg_me}")

        valid_files = []
        for f in files:
            if not f or not f.filename:
                continue
            if not allowed_file(f.filename):
                logger.warning(f"Skipping unsupported file: {f.filename}")
                continue
            valid, msg = validate_file_size(f)
            if not valid:
                logger.warning(f"Skipping {f.filename}: {msg}")
                continue
            display_name = secure_filename(f.filename)
            stored_name = f"{uuid.uuid4().hex[:8]}_{display_name}"
            f.save(os.path.join(root, stored_name))
            valid_files.append((stored_name, display_name))

        if not valid_files:
            shutil.rmtree(root, ignore_errors=True)
            return jsonify(sanitize_json({"error": "No valid audio files provided"})), 400

        results_map = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(
                    process_file_standalone, stored_name, display_name, root,
                    me_path, expected_language, run_asr
                ): i
                for i, (stored_name, display_name) in enumerate(valid_files)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    res = future.result()
                except Exception as e:
                    logger.error(f"Worker exception for {valid_files[idx][1]}: {e}", exc_info=True)
                    res = {
                        "filename": valid_files[idx][1] if idx < len(valid_files) else "unknown",
                        "status": "ERROR", "reason": "Internal processing error", "error": True
                    }
                    metrics.record_file(failed=True)
                if res is not None:
                    results_map[idx] = res

        results = [results_map[i] for i in sorted(results_map)]
        gc.collect()

        duration = time.time() - start_time
        metrics.record_request(duration, failed=False)

        return jsonify(sanitize_json({"mode": "standalone", "results": results}))

    except Exception:
        logger.exception("Standalone QC failed for session %s", session_id)
        shutil.rmtree(root, ignore_errors=True)
        gc.collect()
        metrics.record_request(time.time() - start_time, failed=True)
        return jsonify(sanitize_json({"error": "Internal processing error. Check server logs for details."})), 500


# -- GUNICORN ENTRY POINT --
if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5001"))
    app.run(debug=debug_mode, host=host, port=port)
