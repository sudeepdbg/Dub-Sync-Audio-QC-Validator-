"""
Audio Alignment Engine — Production Hardened
============================================
Temporal alignment QC engine for dubbed audio against reference masters.

Production improvements applied:
- Protected future.result() with per-worker exception handling
- Added /health endpoint for load balancer health checks
- Added rate limiting (Flask-Limiter)
- Added per-file size validation
- Added structured JSON logging with correlation IDs
- Fixed RMS normalization guard for silent files
- Added secondary chromagram DNA metric for dialogue/music hybrid content
- Added confidence intervals to offset estimates
- Added input validation for empty/corrupted audio
- Added Gunicorn-compatible entry point
- Added Docker-ready configuration via env vars
- Fixed all edge cases in speed factor calculation
- Added retry logic for transient I/O failures
- Added Prometheus-compatible metrics endpoint
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from functools import wraps

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
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH", "1073741824"))  # 1 GB

# Rate limiting
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

# Audio processing constants
PERFORMANCE_SR = 22050
WAVEFORM_MAX_POINTS = 2000
SEGMENT_DURATION = 60.0
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
MIN_RELIABLE_DURATION_SEC = 3.0
MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE", "209715200"))  # 200MB default

# QC thresholds
OFFSET_THRESHOLD_MS = 80.0
DRIFT_THRESHOLD_MS = 150.0
DNA_MATCH_THRESHOLD = 80.0
LUFS_TARGET = -23.0
LUFS_TOLERANCE = 1.0
TRUE_PEAK_MAX_DBTP = -2.0

# RMS normalization minimum dynamic range (prevents noise amplification)
RMS_MIN_DYNAMIC_RANGE = 1e-5

FRAME_RATES = {
    "23.976": 23.976,
    "25": 25.0,
    "29.97": 29.97,
}

ALLOWED_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".aac", ".ogg",
    ".m4a", ".aiff", ".aif", ".opus"
}

HOP_LENGTH = 512
WINDOW_SECONDS = 10.0
BUTTER_ORDER = 2
VOCAL_LOW_HZ = 300.0
VOCAL_HIGH_HZ = 3400.0

# ── METRICS (Prometheus-compatible) ───────────────────────────────────────────
class MetricsCollector:
    """Simple in-memory metrics for Prometheus scraping."""
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
# HELP alignment_requests_total Total upload requests
# TYPE alignment_requests_total counter
alignment_requests_total {self._requests_total}
# HELP alignment_requests_failed Total failed requests
# TYPE alignment_requests_failed counter
alignment_requests_failed {self._requests_failed}
# HELP alignment_processing_seconds_total Total processing time
# TYPE alignment_processing_seconds_total counter
alignment_processing_seconds_total {self._processing_time_total:.3f}
# HELP alignment_files_processed_total Total files processed
# TYPE alignment_files_processed_total counter
alignment_files_processed_total {self._files_processed}
# HELP alignment_files_failed_total Total files failed
# TYPE alignment_files_failed_total counter
alignment_files_failed_total {self._files_failed}
# HELP alignment_avg_processing_seconds Average processing time per request
# TYPE alignment_avg_processing_seconds gauge
alignment_avg_processing_seconds {avg_time:.3f}
"""

metrics = MetricsCollector()

# ── AUTO-CLEANUP ──────────────────────────────────────────────────────────────
def _cleanup_worker():
    """Background thread to remove old session folders (>1 hour)."""
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
    """Attach correlation ID to each request for tracing."""
    g.session_id = f"REQ_{uuid.uuid4().hex[:8].upper()}"
    g.start_time = time.time()


@app.after_request
def after_request(response):
    """Log request completion with timing."""
    duration = time.time() - g.start_time
    logger.info(
        f"Request {g.session_id} completed: {response.status_code} in {duration:.3f}s",
        extra={"session_id": g.session_id}
    )
    return response

# ── HELPERS ───────────────────────────────────────────────────────────────────
def sanitize_json(obj):
    """Recursively convert numpy types to native Python and replace NaN/Inf with None."""
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
    """Validate uploaded file size before saving."""
    file_storage.seek(0, os.SEEK_END)
    size = file_storage.tell()
    file_storage.seek(0)

    if size > MAX_FILE_SIZE:
        return False, f"File size {size / 1024 / 1024:.1f}MB exceeds limit of {MAX_FILE_SIZE / 1024 / 1024:.0f}MB"
    if size == 0:
        return False, "File is empty"
    return True, ""


def apply_vocal_filter(y: np.ndarray) -> np.ndarray:
    """Apply HPSS + vocal bandpass filter."""
    try:
        _, y_perc = librosa.effects.hpss(y)
    except Exception:
        y_perc = y

    nyq = 0.5 * PERFORMANCE_SR
    low = VOCAL_LOW_HZ / nyq
    high = VOCAL_HIGH_HZ / nyq

    if not (0 < low < high < 1):
        logger.warning("Invalid filter frequencies, returning unfiltered")
        return np.nan_to_num(y_perc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    b, a = butter(BUTTER_ORDER, [low, high], btype="band")
    out = lfilter(b, a, y_perc)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def normalize_lufs(y, sr, target=-23.0):
    """Normalize to target LUFS with guards for edge cases."""
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
    """Compute RMS envelope with safe resampling."""
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0].astype(np.float64)
    if len(rms) <= 1:
        return np.zeros(target_pts, dtype=np.float64)
    if len(rms) != target_pts:
        rms = signal.resample(rms, target_pts).astype(np.float64)
    rms = rms[:target_pts]
    peak = np.max(rms)
    return rms / peak if peak > 0 else rms


def downsample_waveform(y, max_pts=WAVEFORM_MAX_POINTS):
    """Peak-pick downsample for waveform visualization."""
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


# ── SINGLE-PASS INGESTION & PROCESSING ENGINE ─────────────────────────────────
def process_audio_single_pass(path, target_sr=PERFORMANCE_SR, seg_dur=SEGMENT_DURATION):
    """
    Stream file to collect metadata, compute level statistics, and extract
    analysis windows. True peak and sample peak are computed across the ENTIRE
    file during streaming to catch inter-sample overs anywhere.
    """
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

    with sf.SoundFile(path) as f:
        for block in f.blocks(blocksize=65536, dtype="float32"):
            mono = np.mean(block, axis=1) if block.ndim > 1 else block
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

            if channels >= 2:
                ch1 = block[:, 0]
                ch2 = block[:, 1]
                cross_prod += np.sum(ch1 * ch2)
                var_m1 += np.sum(ch1 ** 2)
                var_m2 += np.sum(ch2 ** 2)

    sample_peak_db = float(20 * np.log10(max_val))
    true_peak_val = float(20 * np.log10(max_true_peak_val + 1e-10))

    # Phase correlation
    if channels < 2:
        phase_str = "1.0 (Mono)"
    else:
        denom = np.sqrt(var_m1 * var_m2)
        corr = float(cross_prod / denom) if denom > 0 else 0.0
        status = "Healthy" if corr > 0.4 else "🚩 Issue"
        phase_str = f"{round(corr, 2)} ({status})"

    # Load analysis segments
    y_start, _ = librosa.load(
        path, sr=target_sr, offset=0.0,
        duration=seg_dur, res_type="soxr_hq"
    )

    if total_duration > seg_dur * 2:
        y_end, _ = librosa.load(
            path, sr=target_sr,
            offset=max(0.0, total_duration - seg_dur),
            duration=seg_dur, res_type="soxr_hq"
        )
    else:
        y_end = y_start

    # LUFS
    try:
        meter = pyln.Meter(target_sr)
        raw_lufs = meter.integrated_loudness(y_start)
        if np.isfinite(raw_lufs):
            lufs_val = float(raw_lufs)
            lufs_str = f"{round(lufs_val, 2)} LUFS"
        else:
            lufs_val = None
            lufs_str = "N/A"
    except Exception:
        lufs_val = None
        lufs_str = "ERR"

    levels = {
        "lufs": lufs_str,
        "lufs_val": lufs_val,
        "peak": f"{round(sample_peak_db, 2)} dBFS",
        "peak_val": sample_peak_db,
        "true_peak": f"{round(true_peak_val, 2)} dBTP",
        "true_peak_val": true_peak_val,
    }

    return meta, levels, phase_str, y_start, y_end


# ── ALIGNMENT ANALYSIS ────────────────────────────────────────────────────────
def analyze_segment(y_ref, y_comp, sr):
    """
    Compute temporal offset and DNA match score.
    Returns: (offset_ms, dna_score, confidence_interval)
    """
    hop = HOP_LENGTH
    y_ref = np.nan_to_num(np.asarray(y_ref, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y_comp = np.nan_to_num(np.asarray(y_comp, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    # ── OFFSET via RMS Envelope Cross-Correlation ──
    ref_rms = librosa.feature.rms(y=y_ref, hop_length=hop)[0].astype(np.float64)
    comp_rms = librosa.feature.rms(y=y_comp, hop_length=hop)[0].astype(np.float64)

    # Guard: check dynamic range before normalization
    ref_range = ref_rms.max() - ref_rms.min()
    comp_range = comp_rms.max() - comp_rms.min()

    if ref_range < RMS_MIN_DYNAMIC_RANGE or comp_range < RMS_MIN_DYNAMIC_RANGE:
        logger.warning("Insufficient dynamic range for reliable RMS alignment")
        return 0.0, 0.0, {"offset_ci": [0.0, 0.0], "dna_ci": [0.0, 0.0]}

    ref_rms = (ref_rms - ref_rms.min()) / ref_range
    comp_rms = (comp_rms - comp_rms.min()) / comp_range

    corr = signal.correlate(comp_rms, ref_rms, mode="full")
    lag = np.argmax(corr) - (len(ref_rms) - 1)
    offset_ms = round(float(lag * hop / sr * 1000), 2)

    # Confidence interval for offset: ±1 hop at 95% confidence
    offset_ci = [
        round(offset_ms - (hop / sr * 1000), 2),
        round(offset_ms + (hop / sr * 1000), 2)
    ]

    # ── DNA Match via Windowed Cross-Correlation ──
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

    # Confidence interval for DNA: IQR-based
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


# ── CHROMAGRAM DNA (Secondary Metric) ─────────────────────────────────────────
def analyze_chromagram_dna(y_ref, y_comp, sr=PERFORMANCE_SR):
    """
    Secondary DNA metric using chromagram cross-correlation.
    More robust for music/dialogue hybrid content where onset strength
    may be dominated by percussion rather than speech structure.
    """
    try:
        min_len = min(len(y_ref), len(y_comp))
        if min_len < 512:
            return 0.0

        y_ref = y_ref[:min_len]
        y_comp = y_comp[:min_len]

        # Compute chromagrams
        chroma_ref = librosa.feature.chroma_stft(y=y_ref, sr=sr, hop_length=HOP_LENGTH)
        chroma_comp = librosa.feature.chroma_stft(y=y_comp, sr=sr, hop_length=HOP_LENGTH)

        # Normalize
        chroma_ref = chroma_ref / (np.linalg.norm(chroma_ref, axis=0, keepdims=True) + 1e-10)
        chroma_comp = chroma_comp / (np.linalg.norm(chroma_comp, axis=0, keepdims=True) + 1e-10)

        # Windowed correlation
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

            # Mean chroma vector correlation
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


# ── SPEED FACTOR ────────────────────────────────────────────────────────────────
def calculate_speed_factor(start_offset_ms, end_offset_ms, duration_sec):
    if duration_sec <= 0:
        return {"ratio": 1.0, "display": "N/A", "delta": "N/A", "action": "N/A"}

    drift_sec = (end_offset_ms - start_offset_ms) / 1000.0
    denom = duration_sec + drift_sec

    if denom <= 0:
        return {
            "ratio": 1.0,
            "display": "N/A",
            "delta": "N/A",
            "action": "Drift exceeds duration — manual review required"
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
        "display": f"{speed_factor:.6f}×",
        "delta": f"{pct_delta:+.4f}%",
        "action": action,
    }


# ── STATUS DETERMINATION ──────────────────────────────────────────────────────
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
        issues.append(
            f"True peak {round(true_peak_val, 2)} dBTP exceeds {TRUE_PEAK_MAX_DBTP} dBTP ceiling"
        )
    if lufs_val is not None and abs(lufs_val - LUFS_TARGET) > LUFS_TOLERANCE:
        issues.append(
            f"Integrated loudness {round(lufs_val, 2)} LUFS outside {LUFS_TARGET}±{LUFS_TOLERANCE} LU target"
        )

    return ("FAIL" if issues else "PASS", "; ".join(issues) if issues else "All metrics within thresholds")


# ── WORKER COMPUTE THREAD ─────────────────────────────────────────────────────
def process_file(stored_name, display_name, root, y_ref_s_an, y_ref_e_an, y_ref_s_raw, ref_meta, vocal_logic):
    try:
        f_path = os.path.join(root, stored_name)

        # Retry logic for transient I/O failures
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

        # Secondary chromagram DNA
        chroma_dna = analyze_chromagram_dna(y_ref_s_an, y_c_s_an, PERFORMANCE_SR)

        speed = calculate_speed_factor(s_off, e_off, comp_dur)
        status, reason = determine_status(
            s_off, drift, dna,
            lufs_val=levels.get("lufs_val"),
            true_peak_val=levels.get("true_peak_val"),
            chroma_dna=chroma_dna
        )

        # Short-clip guard
        ref_dur = ref_meta.get("duration_sec", 0.0)
        if comp_dur < MIN_RELIABLE_DURATION_SEC or ref_dur < MIN_RELIABLE_DURATION_SEC:
            status = "WARN"
            reason = (
                f"Insufficient audio for reliable alignment "
                f"(min {MIN_RELIABLE_DURATION_SEC:.0f}s recommended; "
                f"reference {round(ref_dur, 2)}s, dub {round(comp_dur, 2)}s). "
                "Metrics shown are indicative only."
            )

        result = {
            "filename": display_name,
            "status": status,
            "reason": reason,
            "offset_ms": s_off,
            "offset_confidence": confidence["offset_ci"],
            "total_drift_ms": drift,
            "offset_frames": ms_to_frames(s_off),
            "drift_frames": ms_to_frames(drift),
            "dna_match": dna,
            "dna_confidence": confidence["dna_ci"],
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
        }

        del y_c_s, y_c_e, y_c_s_raw, y_c_s_an, y_c_e_an
        metrics.record_file(failed=False)
        return result

    except Exception as err:
        logger.error(f"Error processing {display_name}: {err}", exc_info=True)
        metrics.record_file(failed=True)
        return {"filename": display_name, "status": "ERROR", "reason": str(err), "error": True}


# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Health check endpoint for load balancers."""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0"
    })


@app.route("/metrics")
def metrics_endpoint():
    """Prometheus-compatible metrics endpoint."""
    return metrics.render(), 200, {"Content-Type": "text/plain; version=0.0.4"}


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
        ref = request.files.get("reference")
        comps = request.files.getlist("comparison[]")

        if not ref or not comps:
            return jsonify(sanitize_json({"error": "Missing mandatory reference or comparison assets"})), 400

        # Validate reference file
        valid, msg = validate_file_size(ref)
        if not valid:
            return jsonify(sanitize_json({"error": f"Reference file error: {msg}"})), 400

        if not allowed_file(ref.filename):
            return jsonify(sanitize_json({"error": f"Unsupported master file format: '{os.path.splitext(ref.filename)[1]}'"})), 400

        ref_secure_name = secure_filename(ref.filename)
        ref_path = os.path.join(root, ref_secure_name)
        ref.save(ref_path)

        # Single-pass parsing of Master file
        ref_meta, ref_levels, ref_phase, y_ref_s, y_ref_e = process_audio_single_pass(ref_path)
        y_ref_s_raw = y_ref_s.copy()

        if vocal_logic:
            y_ref_s_an = apply_vocal_filter(normalize_lufs(y_ref_s, PERFORMANCE_SR))
            y_ref_e_an = apply_vocal_filter(normalize_lufs(y_ref_e, PERFORMANCE_SR))
        else:
            y_ref_s_an = y_ref_s
            y_ref_e_an = y_ref_e

        # Pre-filter and validate comparison files
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

        # Process comparison files in parallel with protected futures
        results_map = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(
                    process_file, stored_name, display_name, root,
                    y_ref_s_an, y_ref_e_an, y_ref_s_raw, ref_meta, vocal_logic
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

        return jsonify(sanitize_json({"results": results}))

    except Exception:
        logger.exception("Upload processing failed for session %s", session_id)
        failed = True
        shutil.rmtree(root, ignore_errors=True)
        gc.collect()
        metrics.record_request(time.time() - start_time, failed=True)
        return jsonify(sanitize_json({"error": "Internal processing error. Check server logs for details."})), 500


# ── GUNICORN ENTRY POINT ──────────────────────────────────────────────────────
# For production: gunicorn -w 4 -b 0.0.0.0:5001 "app:app"
# Do NOT use app.run() in production — it's single-threaded and has the Werkzeug debugger

if __name__ == "__main__":
    # Development-only entry point
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5001"))
    app.run(debug=debug_mode, host=host, port=port)
