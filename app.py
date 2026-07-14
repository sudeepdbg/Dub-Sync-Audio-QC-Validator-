import os
import gc
import uuid
import math
import shutil
import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import signal
from scipy.signal import butter, lfilter, resample_poly
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
# 1 GB cap — Note: Production deployments must stream this to disk using a WSGI middleware
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

PERFORMANCE_SR = 22050
WAVEFORM_MAX_POINTS = 2000
SEGMENT_DURATION = 60
MAX_WORKERS = 4
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.aiff', '.aif', '.opus'}

# Clips shorter than this lack enough material for reliable temporal alignment
MIN_RELIABLE_DURATION_SEC = 3.0

# ── QC GATE THRESHOLDS ──────────────────────────────────────────────────────────
# Sync/DNA thresholds (unchanged from original logic)
OFFSET_THRESHOLD_MS = 80
DRIFT_THRESHOLD_MS = 150
DNA_MATCH_THRESHOLD = 80

# Loudness / true-peak thresholds — tune these to match your actual delivery spec
# (EBU R128 uses -23 LUFS / -1 dBTP; ATSC A/85 uses -24 LKFS; Netflix uses -27 LUFS
# dialogue-gated with its own peak rules). -23 LUFS ±1 LU and -2 dBTP are reasonable
# generic broadcast-safe defaults but should be confirmed against the actual delivery doc.
LUFS_TARGET = -23.0
LUFS_TOLERANCE = 1.0
TRUE_PEAK_MAX_DBTP = -2.0

FRAME_RATES = {
    "23.976": 23.976,
    "25":     25.0,
    "29.97":  29.97,
}

# ── AUTO-CLEANUP ───────────────────────────────────────────────────────────────
def _cleanup_worker():
    while True:
        now = time.time()
        try:
            for folder in os.listdir(DATA_DIR):
                path = os.path.join(DATA_DIR, folder)
                if (os.path.isdir(path) and folder.startswith("SES_")
                        and os.path.getmtime(path) < now - 3600):
                    shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass
        time.sleep(600)

threading.Thread(target=_cleanup_worker, daemon=True).start()

# ── HELPERS ────────────────────────────────────────────────────────────────────
def sanitize_json(obj):
    """Recursively convert numpy scalar types to native python and replace
    NaN / +Inf / -Inf floats with None so Flask emits STRICT, browser-parseable JSON."""
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
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS

def apply_vocal_filter(y: np.ndarray) -> np.ndarray:
    try:
        _, y_perc = librosa.effects.hpss(y)
    except Exception:
        y_perc = y  # Fallback if signal is too short or uniform for HPSS

    nyq = 0.5 * PERFORMANCE_SR
    b, a = butter(2, [300 / nyq, 3400 / nyq], btype='band')
    out = lfilter(b, a, y_perc)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def normalize_lufs(y, sr, target=-23.0):
    try:
        if y is None or len(y) == 0:
            return y
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        # pyloudnorm returns -inf/nan for too-short, silent, or sub-gating-threshold
        # audio. Applying gain against -inf yields infinite gain -> NaN-poisoned
        # signal that breaks JSON and librosa. Skip normalization in that case.
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

def rms_envelope(y, target_pts=2000):
    rms = librosa.feature.rms(y=y, hop_length=64)[0].astype(np.float64)
    if len(rms) != target_pts and len(rms) > 1:
        rms = resample_poly(rms, up=target_pts, down=len(rms)).astype(np.float64)
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

# ── SINGLE-PASS INGESTION & PROCESSING ENGINE ──────────────────────────────────
def process_audio_single_pass(path, target_sr=PERFORMANCE_SR, seg_dur=SEGMENT_DURATION):
    """
    Optimized Processing Architecture:
    Reads the file sequentially in a single pass to collect metadata,
    calculates metrics on downsampled blocks, and extracts analytics windows.

    True peak and sample peak are both computed across the ENTIRE file during
    this streaming pass (not just the analysis segment) so that clipping or
    inter-sample overs anywhere in the file — not only in the first/last
    SEGMENT_DURATION seconds — are caught by the QC gate.
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

    # Stream file to compute level statistics and phase metrics without high memory overhead
    max_val = 1e-10          # for sample peak (dBFS)
    max_true_peak_val = 1e-10  # for true peak (dBTP), across the whole file
    cross_prod = 0.0
    var_m1 = 0.0
    var_m2 = 0.0

    with sf.SoundFile(path) as f:
        for block in f.blocks(blocksize=65536, dtype='float32'):
            mono = np.mean(block, axis=1) if block.ndim > 1 else block

            block_max = np.max(np.abs(mono))
            if block_max > max_val:
                max_val = block_max

            # True-peak: 4x oversample each block to detect inter-sample overs across
            # the whole file. Oversampling per-block (rather than the whole signal at
            # once) can introduce small edge artifacts at block boundaries, but for a
            # max-peak detector this is an accepted trade-off for streaming a large file.
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

    # Process Phase Correlation Status
    if channels < 2:
        phase_str = "1.0 (Mono)"
    else:
        denom = np.sqrt(var_m1 * var_m2)
        corr = float(cross_prod / denom) if denom > 0 else 0.0
        status = "Healthy" if corr > 0.4 else "\U0001F6A9 Issue"
        phase_str = f"{round(corr, 2)} ({status})"

    # Load high-efficiency processing slices using soxr_hq for start/end sync analysis
    y_start, _ = librosa.load(path, sr=target_sr, offset=0.0,
                              duration=seg_dur, res_type='soxr_hq')

    # Only load a separate end segment if the file is long enough that start/end
    # windows wouldn't overlap; otherwise reuse the start segment (drift will read as 0,
    # which is correct — there isn't enough material for a meaningful drift measurement).
    if total_duration > seg_dur * 2:
        y_end, _ = librosa.load(path, sr=target_sr, offset=max(0.0, total_duration - seg_dur),
                                duration=seg_dur, res_type='soxr_hq')
    else:
        y_end = y_start

    # Compute integrated loudness from the extracted segment
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

def analyze_segment(y_ref, y_comp, sr):
    hop = 512
    # Guard: librosa raises "Audio buffer is not finite everywhere" on any NaN/Inf.
    y_ref  = np.nan_to_num(np.asarray(y_ref,  dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y_comp = np.nan_to_num(np.asarray(y_comp, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    # ── OFFSET via RMS Envelope Cross-Correlation
    ref_rms = librosa.feature.rms(y=y_ref,  hop_length=hop)[0].astype(np.float64)
    comp_rms = librosa.feature.rms(y=y_comp, hop_length=hop)[0].astype(np.float64)

    ref_rms = (ref_rms - ref_rms.min()) / (ref_rms.max() - ref_rms.min() + 1e-10)
    comp_rms = (comp_rms - comp_rms.min()) / (comp_rms.max() - comp_rms.min() + 1e-10)

    corr = signal.correlate(comp_rms, ref_rms, mode='full')
    lag = np.argmax(corr) - (len(ref_rms) - 1)
    offset_ms = round(float(lag * hop / sr * 1000), 2)

    # ── DNA Match via Windowed Cross-Correlation
    WIN_SEC = 10
    WIN_FRAMES = int(WIN_SEC * sr / hop)

    ref_onset = librosa.onset.onset_strength(y=y_ref,  sr=sr, hop_length=hop)
    comp_onset = librosa.onset.onset_strength(y=y_comp, sr=sr, hop_length=hop)

    min_len = min(len(ref_onset), len(comp_onset))
    if min_len == 0:
        return offset_ms, 0.0

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

        xcorr = signal.correlate(r_norm, c_norm, mode='same')
        window_scores.append(float(np.max(xcorr)) if len(xcorr) > 0 else 0.0)

    dna_score = round(float(np.median(window_scores)) * 100, 1) if window_scores else 0.0
    dna_score = max(0.0, min(100.0, dna_score))

    if not np.isfinite(offset_ms):
        offset_ms = 0.0
    if not np.isfinite(dna_score):
        dna_score = 0.0

    return offset_ms, dna_score

def calculate_speed_factor(start_offset_ms, end_offset_ms, duration_sec):
    if duration_sec <= 0:
        return {"ratio": 1.0, "display": "N/A", "delta": "N/A", "action": "N/A"}

    drift_sec = (end_offset_ms - start_offset_ms) / 1000.0
    speed_factor = duration_sec / (duration_sec + drift_sec + 1e-10)
    pct_delta = round((speed_factor - 1.0) * 100, 4)

    if abs(pct_delta) < 0.001:
        action = "No time-stretch needed"
    elif pct_delta > 0:
        action = f"Time-compress dub by {abs(pct_delta):.4f}%"
    else:
        action = f"Time-expand dub by {abs(pct_delta):.4f}%"

    return {
        "ratio": round(speed_factor, 6),
        "display": f"{speed_factor:.6f}\u00d7",
        "delta": f"{pct_delta:+.4f}%",
        "action": action,
    }

def determine_status(offset_ms, drift_ms, dna_score, lufs_val=None, true_peak_val=None):
    issues = []
    if abs(offset_ms) > OFFSET_THRESHOLD_MS:
        issues.append(f"Start offset {offset_ms}ms exceeds \u00b1{OFFSET_THRESHOLD_MS}ms threshold")
    if abs(drift_ms) > DRIFT_THRESHOLD_MS:
        issues.append(f"Drift {drift_ms}ms exceeds \u00b1{DRIFT_THRESHOLD_MS}ms threshold")
    if dna_score < DNA_MATCH_THRESHOLD:
        issues.append(f"DNA match {dna_score}% below {DNA_MATCH_THRESHOLD}% threshold")

    # True peak: catches inter-sample clipping. Computed across the whole file
    # (see process_audio_single_pass), so this isn't limited to the sync segment.
    if true_peak_val is not None and true_peak_val > TRUE_PEAK_MAX_DBTP:
        issues.append(
            f"True peak {round(true_peak_val, 2)} dBTP exceeds {TRUE_PEAK_MAX_DBTP} dBTP "
            "ceiling (inter-sample clipping risk)"
        )

    if lufs_val is not None and abs(lufs_val - LUFS_TARGET) > LUFS_TOLERANCE:
        issues.append(
            f"Integrated loudness {round(lufs_val, 2)} LUFS outside "
            f"{LUFS_TARGET}\u00b1{LUFS_TOLERANCE} LU target"
        )

    return ("FAIL" if issues else "PASS", "; ".join(issues) if issues else "All metrics within thresholds")

# ── WORKER COMPUTE THREAD ──────────────────────────────────────────────────────
def process_file(stored_name, display_name, root, y_ref_s_an, y_ref_e_an, y_ref_s_raw, ref_meta, vocal_logic):
    try:
        f_path = os.path.join(root, stored_name)

        # Execute single-pass analysis
        comp_meta, levels, phase, y_c_s, y_c_e = process_audio_single_pass(f_path)
        comp_dur = comp_meta["duration_sec"]
        y_c_s_raw = y_c_s.copy()

        if vocal_logic:
            y_c_s_an = apply_vocal_filter(normalize_lufs(y_c_s, PERFORMANCE_SR))
            y_c_e_an = apply_vocal_filter(normalize_lufs(y_c_e, PERFORMANCE_SR))
        else:
            y_c_s_an = y_c_s
            y_c_e_an = y_c_e

        s_off, dna = analyze_segment(y_ref_s_an, y_c_s_an, PERFORMANCE_SR)
        e_off, _ = analyze_segment(y_ref_e_an, y_c_e_an, PERFORMANCE_SR)
        drift = round(e_off - s_off, 2)

        speed = calculate_speed_factor(s_off, e_off, comp_dur)
        status, reason = determine_status(
            s_off, drift, dna,
            lufs_val=levels.get("lufs_val"),
            true_peak_val=levels.get("true_peak_val"),
        )

        # ── SHORT-CLIP GUARD ────────────────────────────────────────────────
        # Below MIN_RELIABLE_DURATION_SEC there isn't enough material for the
        # cross-correlation to align meaningfully. Metrics still render, but the
        # clip is flagged WARN so operators know results are indicative only.
        ref_dur = ref_meta.get("duration_sec", 0.0)
        if comp_dur < MIN_RELIABLE_DURATION_SEC or ref_dur < MIN_RELIABLE_DURATION_SEC:
            status = "WARN"
            reason = ("Insufficient audio for reliable alignment "
                      f"(min {MIN_RELIABLE_DURATION_SEC:.0f}s recommended; "
                      f"reference {round(ref_dur, 2)}s, dub {round(comp_dur, 2)}s). "
                      "Metrics shown are indicative only.")

        result = {
            "filename": display_name,
            "status": status,
            "reason": reason,
            "offset_ms": s_off,
            "total_drift_ms": drift,
            "offset_frames": ms_to_frames(s_off),
            "drift_frames": ms_to_frames(drift),
            "dna_match": dna,
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
        return result

    except Exception as err:
        return {"filename": display_name, "status": "ERROR", "reason": str(err), "error": True}

# ── ROUTES ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    session_id = f"SES_{uuid.uuid4().hex[:6].upper()}"
    root = os.path.join(DATA_DIR, session_id)
    os.makedirs(root, exist_ok=True)

    try:
        vocal_logic = request.form.get('vocal_logic') == 'true'
        ref = request.files.get('reference')
        comps = request.files.getlist('comparison[]')

        if not ref or not comps:
            return jsonify(sanitize_json({"error": "Missing mandatory reference or comparison assets"})), 400

        # Validate Master file format before executing storage layer allocations
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

        # Pre-filter comparison assets on the request thread to discard illegal formats instantly.
        # Each file is saved under a UUID-prefixed name on disk to avoid collisions when two
        # different uploads sanitize to the same secure_filename (e.g. "Ep 1 (final).wav" and
        # "Ep 1 final.wav" both become "Ep_1_final.wav"); the original display name is kept
        # separately for the response so results are labeled correctly.
        valid_comp_files = []  # list of (stored_name, display_name)
        for f in comps:
            if f and f.filename and allowed_file(f.filename):
                display_name = secure_filename(f.filename)
                stored_name = f"{uuid.uuid4().hex[:8]}_{display_name}"
                f.save(os.path.join(root, stored_name))
                valid_comp_files.append((stored_name, display_name))

        results_map = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(process_file, stored_name, display_name, root, y_ref_s_an, y_ref_e_an,
                            y_ref_s_raw, ref_meta, vocal_logic): i
                for i, (stored_name, display_name) in enumerate(valid_comp_files)
            }
            for future in as_completed(futures):
                idx = futures[future]
                res = future.result()
                if res is not None:
                    results_map[idx] = res

        results = [results_map[i] for i in sorted(results_map)]

        del y_ref_s, y_ref_e, y_ref_s_raw, y_ref_s_an, y_ref_e_an
        gc.collect()

        return jsonify(sanitize_json({"results": results}))

    except Exception:
        # Log full traceback server-side only — never return it to the client,
        # it leaks file paths, library versions, and internal structure.
        app.logger.exception("Upload processing failed for session %s", session_id)
        shutil.rmtree(root, ignore_errors=True)
        gc.collect()
        return jsonify(sanitize_json({"error": "Internal processing error. Check server logs for details."})), 500

if __name__ == '__main__':
    # NEVER run with debug=True and host='0.0.0.0' together: Werkzeug's interactive
    # debugger exposes an unauthenticated code-execution console to anyone who can
    # reach the port. Debug mode is opt-in via env var and defaults to loopback-only.
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    app.run(debug=debug_mode, host=host, port=5001)
