import os
import gc
import uuid
import shutil
import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
import threading
import time
import traceback
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

# Duration mismatch beyond this is flagged — segment-based drift analysis assumes
# ref/comp cover roughly the same programme length.
DURATION_MISMATCH_THRESHOLD_SEC = 2.0

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
def allowed_file(filename: str) -> bool:
    if not filename:
        return False
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS

def apply_vocal_filter(y: np.ndarray):
    """
    Returns (filtered_audio, applied: bool). `applied` is False whenever HPSS
    failed and we silently fell back to the unfiltered signal — callers must
    surface this to the report rather than assuming the requested flag means
    the filter actually ran.
    """
    try:
        _, y_perc = librosa.effects.hpss(y)
    except Exception:
        return y, False

    try:
        nyq = 0.5 * PERFORMANCE_SR
        b, a = butter(2, [300 / nyq, 3400 / nyq], btype='band')
        return lfilter(b, a, y_perc), True
    except Exception:
        return y_perc, False

def normalize_lufs(y, sr, target=-23.0):
    """Returns (normalized_audio, applied: bool)."""
    try:
        meter = pyln.Meter(sr)
        return pyln.normalize.loudness(y, meter.integrated_loudness(y), target), True
    except Exception:
        return y, False

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

def true_peak_db(mono_data: np.ndarray) -> str:
    """Calculates True Peak using an efficient 4x upsampling path."""
    try:
        up = resample_poly(mono_data, up=4, down=1)
        tp_db = 20 * np.log10(np.max(np.abs(up)) + 1e-10)
        return f"{round(float(tp_db), 2)} dBTP"
    except Exception:
        return "N/A"

# ── SINGLE-PASS INGESTION & PROCESSING ENGINE ──────────────────────────────────
def process_audio_single_pass(path, target_sr=PERFORMANCE_SR, seg_dur=SEGMENT_DURATION):
    """
    Streams the file once via SoundFile.blocks() to compute level/phase stats,
    then loads the start/end analysis segments through librosa. Note this is
    "single-pass" for the block-level stats only — the start/end segment loads
    below are a second (partial) read, since librosa.load doesn't accept the
    block-streaming iterator we use for stats. Kept as two lightweight reads
    rather than a full second decode of the whole file.
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
    max_val = 1e-10
    cross_prod = 0.0
    var_m1 = 0.0
    var_m2 = 0.0

    with sf.SoundFile(path) as f:
        for block in f.blocks(blocksize=65536, dtype='float32'):
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

    sample_peak_db = 20 * np.log10(max_val)

    # Process Phase Correlation Status
    if channels < 2:
        phase_str = "1.0 (Mono)"
    else:
        denom = np.sqrt(var_m1 * var_m2)
        corr = float(cross_prod / denom) if denom > 0 else 0.0
        status = "Healthy" if corr > 0.4 else "🚩 Issue"
        phase_str = f"{round(corr, 2)} ({status})"

    # Load high-efficiency processing slices using kaiser_fast
    y_start, _ = librosa.load(path, sr=target_sr, offset=0.0, duration=seg_dur, res_type='kaiser_fast')

    if total_duration > seg_dur:
        y_end, _ = librosa.load(path, sr=target_sr, offset=max(0.0, total_duration - seg_dur),
                                duration=seg_dur, res_type='kaiser_fast')
    else:
        y_end = y_start

    # Compute integrated loudness from the extracted segment
    try:
        meter = pyln.Meter(target_sr)
        lufs_val = meter.integrated_loudness(y_start)
        lufs_str = f"{round(lufs_val, 2)} LUFS"
    except Exception:
        lufs_str = "ERR"

    levels = {
        "lufs": lufs_str,
        "peak": f"{round(sample_peak_db, 2)} dBFS",
        "true_peak": true_peak_db(y_start)  # Compute on segment to preserve I/O cycles
    }

    return meta, levels, phase_str, y_start, y_end

def analyze_segment(y_ref, y_comp, sr):
    hop = 512
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
        "display": f"{speed_factor:.6f}×",
        "delta": f"{pct_delta:+.4f}%",
        "action": action,
    }

def determine_status(offset_ms, drift_ms, dna_score, duration_mismatch=False):
    issues = []
    if abs(offset_ms) > 80:
        issues.append(f"Start offset {offset_ms}ms exceeds ±80ms threshold")
    if abs(drift_ms) > 150:
        issues.append(f"Drift {drift_ms}ms exceeds ±150ms threshold")
    if dna_score < 80:
        issues.append(f"DNA match {dna_score}% below 80% threshold")
    if duration_mismatch:
        issues.append("Reference/dub duration mismatch — drift figures may be unreliable")

    return ("FAIL" if issues else "PASS", "; ".join(issues) if issues else "All metrics within thresholds")

# ── WORKER COMPUTE THREAD ──────────────────────────────────────────────────────
def process_file(filename, root, y_ref_s_an, y_ref_e_an, y_ref_s_raw, ref_meta, vocal_logic):
    try:
        f_path = os.path.join(root, filename)

        # Execute single-pass analysis
        comp_meta, levels, phase, y_c_s, y_c_e = process_audio_single_pass(f_path)
        comp_dur = comp_meta["duration_sec"]
        y_c_s_raw = y_c_s.copy()

        vocal_applied = False
        lufs_applied = False
        if vocal_logic:
            y_c_s_norm, lufs_ok_s = normalize_lufs(y_c_s, PERFORMANCE_SR)
            y_c_s_an, vocal_ok_s = apply_vocal_filter(y_c_s_norm)
            y_c_e_norm, lufs_ok_e = normalize_lufs(y_c_e, PERFORMANCE_SR)
            y_c_e_an, vocal_ok_e = apply_vocal_filter(y_c_e_norm)
            vocal_applied = vocal_ok_s and vocal_ok_e
            lufs_applied = lufs_ok_s and lufs_ok_e
        else:
            y_c_s_an = y_c_s
            y_c_e_an = y_c_e

        s_off, dna = analyze_segment(y_ref_s_an, y_c_s_an, PERFORMANCE_SR)
        e_off, _ = analyze_segment(y_ref_e_an, y_c_e_an, PERFORMANCE_SR)
        drift = round(e_off - s_off, 2)

        speed = calculate_speed_factor(s_off, e_off, comp_dur)

        duration_mismatch = abs(ref_meta["duration_sec"] - comp_dur) > DURATION_MISMATCH_THRESHOLD_SEC
        status, reason = determine_status(s_off, drift, dna, duration_mismatch)

        result = {
            "filename": filename,
            "status": status,
            "reason": reason,
            "offset_ms": s_off,
            "total_drift_ms": drift,
            "offset_frames": ms_to_frames(s_off),
            "drift_frames": ms_to_frames(drift),
            "dna_match": dna,
            "vocal_filter": vocal_logic,
            "vocal_filter_applied": vocal_applied if vocal_logic else False,
            "lufs_normalized": lufs_applied if vocal_logic else False,
            "speed_factor": speed,
            "phase": phase,
            "levels": levels,
            "ref_meta": ref_meta,
            "comp_meta": comp_meta,
            "duration_mismatch": duration_mismatch,
            "wave_rms_master": rms_envelope(y_ref_s_raw).tolist(),
            "wave_rms_dub": (-rms_envelope(y_c_s_raw)).tolist(),
            "wave_raw_master": downsample_waveform(np.abs(normalize_visual(y_ref_s_raw))),
            "wave_raw_dub": downsample_waveform(-np.abs(normalize_visual(y_c_s_raw))),
            "chan_mismatch": ref_meta["channels"] != comp_meta["channels"],
        }

        del y_c_s, y_c_e, y_c_s_raw, y_c_s_an, y_c_e_an
        return result

    except Exception as err:
        return {"filename": filename, "status": "ERROR", "reason": str(err), "error": True}

# ── ROUTES ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/wipe', methods=['POST'])
def wipe():
    """Clears all session data from the data/ directory."""
    try:
        for folder in os.listdir(DATA_DIR):
            path = os.path.join(DATA_DIR, folder)
            if os.path.isdir(path) and folder.startswith("SES_"):
                shutil.rmtree(path, ignore_errors=True)
        return jsonify({"ok": True})
    except Exception as err:
        return jsonify({"ok": False, "error": str(err)}), 500

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
            return jsonify({"error": "Missing mandatory reference or comparison assets"}), 400

        # Validate Master file format before executing storage layer allocations
        if not allowed_file(ref.filename):
            return jsonify({"error": f"Unsupported master file format: '{os.path.splitext(ref.filename)[1]}'"}), 400

        ref_secure_name = secure_filename(ref.filename)
        ref_path = os.path.join(root, ref_secure_name)
        ref.save(ref_path)

        # Single-pass parsing of Master file
        ref_meta, ref_levels, ref_phase, y_ref_s, y_ref_e = process_audio_single_pass(ref_path)
        y_ref_s_raw = y_ref_s.copy()

        if vocal_logic:
            y_ref_s_norm, _ = normalize_lufs(y_ref_s, PERFORMANCE_SR)
            y_ref_s_an, _ = apply_vocal_filter(y_ref_s_norm)
            y_ref_e_norm, _ = normalize_lufs(y_ref_e, PERFORMANCE_SR)
            y_ref_e_an, _ = apply_vocal_filter(y_ref_e_norm)
        else:
            y_ref_s_an = y_ref_s
            y_ref_e_an = y_ref_e

        # Pre-filter comparison assets on the request thread to discard illegal formats instantly
        valid_comp_filenames = []
        for f in comps:
            if f and f.filename and allowed_file(f.filename):
                sec_name = secure_filename(f.filename)
                f.save(os.path.join(root, sec_name))
                valid_comp_filenames.append(sec_name)

        results_map = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(process_file, fname, root, y_ref_s_an, y_ref_e_an,
                            y_ref_s_raw, ref_meta, vocal_logic): i
                for i, fname in enumerate(valid_comp_filenames)
            }
            for future in as_completed(futures):
                idx = futures[future]
                res = future.result()
                if res is not None:
                    results_map[idx] = res

        results = [results_map[i] for i in sorted(results_map)]

        del y_ref_s, y_ref_e, y_ref_s_raw, y_ref_s_an, y_ref_e_an
        gc.collect()

        return jsonify({"results": results})

    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        gc.collect()
        return jsonify({"error": traceback.format_exc()}), 500

if __name__ == '__main__':
    # Debug mode + 0.0.0.0 exposes Werkzeug's interactive debugger on the network,
    # which is a remote code execution risk. Both are now opt-in via env vars so a
    # careless `python app.py` in a shared/production environment is safe by default.
    debug_mode = os.environ.get("SYNC_ENGINE_DEBUG", "false").lower() == "true"
    host = os.environ.get("SYNC_ENGINE_HOST", "127.0.0.1")
    port = int(os.environ.get("SYNC_ENGINE_PORT", "5001"))
    app.run(debug=debug_mode, host=host, port=port)
