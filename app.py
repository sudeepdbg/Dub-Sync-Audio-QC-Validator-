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
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 GB

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

PERFORMANCE_SR      = 22050
WAVEFORM_MAX_POINTS = 2000
SEGMENT_DURATION    = 60       # seconds analysed at start/end of each file
MAX_WORKERS         = 4
ALLOWED_EXTENSIONS  = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.aiff', '.aif', '.opus'}

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
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS

def butter_bandpass(data, lowcut, highcut, fs, order=2):
    nyq  = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return lfilter(b, a, data)

def apply_vocal_filter(y: np.ndarray) -> np.ndarray:
    """Vocal DNA Filter — isolates percussive transient signatures."""
    _, y_perc = librosa.effects.hpss(y)
    nyq  = 0.5 * PERFORMANCE_SR
    b, a = butter(2, [300 / nyq, 3400 / nyq], btype='band')
    return lfilter(b, a, y_perc)

def normalize_lufs(y, sr, target=-23.0):
    meter = pyln.Meter(sr)
    try:
        return pyln.normalize.loudness(y, meter.integrated_loudness(y), target)
    except Exception:
        return y

def normalize_visual(y):
    m = np.max(np.abs(y))
    return y / m if m > 0 else y

def rms_envelope(y, target_pts=2000):
    """Compute smoothed energy curve resampled to fixed coordinates."""
    rms = librosa.feature.rms(y=y, hop_length=64)[0].astype(np.float64)
    if len(rms) != target_pts:
        rms = resample_poly(rms, up=target_pts, down=len(rms)).astype(np.float64)
        rms = rms[:target_pts]
    peak = np.max(rms)
    return rms / peak if peak > 0 else rms

def downsample_waveform(y, max_pts=WAVEFORM_MAX_POINTS):
    if len(y) <= max_pts:
        return y.tolist()
    step    = len(y) // max_pts
    buckets = len(y) // step
    trimmed = y[:buckets * step].reshape(buckets, step)
    idx     = np.argmax(np.abs(trimmed), axis=1)
    return trimmed[np.arange(buckets), idx].tolist()

def ms_to_frames(ms: float) -> dict:
    return {fps_label: round(ms * fps / 1000.0, 2)
            for fps_label, fps in FRAME_RATES.items()}

# ── AUDIO METADATA ─────────────────────────────────────────────────────────────
def get_file_metadata(path):
    try:
        info = sf.info(path)
        return {
            "sr":            info.samplerate,
            "duration_sec":  info.duration,
            "channel_label": ("Stereo" if info.channels == 2 else "Mono" if info.channels == 1 else f"{info.channels} Ch"),
            "format":        info.format,
        }
    except Exception:
        return {"sr": 0, "duration_sec": 0, "channel_label": "N/A", "format": "N/A"}

def true_peak_db(data: np.ndarray, rate: int) -> float:
    try:
        tp    = pyln.meter.true_peak(data, rate)
        return float(20 * np.log10(np.max(np.abs(tp)) + 1e-10))
    except Exception:
        pass
    try:
        mono = np.mean(data, axis=1) if data.ndim > 1 else data
        up   = resample_poly(mono, up=4, down=1)
        return float(20 * np.log10(np.max(np.abs(up)) + 1e-10))
    except Exception:
        return 0.0

def scan_levels(path):
    try:
        data, rate = sf.read(path)
        mono       = np.mean(data, axis=1) if data.ndim > 1 else data
        sample_peak = float(20 * np.log10(np.max(np.abs(mono)) + 1e-10))
        meter    = pyln.Meter(rate)
        lufs_val = float(meter.integrated_loudness(data))
        return {
            "lufs":      lufs_val,
            "peak":      sample_peak,
            "true_peak": true_peak_db(data, rate),
        }
    except Exception:
        return {"lufs": 0.0, "peak": 0.0, "true_peak": 0.0}

def calculate_phase(path):
    try:
        data, _ = sf.read(path)
        if data.ndim < 2 or data.shape[1] < 2:
            return "1.0 (Mono)"
        corr   = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
        status = "Healthy" if corr > 0.4 else "🚩 Issue"
        return f"{round(float(corr), 2)} ({status})"
    except Exception:
        return "N/A"

def load_segment(path, sr, offset=0.0, duration=None):
    return librosa.load(path, sr=sr, offset=offset, duration=duration, res_type='kaiser_fast')

# ── CORE ANALYSIS ──────────────────────────────────────────────────────────────
def analyze_segment(y_ref, y_comp, sr):
    hop = 512
    ref_rms  = librosa.feature.rms(y=y_ref,  hop_length=hop)[0].astype(np.float64)
    comp_rms = librosa.feature.rms(y=y_comp, hop_length=hop)[0].astype(np.float64)

    ref_rms  = (ref_rms  - ref_rms.min())  / (ref_rms.max()  - ref_rms.min()  + 1e-10)
    comp_rms = (comp_rms - comp_rms.min()) / (comp_rms.max() - comp_rms.min() + 1e-10)

    corr      = signal.correlate(comp_rms, ref_rms, mode='full')
    lag       = np.argmax(corr) - (len(ref_rms) - 1)
    offset_ms = round(float(lag * hop / sr * 1000), 2)

    WIN_SEC    = 10
    WIN_FRAMES = int(WIN_SEC * sr / hop)

    ref_onset  = librosa.onset.onset_strength(y=y_ref,  sr=sr, hop_length=hop)
    comp_onset = librosa.onset.onset_strength(y=y_comp, sr=sr, hop_length=hop)

    min_len    = min(len(ref_onset), len(comp_onset))
    ref_onset  = ref_onset[:min_len]
    comp_onset = comp_onset[:min_len]

    n_windows  = max(1, min_len // WIN_FRAMES)
    window_scores = []

    for w in range(n_windows):
        s = w * WIN_FRAMES
        e = s + WIN_FRAMES
        r_win = ref_onset[s:e].astype(np.float64)
        c_win = comp_onset[s:e].astype(np.float64)

        r_norm = r_win  / (np.linalg.norm(r_win)  + 1e-10)
        c_norm = c_win  / (np.linalg.norm(c_win)  + 1e-10)

        xcorr = signal.correlate(r_norm, c_norm, mode='same')
        window_scores.append(float(np.max(xcorr)))

    dna_score = round(float(np.median(window_scores)) * 100, 1)
    return offset_ms, max(0.0, min(100.0, dna_score))

def determine_status(offset_ms, drift_ms, dna_score):
    issues = []
    if abs(offset_ms) > 80:
        issues.append(f"Start offset {offset_ms}ms exceeds ±80ms threshold")
    if abs(drift_ms) > 150:
        issues.append(f"Drift {drift_ms}ms exceeds ±150ms threshold")
    if dna_score < 80:
        issues.append(f"DNA match {dna_score}% below 80% threshold")

    return ("FAIL" if issues else "PASS",
            "; ".join(issues) if issues else "All metrics within standard operational thresholds.")

# ── PER-FILE WORKER ────────────────────────────────────────────────────────────
def process_file(f, root, y_ref_s_an, y_ref_e_an, y_ref_s_raw, ref_meta, ref_levels, vocal_logic):
    if not f or not f.filename:
        return None

    if not allowed_file(f.filename):
        return {"filename": f.filename, "verdict": "FAIL", "verdict_reason": f"Unsupported type.", "error": True}
    try:
        f_path    = os.path.join(root, secure_filename(f.filename))
        f.save(f_path)
        comp_meta = get_file_metadata(f_path)
        comp_levels = scan_levels(f_path)
        comp_dur  = comp_meta["duration_sec"]

        y_c_s, _ = load_segment(f_path, PERFORMANCE_SR, duration=SEGMENT_DURATION)
        y_c_e, _ = load_segment(f_path, PERFORMANCE_SR, offset=max(0.0, comp_dur - SEGMENT_DURATION))

        y_c_s_raw = y_c_s.copy()

        if vocal_logic:
            y_c_s_an = apply_vocal_filter(normalize_lufs(y_c_s, PERFORMANCE_SR))
            y_c_e_an = apply_vocal_filter(normalize_lufs(y_c_e, PERFORMANCE_SR))
        else:
            y_c_s_an = y_c_s
            y_c_e_an = y_c_e

        s_off, dna = analyze_segment(y_ref_s_an, y_c_s_an, PERFORMANCE_SR)
        e_off, _   = analyze_segment(y_ref_e_an, y_c_e_an, PERFORMANCE_SR)
        drift      = round(e_off - s_off, 2)

        # Compute raw clock ratio scalar expected by the chart calculation line
        drift_sec = drift / 1000.0
        speed_val = float(comp_dur / (comp_dur + drift_sec + 1e-10))

        status, reason = determine_status(s_off, drift, dna)

        result = {
            "filename":       f.filename,
            "verdict":        status,
            "verdict_reason": reason,
            "offset_ms":      s_off,
            "drift_ms":       drift,
            "score_pct":      dna,
            "phase_health":   calculate_phase(f_path),
            "speed_factor":   round(speed_val, 6),
            
            # Flattened structural fields explicitly parsed by index.html template matrix
            "master_sample_rate": ref_meta["sr"],
            "dub_sample_rate":    comp_meta["sr"],
            "master_duration":     ref_meta["duration_sec"],
            "dub_duration":        comp_meta["duration_sec"],
            "master_channels":     ref_meta["channel_label"],
            "dub_channels":        comp_meta["channel_label"],
            "master_format":       ref_meta["format"],
            "dub_format":          comp_meta["format"],
            "master_loudness":     ref_levels["lufs"],
            "dub_loudness":        comp_levels["lufs"],
            "master_peak":         ref_levels["peak"],
            "dub_peak":            comp_levels["peak"],
            "master_true_peak":    ref_levels["true_peak"],
            "dub_true_peak":       comp_levels["true_peak"],

            # EChart rendering payload arrays
            "master_rms_envelope": rms_envelope(y_ref_s_raw).tolist(),
            "dub_rms_envelope":    rms_envelope(y_c_s_raw).tolist(),
            "master_raw_peaks":    downsample_waveform(np.abs(normalize_visual(y_ref_s_raw))),
            "dub_raw_peaks":       downsample_waveform(np.abs(normalize_visual(y_c_s_raw)))
        }

        del y_c_s, y_c_e, y_c_s_raw, y_c_s_an, y_c_e_an
        gc.collect()
        return result

    except Exception as err:
        return {"filename": f.filename, "verdict": "FAIL", "verdict_reason": str(err), "error": True}

# ── ROUTES ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    session_id = f"SES_{uuid.uuid4().hex[:6].upper()}"
    root       = os.path.join(DATA_DIR, session_id)
    os.makedirs(root, exist_ok=True)

    try:
        # Match template's optional high-pass configuration flag name
        vocal_logic = request.form.get('vocal_dna') == 'on' or request.form.get('vocal_logic') == 'true'
        
        # Match structural form component names 'master_file' and 'dub_files' from index.html
        ref   = request.files.get('master_file')
        comps = request.files.getlist('dub_files')

        if not ref:
            return jsonify({"error": "No master reference file provided"}), 400
        if not comps or len(comps) == 0 or comps[0].filename == '':
            return jsonify({"error": "No comparison dub file provided"}), 400
        if not allowed_file(ref.filename):
            return jsonify({"error": f"Master file extension not supported."}), 400

        ref_path  = os.path.join(root, secure_filename(ref.filename))
        ref.save(ref_path)
        ref_meta  = get_file_metadata(ref_path)
        ref_levels = scan_levels(ref_path)
        total_dur = ref_meta["duration_sec"]

        y_ref_s, _ = load_segment(ref_path, PERFORMANCE_SR, duration=SEGMENT_DURATION)
        y_ref_e, _ = load_segment(ref_path, PERFORMANCE_SR, offset=max(0.0, total_dur - SEGMENT_DURATION))

        y_ref_s_raw = y_ref_s.copy()

        if vocal_logic:
            y_ref_s_an = apply_vocal_filter(normalize_lufs(y_ref_s, PERFORMANCE_SR))
            y_ref_e_an = apply_vocal_filter(normalize_lufs(y_ref_e, PERFORMANCE_SR))
        else:
            y_ref_s_an = y_ref_s
            y_ref_e_an = y_ref_e

        results_map = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(process_file, f, root, y_ref_s_an, y_ref_e_an, 
                            y_ref_s_raw, ref_meta, ref_levels, vocal_logic): i
                for i, f in enumerate(comps)
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results_map[futures[future]] = result

        results = [results_map[i] for i in sorted(results_map)]

        del y_ref_s, y_ref_e, y_ref_s_raw, y_ref_s_an, y_ref_e_an
        gc.collect()

        # Wrap array response object for the frontend dashboard extractor routine
        return jsonify({"results": results})

    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        gc.collect()
        return jsonify({"error": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
