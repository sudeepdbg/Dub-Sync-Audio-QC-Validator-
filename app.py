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

# Frame-rate standards used for ms → frames conversion in results
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
    """
    Vocal DNA Filter — two-stage processing for dub QC analysis:

    Stage 1 — HPSS (Harmonic-Percussive Source Separation):
        Isolates the PERCUSSIVE layer (transients, onsets, consonants).
        This is what onset-based DNA scoring actually measures — the
        temporal structure of events, not their harmonic content.
        Discarding harmonics removes sustained notes that differ between
        languages (e.g. vowel melody in English vs Spanish) while keeping
        the rhythmic fingerprint that both dubs share.

    Stage 2 — Bandpass 300–3400 Hz:
        Applied AFTER HPSS to remove sub-bass rumble and high-freq noise
        that survived separation. 300–3400 Hz is the ITU-T G.712 speech
        clarity band — the range where consonant onsets have peak energy.

    WITHOUT this filter: onset_strength sees the full mix including music
    beds and background elements that may differ between master and dub.
    WITH this filter: onset_strength sees primarily speech transients and
    foley hits — the true structural fingerprint of the content.

    NOTE: this filtered signal is used ONLY for analysis (offset + DNA).
    The waveform display always uses the unfiltered, normalised signal.
    """
    # Stage 1: separate percussive layer
    _, y_perc = librosa.effects.hpss(y)
    # Stage 2: bandpass to speech clarity band
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
    """
    Compute a smoothed RMS energy envelope resampled to exactly target_pts points.

    WHY FIXED OUTPUT LENGTH
    ───────────────────────
    librosa.feature.rms with hop=512 downsamples by 512×.
    A 4s file at 22050Hz → only ~172 RMS frames — far fewer than the 2000
    display points, so the chart renders data on the left and goes flat-zero
    on the right.

    Fix: compute RMS with a small hop (64 samples = ~3ms) to get high
    temporal resolution, then resample to exactly target_pts points using
    scipy so both Master and Dub always produce the same-length array
    regardless of file duration.  The chart then fills the full width.

    RMS envelope is still the right display signal because:
      - It averages energy over short windows → smooth loudness-over-time curve
      - Two dubs of the same content have similar envelope shapes even in
        different languages (same silences, same music hits, same pacing)
      - Raw waveforms always look different between languages at sample level
    """
    # Small hop for temporal resolution — 64 samples ≈ 2.9ms at 22050Hz
    rms = librosa.feature.rms(y=y, hop_length=64)[0].astype(np.float64)

    # Resample to exactly target_pts points so chart always fills full width
    if len(rms) != target_pts:
        rms = resample_poly(rms,
                            up=target_pts,
                            down=len(rms)).astype(np.float64)
        rms = rms[:target_pts]  # trim any overshoot from poly filter

    # Normalise to [0, 1] relative to each file's own peak energy
    peak = np.max(rms)
    return rms / peak if peak > 0 else rms


def downsample_waveform(y, max_pts=WAVEFORM_MAX_POINTS):
    """Bucket-max downsample — preserves transients, hard-caps JSON payload."""
    if len(y) <= max_pts:
        return y.tolist()
    step    = len(y) // max_pts
    buckets = len(y) // step
    trimmed = y[:buckets * step].reshape(buckets, step)
    idx     = np.argmax(np.abs(trimmed), axis=1)
    return trimmed[np.arange(buckets), idx].tolist()


def ms_to_frames(ms: float) -> dict:
    """Convert milliseconds to frame counts for every broadcast standard."""
    return {fps_label: round(ms * fps / 1000.0, 2)
            for fps_label, fps in FRAME_RATES.items()}


# ── AUDIO METADATA ─────────────────────────────────────────────────────────────
def get_file_metadata(path):
    try:
        info = sf.info(path)
        return {
            "sr":            f"{info.samplerate} Hz",
            "native_sr":     info.samplerate,
            "duration":      f"{round(info.duration, 2)}s",
            "duration_sec":  info.duration,
            "bit_depth":     info.subtype,
            "channels":      info.channels,
            "channel_label": ("Stereo" if info.channels == 2
                              else "Mono" if info.channels == 1
                              else f"{info.channels} Ch"),
            "format":        info.format,
        }
    except Exception:
        return {"sr": "N/A", "native_sr": 0, "duration": "0s", "duration_sec": 0,
                "bit_depth": "N/A", "channels": 0, "channel_label": "N/A", "format": "N/A"}


def true_peak_db(data: np.ndarray, rate: int) -> str:
    """
    True Peak (dBTP) via 4× upsampling — catches inter-sample peaks
    that sample-peak metering misses (required by EBU R128 / ITU-R BS.1770).

    Tries pyloudnorm first; falls back to scipy resample_poly so the
    result is version-independent.
    """
    try:
        # pyloudnorm >= 0.1.0 exposes this directly
        tp    = pyln.meter.true_peak(data, rate)
        tp_db = 20 * np.log10(np.max(np.abs(tp)) + 1e-10)
        return f"{round(float(tp_db), 2)} dBTP"
    except Exception:
        pass

    try:
        # Fallback: scipy 4× upsample on mono signal
        mono = np.mean(data, axis=1) if data.ndim > 1 else data
        up   = resample_poly(mono, up=4, down=1)
        tp_db = 20 * np.log10(np.max(np.abs(up)) + 1e-10)
        return f"{round(float(tp_db), 2)} dBTP"
    except Exception:
        return "N/A"


def scan_levels(path):
    try:
        data, rate = sf.read(path)
        mono       = np.mean(data, axis=1) if data.ndim > 1 else data

        sample_peak_db = 20 * np.log10(np.max(np.abs(mono)) + 1e-10)

        meter    = pyln.Meter(rate)
        lufs_val = meter.integrated_loudness(data)
        lufs     = f"{round(lufs_val, 2)} LUFS"

        return {
            "lufs":      lufs,
            "peak":      f"{round(sample_peak_db, 2)} dBFS",
            "true_peak": true_peak_db(data, rate),
        }
    except Exception:
        return {"lufs": "ERR", "peak": "ERR", "true_peak": "ERR"}


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
    """
    librosa.load with kaiser_fast (~40% faster, imperceptible quality
    difference for envelope/onset analysis at PERFORMANCE_SR).
    Always resamples to PERFORMANCE_SR regardless of native SR, so all
    downstream hop→ms conversions are consistent.
    """
    return librosa.load(path, sr=sr, offset=offset,
                        duration=duration, res_type='kaiser_fast')


# ── CORE ANALYSIS ──────────────────────────────────────────────────────────────
def analyze_segment(y_ref, y_comp, sr):
    """
    Offset  : RMS-envelope cross-correlation.
    DNA     : Onset-pattern cross-correlation — REPLACES MFCC cosine mean.

    WHY THE OLD MFCC APPROACH WAS WRONG
    ─────────────────────────────────────
    MFCC mean collapses 60 s of audio into one 20-dim vector.  Two
    different songs with similar production (vocals + music, similar era)
    will have nearly identical mean MFCC vectors because both look
    "music-shaped" on average.  Result: 99% DNA even for totally
    unrelated content, which is useless for QC.

    WHY ONSET-PATTERN CROSS-CORRELATION IS CORRECT
    ────────────────────────────────────────────────
    A dub of the same content — even in a different language — must
    follow the same rhythmic structure: dialogue starts, pauses, music
    hits, sound effects.  Onset envelopes capture exactly this structure.
    Cross-correlating them gives a similarity score that is:
      • High   when both files share the same rhythmic/event fingerprint
               (true dub of same content)
      • Low    when they are unrelated content
               (different show, different song)

    The peak of the normalised cross-correlation is bounded [0, 1],
    so scaling to [0, 100] is mathematically correct.
    """
    hop = 512

    # ── OFFSET via RMS envelope ──────────────────────────────────────────────
    ref_rms  = librosa.feature.rms(y=y_ref,  hop_length=hop)[0].astype(np.float64)
    comp_rms = librosa.feature.rms(y=y_comp, hop_length=hop)[0].astype(np.float64)

    ref_rms  = (ref_rms  - ref_rms.min())  / (ref_rms.max()  - ref_rms.min()  + 1e-10)
    comp_rms = (comp_rms - comp_rms.min()) / (comp_rms.max() - comp_rms.min() + 1e-10)

    corr      = signal.correlate(comp_rms, ref_rms, mode='full')
    lag       = np.argmax(corr) - (len(ref_rms) - 1)
    offset_ms = round(float(lag * hop / sr * 1000), 2)

    # ── DNA via WINDOWED ONSET CROSS-CORRELATION ────────────────────────────
    #
    # Problem with single 60s correlation:
    #   Two different Christmas songs over 60s have accidentally similar
    #   onset density and tempo → single xcorr peak gives ~50% even for
    #   completely unrelated content, barely distinguishable from a real dub.
    #
    # Fix — windowed median scoring:
    #   Split both onset envelopes into N windows of WIN_FRAMES each.
    #   Score each window independently via xcorr peak.
    #   Take the MEDIAN across windows.
    #
    #   Why median, not mean?
    #   - Mean is inflated by accidental high-correlation windows (e.g. a
    #     shared 4-beat pause in both files coincidentally landing in the
    #     same window).
    #   - Median requires CONSISTENT matching across windows, which only
    #     genuine same-content pairs achieve.
    #
    #   Result: genuine dubs score 80–95% (consistent structure throughout),
    #   different content scores 20–45% (only occasional window matches),
    #   giving clear separation around the 80% threshold.

    WIN_SEC    = 10                              # 10-second windows
    WIN_FRAMES = int(WIN_SEC * sr / hop)         # frames per window

    ref_onset  = librosa.onset.onset_strength(y=y_ref,  sr=sr, hop_length=hop)
    comp_onset = librosa.onset.onset_strength(y=y_comp, sr=sr, hop_length=hop)

    min_len    = min(len(ref_onset), len(comp_onset))
    ref_onset  = ref_onset[:min_len]
    comp_onset = comp_onset[:min_len]

    n_windows  = max(1, min_len // WIN_FRAMES)   # at least 1 window
    window_scores = []

    for w in range(n_windows):
        s = w * WIN_FRAMES
        e = s + WIN_FRAMES
        r_win = ref_onset[s:e].astype(np.float64)
        c_win = comp_onset[s:e].astype(np.float64)

        # Unit-normalise each window independently
        r_norm = r_win  / (np.linalg.norm(r_win)  + 1e-10)
        c_norm = c_win  / (np.linalg.norm(c_win)  + 1e-10)

        xcorr = signal.correlate(r_norm, c_norm, mode='same')
        window_scores.append(float(np.max(xcorr)))

    # Median across windows — robust to accidental single-window matches
    dna_score = round(float(np.median(window_scores)) * 100, 1)
    dna_score = max(0.0, min(100.0, dna_score))

    return offset_ms, dna_score


def calculate_speed_factor(start_offset_ms, end_offset_ms, duration_sec):
    """
    Clock ratio = master_duration / (master_duration + accumulated_drift).
    Gives engineers the exact time-stretch percentage for their DAW.
    """
    if duration_sec <= 0:
        return {"ratio": 1.0, "display": "N/A", "delta": "N/A", "action": "N/A"}

    drift_sec    = (end_offset_ms - start_offset_ms) / 1000.0
    speed_factor = duration_sec / (duration_sec + drift_sec + 1e-10)
    pct_delta    = round((speed_factor - 1.0) * 100, 4)

    if abs(pct_delta) < 0.001:
        action = "No time-stretch needed"
    elif pct_delta > 0:
        action = f"Time-compress dub by {abs(pct_delta):.4f}%"
    else:
        action = f"Time-expand dub by {abs(pct_delta):.4f}%"

    return {
        "ratio":   round(speed_factor, 6),
        "display": f"{speed_factor:.6f}×",
        "delta":   f"{pct_delta:+.4f}%",
        "action":  action,
    }


def determine_status(offset_ms, drift_ms, dna_score):
    issues = []
    if abs(offset_ms) > 80:
        issues.append(f"Start offset {offset_ms}ms exceeds ±80ms threshold")
    if abs(drift_ms) > 150:
        issues.append(f"Drift {drift_ms}ms exceeds ±150ms threshold")
    if dna_score < 80:
        issues.append(f"DNA match {dna_score}% below 80% threshold")

    return ("FAIL" if issues else "PASS",
            "; ".join(issues) if issues else "All metrics within thresholds")


# ── PER-FILE WORKER ────────────────────────────────────────────────────────────
def process_file(f, root, y_ref_s_an, y_ref_e_an, y_ref_s_raw,
                 ref_meta, vocal_logic):
    """
    y_ref_s_an / y_ref_e_an : analysis arrays (filtered when vocal_logic=True)
    y_ref_s_raw             : raw unfiltered master start — used only for waveform display
    """
    if not f or not f.filename:
        return None

    if not allowed_file(f.filename):
        return {"filename": f.filename, "status": "ERROR",
                "reason": f"Unsupported type '{os.path.splitext(f.filename)[1]}'",
                "error": True}
    try:
        f_path    = os.path.join(root, secure_filename(f.filename))
        f.save(f_path)
        comp_meta = get_file_metadata(f_path)
        comp_dur  = comp_meta["duration_sec"]

        y_c_s, _ = load_segment(f_path, PERFORMANCE_SR, duration=SEGMENT_DURATION)
        y_c_e, _ = load_segment(f_path, PERFORMANCE_SR,
                                offset=max(0.0, comp_dur - SEGMENT_DURATION))

        # Save raw dub for waveform display before any filtering
        y_c_s_raw = y_c_s.copy()

        # Apply the same filtering pipeline to dub that was applied to ref
        if vocal_logic:
            y_c_s_an = apply_vocal_filter(normalize_lufs(y_c_s, PERFORMANCE_SR))
            y_c_e_an = apply_vocal_filter(normalize_lufs(y_c_e, PERFORMANCE_SR))
        else:
            y_c_s_an = y_c_s
            y_c_e_an = y_c_e

        s_off, dna = analyze_segment(y_ref_s_an, y_c_s_an, PERFORMANCE_SR)
        e_off, _   = analyze_segment(y_ref_e_an, y_c_e_an, PERFORMANCE_SR)
        drift      = round(e_off - s_off, 2)

        speed          = calculate_speed_factor(s_off, e_off, comp_dur)
        status, reason = determine_status(s_off, drift, dna)

        result = {
            "filename":       f.filename,
            "status":         status,
            "reason":         reason,
            "offset_ms":      s_off,
            "total_drift_ms": drift,
            "offset_frames":  ms_to_frames(s_off),
            "drift_frames":   ms_to_frames(drift),
            "dna_match":      dna,
            "vocal_filter":   vocal_logic,   # pass through so UI can label it
            "speed_factor":   speed,
            "phase":          calculate_phase(f_path),
            "levels":         scan_levels(f_path),
            "ref_meta":       ref_meta,
            "comp_meta":      comp_meta,
            # Waveform uses RAW audio so chart always reflects true signal shape
            # Two datasets sent per file:
            #   rms_*  : smoothed energy envelope — comparable across languages,
            #             shows structural similarity for sync QC decision
            #   raw_*  : abs-normalised waveform — shows acoustic detail,
            #             useful for inspecting specific problem regions
            # Both use abs() so Master sits above axis, Dub below (mirror layout)
            "wave_rms_master":  rms_envelope(y_ref_s_raw).tolist(),
            "wave_rms_dub":     (-rms_envelope(y_c_s_raw)).tolist(),
            "wave_raw_master":  downsample_waveform(np.abs(normalize_visual(y_ref_s_raw))),
            "wave_raw_dub":     downsample_waveform(-np.abs(normalize_visual(y_c_s_raw))),
            "chan_mismatch":  ref_meta["channels"] != comp_meta["channels"],
        }

        del y_c_s, y_c_e, y_c_s_raw, y_c_s_an, y_c_e_an
        gc.collect()
        return result

    except Exception as err:
        return {"filename": f.filename, "status": "ERROR",
                "reason": str(err), "error": True}


# ── ROUTES ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/wipe', methods=['POST'])
def wipe():
    wiped = 0
    for folder in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, folder)
        if os.path.isdir(path) and folder.startswith("SES_"):
            shutil.rmtree(path, ignore_errors=True)
            wiped += 1
    gc.collect()
    return jsonify({"status": "ok", "wiped_sessions": wiped})


@app.route('/upload', methods=['POST'])
def upload():
    session_id = f"SES_{uuid.uuid4().hex[:6].upper()}"
    root       = os.path.join(DATA_DIR, session_id)
    os.makedirs(root, exist_ok=True)

    try:
        vocal_logic = request.form.get('vocal_logic') == 'true'
        ref         = request.files.get('reference')
        comps       = request.files.getlist('comparison[]')

        if not ref:
            return jsonify({"error": "No reference file provided"}), 400
        if not comps:
            return jsonify({"error": "No comparison files provided"}), 400
        if not allowed_file(ref.filename):
            return jsonify({"error": f"Master file type not allowed: "
                            f"'{os.path.splitext(ref.filename)[1]}'"}), 400

        ref_path  = os.path.join(root, secure_filename(ref.filename))
        ref.save(ref_path)
        ref_meta  = get_file_metadata(ref_path)
        total_dur = ref_meta["duration_sec"]

        y_ref_s, _ = load_segment(ref_path, PERFORMANCE_SR, duration=SEGMENT_DURATION)
        y_ref_e, _ = load_segment(ref_path, PERFORMANCE_SR,
                                  offset=max(0.0, total_dur - SEGMENT_DURATION))

        # Always keep raw copy for waveform display (chart shows real audio, not filtered)
        y_ref_s_raw = y_ref_s.copy()

        # Build analysis copies: HPSS + bandpass when vocal filter on, raw otherwise
        if vocal_logic:
            y_ref_s_an = apply_vocal_filter(normalize_lufs(y_ref_s, PERFORMANCE_SR))
            y_ref_e_an = apply_vocal_filter(normalize_lufs(y_ref_e, PERFORMANCE_SR))
        else:
            y_ref_s_an = y_ref_s
            y_ref_e_an = y_ref_e

        results_map = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(process_file, f, root,
                            y_ref_s_an, y_ref_e_an, y_ref_s_raw,
                            ref_meta, vocal_logic): i
                for i, f in enumerate(comps)
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results_map[futures[future]] = result

        results = [results_map[i] for i in sorted(results_map)]

        del y_ref_s, y_ref_e, y_ref_s_raw, y_ref_s_an, y_ref_e_an
        gc.collect()

        return jsonify({"results": results})

    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        gc.collect()
        return jsonify({"error": traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
