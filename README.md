# Audio Sync Engine Pro


Audio Alignment Engine
Version 9.0.0 — Production QC and alignment analysis for dubbed / localized audio against a reference master.
A Flask web service that verifies whether a dubbed audio file is frame-accurate and broadcast-compliant against its reference master, and can also run full standalone quality control on any audio file without a reference. Built for post-production localization workflows (dubbing QC, M&E stem verification, delivery spec checks).
Table of Contents
Overview
Features
Architecture
Requirements
Installation
Configuration
Running the Server
API Reference
QC Checks & Thresholds
Alignment Metrics Explained
Project Structure
Scope & Known Limitations
Observability
Session Storage & Cleanup
Overview
The engine answers two questions:
Sync mode — "Does this dub line up with the master?" Compares one reference/master file against one or more comparison (dub) files and reports start offset, drift, waveform "DNA" similarity, required time-stretch, loudness, true peak, and a full QC battery per file.
Standalone QC mode — "Is this file delivery-ready?" Runs every check that does not require a reference: loudness, true peak, dropouts, hum, rumble, phase, dual-mono, Atmos bed presence, Audio Description detection, spatial loudness, and optional DME / language ID / profanity scans.
Every analysis returns a per-file verdict: PASS, FAIL, WARN (file too short for reliable measurement), or ERROR (processing failure).
Features
Alignment (Sync Mode)
Start offset detection — RMS-envelope cross-correlation, reported in milliseconds and in frames at 23.976 / 25 / 29.97 fps, with a confidence interval.
Drift measurement — independent head- and tail-segment offsets; the difference is total drift over the program.
Speed factor — exact time-stretch ratio plus a human-readable action (e.g. "Time-compress dub by 0.4006%").
DNA match score — windowed onset-strength correlation (0–100%) measuring rhythmic/structural similarity between dub and master.
Chroma DNA score — chromagram cosine-similarity score, robust to pitch-shifted or re-voiced content.
Vocal Logic mode — optional dialogue-focused alignment: loudness-normalizes both files, isolates the 300 Hz–3.4 kHz vocal band (HPSS + Butterworth band-pass) before correlating. Useful when the dub's music/effects bed differs from the master.
Batch comparison — one master against many dubs in a single request, processed in parallel.
Standalone & Advanced QC
Loudness — integrated LUFS (pyloudnorm / ITU-R BS.1770) graded against EBU R128 −23 LUFS ± 1 LU.
True peak — 4× polyphase oversampled peak in dBTP, ceiling −2.0 dBTP.
Dropouts / silence gaps — FFmpeg silencedetect (−50 dB, ≥ 50 ms).
Hum / buzz — 50 Hz / 60 Hz mains interference detection with estimated SNR.
Low-frequency rumble — subsonic energy below 20 Hz.
Dual-mono detection — flags stereo files whose L/R channels are identical.
Inter-channel phase — L/R correlation with mono-collapse risk warning.
Click / spike detection — statistical peak-outlier scan (coarse level-spike check; see Limitations).
Dolby / Atmos metadata — detects Atmos beds and dialnorm values from the bitstream.
Spatial loudness — immersive-mix loudness measured against the Dolby-recommended −27 LUFS target (TP −2.0, LRA 7).
Audio Description (AD) detection — finds AD tracks via container disposition/tags.
DME structural check — detects dialogue leakage into an M&E stem (requires the M&E stem as a separate upload).
Language ID — verifies the spoken language matches the commissioned language (ISO 639-1). Opt-in.
Profanity / censorship scan — transcript-based wordlist flagging. Opt-in.
Architecture
Hybrid pipeline — each tool does what it is best at:
Table
Layer	Technology	Responsibility
Format & metadata	FFmpeg / FFprobe	Container metadata, stream info, silence detection, level stats, spatial loudness (loudnorm)
Temporal alignment	librosa / scipy	RMS envelopes, onset strength, cross-correlation, chromagrams, oversampled true peak
Loudness	pyloudnorm	BS.1770 integrated loudness & normalization
ASR (optional)	faster-whisper	Language identification and profanity scanning — off by default (slowest step)
Web layer	Flask + Flask-Limiter	REST API, rate limiting, session management
plain
audio_align.py               ← main application (routes, ingestion, alignment, QC orchestration)
capability_extensions.py     ← language ID, profanity, DME, AD detection, spatial loudness
templates/index.html         ← web UI (not included in this snippet)
Processing model:
Single-pass ingestion — files are streamed in 65,536-sample blocks; peak, true peak, and phase statistics are accumulated without loading the whole file into memory.
Head/tail analysis — alignment runs on the first and last 60 s of each file, resampled to 22,050 Hz.
Parallel workers — comparison files are processed in a ThreadPoolExecutor (default 4 workers) with one automatic retry per file.
Shared QC pipeline — sync and standalone flows call the same run_advanced_qc() function, so checks can never silently diverge between modes.
Requirements
System:
Python 3.10+
ffmpeg and ffprobe on PATH (or configured via environment variables)
Python packages:
plain
numpy
scipy
librosa
soundfile
pyloudnorm
flask
flask-limiter
werkzeug
Optional (ASR features):
plain
faster-whisper    # language ID + profanity scan
Installation
bash
# 1. System dependency (Debian/Ubuntu example)
sudo apt-get install ffmpeg

# 2. Python dependencies
pip install numpy scipy librosa soundfile pyloudnorm flask flask-limiter werkzeug

# 3. Optional: ASR support (language ID / profanity scan)
pip install faster-whisper
# First run downloads the model (~75 MB for 'base', ~150 MB for 'small').
# CPU works for short QC samples; GPU (CUDA) is much faster if available.

# 4. Project files
#    Place audio_align.py and capability_extensions.py in the same directory,
#    with the Flask UI at templates/index.html.
Configuration
All settings are environment variables; every one has a working default.
Table
Variable	Default	Description
FFMPEG_PATH	ffmpeg	Path to the ffmpeg binary
FFPROBE_PATH	ffprobe	Path to the ffprobe binary
DATA_DIR	./data	Session storage directory
MAX_CONTENT_LENGTH	1073741824 (1 GB)	Max total request size
MAX_FILE_SIZE	209715200 (200 MB)	Max size per uploaded file
MAX_WORKERS	4	ThreadPool workers for parallel file processing
RATE_LIMIT	10 per minute	Global rate limit
UPLOAD_RATE_LIMIT	10 per minute	Rate limit for POST /upload
QC_RATE_LIMIT	10 per minute	Rate limit for POST /qc
LIMITER_STORAGE	memory://	Flask-Limiter storage URI (use Redis in multi-instance deployments)
FLASK_HOST	127.0.0.1	Bind host (dev server)
FLASK_PORT	5001	Bind port (dev server)
FLASK_DEBUG	false	Flask debug mode
ASR tuning is done via constants at the top of capability_extensions.py:
Table
Constant	Default	Description
WHISPER_MODEL_SIZE	base	tiny / base / small / medium — bigger = slower, more accurate
WHISPER_DEVICE	cpu	Set to cuda if a GPU is available
WHISPER_COMPUTE_TYPE	int8	Fastest on CPU with acceptable accuracy loss
WHISPER_SAMPLE_DURATION_SEC	60.0	Only the first N seconds are transcribed, for speed
Running the Server
Development:
bash
python audio_align.py          # serves on http://127.0.0.1:5001
Production (Gunicorn):
bash
gunicorn --bind 0.0.0.0:5001 --workers 2 --timeout 300 audio_align:app
Keep Gunicorn workers low (1–2): each request already fans out across MAX_WORKERS threads, and audio analysis is CPU-heavy. If you scale beyond one process, point LIMITER_STORAGE at Redis so rate limits are shared.
Open http://127.0.0.1:5001/ in a browser for the web UI, or call the API directly.
API Reference
All analysis endpoints accept multipart/form-data and return JSON. Non-finite floats are sanitized to null.
GET /
Serves the web UI (templates/index.html).
GET /health
Liveness probe.
JSON
{ "status": "healthy", "timestamp": 1752768000.0, "version": "9.0.0" }
GET /metrics
Prometheus-compatible plaintext metrics (request/file counters, total and average processing time).
POST /wipe
Deletes all session folders under DATA_DIR (the UI's Clear Cache button).
JSON
{ "status": "ok", "cleared": 3 }
POST /upload — Sync / Alignment Mode
Compares one reference master against one or more dub files.
Table
Field	Type	Required	Description
reference	file	✅	Master / reference audio
comparison[]	file(s)	✅	One or more dub files to align against the master
me_stem	file	–	M&E stem; enables the DME dialogue-leakage check
vocal_logic	"true"	–	Dialogue-band-focused alignment (vocal filter + LUFS normalization)
run_asr	"true"	–	Opt in to language ID + profanity scan (slow)
expected_language	string	–	ISO 639-1 code the dub was commissioned in (e.g. es, fr, hi)
Supported formats: .wav .mp3 .flac .aac .ogg .m4a .aiff .aif .opus .mxf .adm .ec3 .ac3
bash
curl -X POST http://127.0.0.1:5001/upload \
  -F "reference=@master_EN.wav" \
  -F "comparison[]=@dub_ES.wav" \
  -F "comparison[]=@dub_FR.wav" \
  -F "vocal_logic=true"
Response (abridged — one object per comparison file):
JSON
{
  "mode": "sync",
  "results": [
    {
      "filename": "dub_ES.wav",
      "status": "FAIL",
      "reason": "Start offset 132.5ms exceeds ±80.0ms threshold",
      "offset_ms": 132.5,
      "offset_confidence": [109.27, 155.73],
      "total_drift_ms": 240.1,
      "offset_frames": { "23.976": 3.18, "25": 3.31, "29.97": 3.97 },
      "drift_frames":  { "23.976": 5.76, "25": 6.0,  "29.97": 7.2  },
      "dna_match": 91.3,
      "dna_confidence": [84.2, 97.6],
      "chroma_dna": 88.7,
      "vocal_filter": true,
      "speed_factor": {
        "ratio": 0.999841,
        "display": "0.999841×",
        "delta": "-0.0159%",
        "action": "Time-expand dub by 0.0159%"
      },
      "phase": "0.87 (Healthy)",
      "levels": {
        "lufs": "-23.41 LUFS", "lufs_val": -23.41,
        "peak": "-4.12 dBFS",  "peak_val": -4.12,
        "true_peak": "-3.05 dBTP", "true_peak_val": -3.05
      },
      "ref_meta":  { "sr": "48000 Hz", "duration_sec": 183.4, "channels": 2, "...": "..." },
      "comp_meta": { "sr": "48000 Hz", "duration_sec": 183.6, "channels": 2, "...": "..." },
      "chan_mismatch": false,
      "wave_rms_master": [0.0, 0.12, "..."],
      "wave_rms_dub":    [0.0, -0.11, "..."],
      "wave_raw_master": [0.01, 0.44, "..."],
      "wave_raw_dub":    [-0.01, -0.4, "..."],
      "spectrum_master": ["..."],
      "spectrum_dub":    ["..."],
      "qc_checks": { "...": "see QC Checks section" }
    }
  ]
}
Waveform and spectrum arrays are downsampled for direct front-end plotting (RMS envelopes are mirrored: master positive, dub negative).
POST /qc — Standalone QC Mode
Full QC battery on independent files, no reference needed.
Table
Field	Type	Required	Description
files[]	file(s)	✅	Audio files to QC
me_stem	file	–	M&E stem; enables the DME dialogue-leakage check
run_asr	"true"	–	Opt in to language ID + profanity scan
expected_language	string	–	ISO 639-1 code for pass/fail language matching
bash
curl -X POST http://127.0.0.1:5001/qc \
  -F "files[]=@episode01_mix.wav" \
  -F "me_stem=@episode01_ME.wav" \
  -F "run_asr=true" -F "expected_language=es"
Returns {"mode": "standalone", "results": [...]} with status, levels, meta, waveform data, spectrum, and qc_checks per file. Offset/drift/DNA metrics are inherently comparative and are deliberately absent in this mode — not stubbed, not faked.
Error Responses
Table
Code	Meaning
400	Missing/unsupported files, file empty, or per-file size exceeds MAX_FILE_SIZE
429	Rate limit exceeded
500	Internal processing error (session is cleaned up automatically)
Individual file failures never fail the whole batch — the file's result comes back with "status": "ERROR" and a reason.
QC Checks & Thresholds
Graded thresholds (constants in audio_align.py):
Table
Metric	Threshold	Verdict impact
Start offset	± 80 ms	FAIL if exceeded (sync mode)
Total drift	± 150 ms	FAIL if exceeded (sync mode)
DNA match	≥ 80 %	FAIL if below (sync mode)
Chroma DNA	≥ 80 %	FAIL if below (sync mode)
Integrated loudness	−23 LUFS ± 1 LU (EBU R128)	FAIL if outside
True peak	≤ −2.0 dBTP	FAIL if exceeded
Dropouts	< 3 silence gaps (−50 dB / ≥ 50 ms)	FAIL if ≥ 3 (standalone)
Hum / buzz	SNR > 40 dB when detected	FAIL if ≤ 40 dB (standalone)
Dual-mono	L/R correlation > 0.999 & max diff < 1e-5	FAIL if flagged (standalone)
Level spikes	< 5 outliers	FAIL if ≥ 5 (standalone)
Spatial loudness	−27 LUFS ± 1 LU (Dolby immersive target)	reported with within_tolerance
Inter-channel phase	correlation > 0.4 = healthy	mono-collapse warning below
DME leakage	vocal-band energy > −35 dB and dialogue correlation > 0.5	clean: false when suspected
Minimum duration	3 s	below → WARN, metrics indicative only
Every result in qc_checks carries the parameters used (thresholds, sample windows) so the UI can display exactly how a verdict was reached.
Alignment Metrics Explained
Offset (ms) — lag of the maximum of the cross-correlation between the dub's and master's normalized RMS envelopes. Confidence interval = ± one hop (512 samples @ 22,050 Hz ≈ 23 ms).
Drift (ms) — tail-segment offset minus head-segment offset. Non-zero drift means the dub runs at a different speed than the master, not just late.
Speed factor — duration / (duration + drift). Multiply the dub's length by this ratio (or apply the printed % time-stretch) to make it match the master.
DNA match (0–100) — median peak correlation of onset-strength envelopes across 10-second windows; IQR-based confidence interval included. Answers "is this the same performance?" rather than "is it shifted?"
Chroma DNA (0–100) — cosine similarity of windowed chromagrams; stays meaningful when the dub is re-voiced or pitch-shifted.
Vocal Logic — before correlating, both signals are loudness-normalized to −23 LUFS, harmonic/percussive separated, and band-passed to 300 Hz–3.4 kHz (dialogue band). Enable it when music/effects differences would otherwise dominate the correlation.
Project Structure
plain
.
├── audio_align.py             # Flask app: routes, ingestion, alignment engine, QC orchestration
├── capability_extensions.py   # Language ID, profanity scan, DME check, AD detection, spatial loudness
├── templates/
│   └── index.html             # Web UI
├── data/                      # Session storage (auto-created; SES_* folders, auto-purged)
└── README.md
Scope & Known Limitations
These are deliberate design decisions — the tool reports "not measurable" rather than fabricating a result:
AV sync / lip-sync is out of scope. It requires the video stream; this is an audio-only tool. It was removed from scope in v9 rather than faked.
Atmos object count/position is not measurable via ffprobe (requires Dolby Atmos Renderer metadata). atmos_bed_objects.object_count is null and rendered as "Not measurable" — never as zero objects and never as a pass/fail.
Click/pop detection is a coarse level-spike outlier check (FFmpeg astats peak reporting), not sample-accurate discontinuity detection. It is labeled as such in the UI.
DME check is a heuristic (vocal-band energy + envelope correlation against the dialogue track). Always spot-check flagged files by ear.
AD detection is metadata-only. It finds tracks tagged via disposition or title/language tags; untagged AD tracks are not found, and ducking correctness is not verified.
Profanity scan is transcript-based, whole-word, English-starter-wordlist only, and covers just the first 60 s. It intentionally will not match bleeped audio. Wire in a maintained moderation wordlist for production.
ASR features are opt-in (run_asr=true) because transcription is by far the slowest step. The whisper model loads lazily once per process and access is serialized across request threads.
Files shorter than 3 s return WARN — metrics are shown but marked indicative only.
Hum, rumble, and dual-mono checks sample the first 5–10 s of each file, not the full duration.
Observability
Structured JSON logging — every log line is a JSON object (timestamp, level, logger, message, module, funcName); each request carries a REQ_XXXXXXXX ID that appears in all its log lines.
Prometheus metrics at GET /metrics:
plain
alignment_requests_total
alignment_requests_failed
alignment_processing_seconds_total
alignment_files_processed_total
alignment_files_failed_total
alignment_avg_processing_seconds
Session Storage & Cleanup
Each upload creates a SES_XXXXXX folder under DATA_DIR holding the uploaded files.
A background sweeper runs every 10 minutes and deletes session folders older than 1 hour.
POST /wipe clears all sessions immediately.
If a request fails with a 500, its session folder is removed at once.
v9 changelog highlights
Fixed missing /wipe route (the UI's Clear Cache button was 404ing).
Fixed spatial loudness never passing a target to ffmpeg's loudnorm — it silently measured against ffmpeg's −24 LUFS default while the UI graded against −27 LKFS.
Fixed atmos_bed_objects.object_count being hardcoded to 0 (looked like "zero objects found" instead of "not measurable").
Added Audio Description track detection (ffprobe metadata).
Added DME structural check (requires optional M&E stem upload).
Added language ID + profanity scan (opt-in, faster-whisper).
Removed AV Sync / Lip-Sync from scope (requires video; audio-only tool).
