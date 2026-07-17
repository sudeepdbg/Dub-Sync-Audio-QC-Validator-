#Dub Sync & Audio QC Validator (v9)

Broadcast/OTT localization QC tool. Two modes:

1. **Dub Sync + QC** — compares a dub against a master reference: start offset,
   drift, clock-speed factor, onset DNA match, chroma DNA match, plus the full
   advanced QC suite below.
2. **Standalone Audio QC** — no reference needed. Runs every check that
   doesn't require a comparison file, against one or more independent audio
   files.

## Contents

```
.
├── audio_align.py              # Flask app — routes, alignment analysis, orchestration
├── capability_extensions.py    # ASR (language ID / profanity), DME check, AD detection, spatial loudness
├── requirements.txt
└── templates/
    └── index.html              # Frontend (vanilla JS + ECharts)
```

`audio_align.py` imports directly from `capability_extensions.py` — the two
must live in the same folder. `index.html` must be under `templates/` since
Flask's `render_template()` looks there by default.

## Setup

```bash
pip install -r requirements.txt --break-system-packages   # or use a venv
```

You also need `ffmpeg` and `ffprobe` on `PATH` (or set `FFMPEG_PATH` /
`FFPROBE_PATH`). Most of the advanced QC checks shell out to them.

```bash
# Debian/Ubuntu
apt-get install ffmpeg

# macOS
brew install ffmpeg
```

Run it:

```bash
python3 audio_align.py
# or, for production:
gunicorn -w 4 -b 0.0.0.0:5001 audio_align:app
```

By default it binds to `127.0.0.1:5001` — set `FLASK_HOST`/`FLASK_PORT` to
change that, and never set `FLASK_DEBUG=true` in anything reachable from
outside your machine (Flask's debugger allows arbitrary code execution).

### Optional: enable ASR (Language ID + Profanity scan)

```bash
pip install faster-whisper --break-system-packages
```

First run downloads the model (~75MB for the default `base` size). If you're
deploying in a container, bake the model into the image or mount a persistent
cache volume so it doesn't re-download on every cold start. ASR is opt-in per
request (the "Run ASR" checkbox) — it's the slowest step in the pipeline by a
wide margin, so it's off by default.

## requirements.txt

```
Flask>=3.0
flask-limiter>=3.5
numpy>=1.24
librosa>=0.10
soundfile>=0.12
pyloudnorm>=0.1.1
scipy>=1.11
Werkzeug>=3.0
faster-whisper>=1.0.0   # optional — only needed for Run ASR
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `FFMPEG_PATH` | `ffmpeg` | Path to the ffmpeg binary |
| `FFPROBE_PATH` | `ffprobe` | Path to the ffprobe binary |
| `MAX_CONTENT_LENGTH` | `1073741824` (1GB) | Max total request body size |
| `MAX_FILE_SIZE` | `209715200` (200MB) | Max size per uploaded file |
| `MAX_WORKERS` | `4` | ThreadPoolExecutor size for parallel file processing |
| `RATE_LIMIT` | `10 per minute` | Global default rate limit |
| `UPLOAD_RATE_LIMIT` | `10 per minute` | Rate limit on `/upload` (sync mode) |
| `QC_RATE_LIMIT` | `10 per minute` | Rate limit on `/qc` (standalone mode) |
| `DATA_DIR` | `./data` | Session storage; auto-cleaned after 1 hour |
| `FLASK_HOST` | `127.0.0.1` | Bind address |
| `FLASK_PORT` | `5001` | Bind port |
| `FLASK_DEBUG` | `false` | Never set true outside local dev |
| `WHISPER_MODEL_SIZE`* | `base` | faster-whisper model: tiny/base/small/medium |
| `WHISPER_DEVICE`* | `cpu` | `cpu` or `cuda` |

\* Set these by editing the constants at the top of `capability_extensions.py`
— they aren't wired to env vars yet, so change the source directly if needed.

## API

### `POST /upload` — Dub Sync + QC

| Field | Type | Required | Notes |
|---|---|---|---|
| `reference` | file | yes | Master audio file |
| `comparison[]` | file(s) | yes | One or more dub files to check against the master |
| `vocal_logic` | `"true"`/`"false"` | no | HPSS + 300–3400Hz bandpass before DNA scoring |
| `me_stem` | file | no | M&E-only stem — enables the DME structural check |
| `run_asr` | `"true"`/`"false"` | no | Enables Language ID + Profanity (slow) |
| `expected_language` | string | no | ISO-639-1 code, e.g. `es` — graded against detected language |

Returns `{"mode": "sync", "results": [...]}`. Each result includes offset,
drift, DNA match, chroma DNA, speed factor, levels, waveform data, and
`qc_checks`.

### `POST /qc` — Standalone Audio QC

| Field | Type | Required | Notes |
|---|---|---|---|
| `files[]` | file(s) | yes | One or more independent audio files |
| `me_stem` | file | no | Same as above |
| `run_asr` | `"true"`/`"false"` | no | Same as above |
| `expected_language` | string | no | Same as above |

Returns `{"mode": "standalone", "results": [...]}`. Each result includes
levels, waveform, spectrum, and `qc_checks` — no offset/drift/DNA fields,
since there's nothing to compare against.

### Other routes

- `GET /health` — liveness check
- `GET /metrics` — Prometheus-format metrics
- `POST /wipe` — clears all session data under `DATA_DIR`

## QC capability matrix

| Check | Requires | Notes |
|---|---|---|
| Start offset / drift / speed factor | Reference file | Sync mode only |
| Onset DNA / Chroma DNA match | Reference file | Sync mode only |
| Integrated loudness (LUFS) / true peak / sample peak | — | Both modes |
| Inter-channel phase | Stereo+ | Both modes |
| Dropouts / silence gaps | — | ffmpeg `silencedetect` |
| Level spike detection | — | Coarse peak-outlier heuristic, **not** sample-accurate click detection |
| Hum & buzz (50/60Hz) | — | First 10s sample only |
| Low-frequency rumble | — | First 10s sample only |
| Mono-in-stereo (dual-mono) | Stereo | First 5s sample only |
| Spatial loudness | — | Targets -27 LUFS (Dolby immersive-mix guidance) |
| Atmos bed channel presence | Atmos-tagged stream | **Object count/position are not measurable** — ffprobe can't extract Dolby Renderer metadata. Always reported as `null`/"not measurable," never faked |
| Audio Description (AD) detection | — | Container tags/disposition only — untagged AD tracks won't be found; ducking correctness isn't verified |
| DME structural check | M&E stem upload | Heuristic band-energy/correlation check — **verify flagged files by ear** |
| Language ID | `run_asr=true` | faster-whisper — samples first 60s |
| Profanity scan | `run_asr=true` | Text-match on ASR transcript; won't catch mis-transcriptions and by design won't flag bleeped/censored audio |
| AV Sync / Lip-Sync | — | **Not implemented.** Requires a video file; this is an audio-only tool. Not present in the UI. |

## Known limitations

- **First-10s / first-5s sampling** on hum, rumble, and dual-mono checks
  means issues that start mid-file can be missed. Fine for most localization
  QC workflows, but worth knowing if you're chasing an intermittent fault.
- **Level Spike Detection is not true click/pop detection.** It flags
  statistical peak-level outliers via ffmpeg's `astats`, which can't see
  single-sample discontinuities. Treat it as a coarse triage signal.
- **ASR calls are serialized** behind a lock on the shared Whisper model
  instance — with several files and "Run ASR" on, expect them to process one
  at a time even though everything else runs in parallel across
  `MAX_WORKERS` threads.
- **Profanity wordlist is a small starter set** in
  `capability_extensions.py` (`DEFAULT_PROFANITY_WORDLIST`) — swap in a
  maintained moderation list before relying on this for compliance sign-off.
- **DME and Atmos-object checks are explicitly partial.** Both are labeled
  "not run" or "not measurable" rather than a false pass/fail when the
  required input (M&E stem, Dolby Renderer metadata) isn't available.

## Frontend notes

- `qc-list` badge states are `pass` / `warn` / `fail` / `info` / `skip`.
  `skip` means "not run" — it's rendered gray, never as a red failure, so a
  check that wasn't performed can never look like a check that failed.
- Sync-mode cards have 3 tabs (Sync Analysis / Advanced QC / Spectrum);
  standalone cards have 2 (Levels & QC / Spectrum). `switchTab()` is generic
  and works off DOM id matching, so it doesn't need to know which tab set
  it's dealing with.
