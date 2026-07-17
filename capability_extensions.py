"""
capability_extensions.py
=========================
Drop-in additions for Audio Alignment Engine v8 (audio_align.py).

New capabilities:
    - Language Identification (faster-whisper)
    - Profanity / Censorship scan (faster-whisper transcript + wordlist)
    - DME Structural Check (dialogue leakage into M&E stem)
    - Audio Description (AD) track detection (ffprobe metadata, no ASR needed)
    - Spatial Loudness — fixed target mismatch (was silently using ffmpeg's
      -24 LUFS default while the UI displayed a -27 LKFS Atmos target)

Explicitly OUT of scope (see review notes):
    - AV Sync / Lip-Sync — requires a video file; this is an audio-only tool.
    - True Atmos object count/position — requires Dolby Atmos Production Suite /
      Renderer metadata that ffprobe cannot extract. `atmos_bed_objects.object_count`
      stays `None` and is rendered as "Not measurable", never as a pass/fail.

Install:
    pip install faster-whisper
    # First run downloads the model (~75MB for 'base', ~150MB for 'small').
    # CPU works fine for short QC samples; GPU (CUDA) is much faster if available.
"""

from __future__ import annotations

import re
import logging
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from scipy import signal

logger = logging.getLogger("audio_align")

# ── CONFIG ────────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "base"          # tiny/base/small/medium — bigger = slower, more accurate
WHISPER_SAMPLE_DURATION_SEC = 60.0   # only transcribe first N seconds for speed
WHISPER_DEVICE = "cpu"               # set to "cuda" if a GPU is available
WHISPER_COMPUTE_TYPE = "int8"        # int8 = fastest on CPU with acceptable accuracy loss

ATMOS_SPATIAL_LUFS_TARGET = -27.0    # Dolby-recommended target for spatial/immersive mixes
STANDARD_LUFS_TARGET = -23.0         # matches your existing LUFS_TARGET (EBU R128)

DME_LEAKAGE_THRESHOLD_DB = -35.0     # dialogue energy above this in the M&E band = leakage
VOCAL_BAND = (300.0, 3400.0)         # same band your vocal filter already uses

# Minimal starter list — replace/extend with a proper moderation wordlist
# (this is intentionally NOT exhaustive; wire in a maintained list for production).
DEFAULT_PROFANITY_WORDLIST = {
    "fuck", "shit", "bitch", "asshole", "bastard", "dick", "cunt", "damn",
}


# ── LAZY WHISPER LOADER ──────────────────────────────────────────────────
class TranscriptionEngine:
    """
    Lazy-loaded faster-whisper wrapper. The model is loaded once per process
    (not per request) — instantiate a single module-level instance and reuse it.
    """

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE,
                 device: str = WHISPER_DEVICE,
                 compute_type: str = WHISPER_COMPUTE_TYPE):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = None  # loaded on first use
        # A single WhisperModel instance is shared across all request threads.
        # Serialize access to it — concurrent .transcribe() calls into the same
        # CTranslate2 model from multiple threads is not something to rely on
        # being safe, and it was untested under your ThreadPoolExecutor(MAX_WORKERS).
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper is not installed. Run: pip install faster-whisper"
            ) from e
        logger.info(f"Loading faster-whisper model '{self._model_size}' "
                    f"({self._device}/{self._compute_type})…")
        self._model = WhisperModel(
            self._model_size, device=self._device, compute_type=self._compute_type
        )

    def transcribe_sample(self, path: Union[str, Path],
                           duration_sec: float = WHISPER_SAMPLE_DURATION_SEC) -> Dict[str, Any]:
        """
        Transcribes the first `duration_sec` of audio. Returns detected language,
        language confidence, and the transcript text. Any failure returns a dict
        with 'error' set — callers must treat that as "not run", not "failed check".
        """
        try:
            self._ensure_loaded()
        except RuntimeError as e:
            return {"error": str(e), "available": False}

        try:
            with self._lock:
                segments, info = self._model.transcribe(
                    str(path),
                    language=None,           # auto-detect
                    vad_filter=True,         # skip silence, keeps this fast
                    condition_on_previous_text=False,
                )
                text_parts: List[str] = []
                total_sec = 0.0
                for seg in segments:
                    text_parts.append(seg.text)
                    total_sec = seg.end
                    if total_sec >= duration_sec:
                        break

            transcript = " ".join(t.strip() for t in text_parts).strip()
            return {
                "available": True,
                "language": info.language,
                "language_probability": round(float(info.language_probability), 3),
                "transcript_sample": transcript,
                "sampled_seconds": round(total_sec, 1),
            }
        except Exception as e:
            logger.warning(f"Transcription failed for {path}: {e}")
            return {"error": str(e), "available": False}


# Module-level singleton — reuse across requests so the model loads once.
transcription_engine = TranscriptionEngine()


def check_language_id(path: Union[str, Path], expected_language: Optional[str] = None) -> Dict[str, Any]:
    """
    Language Identification check.

    `expected_language` should be an ISO-639-1 code (e.g. "es", "fr", "hi") —
    pass it from your intake metadata (the language the dub was commissioned for).
    If omitted, this only reports what was detected, with no pass/fail.
    """
    result = transcription_engine.transcribe_sample(path)
    if not result.get("available"):
        return {"checked": False, "reason": result.get("error", "ASR unavailable")}

    detected = result["language"]
    out = {
        "checked": True,
        "detected": detected,
        "confidence": result["language_probability"],
        "expected": expected_language,
    }
    if expected_language:
        out["match"] = (detected == expected_language)
    return out


def check_profanity(path: Union[str, Path],
                     wordlist: Optional[set] = None) -> Dict[str, Any]:
    """
    Profanity / Censorship scan.

    Transcribes the sample and matches whole words against `wordlist`
    (case-insensitive). This is a coarse text-match, not acoustic detection —
    it will miss bleeped/censored audio (which is the point: bleeped audio
    should NOT match) and won't catch profanity outside the transcribed window.
    """
    wordlist = wordlist or DEFAULT_PROFANITY_WORDLIST
    result = transcription_engine.transcribe_sample(path)
    if not result.get("available"):
        return {"scanned": False, "reason": result.get("error", "ASR unavailable")}

    transcript = result.get("transcript_sample", "")
    words = re.findall(r"[a-zA-Z']+", transcript.lower())
    hits = [w for w in words if w in wordlist]

    return {
        "scanned": True,
        "flagged": len(hits),
        "flagged_words": hits[:10],   # cap for payload size
        "sampled_seconds": result.get("sampled_seconds"),
        "language": result.get("language"),
    }


# ── DME STRUCTURAL CHECK ─────────────────────────────────────────────────
def check_dme_structural(me_path: Union[str, Path],
                          y_dialogue_ref: Optional[np.ndarray],
                          sr: int,
                          hop_length: int = 512) -> Dict[str, Any]:
    """
    Checks whether dialogue has leaked into an M&E (Music & Effects) stem.

    Requires the M&E stem to be uploaded separately — there is no way to detect
    this from the dub file alone, since the dub *should* contain dialogue.

    Method: band-pass the M&E stem to the vocal range, and check whether its
    energy envelope correlates with the *reference dialogue* envelope
    (from the full mix). High correlation + energy above threshold in the
    vocal band = likely dialogue leakage. This is a heuristic, not a
    guarantee — always spot-check flagged files.
    """
    if y_dialogue_ref is None:
        return {"checked": False, "reason": "No dialogue reference track provided"}

    try:
        import librosa
        from scipy.signal import butter, lfilter

        y_me, me_sr = librosa.load(str(me_path), sr=sr, duration=60.0)

        nyq = 0.5 * sr
        low, high = VOCAL_BAND[0] / nyq, VOCAL_BAND[1] / nyq
        b, a = butter(2, [low, high], btype="band")
        y_me_band = lfilter(b, a, y_me)

        me_rms = librosa.feature.rms(y=y_me_band, hop_length=hop_length)[0]
        ref_rms = librosa.feature.rms(
            y=y_dialogue_ref[: len(y_me)], hop_length=hop_length
        )[0]

        min_len = min(len(me_rms), len(ref_rms))
        if min_len < 4:
            return {"checked": False, "reason": "Sample too short for analysis"}
        me_rms, ref_rms = me_rms[:min_len], ref_rms[:min_len]

        me_db = 20 * np.log10(np.mean(me_rms) + 1e-10)
        correlation = float(np.corrcoef(me_rms, ref_rms)[0, 1]) if np.std(me_rms) > 0 and np.std(ref_rms) > 0 else 0.0

        leakage_suspected = (me_db > DME_LEAKAGE_THRESHOLD_DB) and (correlation > 0.5)

        return {
            "checked": True,
            "clean": not leakage_suspected,
            "vocal_band_energy_db": round(float(me_db), 1),
            "correlation_with_dialogue": round(correlation, 3),
            "dialogue_leakage_db": round(float(me_db), 1) if leakage_suspected else None,
            "threshold_db": DME_LEAKAGE_THRESHOLD_DB,
            "note": "Heuristic band-energy/correlation check — verify flagged files by ear.",
        }
    except Exception as e:
        logger.warning(f"DME structural check failed: {e}")
        return {"checked": False, "reason": str(e)}


# ── AUDIO DESCRIPTION (AD) DETECTION ─────────────────────────────────────
def check_audio_description(ffprobe_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detects an Audio Description track from ffprobe stream metadata.
    Takes the already-computed `ff_meta` dict from FFmpegAnalyzer.get_full_metadata
    (avoids a second ffprobe call).

    AD tracks are conventionally tagged via one of:
        - stream disposition 'visual_impaired' = 1
        - language/title tags containing "AD", "audio description", "described video", "DVS"
    Coverage depends entirely on how the source was muxed/tagged — untagged AD
    tracks will not be found by this method.
    """
    ad_streams = []
    for stream in ffprobe_meta.get("streams", []):
        if stream.get("codec_type") != "audio":
            continue
        disposition = stream.get("disposition", {})
        tags = stream.get("tags", {})
        tag_text = " ".join(str(v) for v in tags.values()).lower()

        is_ad = (
            disposition.get("visual_impaired") == 1
            or any(kw in tag_text for kw in ["audio description", "described video", " ad ", "dvs", "narration"])
        )
        if is_ad:
            ad_streams.append({
                "index": stream.get("index"),
                "language": tags.get("language"),
                "title": tags.get("title"),
                "channels": stream.get("channels"),
            })

    return {
        "checked": True,
        "present": len(ad_streams) > 0,
        "streams": ad_streams,
        "note": "Detected via container tags/disposition only — untagged AD tracks won't be found. "
                "Ducking correctness (AD track lowering under narration) is not verified here.",
    }


# ── FIXED SPATIAL LOUDNESS ───────────────────────────────────────────────
def get_spatial_loudness(ffmpeg_path: str, path: Union[str, Path],
                          target_lufs: float = ATMOS_SPATIAL_LUFS_TARGET,
                          timeout: int = 60) -> Dict[str, Any]:
    """
    Fixed version of FFmpegAnalyzer.get_loudness_stats: the original call never
    passed a target (`I=`) to the loudnorm filter, so ffmpeg silently used its
    own default (-24 LUFS) while your frontend displayed and graded against
    -27 LKFS. Measurement and pass/fail threshold now agree.
    """
    import json as _json

    cmd = [
        ffmpeg_path, "-i", str(path),
        "-af", f"loudnorm=I={target_lufs}:TP=-2.0:LRA=7:print_format=json",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        json_match = re.search(r"\{[^}]+\}", result.stderr.replace("\n", ""))
        if not json_match:
            return {}
        stats = _json.loads(json_match.group())
        measured = float(stats.get("input_i", 0))
        return {
            "lufs": round(measured, 2),
            "true_peak": round(float(stats.get("input_tp", 0)), 2),
            "lra": round(float(stats.get("input_lra", 0)), 2),
            "target_lufs": target_lufs,
            "within_tolerance": abs(measured - target_lufs) <= 1.0,
            "source": "ffmpeg_loudnorm",
        }
    except (subprocess.TimeoutExpired, ValueError, Exception) as e:
        logger.warning(f"Spatial loudness measurement failed for {path}: {e}")
        return {}
