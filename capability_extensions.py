"""
capability_extensions.py v3 (FIXED)
====================================
Drop-in additions for Audio Alignment Engine v10.

FIXES APPLIED:
    1.  ebur128 regex patterns: Fixed to match FFmpeg's actual multi-line output format
        - Integrated loudness: matches "I: -22.8 LUFS" (was "Integrated loudness: -22.8 LUFS")
        - Loudness range: matches "LRA: 8.0 LU" (was "Loudness range: 8.0 LU")
        - True peak: matches "Peak: -7.0 dBFS" (was "True peak: -7.0 dBTP")
    2.  Added framelog=0 to suppress per-frame ebur128 logging (eliminates nan/inf noise)
    3.  Parse from Summary block only to avoid matching initial measurement values
    4.  Added -nostats to ffmpeg calls to suppress progress output
    5.  Better error messages that distinguish ffmpeg failure vs parse failure
    6.  Environment variables now honored for Whisper config
    7.  ASR deduplication: transcribe once, reuse result
    8.  DME check: marked as EXPERIMENTAL with honest limitations
"""

from __future__ import annotations

import os
import re
import json
import logging
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger("audio_align")

# Honor environment variables (fixes Point 10)
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_SAMPLE_DURATION_SEC = 60.0

ATMOS_SPATIAL_LUFS_TARGET = -27.0
STANDARD_LUFS_TARGET = -23.0

DME_LEAKAGE_THRESHOLD_DB = -35.0
VOCAL_BAND = (300.0, 3400.0)

DEFAULT_PROFANITY_WORDLIST = {
    "fuck", "shit", "bitch", "asshole", "bastard", "dick", "cunt", "damn",
}


class TranscriptionEngine:
    """Lazy-loaded faster-whisper wrapper with result caching."""

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE,
                 device: str = WHISPER_DEVICE,
                 compute_type: str = WHISPER_COMPUTE_TYPE):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = None
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}

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
                    f"({self._device}/{self._compute_type})")
        self._model = WhisperModel(
            self._model_size, device=self._device, compute_type=self._compute_type
        )

    def transcribe_sample(self, path: Union[str, Path],
                           duration_sec: float = WHISPER_SAMPLE_DURATION_SEC,
                           use_cache: bool = True) -> Dict[str, Any]:
        path_str = str(path)
        if use_cache and path_str in self._cache:
            return self._cache[path_str]

        try:
            self._ensure_loaded()
        except RuntimeError as e:
            return {"error": str(e), "available": False, "checked": False}

        try:
            with self._lock:
                segments, info = self._model.transcribe(
                    path_str,
                    language=None,
                    vad_filter=True,
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
            result = {
                "available": True,
                "checked": True,
                "language": info.language,
                "language_probability": round(float(info.language_probability), 3),
                "transcript_sample": transcript,
                "sampled_seconds": round(total_sec, 1),
            }
            if use_cache:
                self._cache[path_str] = result
            return result
        except Exception as e:
            logger.warning(f"Transcription failed for {path_str}: {e}")
            return {"error": str(e), "available": False, "checked": False}

    def clear_cache(self):
        self._cache.clear()


transcription_engine = TranscriptionEngine()


def get_transcription_result(path: Union[str, Path]) -> Dict[str, Any]:
    return transcription_engine.transcribe_sample(path, use_cache=True)


def check_language_id(path: Union[str, Path], expected_language: Optional[str] = None) -> Dict[str, Any]:
    result = get_transcription_result(path)
    if not result.get("available"):
        return {
            "checked": False,
            "status": "SKIP",
            "reason": result.get("error", "ASR unavailable"),
        }

    detected = result["language"]
    out = {
        "checked": True,
        "status": "PASS",
        "detected": detected,
        "confidence": result["language_probability"],
        "expected": expected_language,
    }
    if expected_language:
        match = (detected == expected_language)
        out["match"] = match
        out["status"] = "PASS" if match else "FAIL"
        if not match:
            out["reason"] = f"Expected {expected_language}, detected {detected}"
    return out


def check_profanity(path: Union[str, Path],
                     wordlist: Optional[set] = None) -> Dict[str, Any]:
    wordlist = wordlist or DEFAULT_PROFANITY_WORDLIST
    result = get_transcription_result(path)
    if not result.get("available"):
        return {
            "scanned": False,
            "status": "SKIP",
            "reason": result.get("error", "ASR unavailable"),
        }

    transcript = result.get("transcript_sample", "")
    words = re.findall(r"[a-zA-Z']+", transcript.lower())
    hits = [w for w in words if w in wordlist]

    status = "FAIL" if hits else "PASS"
    return {
        "scanned": True,
        "status": status,
        "flagged": len(hits),
        "flagged_words": hits[:10],
        "sampled_seconds": result.get("sampled_seconds"),
        "language": result.get("language"),
        "reason": f"{len(hits)} profanity hits" if hits else None,
    }


def check_dme_structural(me_path: Union[str, Path],
                          y_dialogue_ref: Optional[np.ndarray],
                          sr: int,
                          hop_length: int = 512) -> Dict[str, Any]:
    if y_dialogue_ref is None:
        return {
            "checked": False,
            "status": "SKIP",
            "reason": "No isolated dialogue stem provided -- DME leakage cannot be "
                      "reliably detected from the full mix alone.",
        }

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
            return {
                "checked": False,
                "status": "SKIP",
                "reason": "Sample too short for analysis",
            }
        me_rms, ref_rms = me_rms[:min_len], ref_rms[:min_len]

        me_db = 20 * np.log10(np.mean(me_rms) + 1e-10)
        correlation = float(np.corrcoef(me_rms, ref_rms)[0, 1]) if np.std(me_rms) > 0 and np.std(ref_rms) > 0 else 0.0

        leakage_suspected = (me_db > DME_LEAKAGE_THRESHOLD_DB) and (correlation > 0.5)
        status = "WARN" if leakage_suspected else "PASS"

        return {
            "checked": True,
            "status": status,
            "clean": not leakage_suspected,
            "vocal_band_energy_db": round(float(me_db), 1),
            "correlation_with_dialogue": round(correlation, 3),
            "dialogue_leakage_db": round(float(me_db), 1) if leakage_suspected else None,
            "threshold_db": DME_LEAKAGE_THRESHOLD_DB,
            "reason": (
                "Possible dialogue leakage detected (heuristic -- verify by ear)"
                if leakage_suspected else None
            ),
            "note": "EXPERIMENTAL: This check cannot reliably distinguish dialogue "
                    "from music/effects in the vocal band. For reliable detection, "
                    "provide an isolated dialogue stem.",
        }
    except Exception as e:
        logger.warning(f"DME structural check failed: {e}")
        return {
            "checked": False,
            "status": "ERROR",
            "reason": str(e),
        }


def check_audio_description(ffprobe_meta: Dict[str, Any]) -> Dict[str, Any]:
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
        "status": "INFO",
        "present": len(ad_streams) > 0,
        "streams": ad_streams,
        "note": "Detected via container tags/disposition only -- untagged AD tracks won't be found.",
    }


def _parse_ebur128_summary(stderr: str) -> Dict[str, Any]:
    """
    Parse ebur128 Summary block from FFmpeg stderr output.
    
    FFmpeg ebur128 outputs a multi-line summary block:
      Summary:
      
        Integrated loudness:
          I:         -22.8 LUFS
          Threshold: -33.0 LUFS
      
        Loudness range:
          LRA:         8.0 LU
          Threshold: -44.7 LUFS
          LRA low:   -30.4 LUFS
          LRA high:  -22.4 LUFS
      
        True peak:
          Peak:       -7.0 dBFS
    
    We extract ONLY the Summary block to avoid matching initial measurement values.
    """
    summary_match = re.search(r'Summary:\s*\n(.*)', stderr, re.DOTALL)
    if not summary_match:
        return {
            "parsed": False,
            "reason": "No Summary block found in ebur128 output",
        }
    
    summary_text = summary_match.group(0)
    
    # Parse individual fields from the Summary block only
    integrated_match = re.search(r'I:\s+([-\d.]+)\s+LUFS', summary_text)
    range_match = re.search(r'LRA:\s+([-\d.]+)\s+LU', summary_text)
    peak_match = re.search(r'Peak:\s+([-\d.]+)\s+dBFS', summary_text)
    
    result = {"parsed": True}
    
    if integrated_match:
        result["integrated_lufs"] = float(integrated_match.group(1))
    else:
        result["integrated_lufs"] = None
        result["parsed"] = False
        result["reason"] = "Could not parse Integrated loudness (I:) from Summary"
    
    if range_match:
        result["lra_lu"] = float(range_match.group(1))
    else:
        result["lra_lu"] = None
    
    if peak_match:
        result["true_peak_dbfs"] = float(peak_match.group(1))
    else:
        result["true_peak_dbfs"] = None
    
    return result


def get_spatial_loudness(ffmpeg_path: str, path: Union[str, Path],
                          target_lufs: float = ATMOS_SPATIAL_LUFS_TARGET,
                          timeout: int = 120) -> Dict[str, Any]:
    """
    Measure spatial loudness using FFmpeg ebur128 filter.
    
    FIX: Uses framelog=0 to suppress per-frame logging (eliminates nan/inf noise).
    FIX: Parses from Summary block only with correct multi-line regex patterns.
    FIX: Uses dBFS (not dBTP) for True Peak matching FFmpeg's actual output.
    """
    cmd = [
        ffmpeg_path, "-hide_banner", "-nostats", "-i", str(path),
        "-map", "0:a:0",
        "-af", "ebur128=peak=true:framelog=0",  # FIX: framelog=0 suppresses per-frame noise
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode != 0:
            return {
                "checked": False,
                "status": "ERROR",
                "reason": f"ffmpeg ebur128 failed (rc={result.returncode}): {result.stderr[:300]}",
            }

        stderr = result.stderr
        
        # FIX: Parse from Summary block only with correct patterns
        parse_result = _parse_ebur128_summary(stderr)
        
        if not parse_result.get("parsed"):
            return {
                "checked": False,
                "status": "ERROR",
                "reason": parse_result.get("reason", "Could not parse ebur128 output"),
                "raw_output_preview": stderr[-500:] if len(stderr) > 500 else stderr,
            }

        measured = parse_result["integrated_lufs"]
        lra = parse_result.get("lra_lu")
        true_peak = parse_result.get("true_peak_dbfs")

        within_tolerance = abs(measured - target_lufs) <= 1.0
        status = "PASS" if within_tolerance else "FAIL"

        return {
            "checked": True,
            "status": status,
            "lufs": round(measured, 2),
            "true_peak": round(true_peak, 2) if true_peak is not None else None,
            "lra": round(lra, 2) if lra is not None else None,
            "target_lufs": target_lufs,
            "within_tolerance": within_tolerance,
            "source": "ffmpeg_ebur128",
            "reason": (
                f"Integrated loudness {measured:.2f} LUFS outside target "
                f"{target_lufs} plus/minus 1.0 LU" if not within_tolerance else None
            ),
        }
    except subprocess.TimeoutExpired:
        return {
            "checked": False,
            "status": "ERROR",
            "reason": f"ffmpeg ebur128 timed out after {timeout}s",
        }
    except Exception as e:
        logger.warning(f"Spatial loudness measurement failed for {path}: {e}")
        return {
            "checked": False,
            "status": "ERROR",
            "reason": str(e),
        }


def get_full_file_loudness(ffmpeg_path: str, path: Union[str, Path],
                            timeout: int = 120) -> Dict[str, Any]:
    """
    Measure full-file loudness using FFmpeg ebur128 filter.
    
    FIX: Uses framelog=0 to suppress per-frame logging.
    FIX: Parses from Summary block only with correct multi-line regex patterns.
    FIX: Uses dBFS (not dBTP) for True Peak matching FFmpeg's actual output.
    """
    cmd = [
        ffmpeg_path, "-hide_banner", "-nostats", "-i", str(path),
        "-map", "0:a:0",
        "-af", "ebur128=peak=true:framelog=0",  # FIX: framelog=0 suppresses per-frame noise
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode != 0:
            return {
                "checked": False,
                "status": "ERROR",
                "reason": f"ffmpeg ebur128 failed (rc={result.returncode}): {result.stderr[:300]}",
                "lufs_val": None,
                "true_peak_val": None,
            }

        stderr = result.stderr
        
        # FIX: Parse from Summary block only with correct patterns
        parse_result = _parse_ebur128_summary(stderr)
        
        if not parse_result.get("parsed"):
            return {
                "checked": False,
                "status": "ERROR",
                "reason": parse_result.get("reason", "Could not parse ebur128 output"),
                "lufs_val": None,
                "true_peak_val": None,
                "raw_output_preview": stderr[-500:] if len(stderr) > 500 else stderr,
            }

        lufs_val = parse_result.get("integrated_lufs")
        true_peak_val = parse_result.get("true_peak_dbfs")
        lra = parse_result.get("lra_lu")

        return {
            "checked": True,
            "status": "PASS",
            "lufs_val": lufs_val,
            "true_peak_val": true_peak_val,
            "lra": lra,
            "source": "ffmpeg_ebur128_full",
            "reason": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "checked": False,
            "status": "ERROR",
            "reason": f"ffmpeg ebur128 timed out after {timeout}s",
            "lufs_val": None,
            "true_peak_val": None,
        }
    except Exception as e:
        logger.warning(f"Full-file loudness measurement failed for {path}: {e}")
        return {
            "checked": False,
            "status": "ERROR",
            "reason": str(e),
            "lufs_val": None,
            "true_peak_val": None,
        }
