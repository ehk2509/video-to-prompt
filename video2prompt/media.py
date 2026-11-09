"""Video inspection with ffmpeg: metadata, shot boundaries, frames, speech."""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool


@dataclass(frozen=True)
class Shot:
    index: int
    start: float
    end: float
    continues_previous: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    except FileNotFoundError as exc:
        raise MediaError(f"{cmd[0]} est introuvable dans le PATH.") from exc


def _parse_fps(rate: str) -> float:
    num, _, den = rate.partition("/")
    try:
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(path: Path) -> VideoInfo:
    result = _run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]
    )
    if result.returncode != 0:
        raise MediaError(f"Fichier illisible : {path}\n{result.stderr.strip()}")

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError(f"Aucune piste vidéo dans {path}")

    raw_duration = data.get("format", {}).get("duration") or video.get("duration")
    if raw_duration is None:
        raise MediaError(f"Durée inconnue pour {path}")

    return VideoInfo(
        duration=float(raw_duration),
        width=int(video["width"]),
        height=int(video["height"]),
        fps=_parse_fps(video.get("avg_frame_rate", "0/1")),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def parse_cut_times(ffmpeg_log: str) -> list[float]:
    return [float(t) for t in re.findall(r"pts_time:\s*([0-9.]+)", ffmpeg_log)]


def detect_cuts(path: Path, threshold: float) -> list[float]:
    # Scoring on a small copy is much faster and barely changes the result.
    vf = f"scale=320:-2,select='gt(scene,{threshold})',showinfo"
    result = _run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-i", str(path), "-map", "0:v:0",
         "-vf", vf, "-an", "-f", "null", "-"]
    )
    if result.returncode != 0:
        raise MediaError(f"Échec de la détection des plans :\n{result.stderr[-1500:]}")
    return parse_cut_times(result.stderr)


def build_shots(
    cuts: list[float], duration: float, min_shot: float = 0.6, max_segment: float = 10.0
) -> list[Shot]:
    """Turn cut times into shots, ignoring flash cuts and splitting long takes.

    Long takes are split because generators only produce a few seconds per clip,
    and a handful of frames can't describe a minute of action.
    """
    bounds = [0.0]
    for cut in sorted(cuts):
        if cut - bounds[-1] >= min_shot and duration - cut >= min_shot:
            bounds.append(cut)
    bounds.append(duration)

    shots: list[Shot] = []
    for start, end in pairwise(bounds):
        parts = max(1, math.ceil((end - start) / max_segment))
        step = (end - start) / parts
        for part in range(parts):
            shots.append(
                Shot(
                    index=len(shots) + 1,
                    start=start + part * step,
                    end=start + (part + 1) * step,
                    continues_previous=part > 0,
                )
            )
    return shots


def select_shots(shots: list[Shot], max_shots: int) -> list[Shot]:
    """Keep the longest shots, in their original order."""
    if len(shots) <= max_shots:
        return shots
    longest = sorted(shots, key=lambda s: s.duration, reverse=True)[:max_shots]
    return sorted(longest, key=lambda s: s.start)


def frame_times(shot: Shot, count: int) -> list[float]:
    """Evenly spaced times, never on the shot edges where transitions sit."""
    count = min(count, max(2, math.ceil(shot.duration * 2)))
    return [shot.start + shot.duration * (i + 0.5) / count for i in range(count)]


def extract_frames(path: Path, shot: Shot, count: int, out_dir: Path, max_side: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = (
        f"scale='if(gt(iw,ih),min(iw,{max_side}),-2)':'if(gt(iw,ih),-2,min(ih,{max_side}))'"
    )
    frames: list[Path] = []
    for number, time in enumerate(frame_times(shot, count), start=1):
        target = out_dir / f"shot_{shot.index:03d}_f{number}.jpg"
        result = _run(
            ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-ss", f"{time:.3f}", "-i", str(path),
             "-map", "0:v:0", "-frames:v", "1", "-vf", scale, "-q:v", "3", str(target)]
        )
        if result.returncode == 0 and target.exists():
            frames.append(target)
    if not frames:
        raise MediaError(f"Aucune image extraite pour le plan {shot.index}.")
    return frames


def _decode_audio(path: Path):
    """Decode to Whisper's input format with ffmpeg (faster-whisper's PyAV path breaks on current PyAV)."""
    import numpy as np

    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path),
         "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise MediaError(f"Décodage audio impossible : {detail}")
    return np.frombuffer(result.stdout, dtype=np.float32)


def transcribe(path: Path, model_size: str) -> list[dict]:
    """Return speech segments. Runs on CPU so it never competes with the vision model for VRAM."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise MediaError(
            "faster-whisper n'est pas installé. Lance : pip install -e .[transcribe]"
        ) from exc

    try:
        # Use the cached model first, so a flaky connection can't break a later run.
        model = WhisperModel(model_size, device="cpu", compute_type="int8", local_files_only=True)
    except Exception:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    # The voice filter prevents Whisper from inventing words over music.
    segments, _info = model.transcribe(_decode_audio(path), vad_filter=True)
    return [
        {"start": s.start, "end": s.end, "text": s.text.strip()}
        for s in segments
        if s.text.strip()
    ]


def speech_for_shot(segments: list[dict], shot: Shot) -> str:
    return " ".join(
        s["text"] for s in segments if shot.start <= (s["start"] + s["end"]) / 2 < shot.end
    )
