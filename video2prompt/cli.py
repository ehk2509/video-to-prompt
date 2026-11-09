"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import media
from .backends import BackendError, create_backend
from .pipeline import Options, run
from .targets import TARGETS


def download(url: str, output_root: Path) -> Path:
    try:
        import yt_dlp
    except ImportError as exc:
        raise media.MediaError("yt-dlp n'est pas installé. Lance : pip install -e .[download]") from exc

    options = {
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "outtmpl": str(output_root / "_downloads" / "%(title).60B-%(id)s.%(ext)s"),
    }
    try:
        with yt_dlp.YoutubeDL(options) as client:
            info = client.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise media.MediaError(f"Téléchargement impossible : {exc}") from exc
    return Path(info["requested_downloads"][0]["filepath"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video2prompt",
        description="Transforme une vidéo en prompts pour générer des vidéos similaires.",
    )
    parser.add_argument("input", help="fichier vidéo ou URL")
    parser.add_argument(
        "-t", "--target", default="ltx,wan",
        help=f"cibles séparées par des virgules : {', '.join(TARGETS)} (défaut : ltx,wan)",
    )
    parser.add_argument("-b", "--backend", choices=["ollama", "claude-code", "claude"], default="ollama")
    parser.add_argument("-m", "--model", help="modèle à utiliser (défaut selon le backend)")
    parser.add_argument("--writer-model", help="modèle pour écrire les prompts (défaut : le même)")
    parser.add_argument("-o", "--output", type=Path, default=Path("output"))
    parser.add_argument("--variations", type=int, default=3, help="variations à créer (défaut : 3)")
    parser.add_argument("--max-shots", type=int, default=24, help="plans analysés au maximum")
    parser.add_argument("--frames-per-shot", type=int, default=8)
    parser.add_argument("--scene-threshold", type=float, default=0.3, help="sensibilité aux coupes, 0 à 1")
    parser.add_argument("--no-transcribe", action="store_true", help="ne pas transcrire la parole")
    parser.add_argument("--whisper-model", default="small")
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    options = Options(
        targets=tuple(t.strip() for t in args.target.split(",") if t.strip()),
        variations=args.variations,
        max_shots=args.max_shots,
        frames_per_shot=args.frames_per_shot,
        scene_threshold=args.scene_threshold,
        transcribe=not args.no_transcribe,
        whisper_model=args.whisper_model,
    )

    try:
        if args.input.startswith(("http://", "https://")):
            print("Téléchargement de la vidéo...")
            video = download(args.input, args.output)
        else:
            video = Path(args.input)
            if not video.is_file():
                raise media.MediaError(f"Fichier introuvable : {video}")
        writer = create_backend(args.backend, args.writer_model) if args.writer_model else None
        output_dir = run(
            video, args.output, create_backend(args.backend, args.model), options, writer=writer
        )
    except (media.MediaError, BackendError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1

    print(f"Terminé : {output_dir / 'prompts.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
