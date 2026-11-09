"""Local web interface: submit a video, follow the analysis, copy the prompts."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import threading
import traceback
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import media
from ..backends import BackendError, create_backend, ollama_vision_models
from ..cli import download
from ..pipeline import Options, run, slugify
from ..targets import TARGETS

DEFAULT_OLLAMA_MODEL = "qwen3-vl:8b-instruct"
STATIC_DIR = Path(__file__).parent / "static"
OUTPUT_ROOT = Path.cwd() / "output" / "web"
LOCAL_HOSTS = {"127.0.0.1", "localhost"}
FRAME_NAME = re.compile(r"shot_\d+_f\d+\.jpg")
DOWNLOADS = {"prompts.md": "text/markdown", "result.json": "application/json"}


class JobCancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    name: str
    root: Path
    backend: str
    model: str | None
    options: Options
    url: str = ""
    upload: Path | None = None
    status: str = "queued"
    stage: str = ""
    done: int = 0
    total: int = 0
    preview: str = ""
    logs: list[str] = field(default_factory=list)
    error: str = ""
    output_dir: Path | None = None
    result: dict | None = None
    cancel_requested: bool = False

    def update(self, stage: str, done: int, total: int, preview: str | None = None) -> None:
        if self.cancel_requested:
            raise JobCancelled
        self.stage, self.done, self.total = stage, done, total
        if preview:
            self.preview = preview

    def public(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "stage": self.stage,
            "done": self.done,
            "total": self.total,
            "preview": self.preview,
            "logs": self.logs,
            "error": self.error,
            "from_url": bool(self.url),
            "transcribes": self.options.transcribe,
            "result": self.result,
        }


jobs: dict[str, Job] = {}
# One at a time: every job needs the same GPU.
worker = ThreadPoolExecutor(max_workers=1)


def execute(job: Job) -> None:
    if job.cancel_requested:
        job.status = "cancelled"
        return
    job.status = "running"
    try:
        if job.url:
            job.update("download", 0, 1)
            video = download(job.url, job.root)
            job.name = video.stem
        else:
            video = job.upload
        job.output_dir = job.root / slugify(video.stem)
        run(
            video,
            job.root,
            create_backend(job.backend, job.model),
            job.options,
            log=job.logs.append,
            on_progress=job.update,
        )
        job.result = json.loads((job.output_dir / "result.json").read_text(encoding="utf-8"))
        job.status = "done"
    except JobCancelled:
        job.status = "cancelled"
    except (media.MediaError, BackendError, ValueError) as exc:
        job.error, job.status = str(exc), "error"
    except Exception as exc:
        traceback.print_exc()
        job.error, job.status = f"Erreur inattendue : {exc}", "error"


app = FastAPI(title="video2prompt", docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def only_local_pages(request: Request, call_next):
    """Stop other websites from driving this server through the visitor's browser."""
    host_header = request.headers.get("host", "")
    if host_header.rsplit(":", 1)[0] not in LOCAL_HOSTS:
        return JSONResponse({"detail": "Hôte non autorisé."}, status_code=403)
    origin = request.headers.get("origin")
    if request.method != "GET" and origin and urlsplit(origin).netloc != host_header:
        return JSONResponse({"detail": "Origine non autorisée."}, status_code=403)
    return await call_next(request)


@app.get("/api/config")
def config() -> dict:
    models = ollama_vision_models()
    return {
        "targets": [{"key": t.key, "label": t.label} for t in TARGETS.values()],
        "ollama": {
            "ok": models is not None,
            "models": models or [],
            "default": DEFAULT_OLLAMA_MODEL,
        },
        "claude_available": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }


@app.post("/api/jobs")
def create_job(
    file: Annotated[UploadFile | None, File()] = None,
    url: Annotated[str, Form()] = "",
    targets: Annotated[str, Form()] = "ltx,wan",
    backend: Annotated[str, Form()] = "ollama",
    model: Annotated[str, Form()] = "",
    variations: Annotated[int, Form(ge=0, le=10)] = 3,
    max_shots: Annotated[int, Form(ge=1, le=100)] = 24,
    frames_per_shot: Annotated[int, Form(ge=2, le=8)] = 4,
    scene_threshold: Annotated[float, Form(ge=0.05, le=0.9)] = 0.3,
    transcribe: Annotated[bool, Form()] = True,
    whisper_model: Annotated[str, Form(pattern=r"^[\w.-]+$")] = "small",
) -> dict:
    url = url.strip()
    if bool(file and file.filename) == bool(url):
        raise HTTPException(400, "Choisis un fichier ou un lien, pas les deux.")
    if url and not url.startswith(("http://", "https://")):
        raise HTTPException(400, "Le lien doit commencer par http:// ou https://.")
    chosen = tuple(t for t in targets.split(",") if t)
    if not chosen or any(t not in TARGETS for t in chosen):
        raise HTTPException(400, "Choisis au moins un générateur valide.")
    if backend not in ("ollama", "claude"):
        raise HTTPException(400, "Moteur inconnu.")

    job_id = secrets.token_hex(6)
    root = OUTPUT_ROOT / job_id
    upload = None
    name = url
    if file and file.filename:
        name = Path(file.filename.replace("\\", "/")).name or "video"
        upload_dir = root / "_upload"
        upload_dir.mkdir(parents=True)
        upload = upload_dir / name
        with upload.open("wb") as target:
            shutil.copyfileobj(file.file, target, 1024 * 1024)

    job = Job(
        id=job_id,
        name=name,
        root=root,
        backend=backend,
        model=model or None,
        url=url,
        upload=upload,
        options=Options(
            targets=chosen,
            variations=variations,
            max_shots=max_shots,
            frames_per_shot=frames_per_shot,
            scene_threshold=scene_threshold,
            transcribe=transcribe,
            whisper_model=whisper_model,
        ),
    )
    jobs[job_id] = job
    worker.submit(execute, job)
    return {"id": job_id}


def _job(job_id: str) -> Job:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Analyse introuvable.")
    return job


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    return _job(job_id).public()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    _job(job_id).cancel_requested = True
    return {"ok": True}


@app.get("/api/jobs/{job_id}/frames/{name}")
def frame(job_id: str, name: str) -> Response:
    job = _job(job_id)
    path = job.output_dir / "frames" / name if job.output_dir else None
    if not FRAME_NAME.fullmatch(name) or path is None or not path.is_file():
        raise HTTPException(404, "Image introuvable.")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/download/{name}")
def download_file(job_id: str, name: str) -> Response:
    job = _job(job_id)
    if name not in DOWNLOADS or job.status != "done" or job.output_dir is None:
        raise HTTPException(404, "Fichier introuvable.")
    return FileResponse(job.output_dir / name, media_type=DOWNLOADS[name], filename=name)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(prog="video2prompt-web")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    address = f"http://127.0.0.1:{args.port}"
    print(f"Interface disponible sur {address}")
    if not args.no_browser:
        threading.Timer(1.2, webbrowser.open, [address]).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
