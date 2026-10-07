"""Local inference API without application-level upload storage."""
import json
import os
import threading
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .inference import Detector, png_uri, read_image
from .simulator import ACTIONS, SCENARIOS, make_scene

app = FastAPI(title="TRACE local inspection", version=__version__)
inference_lock = threading.Lock()


@lru_cache(maxsize=1)
def detector():
    import torch
    torch.set_num_threads(4)
    return Detector(os.environ.get("TRACE_MODEL", "artifacts/trace"))


def ready_detector():
    try:
        return detector()
    except FileNotFoundError as exc:
        raise HTTPException(503, "Trained checkpoint or calibration missing. Train TRACE first.") from exc


class SceneRequest(BaseModel):
    seed: int = Field(default=2_000_021, ge=0, le=99_999_999)
    scenario: str = "moved"
    action: str | None = None


def create_scene(request):
    if request.scenario not in SCENARIOS:
        raise HTTPException(422, "Unknown scenario")
    if request.action is not None and request.action not in ACTIONS:
        raise HTTPException(422, "Unknown re-observation action")
    scene = make_scene(request.seed, scenario=request.scenario)
    observation, _ = scene.observe(request.action)
    return scene.reference, observation


@app.get("/api/health")
def health():
    model = ready_detector()
    return {"status": "ready", "checkpoint_sha256": model.sha256,
            "policy_sha256": model.policy_sha256,
            "variant": model.variant, "policy_ready": model.policy is not None,
            "processing_size": [96, 96], "device": str(model.device)}


@app.post("/api/scene")
def scene(request: SceneRequest):
    reference, observation = create_scene(request)
    return {"reference": png_uri(reference), "observation": png_uri(observation),
            "seed": request.seed, "scenario": request.scenario, "action": request.action}


@app.post("/api/inspect-scene")
def inspect_scene(request: SceneRequest):
    reference, observation = create_scene(request)
    with inference_lock:
        result = ready_detector().evidence(reference, observation)
    result.update(source="simulator", seed=request.seed, scenario=request.scenario, action=request.action,
                  reference=png_uri(reference), observation=png_uri(observation))
    return result


@app.post("/api/inspect")
async def inspect(reference: UploadFile = File(...), observation: UploadFile = File(...)):
    arrays, dimensions = [], []
    try:
        for upload in (reference, observation):
            data = await upload.read(10_000_001)
            if len(data) > 10_000_000:
                raise HTTPException(413, "Each image must be smaller than 10 MB")
            array, size = read_image(data)
            arrays.append(array)
            dimensions.append(size)
        if dimensions[0] != dimensions[1]:
            raise HTTPException(422, "Image dimensions must match")
    except (ValueError, OSError) as exc:
        raise HTTPException(422, "Unable to read this image pair: " + str(exc)) from exc
    # Keep synchronous model execution off the async event loop.
    from starlette.concurrency import run_in_threadpool
    def run():
        with inference_lock:
            return ready_detector().evidence(*arrays)
    result = await run_in_threadpool(run)
    result.update(source="upload", original_dimensions=dimensions[0],
                  reference=png_uri(arrays[0]), observation=png_uri(arrays[1]))
    return result


@app.get("/api/research")
def research():
    directory = Path(os.environ.get("TRACE_MODEL", "artifacts/trace"))
    reports = {}
    for name in ("provenance", "calibration", "policy-report"):
        path = directory / f"{name}.json"
        if path.exists():
            reports[name] = json.loads(path.read_text())
    evaluation = Path("artifacts/evaluation/report.json")
    if evaluation.exists():
        reports["evaluation"] = json.loads(evaluation.read_text())
    return reports


# Built studio and API use one origin; Vite proxies /api during development.
web = Path(__file__).resolve().parents[2] / "web" / "dist"
if web.exists():
    app.mount("/assets", StaticFiles(directory=web / "assets"), name="assets")

    @app.get("/")
    def studio():
        return FileResponse(web / "index.html")
