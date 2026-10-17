from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from wafer_inspect import CLASSES, __version__
from wafer_inspect.config import settings
from wafer_inspect.logging_utils import configure_logging, request_id_var
from wafer_inspect.model import WaferModel
from wafer_inspect.review_queue import ReviewQueue
from wafer_inspect.synth import DEFAULT_SIZE

configure_logging(settings.LOG_LEVEL)
logger = logging.getLogger("wafer_inspect")

state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["queue"] = ReviewQueue(settings.QUEUE_DB_PATH)
    try:
        state["model"] = WaferModel.load(settings.MODEL_PATH)
        logger.info("model loaded")
    except Exception:
        logger.exception("failed to load model from %s", settings.MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="WaferDefectX inspection API", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex
    token = request_id_var.set(rid)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        logger.info(
            "request",
            extra={
                "fields": {
                    "method": request.method,
                    "path": request.url.path,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                }
            },
        )
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = rid
    return response


class PredictionOut(BaseModel):
    request_id: str
    label: str
    confidence: float
    uncertainty: float
    probabilities: Dict[str, float]
    needs_review: bool
    queue_id: Optional[int] = None


class LabelIn(BaseModel):
    label: str


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail="Not a decodable image")
    if img.shape != (DEFAULT_SIZE, DEFAULT_SIZE):
        img = cv2.resize(img, (DEFAULT_SIZE, DEFAULT_SIZE), interpolation=cv2.INTER_AREA)
    return img


def _queue() -> ReviewQueue:
    return state["queue"]


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "healthy" if "model" in state else "degraded", "version": __version__}


@app.get("/readyz")
def readyz() -> dict:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded; run `python -m wafer_inspect.train`")
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionOut)
async def predict(file: UploadFile = File(...)) -> PredictionOut:
    """Classify a wafer map. Low-confidence wafers are added to the review queue."""
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    data = await file.read(settings.MAX_UPLOAD_BYTES + 1)
    if len(data) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload too large")
    gray = decode_image(data)
    pred = state["model"].predict_image(gray)
    rid = request_id_var.get()
    needs_review = pred.confidence < settings.REVIEW_CONFIDENCE_THRESHOLD
    queue_id = None
    if needs_review:
        ok, png = cv2.imencode(".png", gray)
        queue_id = _queue().add(rid, pred, png.tobytes()) if ok else None
    return PredictionOut(
        request_id=rid,
        label=pred.label,
        confidence=pred.confidence,
        uncertainty=pred.uncertainty,
        probabilities=pred.probabilities,
        needs_review=needs_review,
        queue_id=queue_id,
    )


@app.get("/queue")
def list_queue(status: Optional[str] = "pending", limit: int = 100) -> List[dict]:
    """Review queue, most uncertain first. ``status`` is pending, labelled or empty for all."""
    return _queue().list(status or None, min(max(limit, 1), 1000))


@app.get("/queue/stats")
def queue_stats() -> dict:
    return {**_queue().stats(), "review_confidence_threshold": settings.REVIEW_CONFIDENCE_THRESHOLD}


@app.get("/queue/export")
def export_queue(status: Optional[str] = None, format: str = "json") -> Response:
    """Export queue entries for the labelling team (json or csv)."""
    if format == "csv":
        return Response(_queue().export_csv(status), media_type="text/csv")
    if format != "json":
        raise HTTPException(status_code=400, detail="format must be json or csv")
    return JSONResponse(_queue().list(status, limit=1_000_000))


@app.get("/queue/{item_id}/image")
def queue_image(item_id: int) -> Response:
    png = _queue().image(item_id)
    if png is None:
        raise HTTPException(status_code=404, detail="No such queue item")
    return Response(png, media_type="image/png")


@app.post("/queue/{item_id}/label")
def label_item(item_id: int, body: LabelIn) -> dict:
    """Record the reviewer's label; labelled items can be exported as new training data."""
    if body.label not in CLASSES:
        raise HTTPException(status_code=422, detail=f"label must be one of {list(CLASSES)}")
    if not _queue().label(item_id, body.label):
        raise HTTPException(status_code=404, detail="No such queue item")
    return {"id": item_id, "status": "labelled", "true_label": body.label}


@app.get("/metrics-report")
def metrics_report() -> dict:
    """Metrics written by the training run (synthetic held-out set)."""
    path = Path(settings.METRICS_PATH)
    if not path.exists():
        raise HTTPException(status_code=404, detail="No metrics file; run training")
    return json.loads(path.read_text())


@app.get("/samples")
def list_samples() -> List[str]:
    folder = Path(settings.SAMPLES_DIR)
    return sorted(p.name for p in folder.glob("*.png")) if folder.is_dir() else []


@app.get("/samples/{name}")
def get_sample(name: str) -> Response:
    path = Path(settings.SAMPLES_DIR) / Path(name).name
    if path.suffix != ".png" or not path.is_file():
        raise HTTPException(status_code=404, detail="No such sample")
    return Response(path.read_bytes(), media_type="image/png")


_PAGE = """<!doctype html><meta charset=utf-8><title>WaferDefectX demo</title>
<style>body{font:15px system-ui;max-width:900px;margin:2rem auto;padding:0 1rem}
.g{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;width:160px;cursor:pointer}
img{width:160px;height:160px;border:1px solid #ccc}pre{background:#f4f4f4;padding:1rem;overflow:auto}</style>
<h1>Wafer inspection with a review queue</h1>
<p>Click a sample wafer map (synthetic). Wafers below the confidence threshold are queued for review:
<a href=/queue>/queue</a>, <a href="/queue/export?format=csv">export</a>, <a href=/docs>API docs</a>.</p>
<div class=g id=g></div><pre id=out>(pick a wafer)</pre>
<script>
async function init(){const names=await (await fetch('/samples')).json();
for(const n of names){const f=document.createElement('figure');
f.innerHTML=`<img src="/samples/${n}"><figcaption>${n}</figcaption>`;f.onclick=()=>run(n);g.appendChild(f)}}
async function run(n){const blob=await (await fetch('/samples/'+n)).blob();const fd=new FormData();fd.append('file',blob,n);
const r=await fetch('/predict',{method:'POST',body:fd});out.textContent=JSON.stringify(await r.json(),null,2)}
init()
</script>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo_page() -> str:
    return _PAGE
