"""
FastAPI server for the code-completion IDE demo (multi-page version).

Run:
    cd backend
    uvicorn main:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000 in a browser.

Pages:
    /            home / intro
    /analysis    saved example cases (from examples_data.py)
    /playground  the Monaco-editor IDE, calls /complete
    /about       about page

API:
    GET  /models           list of registered models (for the dropdown)
    POST /complete          { prefix, suffix, model_key, max_tokens } -> { completion }
    POST /warmup/{model_key} force-load a model ahead of time
    GET  /health            device + which models are currently loaded
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import model_backend
from examples_data import EXAMPLES

app = FastAPI(title="Code Completion Demo")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


class CompletionRequest(BaseModel):
    prefix: str
    suffix: str = ""
    model_key: str = model_backend.DEFAULT_MODEL_KEY
    max_tokens: int = 40
    use_rag: bool = False      # ← add this


class CompletionResponse(BaseModel):
    completion: str
    retrieved_context: str = ""   # ← add
    prompt_sent: str = ""         # ← add


# --- Pages ----------------------------------------------------------------
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "home.html", {"active": "home"})


@app.get("/analysis")
def analysis(request: Request):
    return templates.TemplateResponse(
        request,
        "analysis.html",
        {
            "active": "analysis",
            "examples": EXAMPLES,
            "models": model_backend.MODEL_REGISTRY,
        },
    )


@app.get("/playground")
def playground(request: Request):
    return templates.TemplateResponse(request, "playground.html", {"active": "playground"})


@app.get("/about")
def about(request: Request):
    return templates.TemplateResponse(request, "about.html", {"active": "about"})


# --- API --------------------------------------------------------------------
@app.get("/models")
def list_models():
    return {
        "models": [
            {"key": key, "label": cfg["label"], "is_instruct": cfg["is_instruct"]}
            for key, cfg in model_backend.MODEL_REGISTRY.items()
        ],
        "default": model_backend.DEFAULT_MODEL_KEY,
    }


@app.post("/complete", response_model=CompletionResponse)
def complete(req: CompletionRequest):
    completion, debug = model_backend.generate(
        prefix=req.prefix,
        suffix=req.suffix,
        model_key=req.model_key,
        max_new_tokens=min(req.max_tokens, 40),
        use_rag=req.use_rag,
    )
    return CompletionResponse(
        completion=completion,
        retrieved_context=debug.get("retrieved_context", ""),
        prompt_sent=debug.get("prompt_sent", ""),
    )


@app.post("/warmup/{model_key}")
def warmup(model_key: str):
    """Force-load a model ahead of time — handy to hit once for each model
    right before your defense starts, so switching is instant on stage."""
    model_backend.warmup(model_key)
    return {"status": "ok", "model_key": model_key, "loaded_models": model_backend.loaded_models()}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": model_backend.DEVICE,
        "loaded_models": model_backend.loaded_models(),
    }