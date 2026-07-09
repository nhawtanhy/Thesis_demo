"""
FastAPI server for the code-completion IDE demo (multi-page version).

Run:
    cd backend
    uvicorn main:app --host 0.0.0.0 --port 8000

Pages:
    /            home / intro
    /analysis    saved example cases + live "add & test" feature
    /playground  the Monaco-editor IDE, calls /complete
    /about       about page

API:
    GET  /models              list of registered models
    POST /complete             { prefix, suffix, model_key, max_tokens,
                                  rag_method, intention } -> { completion, ... }
    POST /warmup/{model_key}   force-load a model ahead of time
    GET  /health               device + which model is currently loaded
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
    max_tokens: int = 128
    rag_method: str = "none"   # "none" | "m1" | "m2"
    intention: str = ""        # known intent, if any — used by M2; ignored otherwise


class CompletionResponse(BaseModel):
    completion: str
    retrieved_context: str = ""
    prompt_sent: str = ""
    intention_used: str = ""


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
        max_new_tokens=min(req.max_tokens, 128),
        rag_method=req.rag_method,
        intention=req.intention,
    )
    return CompletionResponse(
        completion=completion,
        retrieved_context=debug.get("retrieved_context", ""),
        prompt_sent=debug.get("prompt_sent", ""),
        intention_used=debug.get("intention_used", ""),
    )


@app.post("/warmup/{model_key}")
def warmup(model_key: str):
    """Force-load a model ahead of time. Only one model stays resident at
    a time (see model_backend.py eviction) — calling this for a new model
    evicts whatever was previously loaded."""
    model_backend.warmup(model_key)
    return {"status": "ok", "model_key": model_key, "loaded_models": model_backend.loaded_models()}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": model_backend.DEVICE,
        "loaded_models": model_backend.loaded_models(),
    }