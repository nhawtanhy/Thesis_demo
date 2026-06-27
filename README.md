# Code Completion IDE Demo

A multi-page web demo: an intro page, a page of saved example cases, a
live Monaco-editor playground with a model picker, and an about page.
Server-rendered with FastAPI + Jinja2 — no frontend build step, no npm.

## Project structure

```
backend/
├── main.py              # FastAPI app: page routes, /models, /complete, /warmup, /health
├── model_backend.py      # Model registry + ALL prompt logic lives here
├── examples_data.py       # Saved example cases shown on /analysis
├── requirements.txt
├── templates/
│   ├── base.html           # shared nav + layout, other pages extend this
│   ├── home.html
│   ├── analysis.html
│   ├── playground.html
│   └── about.html
└── static/
    ├── style.css            # shared dark IDE theme
    └── playground.js         # Monaco editor + model picker + /complete calls
```

## Setup

```bash
cd backend
pip install -r requirements.txt
```

## Run

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**:

- `/` — home / intro
- `/analysis` — saved example cases (edit `examples_data.py` to add real ones)
- `/playground` — the live editor with a model dropdown, calls `/complete`
- `/about` — about page (has `[bracket placeholders]` — fill in your name,
  thesis title, advisor)

## Models

`model_backend.py` has a registry — currently two models, side by side,
selectable from the playground dropdown:

```python
MODEL_REGISTRY = {
    "deepseek-coder-1.3b-instruct": {
        "label": "DeepSeek-Coder 1.3B (instruct)",
        "model_id": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "is_instruct": True,
    },
    "codegen-2b-mono": {
        "label": "CodeGen 2B mono (base)",
        "model_id": "Salesforce/codegen-2B-mono",
        "is_instruct": False,
    },
}
```

Models are **lazy-loaded and cached** — nothing loads until you actually
select it and hit Complete, then it stays in memory for the rest of the
session. This matters for cross-model demos: don't load everything eagerly
at startup, since adding bigger models later (StarCoder2-3B, CodeLlama-7B-
Python) could otherwise eat a lot of VRAM you don't need loaded at once.

**To add another model**, just add an entry to `MODEL_REGISTRY` — the
dropdown and `/complete` routing pick it up automatically, nothing else
needs to change. Set `is_instruct: True` for instruction-tuned checkpoints
(gets the full instruction-wrapped prompt) or `False` for base/raw
completion checkpoints (gets the code as-is, no instruction wrapper —
base models don't follow instructions, they just continue text).

**To swap in your finetuned checkpoint**: change `model_id` for the
relevant entry to your local checkpoint path, e.g.
`/home/user04/checkpoints/deepseek-coder-1.3b-ft`.

### Pre-warming before your defense

Since loading is lazy, the *first* completion request for a given model
will be slow (disk load). Avoid that surprise on stage by warming up every
model you plan to demo beforehand:

```bash
curl -X POST http://localhost:8000/warmup/deepseek-coder-1.3b-instruct
curl -X POST http://localhost:8000/warmup/codegen-2b-mono
```

Or just click through each option in the playground dropdown and hit
Complete once, before your talk starts. The status bar (bottom of the
playground) shows "ready" vs "not loaded yet" per selected model.

## The prompt format

For **instruct models**, `build_prompt()` uses your thesis's exact prompt,
sent as a single user-turn message (no separate system role) via
`tokenizer.apply_chat_template`:

```python
def build_prompt(probing, context="", is_instruct=True):
    if not is_instruct:
        return probing   # base models: raw continuation, no wrapper
    if context:
        return (...)      # includes EVOLUTION INFO block
    return (...)           # CODE-only branch
```

For **base models** (`is_instruct=False`), the prompt is just the raw code
— base models complete text, they don't follow instructions, so wrapping
them in "You are an expert Python programmer..." would just become part of
the text they're continuing, not an instruction they understand.

Output is passed through `clean_completion()`, which extracts just the code
regardless of where the model puts conversational wrapper text (mostly
relevant for instruct models — base-model output rarely has this problem).

This prompt is Python-specific by design, matching your thesis scope —
there's no language switcher in the UI anymore.

## Wiring in retrieval (RAG)

`model_backend.py` has a stub:

```python
def retrieve_context(prefix: str) -> str:
    return ""   # <- replace with your M3.2 retriever call
```

Call your hybrid BM25+dense retriever there and return the formatted
"EVOLUTION INFO" string. `generate()` already threads `context` through to
`build_prompt()` for instruct models (it's a no-op for base models, since
they don't get the context block at all).

If you want to *show* the retrieved context in the playground UI
(recommended — it visually proves the RAG mechanism), add a
`retrieved_context` field to `CompletionResponse` in `main.py`, return it
from `/complete`, and render it in a new panel in `templates/playground.html`
/ `static/playground.js`.

## Adding real examples to the Analysis page

Edit `backend/examples_data.py` — a plain list of dicts:

```python
EXAMPLES = [
    {
        "title": "...",
        "language": "python",
        "deprecated_snippet": "...",
        "context_used": "...",   # "" if no retrieval used
        "completion": "...",
        "notes": "...",          # optional
    },
    ...
]
```

These are static/pre-computed, not live model calls — keeps the Analysis
page fast and reproducible during the defense regardless of whether the
model server is up.

## Defense-day logistics

**Run on the demo laptop directly.** Safest if venue wifi is unreliable.
Pre-load (warm up) every model you'll show on that laptop ahead of time.

**Run on DGX01, tunnel to the laptop:**
```bash
ssh -L 8000:localhost:8000 user04@<dgx01-address>
# then on DGX01: cd backend && uvicorn main:app --port 8000
```
Open `http://localhost:8000` on the laptop. Test this from the actual venue
network beforehand.

**Do a full dry run on the actual laptop you'll demo from**, including
clicking through every model in the dropdown so they're all warm.