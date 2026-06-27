"""
Saved example cases shown on the Analysis page.

Add one dict per example. `model_key` must match a key in
model_backend.MODEL_REGISTRY — it's what drives the model tabs on the
Analysis page. `context_used` can be left as "" for examples that didn't
use retrieval (e.g. base-model completions, which never get retrieval
context — see model_backend.build_prompt()).

This is intentionally separate from model_backend.py: these are pre-computed
results from your offline experiments, not live model calls, so the
Analysis page stays fast and reproducible for the defense regardless of
whether the model server is running.
"""

EXAMPLES = [
    {
        "title": "requests: Response.json() vs deprecated .json (property)",
        "language": "python",
        "model_key": "deepseek-coder-1.3b-instruct",
        "deprecated_snippet": (
            "import requests\n\n"
            "resp = requests.get(url)\n"
            "data = resp.json  # missing call — deprecated usage pattern\n"
        ),
        "context_used": (
            "requests >=2.x: `Response.json` is a method, not a property.\n"
            "Call it as `resp.json()`."
        ),
        "completion": "data = resp.json()",
        "notes": "Placeholder example — replace with a real saved run from your experiments.",
    },
    {
        "title": "[Add a CodeGen-2B example here]",
        "language": "python",
        "model_key": "codegen-2b-mono",
        "deprecated_snippet": "# paste a real code snippet from your eval set",
        "context_used": "",
        "completion": "# paste the model's actual completion here",
        "notes": "Placeholder — CodeGen is a base model, so no retrieval context applies.",
    },
]