"""
Saved example cases shown on the Analysis page.

Structure: each entry is a CASE (one deprecated-API scenario), containing
a `results` list of per-model outcomes. This lets the Analysis page show
"base model used the deprecated API, but DPO and GRPO didn't" side by side
for the same input, instead of one example per model.

Fields per case:
    title               - short description of the scenario
    language            - for the syntax tag (usually "python")
    deprecated_snippet  - the code PREFIX shown to the model (same text
                          used as `prefix` when sent to /complete)
    context_used        - RAG context, if any was retrieved for this case
                          (leave "" if not using RAG for this example)
    results             - list of dicts, one per model shown for this case:
        model_key   - must match a key in model_backend.MODEL_REGISTRY
        completion  - the model's actual output for this input
        outcome     - "deprecated" | "replacement" | "other"
                      drives the color-coded badge (red/green/gray)
    notes               - optional free text shown under the case

`model_key` values used below must exist in MODEL_REGISTRY (see
model_backend.py) for the "Try live" button to correctly preselect the
right tab in the Playground.
"""

EXAMPLES = [
    {
        "title": "numpy.product \u2192 numpy.prod",
        "language": "python",
        "deprecated_snippet": (
            "import numpy as np\n\n"
            "# Multiply all elements\n"
            "arr = np.array([2, 3, 4, 5])\n"
            "result = np."
        ),
        "context_used": (
            "Replacement API: numpy.prod\n"
            "Deprecated: ['numpy.product']\n"
            "Example: npy=np.arange(np.prod(shape)).reshape(shape)"
        ),
        "results": [
            {
                "model_key": "deepseek-coder-1.3b-instruct",
                "completion": "product(arr)",
                "outcome": "deprecated",
            },
            {
                "model_key": "deepseek-coder-1.3b-dpo",
                "completion": "prod(arr)",
                "outcome": "replacement",
            },
            {
                "model_key": "deepseek-coder-1.3b-grpo",
                "completion": "prod(arr)",
                "outcome": "replacement",
            },
        ],
        "notes": (
            "Placeholder values for deepseek-coder-1.3b-instruct's completion "
            "— replace with a real saved M0 run where the base model actually "
            "produced the deprecated call. DPO/GRPO completions can be swapped "
            "for real saved outputs once available."
        ),
    },
    {
        "title": "torch.norm \u2192 torch.linalg.norm",
        "language": "python",
        "deprecated_snippet": (
            "import torch\n\n"
            "x = torch.randn(3, 3)\n"
            "norm = torch."
        ),
        "context_used": "",
        "results": [
            {
                "model_key": "deepseek-coder-1.3b-instruct",
                "completion": "norm(x)",
                "outcome": "deprecated",
            },
            {
                "model_key": "deepseek-coder-1.3b-dpo",
                "completion": "linalg.norm(x)",
                "outcome": "replacement",
            },
            {
                "model_key": "deepseek-coder-1.3b-grpo",
                "completion": "norm(x)",
                "outcome": "deprecated",
            },
        ],
        "notes": (
            "Illustrates a case where GRPO does NOT correct the deprecated "
            "call while DPO does \u2014 placeholder values, replace with a "
            "real case from results.json where this pattern actually occurs."
        ),
    },
    {
        "title": "seaborn.tsplot \u2192 seaborn.lineplot",
        "language": "python",
        "deprecated_snippet": (
            "import seaborn as sns\n\n"
            "def plot_trend(data, label):\n"
            "    sns."
        ),
        "context_used": "",
        "results": [
            {
                "model_key": "codegen-2b-mono",
                "completion": "tsplot(data, label=label)",
                "outcome": "deprecated",
            },
            {
                "model_key": "codegen-2b-dpo",
                "completion": "tsplot(data, label=label)",
                "outcome": "deprecated",
            },
            {
                "model_key": "codegen-2b-grpo",
                "completion": "lineplot(data=data, label=label)",
                "outcome": "replacement",
            },
        ],
        "notes": (
            "Placeholder \u2014 illustrates GRPO succeeding where both the base "
            "model and DPO fail to move off the deprecated call. Replace with "
            "a real CodeGen case once available."
        ),
    },
]