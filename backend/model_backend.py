"""
Model backend for the code-completion IDE demo.

Supports multiple models via MODEL_REGISTRY below — each entry is either
an instruction-tuned model (gets the instruction-wrapped prompt) or a base
model (gets raw code as-is, since base models just continue text and don't
follow instructions). Models are loaded lazily on first use and cached in
memory afterwards — see warmup() if you want to pre-load before a live demo.

To add another model for cross-model comparison (e.g. StarCoder2-3B,
CodeLlama-7B-Python): add an entry to MODEL_REGISTRY. Nothing else needs
to change — main.py and the frontend read the registry dynamically.

To swap in your finetuned checkpoint: change "model_id" for the relevant
entry to your local checkpoint path.
"""

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# --- Model registry -------------------------------------------------------
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
DEFAULT_MODEL_KEY = "deepseek-coder-1.3b-instruct"

_cache: dict[str, tuple] = {}  # key -> (tokenizer, model)


def _get_model(key: str):
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: {key!r}. Known keys: {list(MODEL_REGISTRY)}")
    if key in _cache:
        return _cache[key]

    cfg = MODEL_REGISTRY[key]
    print(f"[model_backend] Loading {cfg['model_id']} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        # GPT2-family tokenizers (CodeGen included) often have no pad token.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        trust_remote_code=True,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    model.eval()

    _cache[key] = (tokenizer, model)
    print(f"[model_backend] {cfg['model_id']} loaded.")
    return _cache[key]


def warmup(key: str = DEFAULT_MODEL_KEY) -> None:
    """Force-load a model ahead of time. Call this for every model you
    plan to demo, before your defense starts, so switching is instant."""
    _get_model(key)


def loaded_models() -> list[str]:
    return list(_cache.keys())


# --- Prompt construction (matches thesis prompt format) --------------------
def build_prompt(probing: str, context: str = "", is_instruct: bool = True) -> str:
    if not is_instruct:
        # Base models: no instruction-following — raw continuation only.
        return probing

    if context:
        return (
            "You are an expert Python programmer.\n"
            "Complete the code using ONLY modern non-deprecated APIs.\n"
            "Output ONLY the missing code. No markdown. No explanation.\n\n"
            f"EVOLUTION INFO:\n{context}\n\n"
            f"CODE:\n{probing}"
        )
    return (
        "You are an expert Python programmer. Complete the code.\n"
        "Output ONLY the missing code. No markdown. No explanation.\n\n"
        f"CODE:\n{probing}"
    )


def retrieve_context(prefix: str) -> str:
    try:
        import rag_backend
        return rag_backend.retrieve(prefix)
    except Exception:
        return "" 


# --- Output cleaning ---------------------------------------------------------
_FENCE_RE = re.compile(r"```[a-zA-Z]*\n?(.*?)```", re.DOTALL)
_PREAMBLE_RE = re.compile(
    r"^(sure|here'?s?|certainly|of course|okay|ok)[^\n]*\n+",
    re.IGNORECASE,
)

# Once a completion hits one of these, the model has wandered past "complete
# the current line/function" into "write a whole new top-level definition" —
# same idea as the stop-sequence trimming in your HumanEval evaluation
# harness. Base models especially will keep going for the full token budget
# with no natural stopping point unless cut off here. Add more patterns if
# you see a model wander off in a different direction.
_STOP_SEQUENCES = ["\ndef ", "\nclass ", "\nif __name__", "\n\n\n", "\nimport "]


def truncate_at_stop_sequence(text: str) -> str:
    cut_points = [i for i in (text.find(s) for s in _STOP_SEQUENCES) if i != -1]
    if cut_points:
        text = text[: min(cut_points)]
    return text.rstrip()


def clean_completion(text: str, prefix: str = "") -> str:
    # BPE fix
    text = text.replace("Ċ", "\n").replace("Ġ", " ")

    # strip echoed prefix if model repeated it
    if prefix and text.lstrip().startswith(prefix.lstrip()):
        text = text.lstrip()[len(prefix.lstrip()):]

    # cut at stop sequences
    cuts = [i for i in (text.find(s) for s in _STOP_SEQUENCES) if i != -1]
    if cuts:
        text = text[:min(cuts)]

    text = text.strip()

    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    else:
        text = _PREAMBLE_RE.sub("", text, count=1).strip()

    return text.rstrip()


# --- Generation --------------------------------------------------------------
@torch.inference_mode()
def generate(prefix, suffix="", model_key=DEFAULT_MODEL_KEY,
             max_new_tokens=128, use_rag=False):
    # NOTE: suffix is accepted for API/frontend stability but intentionally
    # unused — the thesis prompt format is prefix(+context)-only.
    cfg = MODEL_REGISTRY[model_key]
    tokenizer, model = _get_model(model_key)

    context = retrieve_context(prefix) if use_rag else ""
    prompt_text = build_prompt(prefix, context, is_instruct=cfg["is_instruct"])

    if cfg["is_instruct"]:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            return_tensors="pt", add_generation_prompt=True
        )
        # newer transformers returns BatchEncoding, older returns plain tensor
        if hasattr(encoded, 'input_ids'):
            input_ids = encoded.input_ids.to(DEVICE)
            attention_mask = encoded.attention_mask.to(DEVICE)
        else:
            input_ids = encoded.to(DEVICE)
            attention_mask = torch.ones_like(input_ids)
    else:
        encoded = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        no_repeat_ngram_size=4,  # blocks literal copy-paste loops (e.g. repeated if-blocks)
        pad_token_id=tokenizer.pad_token_id,
    )

    completion_ids = output[0][input_ids.shape[1]:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False)
    completion = clean_completion(text, prefix=prefix)
    return completion, {
        "retrieved_context": context,
        "prompt_sent": prompt_text,
    }