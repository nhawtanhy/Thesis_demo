"""
Model backend for the code-completion IDE demo.

Supports multiple models via MODEL_REGISTRY below — each entry is either
an instruction-tuned model (gets the instruction-wrapped prompt) or a base
model (gets raw code as-is, since base models just continue text and don't
follow instructions).

MEMORY MODEL: only ONE model is kept resident on the GPU at a time.
Switching models (clicking a different tab) evicts whatever is currently
loaded and loads the new one — this costs roughly 10-20 seconds per swap,
but avoids the OOM crashes that occurred when trying to keep multiple
models (even small 1-3B ones) simultaneously resident on a T4's ~14.5GB.
This is a deliberate reliability trade-off for live-demo conditions.

RAG (rag_backend.py — BM25 index + reranker) is a SEPARATE, always-resident
component, unaffected by generation-model eviction. Once initialized, RAG
stays loaded regardless of which generation model tab is active.

Adapter entries (adapter_path set) load a LoRA/PEFT adapter on top of the
same base model_id and merge it in — used for the M3 (DPO) and M4 (GRPO)
fine-tuned variants.

To swap in a different checkpoint: change "model_id" or "adapter_path" for
the relevant entry.
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
        "adapter_path": None,
    },
    "deepseek-coder-1.3b-dpo": {
        "label": "DeepSeek-Coder 1.3B + DPO (M3)",
        "model_id": "deepseek-ai/deepseek-coder-1.3b-instruct",   # same base as deepseek-coder-1.3b-instruct
        "is_instruct": True,
        "adapter_path": "/content/drive/MyDrive/thesis/Adapters/m5_dpo_deepseek",
    },
    "deepseek-coder-1.3b-grpo": {
        "label": "DeepSeek-Coder 1.3B + GRPO (M4)",
        "model_id": "deepseek-ai/deepseek-coder-1.3b-instruct",   # same base as deepseek-coder-1.3b-instruct
        "is_instruct": True,
        "adapter_path": "/content/drive/MyDrive/thesis/Adapters/m6_grpo_deepseek",
    },
    "codegen-2b-mono": {
        "label": "CodeGen 2B (M0 base)",
        "model_id": "Salesforce/codegen-2B-mono",
        "is_instruct": False,
        "adapter_path": None,
    },
    "codegen-2b-dpo": {
        "label": "CodeGen 2B + DPO (M3)",
        "model_id": "Salesforce/codegen-2B-mono",   # same base as codegen-2b-mono
        "is_instruct": False,
        "adapter_path": "/content/drive/MyDrive/thesis/Adapters/m5_dpo_codegen",
    },
    "codegen-2b-grpo": {
        "label": "CodeGen 2B + GRPO (M4)",
        "model_id": "Salesforce/codegen-2B-mono",   # same base as codegen-2b-mono
        "is_instruct": False,
        "adapter_path": "/content/drive/MyDrive/thesis/Adapters/m6_grpo_codegen",
    },
}
DEFAULT_MODEL_KEY = "deepseek-coder-1.3b-instruct"

_cache: dict[str, tuple] = {}  # key -> (tokenizer, model)


def _get_model(key: str):
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: {key!r}. Known keys: {list(MODEL_REGISTRY)}")
    if key in _cache:
        return _cache[key]

    import gc

    # Eviction: only keep ONE model resident in GPU memory at a time.
    # T4's usable memory has proven unreliable for holding even two small
    # (1-3B) models simultaneously — swapping on tab click costs ~10-20s
    # but avoids OOM crashes during a live demo.
    if _cache:
        evicted_key = next(iter(_cache))
        print(f"[model_backend] Evicting {evicted_key} to free memory ...")
        _, old_model = _cache.pop(evicted_key)
        del old_model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    cfg = MODEL_REGISTRY[key]
    print(f"[model_backend] Loading {cfg['model_id']} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        trust_remote_code=True,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(DEVICE)

    if cfg.get("adapter_path"):
        from peft import PeftModel
        print(f"[model_backend] Applying LoRA adapter from {cfg['adapter_path']} ...")
        model = PeftModel.from_pretrained(model, cfg["adapter_path"])
        model = model.merge_and_unload()   # merge LoRA weights into base for faster inference
        print(f"[model_backend] Adapter merged.")

    model.eval()
    _cache[key] = (tokenizer, model)
    print(f"[model_backend] {cfg['model_id']} ready.")
    return _cache[key]


def warmup(key: str = DEFAULT_MODEL_KEY) -> None:
    """Pre-load ONE model before the demo starts, so the first click on
    that tab is instant. Only call this for the model you'll show FIRST —
    calling it for a second model will evict the first one immediately
    (same eviction behavior as clicking a different tab), so there's no
    benefit to warming up more than one model in advance."""
    _get_model(key)


def loaded_models() -> list[str]:
    """Returns the currently resident model (0 or 1 entries — eviction
    means at most one model is ever loaded at once)."""
    return list(_cache.keys())


# --- Prompt construction (matches thesis prompt format) --------------------
def build_prompt(probing: str, context: str = "", is_instruct: bool = True) -> str:
    if not is_instruct:
        # Base models: no instruction-following, but RAG context IS used —
        # presented as Python comments directly above the code, matching
        # the thesis's M1/M2 base-model prompt format (see
        # M1_dtrain_dtest_codegen.py's build_prompt). This is the pattern
        # base models see constantly during pretraining (a comment hinting
        # at intent, immediately followed by code), so it acts as a
        # natural continuation cue rather than an instruction to "understand."
        if context:
            commented_context = "\n".join(
                f"# {line}" for line in context.splitlines() if line.strip()
            )
            return f"{commented_context}\n{probing}"
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


def retrieve_context(prefix: str, rag_method: str = "none", intention: str = "") -> tuple[str, str]:
    """
    Returns (context_string, intention_used).

    rag_method: "none" | "m1" | "m2"
      "none" -> no retrieval, both return values are ""
      "m1"   -> BM25 + bi-encoder rerank (rag_backend.retrieve)
      "m2"   -> intent-extended RRF fusion (rag_backend.retrieve_m2).
                If `intention` is empty (no precomputed intent known for
                this input — i.e. live/free-typed Playground input), falls
                back to generating one via intent_backend (CPU, ~9-11s).
    """
    if rag_method == "none":
        return "", ""

    try:
        import rag_backend

        if rag_method == "m1":
            return rag_backend.retrieve(prefix), ""

        if rag_method == "m2":
            used_intention = intention
            if not used_intention.strip():
                try:
                    import intent_backend
                    used_intention = intent_backend.generate_intent(prefix)
                    print(f"[model_backend] Generated intent (CPU fallback): {used_intention!r}")
                except Exception as e:
                    print(f"[model_backend] Intent generation failed: {e}")
                    used_intention = ""
            return rag_backend.retrieve_m2(prefix, used_intention), used_intention

        return "", ""
    except Exception as e:
        print(f"[model_backend] retrieve_context failed: {e}")
        return "", ""


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


_UNCLOSED_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?")


def clean_completion(text: str, prefix: str = "") -> str:
    # BPE fix
    text = text.replace("Ċ", "\n").replace("Ġ", " ")

    # strip echoed prefix if model repeated it
    if prefix:
        text_norm = re.sub(r'\n+', '\n', text.strip())
        # strip up to the last line break in prefix only — keep the partial token
        prefix_for_cmp = re.sub(r'[^\n]*$', '', prefix.strip())  # drop last partial line
        prefix_norm = re.sub(r'\n+', '\n', prefix_for_cmp)
        if prefix_norm and text_norm.startswith(prefix_norm):
            text = text_norm[len(prefix_norm):]

    # cut at stop sequences — but IGNORE matches too close to the start.
    # When RAG context is present, the model sometimes restarts/echoes the
    # prompt's CODE section from scratch (e.g. "\n\nimport torch\n\n#...")
    # instead of continuing directly. A stop sequence like "\nimport " can
    # then fire within the first few characters, cutting the ENTIRE
    # completion down to nothing. A genuine "model wandered into a new
    # unrelated definition after giving a real answer" case never happens
    # this early — there's no room for an answer before position ~5 — so
    # matches this close to the start are always the restart artifact, not
    # useful signal, and are skipped.
    MIN_STOP_POSITION = 5
    cuts = [i for i in (text.find(s) for s in _STOP_SEQUENCES) if i > MIN_STOP_POSITION]
    if cuts:
        text = text[:min(cuts)]

    text = text.strip()

    m = _FENCE_RE.search(text)
    if m:
        # properly closed fence — extract just the code between the markers
        text = m.group(1).strip()
    else:
        # no closed fence found — still strip a lone UNCLOSED opening fence
        # (e.g. "```python" with no matching closing "```", which happens
        # when generation gets cut off by a stop sequence or token limit
        # before the model closes its own fence) and any preamble text
        text = _UNCLOSED_FENCE_RE.sub("", text, count=1)
        text = _PREAMBLE_RE.sub("", text, count=1).strip()

    return text.rstrip()


# --- Generation --------------------------------------------------------------
@torch.inference_mode()
def generate(prefix, suffix="", model_key=DEFAULT_MODEL_KEY,
             max_new_tokens=128, rag_method="none", intention=""):
    # NOTE: suffix is accepted for API/frontend stability but intentionally
    # unused — the thesis prompt format is prefix(+context)-only.
    cfg = MODEL_REGISTRY[model_key]
    tokenizer, model = _get_model(model_key)

    context, intention_used = retrieve_context(prefix, rag_method, intention)
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
        "intention_used": intention_used,
    }