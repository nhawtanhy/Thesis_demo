const API_BASE = ""; // same-origin

require.config({ paths: { vs: "https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.49.0/min/vs" } });

const STARTER_CODE = `import requests

def fetch_data(url):
    # Cursor goes here — try completing this function.
    resp = `;

let editor;
let healthData = null;
let lastSuggestion = null;
let insertPosition = null;

require(["vs/editor/editor.main"], function () {
  editor = monaco.editor.create(document.getElementById("editor"), {
    value: STARTER_CODE,
    language: "python",
    theme: "vs-dark",
    fontFamily: "JetBrains Mono, monospace",
    fontSize: 13,
    minimap: { enabled: false },
    automaticLayout: true,
    scrollBeyondLastLine: false,
    padding: { top: 14 },
  });

  editor.onDidChangeCursorPosition((e) => {
    document.getElementById("cursorPos").textContent =
      `Ln ${e.position.lineNumber}, Col ${e.position.column}`;
  });

  editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, requestCompletion);
});

document.getElementById("completeBtn").addEventListener("click", requestCompletion);

async function requestCompletion() {
  if (!editor) return;
  const model = editor.getModel();
  const pos = editor.getPosition();
  insertPosition = pos;

  const prefix = model.getValueInRange({
    startLineNumber: 1, startColumn: 1,
    endLineNumber: pos.lineNumber, endColumn: pos.column,
  });
  const suffix = model.getValueInRange({
    startLineNumber: pos.lineNumber, startColumn: pos.column,
    endLineNumber: model.getLineCount(), endColumn: model.getLineMaxColumn(model.getLineCount()),
  });
  const modelKey = selectedModelKey;

  setLoading(true);
  setSuggestion(null);

  try {
    const res = await fetch(`${API_BASE}/complete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prefix, suffix, model_key: selectedModelKey, max_tokens: 128, use_rag: useRag,}),
    });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const data = await res.json();
    lastSuggestion = data.completion;
    setSuggestion(lastSuggestion);
    checkHealth(); // refresh "ready" status — this model is now warm
  } catch (err) {
    setError(`Couldn't reach the backend (${err.message}). Is uvicorn running on this machine?`);
  } finally {
    setLoading(false);
  }
}

function setLoading(isLoading) {
  const btn = document.getElementById("completeBtn");
  btn.disabled = isLoading;
  btn.innerHTML = isLoading ? `<span class="spinner"></span>Generating…` : "Complete";
}

function setSuggestion(text) {
  const box = document.getElementById("suggestionBox");
  const insertBtn = document.getElementById("insertBtn");
  const discardBtn = document.getElementById("discardBtn");
  box.classList.remove("error");
  if (text === null) {
    box.textContent = "";
    box.classList.add("empty");
    insertBtn.disabled = true;
    discardBtn.disabled = true;
    return;
  }
  box.classList.remove("empty");
  box.textContent = text || "(model returned empty completion)";
  insertBtn.disabled = !text;
  discardBtn.disabled = !text;
}

function setError(msg) {
  const box = document.getElementById("suggestionBox");
  box.classList.remove("empty");
  box.classList.add("error");
  box.textContent = msg;
  document.getElementById("insertBtn").disabled = true;
  document.getElementById("discardBtn").disabled = true;
}

document.getElementById("insertBtn").addEventListener("click", () => {
  if (!lastSuggestion || !insertPosition) return;
  editor.executeEdits("insert-completion", [{
    range: new monaco.Range(
      insertPosition.lineNumber, insertPosition.column,
      insertPosition.lineNumber, insertPosition.column
    ),
    text: lastSuggestion,
  }]);
  editor.focus();
  setSuggestion(null);
  lastSuggestion = null;
});

document.getElementById("discardBtn").addEventListener("click", () => {
  setSuggestion(null);
  lastSuggestion = null;
});

let selectedModelKey = null;

async function loadModelOptions() {
  const container = document.getElementById("modelTabs");
  try {
    const res = await fetch(`${API_BASE}/models`);
    const data = await res.json();
    container.innerHTML = "";
    data.models.forEach((m, i) => {
      const btn = document.createElement("button");
      btn.className = "tab-btn" + (i === 0 ? " active" : "");
      btn.dataset.model = m.key;
      btn.textContent = m.label + (m.is_instruct ? "" : " · base");
      btn.addEventListener("click", () => selectModel(m.key));
      container.appendChild(btn);
    });
    selectedModelKey = data.default || (data.models[0] && data.models[0].key) || null;
  } catch (err) {
    container.innerHTML = `<span class="hint">Couldn't load model list</span>`;
  }
}

let useRag = false;

document.querySelectorAll("#methodTabs .tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#methodTabs .tab-btn")
            .forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    useRag = btn.dataset.rag === "true";
  });
});

function selectModel(key) {
  selectedModelKey = key;
  document.querySelectorAll("#modelTabs .tab-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.model === key);
  });
  updateModelStatusDisplay();
}

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    healthData = await res.json();
  } catch (err) {
    healthData = null;
  }
  updateModelStatusDisplay();
}

function updateModelStatusDisplay() {
  const statusEl = document.getElementById("modelStatus");
  if (!healthData) {
    statusEl.innerHTML = `<span class="dot off"></span>backend unreachable`;
    return;
  }
  const isLoaded = selectedModelKey && healthData.loaded_models.includes(selectedModelKey);
  const activeBtn = document.querySelector(`#modelTabs .tab-btn[data-model="${selectedModelKey}"]`);
  const label = activeBtn ? activeBtn.textContent.trim() : selectedModelKey;
  statusEl.innerHTML = isLoaded
    ? `<span class="dot"></span>${label} · ${healthData.device} · ready`
    : `<span class="dot off"></span>${label} · ${healthData.device} · not loaded yet — first request will be slower`;
}

loadModelOptions().then(checkHealth);