document.addEventListener("DOMContentLoaded", () => {
  setupTryLiveButtons();
  setupLiveTestForm();
});

// ── "Try live" handoff (existing saved cases) ──────────────────────────────
function setupTryLiveButtons() {
  document.querySelectorAll(".try-live-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const payload = {
        prefix: btn.dataset.prefix,
        modelKey: btn.dataset.model,
        ragMethod: btn.dataset.ragMethod || "none",
        intention: btn.dataset.intention || "",
      };
      sessionStorage.setItem("demoLoadCase", JSON.stringify(payload));
      window.location.href = "/playground";
    });
  });
}

// ── Live "add & test a new case" form ───────────────────────────────────────
function setupLiveTestForm() {
  const toggleBtn = document.getElementById("toggleLiveForm");
  const form = document.getElementById("liveTestForm");
  const ragSelect = document.getElementById("liveRagMethod");
  const intentionRow = document.getElementById("liveIntentionRow");
  const modelChecksContainer = document.getElementById("liveModelChecks");

  if (!toggleBtn || !form) return;

  toggleBtn.addEventListener("click", () => {
    const isHidden = form.style.display === "none";
    form.style.display = isHidden ? "block" : "none";
    toggleBtn.textContent = isHidden ? "\u2212 Hide form" : "+ Add & test a new case live";
  });

  ragSelect.addEventListener("change", () => {
    intentionRow.style.display = ragSelect.value === "m2" ? "block" : "none";
  });

  // Populate model checkboxes from the same /models endpoint the Playground uses
  fetch("/models")
    .then((res) => res.json())
    .then((data) => {
      modelChecksContainer.innerHTML = "";
      data.models.forEach((m, i) => {
        const label = document.createElement("label");
        label.className = "checkbox-item";
        label.innerHTML = `
          <input type="checkbox" value="${m.key}" ${i === 0 ? "checked" : ""}>
          <span>${m.label}${m.is_instruct ? "" : " · base"}</span>
        `;
        modelChecksContainer.appendChild(label);
      });
    })
    .catch(() => {
      modelChecksContainer.innerHTML = `<span class="hint">Couldn't load model list</span>`;
    });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const snippet = document.getElementById("liveSnippet").value;
    const title = document.getElementById("liveTitle").value.trim() || "Live test case";
    const ragMethod = ragSelect.value;
    const intention = document.getElementById("liveIntention").value.trim();
    const checkedModels = Array.from(
      modelChecksContainer.querySelectorAll("input[type=checkbox]:checked")
    ).map((cb) => cb.value);

    if (!snippet.trim()) {
      alert("Paste some code first.");
      return;
    }
    if (checkedModels.length === 0) {
      alert("Select at least one model to test.");
      return;
    }

    const runBtn = document.getElementById("liveRunBtn");
    const status = document.getElementById("liveRunStatus");
    runBtn.disabled = true;

    const results = [];
    for (let i = 0; i < checkedModels.length; i++) {
      const modelKey = checkedModels[i];
      status.textContent = `Running ${i + 1}/${checkedModels.length} (${modelKey})\u2026 this may trigger a model swap (~15-20s) if switching from what's currently loaded.`;
      try {
        const res = await fetch("/complete", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prefix: snippet,
            model_key: modelKey,
            max_tokens: 128,
            rag_method: ragMethod,
            intention: intention,
          }),
        });
        const data = await res.json();
        results.push({
          modelKey,
          completion: res.ok ? data.completion : `[error: server returned ${res.status}]`,
          context: data.retrieved_context || "",
          intentionUsed: data.intention_used || "",
        });
      } catch (err) {
        results.push({ modelKey, completion: `[error: ${err.message}]`, context: "", intentionUsed: "" });
      }
    }

    status.textContent = "Done.";
    runBtn.disabled = false;
    renderLiveCase(title, snippet, ragMethod, results);
  });
}

function renderLiveCase(title, snippet, ragMethod, results) {
  const container = document.getElementById("liveCasesContainer");

  const card = document.createElement("article");
  card.className = "case-card live-case-card";

  const methodLabel = { none: "M0 · No RAG", m1: "M1 · BM25 + Rerank", m2: "M2 · Intent-Extended" }[ragMethod];

  let resultsHtml = "";
  results.forEach((r) => {
    resultsHtml += `
      <div class="result-row outcome-other">
        <span class="outcome-dot"></span>
        <span class="result-model">${r.modelKey} (${methodLabel})</span>
        <code class="result-completion">${escapeHtml(r.completion)}</code>
        <span class="outcome-label">live result</span>
      </div>`;
    if (r.context) {
      resultsHtml += `
        <div class="result-context">
          <span class="result-context-label">Retrieved context:</span>
          <pre>${escapeHtml(r.context)}</pre>
        </div>`;
    }
  });

  card.innerHTML = `
    <header>
      <h3>${escapeHtml(title)} <span class="live-badge">LIVE &middot; session only</span></h3>
      <span class="tag">python</span>
    </header>
    <div class="case-input">
      <h4>Input</h4>
      <pre>${escapeHtml(snippet)}</pre>
    </div>
    <div class="case-results">
      <h4>Method outcomes</h4>
      ${resultsHtml}
    </div>
    <p class="notes">Generated live in this browser session &mdash; not saved to examples_data.py. Copy the results manually if you want to keep them permanently.</p>
  `;

  container.insertBefore(card, container.firstChild);
  card.scrollIntoView({ behavior: "smooth", block: "start" });
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}