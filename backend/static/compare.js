document.addEventListener("DOMContentLoaded", () => {
  const runBtn = document.getElementById("cmpRunBtn");
  const toggleBtn = document.getElementById("cmpToggleControls");
  const controls = document.getElementById("compareControls");
  const statusEl = document.getElementById("cmpStatus");

  runBtn.addEventListener("click", runComparison);

  toggleBtn.addEventListener("click", () => {
    const isHidden = controls.style.display === "none";
    controls.style.display = isHidden ? "flex" : "none";
    toggleBtn.textContent = isHidden ? "Hide controls before screenshot" : "Show controls";
  });

  async function runComparison() {
    const title = document.getElementById("cmpTitle").value.trim();
    const snippet = document.getElementById("cmpSnippet").value;
    const family = document.getElementById("cmpFamily").value;
    const intention = document.getElementById("cmpIntention").value.trim();

    if (!snippet.trim()) {
      alert("Paste some code first.");
      return;
    }

    const baseKey = family === "deepseek-coder-1.3b"
      ? "deepseek-coder-1.3b-instruct"
      : "codegen-2b-mono";
    const dpoKey = family === "deepseek-coder-1.3b"
      ? "deepseek-coder-1.3b-dpo"
      : "codegen-2b-dpo";
    const grpoKey = family === "deepseek-coder-1.3b"
      ? "deepseek-coder-1.3b-grpo"
      : "codegen-2b-grpo";

    const methods = [
      { key: "M0", label: "M0 \u00b7 Base (no RAG)", modelKey: baseKey, ragMethod: "none" },
      { key: "M1", label: "M1 \u00b7 BM25 + Rerank RAG", modelKey: baseKey, ragMethod: "m1" },
      { key: "M2", label: "M2 \u00b7 Intent-Extended RAG", modelKey: baseKey, ragMethod: "m2" },
      { key: "M3", label: "M3 \u00b7 DPO", modelKey: dpoKey, ragMethod: "none" },
      { key: "M4", label: "M4 \u00b7 GRPO", modelKey: grpoKey, ragMethod: "none" },
    ];

    runBtn.disabled = true;
    const results = [];

    for (let i = 0; i < methods.length; i++) {
      const m = methods[i];
      statusEl.textContent = `Running ${i + 1}/5 (${m.key}: ${m.modelKey})\u2026 model swap can take ~15-20s.`;
      try {
        const res = await fetch("/complete", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prefix: snippet,
            model_key: m.modelKey,
            max_tokens: 128,
            rag_method: m.ragMethod,
            intention: intention,
          }),
        });
        const data = await res.json();
        results.push({
          ...m,
          completion: res.ok ? data.completion : `[error: server returned ${res.status}]`,
          context: data.retrieved_context || "",
          intentionUsed: data.intention_used || "",
        });
      } catch (err) {
        results.push({ ...m, completion: `[error: ${err.message}]`, context: "", intentionUsed: "" });
      }
    }

    statusEl.textContent = "Done.";
    runBtn.disabled = false;
    renderSnapshot(title, snippet, results);
  }

  function renderSnapshot(title, snippet, results) {
    document.getElementById("snapshotTitle").textContent = title || "Method comparison";
    document.getElementById("snapshotInput").textContent = snippet;

    const grid = document.getElementById("snapshotGrid");
    grid.innerHTML = "";

    results.forEach((r) => {
      const col = document.createElement("div");
      col.className = "snapshot-col";

      let contextHtml = "";
      if (r.ragMethod !== "none") {
        contextHtml = `
          <div class="snapshot-context">
            <span class="snapshot-sublabel">Retrieved context</span>
            <pre>${r.context ? escapeHtml(r.context) : "\u2014 none / no hits \u2014"}</pre>
          </div>`;
      }

      col.innerHTML = `
        <div class="snapshot-col-header">${escapeHtml(r.label)}</div>
        ${contextHtml}
        <div class="snapshot-output">
          <span class="snapshot-sublabel">Output</span>
          <pre>${r.completion ? escapeHtml(r.completion) : "\u2014 empty \u2014"}</pre>
        </div>
      `;
      grid.appendChild(col);
    });

    document.getElementById("compareSnapshot").style.display = "block";
    toggleBtn.style.display = "inline-block";
    document.getElementById("compareSnapshot").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
});