document.addEventListener("DOMContentLoaded", () => {
  const tabs = document.querySelectorAll("#compareCaseTabs .tab-btn");
  const cases = document.querySelectorAll(".compare-case");

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const index = tab.dataset.caseIndex;

      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");

      cases.forEach((c) => {
        c.style.display = c.dataset.caseIndex === index ? "block" : "none";
      });
    });
  });

  highlightAllCases();
  setupPromptPreviews();
});

// ── API name highlighting (red=deprecated, teal=replacement) ───────────────

// Parses "numpy.alltrue \u2192 numpy.all (rms_from_ivar)" into
// { deprecatedApi: "numpy.alltrue", replacementApi: "numpy.all" }
function parseTitleForApis(title) {
  const parts = title.split("\u2192");
  if (parts.length < 2) return { deprecatedApi: "", replacementApi: "" };
  const deprecatedApi = parts[0].trim();
  const replacementApi = parts[1].replace(/\s*\([^)]*\)\s*$/, "").trim();
  return { deprecatedApi, replacementApi };
}

function escapeRegExp(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function wrapMatches(escapedText, apiName, cssClass) {
  if (!apiName) return escapedText;
  const pattern = new RegExp("\\b" + escapeRegExp(apiName) + "\\b", "g");
  return escapedText.replace(pattern, (match) => `<span class="${cssClass}">${match}</span>`);
}

function highlightAllCases() {
  document.querySelectorAll(".compare-case").forEach((caseEl) => {
    const titleEl = caseEl.querySelector(".snapshot-title");
    if (!titleEl) return;
    const { deprecatedApi, replacementApi } = parseTitleForApis(titleEl.textContent);
    if (!deprecatedApi && !replacementApi) return;

    caseEl.querySelectorAll(".snapshot-output pre").forEach((pre) => {
      const raw = pre.textContent;
      let escaped = escapeHtml(raw);
      escaped = wrapMatches(escaped, replacementApi, "hl-replacement");
      escaped = wrapMatches(escaped, deprecatedApi, "hl-deprecated");
      pre.innerHTML = escaped;
    });
  });
}

// ── Prompt reconstruction (mirrors model_backend.py's build_prompt exactly) ─

function buildPromptNoContext(probing, isInstruct) {
  if (!isInstruct) return probing;
  return (
    "You are an expert Python programmer. Complete the code.\n" +
    "Output ONLY the missing code. No markdown. No explanation.\n\n" +
    "CODE:\n" + probing
  );
}

function buildPromptWithContext(probing, context, isInstruct) {
  if (!context) return buildPromptNoContext(probing, isInstruct);
  if (!isInstruct) {
    const commented = context
      .split("\n")
      .filter((l) => l.trim())
      .map((l) => "# " + l)
      .join("\n");
    return commented + "\n" + probing;
  }
  return (
    "You are an expert Python programmer.\n" +
    "Complete the code using ONLY modern non-deprecated APIs.\n" +
    "Output ONLY the missing code. No markdown. No explanation.\n\n" +
    "EVOLUTION INFO:\n" + context + "\n\n" +
    "CODE:\n" + probing
  );
}

function setupPromptPreviews() {
  document.querySelectorAll(".compare-case").forEach((caseEl) => {
    const probingEl = caseEl.querySelector(".snapshot-input pre");
    if (!probingEl) return;
    const probing = probingEl.textContent;

    // Find a "no RAG" column to get is_instruct for the bare prompt
    const noRagCol = caseEl.querySelector('.snapshot-col[data-rag-method="none"]');
    const isInstructNoRag = noRagCol ? noRagCol.dataset.isInstruct === "true" : true;
    const noContextPrompt = buildPromptNoContext(probing, isInstructNoRag);

    // Find the first RAG column (m1 or m2) that has retrieved context
    const ragCol = caseEl.querySelector(
      '.snapshot-col[data-rag-method="m1"], .snapshot-col[data-rag-method="m2"]'
    );
    let withContextPrompt = null;
    if (ragCol) {
      const contextPre = ragCol.querySelector(".snapshot-context pre");
      const isInstructRag = ragCol.dataset.isInstruct === "true";
      if (contextPre) {
        withContextPrompt = buildPromptWithContext(probing, contextPre.textContent, isInstructRag);
      }
    }

    const previewBox = caseEl.querySelector(".prompt-preview-content");
    const promptTabs = caseEl.querySelectorAll(".prompt-tabs .tab-btn");
    if (!previewBox) return;

    // Default to the "no RAG" prompt
    previewBox.textContent = noContextPrompt;

    // If no RAG data exists for this case, disable the second tab
    const contextTabBtn = caseEl.querySelector('.prompt-tabs .tab-btn[data-prompt-type="context"]');
    if (!withContextPrompt && contextTabBtn) {
      contextTabBtn.disabled = true;
      contextTabBtn.title = "No RAG context available for this case";
      contextTabBtn.style.opacity = "0.4";
      contextTabBtn.style.cursor = "not-allowed";
    }

    promptTabs.forEach((btn) => {
      btn.addEventListener("click", () => {
        if (btn.disabled) return;
        promptTabs.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        previewBox.textContent =
          btn.dataset.promptType === "context" ? withContextPrompt : noContextPrompt;
      });
    });
  });
}