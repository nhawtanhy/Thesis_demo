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
});

// Parses "numpy.alltrue \u2192 numpy.all (rms_from_ivar)" into
// { deprecatedApi: "numpy.alltrue", replacementApi: "numpy.all" }
// (strips a trailing " (...)" annotation from the replacement side).
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

// text is ALREADY html-escaped (browser did this via textContent originally,
// we work from the raw pre.textContent instead, escaping ourselves here).
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