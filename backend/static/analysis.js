document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".try-live-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const payload = {
        prefix: btn.dataset.prefix,
        modelKey: btn.dataset.model,
      };
      // sessionStorage survives the navigation to /playground; playground.js
      // reads this once on load, applies it, then clears it so a manual
      // refresh afterwards doesn't keep re-loading the same case.
      sessionStorage.setItem("demoLoadCase", JSON.stringify(payload));
      window.location.href = "/playground";
    });
  });
});