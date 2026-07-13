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
});