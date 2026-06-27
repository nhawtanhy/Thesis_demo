document.addEventListener("DOMContentLoaded", () => {
  const tabs = document.querySelectorAll("#modelTabs .tab-btn");
  const cards = document.querySelectorAll(".example-card");
  const emptyMsg = document.getElementById("noExamplesForModel");

  function applyFilter(modelKey) {
    let visibleCount = 0;
    cards.forEach((card) => {
      const match = card.dataset.model === modelKey;
      card.style.display = match ? "" : "none";
      if (match) visibleCount++;
    });
    if (emptyMsg) emptyMsg.style.display = visibleCount === 0 ? "block" : "none";
  }

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      applyFilter(tab.dataset.model);
    });
  });

  if (tabs.length) applyFilter(tabs[0].dataset.model);
});