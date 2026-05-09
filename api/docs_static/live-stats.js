(function () {
  function pick(obj, path) {
    return path.split(".").reduce(function (o, k) {
      return o != null ? o[k] : undefined;
    }, obj);
  }

  function fmtNumber(n) {
    if (typeof n !== "number" || !isFinite(n)) return null;
    return n.toLocaleString("zh-CN");
  }

  async function loadLiveStats() {
    var els = document.querySelectorAll("[data-live-stat]");
    if (!els.length) return;
    try {
      var r = await fetch("/api/v1/stats", { credentials: "same-origin" });
      if (!r.ok) return;
      var j = await r.json();
      if (!j || j.error) return;
      els.forEach(function (el) {
        var key = el.getAttribute("data-live-stat");
        if (!key) return;
        var v = pick(j, key);
        var s = fmtNumber(v);
        if (s != null) el.textContent = s;
      });
    } catch (e) {
      /* 保留页面上的占位文案 */
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", loadLiveStats);
  } else {
    loadLiveStats();
  }
})();
