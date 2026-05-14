document.addEventListener('DOMContentLoaded', function () {
  var toc = document.querySelector('.toc');
  var curPath = location.pathname.replace(/\.html$/, '').replace(/\/$/, '');

  /* ── 0. Replace back-link with search form ── */
  var backLink = document.querySelector('.back-link');
  if (backLink) {
    var form = document.createElement('form');
    form.className = 'search-form';
    form.setAttribute('action', '/docs/search');
    form.setAttribute('method', 'get');
    form.innerHTML =
      '<span class="search-icon">&#128269;</span>' +
      '<input class="search-input" type="text" name="q" placeholder="搜索文档..." autocomplete="off">';
    backLink.parentNode.replaceChild(form, backLink);
  }

  if (!toc) return;

  /* ── 1. Wrap parent + sub links into .toc-group ── */
  var allLinks = Array.prototype.slice.call(toc.querySelectorAll('a'));
  var i = 0;
  while (i < allLinks.length) {
    var link = allLinks[i];
    if (link.classList.contains('sub')) { i++; continue; }

    var subs = [];
    for (var j = i + 1; j < allLinks.length; j++) {
      if (!allLinks[j].classList.contains('sub')) break;
      subs.push(allLinks[j]);
    }
    if (subs.length === 0) { i++; continue; }

    var group = document.createElement('div');
    group.className = 'toc-group';
    link.parentNode.insertBefore(group, link);
    group.appendChild(link);

    var children = document.createElement('div');
    children.className = 'toc-children';
    subs.forEach(function (s) { children.appendChild(s); });
    group.appendChild(children);

    var arrow = document.createElement('span');
    arrow.className = 'toc-arrow';
    arrow.textContent = '+';
    link.insertBefore(arrow, link.firstChild);

    i += 1 + subs.length;
  }

  /* ── 2. Decide open/closed state ── */
  var groups = toc.querySelectorAll('.toc-group');
  var curHash = location.hash;

  function setOpen(g, open) {
    if (open) {
      g.classList.add('open');
      var a = g.querySelector('.toc-arrow');
      if (a) a.textContent = '\u2212';
    } else {
      g.classList.remove('open');
      var a2 = g.querySelector('.toc-arrow');
      if (a2) a2.textContent = '+';
    }
  }

  /* helper: find the group whose parent link points to curPath */
  var localGroup = null;
  var localParent = null;
  var localSubs = [];

  groups.forEach(function (g) {
    /* default: every group starts CLOSED, regardless of any prior state */
    setOpen(g, false);

    var parent = g.querySelector('a:first-child');
    var childrenDiv = g.querySelector('.toc-children');
    if (!parent || !childrenDiv) return;

    var parentHref = (parent.getAttribute('href') || '').split('#')[0].replace(/\/$/, '');
    var onParentPage = curPath === parentHref;

    /* Open ONLY when the parent link points at the current page. We deliberately
       do NOT auto-open groups whose subs target other pages (e.g. on
       /docs/modules/validation we must NOT expand the "events" group just
       because its subs share the same hash style). */
    if (onParentPage) {
      setOpen(g, true);
      localGroup = g;
      localParent = parent;
      localSubs = Array.prototype.slice.call(childrenDiv.querySelectorAll('a'));

      if (curHash) {
        localSubs.forEach(function (cl) {
          var h = cl.getAttribute('href') || '';
          var hHash = h.indexOf('#') !== -1 ? h.substring(h.indexOf('#')) : '';
          if (hHash && hHash === curHash) {
            parent.classList.remove('active');
            cl.classList.add('active');
          }
        });
      }
    }
  });

  /* ── helper: highlight a specific anchor (or parent if null) ── */
  function highlightAnchor(anchorId) {
    if (!localGroup) return;
    if (!anchorId) {
      localSubs.forEach(function (s) { s.classList.remove('active'); });
      localParent.classList.add('active');
      return;
    }
    var matched = false;
    localSubs.forEach(function (s) {
      var h = s.getAttribute('href') || '';
      var hHash = h.indexOf('#') !== -1 ? h.substring(h.indexOf('#')) : '';
      if (hHash === '#' + anchorId) {
        s.classList.add('active');
        matched = true;
      } else {
        s.classList.remove('active');
      }
    });
    if (matched) {
      localParent.classList.remove('active');
    } else {
      localParent.classList.add('active');
    }
  }

  /* ── 3. Toggle on arrow click ── */
  toc.addEventListener('click', function (e) {
    var arrow = e.target.closest('.toc-arrow');
    if (!arrow) return;
    e.preventDefault();
    e.stopPropagation();
    var grp = arrow.closest('.toc-group');
    if (grp) {
      var isOpen = grp.classList.contains('open');
      setOpen(grp, !isOpen);
    }
  });

  /* ── 4. Same-page parent click → scroll to top + highlight parent ── */
  toc.addEventListener('click', function (e) {
    if (e.target.closest('.toc-arrow')) return;
    var link = e.target.closest('a');
    if (!link || link.classList.contains('sub')) return;
    var href = (link.getAttribute('href') || '').split('#')[0].replace(/\/$/, '');
    if (href === curPath) {
      e.preventDefault();
      history.replaceState(null, '', curPath);
      window.scrollTo({ top: 0, behavior: 'smooth' });
      highlightAnchor(null);
    }
  });

  /* ── 5. Hash change → update sub highlight ── */
  window.addEventListener('hashchange', function () {
    var hash = location.hash;
    if (hash) {
      highlightAnchor(hash.substring(1));
    } else {
      highlightAnchor(null);
    }
  });

  /* ── 6. Scroll spy: track which section heading is on screen ── */
  if (localGroup && localSubs.length > 0) {
    var anchorIds = [];
    localSubs.forEach(function (s) {
      var h = s.getAttribute('href') || '';
      var idx = h.indexOf('#');
      if (idx !== -1) anchorIds.push(h.substring(idx + 1));
    });

    if (anchorIds.length > 0) {
      var headingEls = [];
      anchorIds.forEach(function (id) {
        var el = document.getElementById(id);
        if (el) headingEls.push({ id: id, el: el });
      });

      if (headingEls.length > 0) {
        var scrollSuppressed = false;

        var observer = new IntersectionObserver(function (entries) {
          if (scrollSuppressed) return;
          var visible = [];
          entries.forEach(function (entry) {
            if (entry.isIntersecting) {
              visible.push(entry.target);
            }
          });
          if (visible.length === 0) return;

          var topmost = null;
          var topY = Infinity;
          visible.forEach(function (v) {
            var r = v.getBoundingClientRect();
            if (r.top < topY) { topY = r.top; topmost = v; }
          });
          if (topmost && topmost.id) {
            highlightAnchor(topmost.id);
          }
        }, { rootMargin: '-10% 0px -70% 0px', threshold: 0 });

        headingEls.forEach(function (h) { observer.observe(h.el); });

        /* when at top of page, highlight parent */
        window.addEventListener('scroll', function () {
          if (scrollSuppressed) return;
          if (window.scrollY < 80) {
            highlightAnchor(null);
          }
        }, { passive: true });

        /* suppress scroll spy briefly when clicking a sub-link */
        toc.addEventListener('click', function (e) {
          var link = e.target.closest('a.sub');
          if (!link) return;
          scrollSuppressed = true;
          var h = link.getAttribute('href') || '';
          var idx = h.indexOf('#');
          if (idx !== -1) highlightAnchor(h.substring(idx + 1));
          setTimeout(function () { scrollSuppressed = false; }, 800);
        });
      }
    }
  }
});

/* ──────────────────────────────────────────────────────────────────────
   Smart API base-URL rewrite.
   Works for:
     - local dev   : http://localhost:8502/docs/...     -> origin
     - cloudflare  : https://debris-api.he-ting.com/... -> origin
     - streamlit   : http://localhost:8501/...          -> :8502 sibling
   This rewrites every <code>/<pre> placeholder like
   "http://<your-host>:8000", "http://<your-host>:8502", and
   "http://localhost:850(0|2)" to the current origin so users can copy-paste
   the curl examples no matter how they accessed the docs.
   It also fixes the Swagger / ReDoc launcher buttons on api.html that were
   hard-coded to ":8000".
   ────────────────────────────────────────────────────────────────────── */
(function () {
  function apiBase() {
    var loc = window.location;
    if (loc.port === "8501") {
      return loc.protocol + "//" + loc.hostname + ":8502";
    }
    return loc.origin;
  }

  document.addEventListener('DOMContentLoaded', function () {
    var base = apiBase();

    var bu = document.getElementById('base-url');
    if (bu) bu.textContent = base;

    var patterns = [
      /http:\/\/&lt;your-host&gt;:(?:8000|8502)/g,
      /http:\/\/&lt;your-host&gt;/g,
      /http:\/\/localhost:(?:8000|8502)/g,
      /http:\/\/127\.0\.0\.1:(?:8000|8502)/g
    ];
    document.querySelectorAll('pre, code').forEach(function (el) {
      var html = el.innerHTML;
      var changed = false;
      patterns.forEach(function (p) {
        if (p.test(html)) { html = html.replace(p, base); changed = true; }
      });
      if (changed) el.innerHTML = html;
    });

    document.querySelectorAll(
      'a[onclick*=":8502/api/"], a[onclick*=":8000/api/"]'
    ).forEach(function (a) {
      var m = (a.getAttribute('onclick') || '').match(/\/api\/(docs|redoc)/);
      if (!m) return;
      a.setAttribute('href', base + '/api/' + m[1]);
      a.removeAttribute('onclick');
      a.setAttribute('target', '_blank');
    });
  });
})();
