# -*- coding: utf-8 -*-
"""Minimal line-art SVG icons for navigation (research / technical style)."""
from __future__ import annotations

_STROKE = "currentColor"
_SW = "1.55"


def _svg(inner: str, *, vb: str = "0 0 24 24") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" '
        f'fill="none" stroke="{_STROKE}" stroke-width="{_SW}" '
        'stroke-linecap="round" stroke-linejoin="round">'
        f"{inner}</svg>"
    )


_ICONS: dict[str, str] = {
    "brand": _svg(
        '<circle cx="12" cy="12" r="2.8"/>'
        '<ellipse cx="12" cy="12" rx="10" ry="3.8" transform="rotate(-22 12 12)"/>'
        '<path d="M2 12h4M18 12h4M12 2v3M12 19v3"/>'
    ),
    "overview": _svg(
        '<circle cx="12" cy="12" r="8"/>'
        '<path d="M4 12h16M12 4a16 16 0 0 1 0 16"/>'
        '<path d="M12 4a16 16 0 0 0 0 16"/>'
    ),
    "viz": _svg(
        '<path d="M4 19V5M4 19h16"/>'
        '<circle cx="8" cy="14" r="1.6"/><circle cx="12" cy="9" r="1.6"/>'
        '<circle cx="16" cy="12" r="1.6"/><circle cx="10" cy="17" r="1.6"/>'
    ),
    "catalog": _svg(
        '<rect x="5" y="4" width="14" height="16" rx="1.5"/>'
        '<path d="M8 8h8M8 12h8M8 16h5"/>'
    ),
    "segments": _svg(
        '<path d="M3 17c3-8 6-12 9-12s6 4 9 12"/>'
        '<circle cx="6" cy="14" r="1.3"/><circle cx="12" cy="8" r="1.3"/>'
        '<circle cx="18" cy="14" r="1.3"/>'
    ),
    "sim": _svg(
        '<path d="M12 2l3 7h4l-6 4 2 8-3-5-3 5 2-8-6-4h4z"/>'
        '<path d="M12 22v-3"/>'
    ),
    "oem": _svg(
        '<path d="M6 4h12v16H6z"/><path d="M9 8h6M9 12h6M9 16h4"/>'
    ),
    "lcola": _svg(
        '<path d="M12 3L4 20h16z"/>'
        '<path d="M12 9v4M12 16h.01"/>'
    ),
    "collision": _svg(
        '<circle cx="9" cy="12" r="4"/><circle cx="15" cy="12" r="4"/>'
        '<path d="M11 10l6 4M17 10l-6 4"/>'
    ),
    "longterm": _svg(
        '<circle cx="12" cy="12" r="9"/>'
        '<path d="M12 6v6l4 2"/>'
        '<path d="M5.5 18.5l1.5-1.5M17 7l1.5-1.5M18.5 18.5L17 17M7 7L5.5 5.5"/>'
    ),
    "ai": _svg(
        '<rect x="5" y="7" width="14" height="10" rx="2"/>'
        '<path d="M9 7V5M15 7V5M9 17v2M15 17v2"/>'
        '<path d="M9 11h2M13 11h2M9 14h6"/>'
    ),
    # ── UI fragments (tabs / filters / actions) ─────────────────────────────
    "chart_bar": _svg(
        '<path d="M4 19V5M4 19h16"/>'
        '<path d="M7 15v-4M11 15V9M15 15v-7M19 15v-3"/>'
    ),
    "chart_line": _svg(
        '<path d="M4 19V5M4 19h16"/>'
        '<path d="M6 16l4-5 3 2 5-8 4 6"/>'
    ),
    "globe_meridians": _svg(
        '<circle cx="12" cy="12" r="8"/>'
        '<path d="M4 12h16M12 4c3 3 3 13 0 16M12 4c-3 3-3 13 0 16"/>'
    ),
    "download": _svg(
        '<path d="M12 3v10M8 11l4 4 4-4"/>'
        '<path d="M5 21h14"/>'
    ),
    "magnifier": _svg(
        '<circle cx="10" cy="10" r="6"/>'
        '<path d="M15 15l5 5"/>'
    ),
    "gear": _svg(
        '<circle cx="12" cy="12" r="3"/>'
        '<path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/>'
    ),
    "beaker": _svg(
        '<path d="M9 3h6M8 21h8l-1-9H9z"/>'
        '<path d="M10 12h4"/>'
    ),
    "play": _svg('<path d="M8 5v14l11-7z"/>'),
    "clipboard": _svg(
        '<path d="M9 4h6l1 2h3v14H5V6h3z"/>'
        '<path d="M9 10h6M9 14h4"/>'
    ),
    "wrench": _svg(
        '<path d="M14.7 6.3a4 4 0 0 0-5.6 5.6L4 17l2 2 5.1-5.1a4 4 0 0 0 5.6-5.6"/>'
    ),
    "idea": _svg(
        '<path d="M9 18h6M10 22h4"/>'
        '<path d="M12 2a5 5 0 0 0-3 9v2h6v-2a5 5 0 0 0-3-9z"/>'
    ),
    "trash": _svg(
        '<path d="M4 7h16M10 11v6M14 11v6M7 7l1 12h8l1-12"/>'
        '<path d="M9 7V5h6v2"/>'
    ),
    "folder_open": _svg(
        '<path d="M4 8h6l2-2h8v4H4z"/>'
        '<path d="M4 10v10h16V10"/>'
    ),
    "satellite_small": _svg(
        '<rect x="8" y="10" width="8" height="4" rx="0.5"/>'
        '<path d="M8 12H5M16 12h3M12 8V5M12 16v3"/>'
    ),
    "sliders": _svg(
        '<path d="M4 7h4M10 7h10M4 12h10M16 12h4M8 17h12"/>'
        f'<circle cx="7" cy="7" r="1.8" fill="{_STROKE}" stroke="none"/>'
        f'<circle cx="14" cy="12" r="1.8" fill="{_STROKE}" stroke="none"/>'
        f'<circle cx="18" cy="17" r="1.8" fill="{_STROKE}" stroke="none"/>'
    ),
    "refresh": _svg(
        '<path d="M21 12a9 9 0 1 1-2.64-6.36"/>'
        '<path d="M21 3v7h-7"/>'
    ),
    "globe_flat": _svg(
        '<circle cx="12" cy="12" r="8"/>'
        '<path d="M4 12h16"/>'
    ),
    "layers": _svg(
        '<path d="M12 2L3 7l9 5 9-5-9-5z"/>'
        '<path d="M3 12l9 5 9-5"/>'
        '<path d="M3 17l9 5 9-5"/>'
    ),
    "timeline": _svg(
        '<path d="M4 19h16"/>'
        '<circle cx="8" cy="19" r="2"/><circle cx="16" cy="19" r="2"/>'
        '<path d="M8 19V9M16 19V5"/>'
    ),
    "rocket_tab": _svg(
        '<path d="M12 3l2 6h3l-4 3 1 7-2-4-2 4 1-7-4-3h3z"/>'
    ),
    "check": _svg(
        '<path d="M4 12l4 4L20 6"/>'
    ),
    "ban": _svg(
        '<circle cx="12" cy="12" r="8"/>'
        '<path d="M5 5l14 14"/>'
    ),
    "docs": _svg(
        '<path d="M4 4h10l4 4v12H4z"/>'
        '<path d="M14 4v4h4"/>'
        '<path d="M7 13h8M7 16h5M7 10h3"/>'
    ),
}

_RISK_FILL = {
    "RED": "#dc2626",
    "AMBER": "#ea580c",
    "YELLOW": "#ca8a04",
    "GREEN": "#15803d",
}


def icon_inline(name: str, size: int = 22) -> str:
    """Return HTML span wrapping a sized SVG (for st.markdown(..., unsafe_allow_html=True))."""
    body = _ICONS.get(name) or _ICONS["brand"]
    return (
        f'<span style="display:inline-flex;align-items:center;justify-content:center;'
        f'width:{size}px;height:{size}px;flex-shrink:0">{body}</span>'
    )


def title_row(icon_name: str, title_text: str, *, icon_size: int = 34) -> str:
    """Single h1 row: icon + title (serif-free, compact).

    Uses color:inherit so the text is readable in both Streamlit light and dark mode.
    """
    ic = icon_inline(icon_name, size=icon_size)
    return (
        '<h1 style="display:flex;align-items:center;gap:14px;font-size:1.85rem;'
        'font-weight:600;margin:0.15em 0 0.35em 0;color:inherit;letter-spacing:0.02em">'
        f"{ic}<span>{title_text}</span></h1>"
    )


def sidebar_brand_row() -> str:
    ic = icon_inline("brand", size=28)
    return (
        '<div style="display:flex;align-items:center;gap:10px;margin:0 0 14px 0;padding:4px 0">'
        f"{ic}"
        '<span style="font-size:1.05rem;font-weight:600;color:inherit">空间碎片监测系统</span>'
        "</div>"
    )


def section_title(icon_name: str, text: str, *, level: int = 3, icon_size: int = 22) -> str:
    """Section heading (h2–h4) with inline SVG + text.

    Uses color:inherit for dark-mode compatibility.
    """
    lv = min(max(level, 2), 4)
    tag = f"h{lv}"
    sz = {2: "1.35rem", 3: "1.12rem", 4: "1.02rem"}.get(lv, "1.12rem")
    ic = icon_inline(icon_name, size=icon_size)
    return (
        f'<{tag} style="display:flex;align-items:center;gap:10px;font-size:{sz};'
        'font-weight:600;color:inherit;margin:0.85em 0 0.45em 0">'
        f"{ic}<span>{text}</span></{tag}>"
    )


def risk_dot_html(level: str, *, size: int = 11) -> str:
    """Filled circle for risk level (use inside st.markdown, unsafe_allow_html=True)."""
    c = _RISK_FILL.get(level, "#64748b")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        'viewBox="0 0 12 12" style="vertical-align:-2px;display:inline-block;margin-right:4px">'
        f'<circle cx="6" cy="6" r="5" fill="{c}"/></svg>'
    )


# Streamlit 默认 primary 常为红/品红 — 全站统一为科研蓝（侧栏导航 + 可视化子标签等）
SIDEBAR_NAV_BLUE_CSS = """
<style>
button[kind="primary"] {
  background-color: #1d4ed8 !important;
  border: 1px solid #1e40af !important;
  color: #ffffff !important;
}
button[kind="primary"]:hover {
  background-color: #1e3a8a !important;
  border-color: #172554 !important;
  color: #ffffff !important;
}
button[kind="primary"]:focus-visible {
  box-shadow: 0 0 0 2px #bfdbfe !important;
}
</style>
"""
