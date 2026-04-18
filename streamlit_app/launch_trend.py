"""Historical launch trend charts for the space debris monitoring dashboard.

Data sources (priority order):
  1. Jonathan McDowell GCAT  (external_country_yearly_payload / external_cumulative_onorbit)
  2. UNOOSA / Our World in Data  (external_unoosa_launches) — cross-reference
  3. UCS Satellite Database  (external_ucs_satellites) — satellite purpose analytics
  4. ESA DISCOS  (external_esa_discos) — mass/cross-section analytics
  5. Space-Track catalog_objects — fallback

Provides public functions:
  - load_launch_history()        → dict of DataFrames for all charts
  - make_annual_launch_fig(df)   → Plotly stacked-bar: launches per year by region
  - make_cumulative_fig(df)      → Plotly line: cumulative on-orbit objects over time
  - make_decade_summary_fig(df)  → Plotly stacked-bar: by-decade summary
  - make_country_trend_fig(df)   → Plotly line: country trends
  - make_recent_country_bar(df)  → Plotly horizontal bar
  - make_ucs_purpose_fig(df)     → Plotly pie/bar: satellite purpose distribution
  - make_unoosa_comparison_fig() → Plotly overlay: GCAT vs UNOOSA annual comparison
"""
from __future__ import annotations

import logging
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import text

log = logging.getLogger(__name__)

# ── Country / region grouping ─────────────────────────────────────────────────
_REGION_MAP: dict[str, str] = {
    "US":   "美国",
    "USA":  "美国",
    "CIS":  "俄罗斯/苏联",
    "SU":   "俄罗斯/苏联",
    "RU":   "俄罗斯/苏联",
    "PRC":  "中国",
    "CN":   "中国",
    "China":"中国",
    "UK":   "欧洲",
    "FR":   "欧洲",
    "ESA":  "欧洲",
    "EUTE": "欧洲",
    "D":    "欧洲",        # Germany
    "I":    "欧洲",        # Italy
    "NL":   "欧洲",
    "JPN":  "日本",
    "IND":  "印度",
}
_REGION_COLORS = {
    "美国":        "#3B82F6",
    "俄罗斯/苏联": "#EF4444",
    "中国":        "#F59E0B",
    "欧洲":        "#10B981",
    "日本":        "#8B5CF6",
    "印度":        "#EC4899",
    "其他":        "#94A3B8",
}
_REGION_ORDER = ["美国", "俄罗斯/苏联", "中国", "欧洲", "日本", "印度", "其他"]


# ── SQL helpers ───────────────────────────────────────────────────────────────

_SQL_ANNUAL_LAUNCHES = """
SELECT
    EXTRACT(YEAR FROM launch_date)::int AS yr,
    country_code,
    object_type,
    COUNT(*) AS n
FROM v_unified_objects
WHERE launch_date IS NOT NULL
  AND EXTRACT(YEAR FROM launch_date) BETWEEN 1957 AND EXTRACT(YEAR FROM NOW())
GROUP BY yr, country_code, object_type
ORDER BY yr, n DESC
"""

_SQL_ANNUAL_OBJECTS = """
WITH launch_by_yr AS (
    SELECT EXTRACT(YEAR FROM launch_date)::int AS yr,
           object_type, COUNT(*) AS n
    FROM v_unified_objects
    WHERE launch_date IS NOT NULL
    GROUP BY yr, object_type
),
decay_by_yr AS (
    SELECT EXTRACT(YEAR FROM decay_date)::int AS yr,
           object_type, COUNT(*) AS n
    FROM v_unified_objects
    WHERE decay_date IS NOT NULL
    GROUP BY yr, object_type
),
yrs AS (
    SELECT generate_series(1957, EXTRACT(YEAR FROM NOW())::int) AS yr
)
SELECT
    y.yr,
    COALESCE(SUM(l.n) FILTER (WHERE l.object_type = 'PAYLOAD'),      0)::int AS payload_launch,
    COALESCE(SUM(l.n) FILTER (WHERE l.object_type = 'DEBRIS'),       0)::int AS debris_launch,
    COALESCE(SUM(l.n) FILTER (WHERE l.object_type = 'ROCKET BODY'),  0)::int AS rocket_launch,
    COALESCE(SUM(d.n) FILTER (WHERE d.object_type = 'PAYLOAD'),      0)::int AS payload_decay,
    COALESCE(SUM(d.n) FILTER (WHERE d.object_type = 'DEBRIS'),       0)::int AS debris_decay,
    COALESCE(SUM(d.n) FILTER (WHERE d.object_type = 'ROCKET BODY'),  0)::int AS rocket_decay
FROM yrs y
LEFT JOIN launch_by_yr l ON l.yr = y.yr
LEFT JOIN decay_by_yr  d ON d.yr = y.yr
GROUP BY y.yr
ORDER BY y.yr
"""


# ── Data loading ──────────────────────────────────────────────────────────────

def _try_load_external(sess) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Try loading GCAT-McDowell external data if table exists."""
    try:
        ext_payload = pd.DataFrame(
            sess.execute(text(
                "SELECT launch_year AS yr, country_code, count AS n "
                "FROM external_country_yearly_payload ORDER BY launch_year"
            )).fetchall(),
            columns=["yr", "country_code", "n"],
        )
        ext_cum = pd.DataFrame(
            sess.execute(text(
                "SELECT year AS yr, object_type, on_orbit "
                "FROM external_cumulative_onorbit ORDER BY year"
            )).fetchall(),
            columns=["yr", "object_type", "on_orbit"],
        )
        return ext_payload, ext_cum
    except Exception:
        sess.rollback()
        return pd.DataFrame(), pd.DataFrame()


def _try_load_unoosa(sess) -> pd.DataFrame:
    """Load UNOOSA/OWID annual launch totals for cross-validation."""
    try:
        df = pd.DataFrame(
            sess.execute(text(
                "SELECT entity, year AS yr, annual_launches AS n "
                "FROM external_unoosa_launches "
                "WHERE entity = 'World' ORDER BY year"
            )).fetchall(),
            columns=["entity", "yr", "n"],
        )
        return df
    except Exception:
        sess.rollback()
        return pd.DataFrame()


def _try_load_ucs(sess) -> pd.DataFrame:
    """Load UCS satellite database for purpose/usage analytics."""
    try:
        df = pd.DataFrame(
            sess.execute(text(
                "SELECT name, country, users, purpose, detailed_purpose, "
                "       orbit_class, perigee_km, apogee_km, inclination_deg, "
                "       launch_mass_kg, launch_date, expected_lifetime_yr "
                "FROM external_ucs_satellites"
            )).fetchall(),
            columns=["name", "country", "users", "purpose", "detailed_purpose",
                     "orbit_class", "perigee_km", "apogee_km", "inclination_deg",
                     "launch_mass_kg", "launch_date", "expected_lifetime_yr"],
        )
        return df
    except Exception:
        sess.rollback()
        return pd.DataFrame()


def _try_load_esa(sess) -> pd.DataFrame:
    """Load ESA DISCOS objects for mass/cross-section analytics."""
    try:
        df = pd.DataFrame(
            sess.execute(text(
                "SELECT name, \"objectClass\", mass, \"xSectAvg\", "
                "       \"firstEpoch\", mission, active "
                "FROM external_esa_discos"
            )).fetchall(),
            columns=["name", "objectClass", "mass", "xSectAvg",
                     "firstEpoch", "mission", "active"],
        )
        return df
    except Exception:
        sess.rollback()
        return pd.DataFrame()


def load_launch_history() -> dict[str, pd.DataFrame]:
    """Load all historical launch / on-orbit data from the database.

    Prefers external GCAT-McDowell data (68k+ objects, 1957–present) over
    Space-Track catalog_objects when available (more complete history).

    Returns a dict with keys:
      'annual_by_region' – per-year per-region payload launches (pivot-ready)
      'cumulative'  – per-year cumulative on-orbit object counts
      'source' – data source label
    """
    from database.db import session_scope

    result: dict[str, pd.DataFrame] = {}
    try:
        with session_scope() as sess:
            ext_payload, ext_cum = _try_load_external(sess)

            if not ext_payload.empty:
                log.info("Using GCAT-McDowell external data (%d rows)", len(ext_payload))
                raw_payload = ext_payload
                result["source"] = "GCAT (Jonathan McDowell)"
            else:
                raw = pd.DataFrame(
                    sess.execute(text(_SQL_ANNUAL_LAUNCHES)).fetchall(),
                    columns=["yr", "country_code", "object_type", "n"],
                )
                if raw.empty:
                    return {}
                raw_payload = raw[raw["object_type"] == "PAYLOAD"][
                    ["yr", "country_code", "n"]
                ].copy()
                result["source"] = "Space-Track"

            # Cumulative on-orbit
            if not ext_cum.empty:
                pivot = ext_cum.pivot_table(
                    index="yr", columns="object_type", values="on_orbit",
                    aggfunc="sum", fill_value=0,
                ).reset_index()
                pivot.columns.name = None
                for col, alias in [("PAYLOAD", "payload_onorbit"),
                                    ("DEBRIS", "debris_onorbit"),
                                    ("ROCKET BODY", "rocket_onorbit")]:
                    if col in pivot.columns:
                        pivot.rename(columns={col: alias}, inplace=True)
                    else:
                        pivot[alias] = 0
                pivot["total_onorbit"] = (
                    pivot["payload_onorbit"]
                    + pivot["debris_onorbit"]
                    + pivot["rocket_onorbit"]
                )
                result["cumulative"] = pivot
            else:
                cum_raw = pd.DataFrame(
                    sess.execute(text(_SQL_ANNUAL_OBJECTS)).fetchall(),
                    columns=["yr",
                             "payload_launch", "debris_launch", "rocket_launch",
                             "payload_decay",  "debris_decay",  "rocket_decay"],
                )
                if not cum_raw.empty:
                    cum_raw = cum_raw.sort_values("yr")
                    cum_raw["payload_onorbit"] = (
                        cum_raw["payload_launch"].cumsum()
                        - cum_raw["payload_decay"].cumsum()
                    )
                    cum_raw["debris_onorbit"] = (
                        cum_raw["debris_launch"].cumsum()
                        - cum_raw["debris_decay"].cumsum()
                    )
                    cum_raw["rocket_onorbit"] = (
                        cum_raw["rocket_launch"].cumsum()
                        - cum_raw["rocket_decay"].cumsum()
                    )
                    cum_raw["total_onorbit"] = (
                        cum_raw["payload_onorbit"]
                        + cum_raw["debris_onorbit"]
                        + cum_raw["rocket_onorbit"]
                    )
                    result["cumulative"] = cum_raw

    except Exception as exc:
        log.warning("launch_trend DB error: %s", exc)
        return {}

    if raw_payload.empty:
        return {}

    # Map country codes → regions
    raw_payload["region"] = raw_payload["country_code"].map(_REGION_MAP).fillna("其他")
    by_region = (
        raw_payload.groupby(["yr", "region"], as_index=False)["n"].sum()
    )
    result["annual_by_region"] = by_region

    # ── Load supplementary data sources ───────────────────────────────────
    try:
        with session_scope() as sess:
            unoosa = _try_load_unoosa(sess)
            if not unoosa.empty:
                result["unoosa_world"] = unoosa
                log.info("Loaded %d UNOOSA/OWID world rows", len(unoosa))

            ucs = _try_load_ucs(sess)
            if not ucs.empty:
                result["ucs"] = ucs
                log.info("Loaded %d UCS satellite rows", len(ucs))

            esa = _try_load_esa(sess)
            if not esa.empty:
                result["esa"] = esa
                log.info("Loaded %d ESA DISCOS rows", len(esa))
    except Exception as exc:
        log.warning("supplementary data load error: %s", exc)

    return result


# ── Figure builders ───────────────────────────────────────────────────────────

def make_annual_launch_fig(by_region: pd.DataFrame, title: str = "",
                           start_year: int | None = None,
                           show_legend: bool = False) -> go.Figure:
    """Stacked-bar chart: payload launches per year coloured by region.

    If start_year is None, auto-pick last 15 full years to keep the chart
    readable (pre-2010 data is too sparse relative to the Starlink era).
    """
    if start_year is None:
        start_year = max(2010, pd.Timestamp.now().year - 15)

    sub = by_region[by_region["yr"] >= start_year]
    fig = go.Figure()
    pivot = sub.pivot_table(
        index="yr", columns="region", values="n", aggfunc="sum", fill_value=0
    )
    cols = [r for r in _REGION_ORDER if r in pivot.columns]
    for region in cols:
        fig.add_trace(go.Bar(
            x=pivot.index,
            y=pivot[region],
            name=region,
            marker_color=_REGION_COLORS.get(region, "#94A3B8"),
            hovertemplate="%{x}年<br>" + region + ": %{y:,}颗<extra></extra>",
            showlegend=show_legend,
        ))
    fig.update_layout(
        barmode="stack",
        title=dict(text=title, font=dict(size=13)) if title else {},
        xaxis=dict(title="年份", tickmode="linear", dtick=2),
        yaxis=dict(title="发射数量（颗）", rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(size=11)),
        template="plotly_white",
        height=340,
        margin=dict(t=40, b=40, l=50, r=10),
        plot_bgcolor="#FAFAFA",
    )
    return fig


def make_decade_summary_fig(by_region: pd.DataFrame) -> go.Figure:
    """Grouped-bar chart: total launches aggregated by decade and region."""
    df = by_region.copy()
    now_yr = pd.Timestamp.now().year
    def _decade_label(yr: int) -> str:
        if yr < 1970:
            return "1957–69"
        if yr < 1980:
            return "1970s"
        if yr < 1990:
            return "1980s"
        if yr < 2000:
            return "1990s"
        if yr < 2010:
            return "2000s"
        if yr < 2020:
            return "2010s"
        return f"2020–{now_yr % 100:02d}"

    df["decade"] = df["yr"].apply(_decade_label)
    _decade_order = ["1957–69", "1970s", "1980s", "1990s", "2000s", "2010s",
                     f"2020–{now_yr % 100:02d}"]
    agg = df.groupby(["decade", "region"], as_index=False)["n"].sum()

    fig = go.Figure()
    for region in _REGION_ORDER:
        sub = agg[agg["region"] == region]
        if sub.empty or sub["n"].sum() == 0:
            continue
        ordered = []
        for d in _decade_order:
            row = sub[sub["decade"] == d]
            ordered.append(int(row["n"].iloc[0]) if not row.empty else 0)
        fig.add_trace(go.Bar(
            x=_decade_order,
            y=ordered,
            name=region,
            marker_color=_REGION_COLORS.get(region, "#94A3B8"),
            hovertemplate="%{x}<br>" + region + ": %{y:,}颗<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        xaxis=dict(title="时期", categoryorder="array",
                   categoryarray=_decade_order),
        yaxis=dict(title="发射总量（颗）", rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(size=11)),
        template="plotly_white",
        height=340,
        margin=dict(t=40, b=40, l=50, r=10),
        plot_bgcolor="#FAFAFA",
    )
    return fig


def make_cumulative_fig(cumulative: pd.DataFrame) -> go.Figure:
    """Line chart: cumulative on-orbit objects by type."""
    fig = go.Figure()
    series = [
        ("payload_onorbit", "有效载荷", "#3B82F6"),
        ("debris_onorbit",  "空间碎片", "#EF4444"),
        ("rocket_onorbit",  "火箭箭体", "#F59E0B"),
        ("total_onorbit",   "总计",     "#1E293B"),
    ]
    for col, name, color in series:
        if col not in cumulative.columns:
            continue
        dash = "dot" if name != "总计" else "solid"
        width = 3 if name == "总计" else 1.5
        fig.add_trace(go.Scatter(
            x=cumulative["yr"],
            y=cumulative[col].clip(lower=0),
            mode="lines",
            name=name,
            line=dict(color=color, width=width, dash=dash),
            hovertemplate="%{x}年<br>" + name + ": %{y:,}个<extra></extra>",
        ))
    fig.update_layout(
        xaxis=dict(title="年份", tickmode="linear", dtick=5,
                   range=[1956, pd.Timestamp.now().year + 1]),
        yaxis=dict(title="在轨数量（估算）", rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(size=11)),
        template="plotly_white",
        height=340,
        margin=dict(t=40, b=40, l=60, r=10),
        plot_bgcolor="#FAFAFA",
    )
    return fig


def make_country_trend_fig(by_region: pd.DataFrame,
                           start_year: int = 2000) -> go.Figure:
    """Line chart: payload launches per year per region (from start_year)."""
    sub = by_region[by_region["yr"] >= start_year]
    fig = go.Figure()
    for region in _REGION_ORDER:
        df_r = sub[sub["region"] == region]
        if df_r.empty or df_r["n"].sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=df_r["yr"],
            y=df_r["n"],
            mode="lines+markers",
            name=region,
            line=dict(color=_REGION_COLORS.get(region, "#94A3B8"), width=2),
            marker=dict(size=5),
            hovertemplate="%{x}年<br>" + region + ": %{y:,}颗<extra></extra>",
        ))
    fig.update_layout(
        xaxis=dict(title="年份", tickmode="linear", dtick=2),
        yaxis=dict(title="发射数量（颗）", rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(size=11)),
        template="plotly_white",
        height=320,
        margin=dict(t=40, b=40, l=50, r=10),
        plot_bgcolor="#FAFAFA",
    )
    return fig


def make_unoosa_comparison_fig(by_region: pd.DataFrame,
                                unoosa_world: pd.DataFrame) -> go.Figure:
    """Overlay: GCAT total vs UNOOSA 'World' annual launches for cross-validation."""
    gcat_total = by_region.groupby("yr")["n"].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gcat_total["yr"], y=gcat_total["n"],
        mode="lines+markers", name="GCAT (McDowell)",
        line=dict(color="#3B82F6", width=2),
        marker=dict(size=4),
    ))
    if not unoosa_world.empty:
        fig.add_trace(go.Scatter(
            x=unoosa_world["yr"], y=unoosa_world["n"],
            mode="lines+markers", name="UNOOSA / OWID",
            line=dict(color="#EF4444", width=2, dash="dot"),
            marker=dict(size=4),
        ))
    fig.update_layout(
        xaxis=dict(title="年份", tickmode="linear", dtick=5),
        yaxis=dict(title="年度发射数量", rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(size=11)),
        template="plotly_white", height=320,
        margin=dict(t=40, b=40, l=50, r=10),
        plot_bgcolor="#FAFAFA",
    )
    return fig


def make_ucs_purpose_fig(ucs: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart: satellite count by purpose from UCS database."""
    if ucs.empty or "purpose" not in ucs.columns:
        return go.Figure()
    purpose_counts = ucs["purpose"].dropna().value_counts().head(12)
    purpose_counts = purpose_counts.sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        x=purpose_counts.values,
        y=purpose_counts.index,
        orientation="h",
        marker_color="#3B82F6",
        text=purpose_counts.values,
        textposition="outside",
    ))
    fig.update_layout(
        xaxis=dict(title="卫星数量", rangemode="tozero"),
        yaxis=dict(title=""),
        template="plotly_white", height=360,
        margin=dict(t=20, b=40, l=140, r=60),
        plot_bgcolor="#FAFAFA",
    )
    return fig


def make_ucs_users_fig(ucs: pd.DataFrame) -> go.Figure:
    """Pie chart: satellite usage type (military/civil/commercial) from UCS."""
    if ucs.empty or "users" not in ucs.columns:
        return go.Figure()
    users_counts = ucs["users"].dropna().value_counts()
    _MERGE = {
        "Commercial": "商业",
        "Government": "政府",
        "Military": "军事",
        "Civil": "民用",
    }
    labels, values = [], []
    other = 0
    for lbl, cnt in users_counts.items():
        mapped = _MERGE.get(str(lbl))
        if mapped:
            labels.append(mapped)
            values.append(int(cnt))
        else:
            other += int(cnt)
    if other > 0:
        labels.append("其他/混合")
        values.append(other)
    total = sum(values)
    positions = ["outside" if v / total < 0.05 else "inside" for v in values]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=["#3B82F6", "#10B981", "#EF4444",
                             "#F59E0B", "#94A3B8"]),
        textinfo="label+value+percent",
        textfont=dict(size=12),
        textposition=positions,
        insidetextorientation="horizontal",
    ))
    fig.update_layout(
        template="plotly_white", height=380,
        margin=dict(t=10, b=30, l=60, r=60),
        showlegend=False,
    )
    return fig


def make_ucs_orbit_fig(ucs: pd.DataFrame) -> go.Figure:
    """Bar chart: satellite count by orbit class from UCS database."""
    if ucs.empty or "orbit_class" not in ucs.columns:
        return go.Figure()
    _oc = ucs["orbit_class"].dropna().str.strip().str.upper()
    orbit_counts = _oc.value_counts()
    colors = {"LEO": "#FF6B6B", "MEO": "#6BCB77", "GEO": "#4D96FF",
              "ELLIPTICAL": "#FF9F45"}
    fig = go.Figure(go.Bar(
        x=orbit_counts.index,
        y=orbit_counts.values,
        marker_color=[colors.get(c, "#94A3B8") for c in orbit_counts.index],
        text=[f"{v:,}" for v in orbit_counts.values],
        textposition="outside",
    ))
    y_max = int(orbit_counts.max() * 1.07)
    fig.update_layout(
        xaxis=dict(title="轨道类别"),
        yaxis=dict(title="卫星数量", rangemode="tozero", range=[0, y_max]),
        template="plotly_white", height=340,
        margin=dict(t=30, b=40, l=50, r=10),
        plot_bgcolor="#FAFAFA",
    )
    return fig


def make_recent_country_bar(by_region: pd.DataFrame,
                             start_year: int = 2020) -> go.Figure:
    """Horizontal bar chart: total payload launches by region since start_year."""
    sub = by_region[by_region["yr"] >= start_year].groupby("region")["n"].sum().reset_index()
    sub = sub.sort_values("n", ascending=True)
    fig = go.Figure(go.Bar(
        x=sub["n"],
        y=sub["region"],
        orientation="h",
        marker_color=[_REGION_COLORS.get(r, "#94A3B8") for r in sub["region"]],
        text=sub["n"].apply(lambda v: f"{v:,}"),
        textposition="outside",
        hovertemplate="%{y}: %{x:,}颗<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title=f"{start_year}年以来发射量", rangemode="tozero"),
        yaxis=dict(title=""),
        template="plotly_white",
        height=300,
        margin=dict(t=20, b=40, l=80, r=60),
        plot_bgcolor="#FAFAFA",
    )
    return fig
