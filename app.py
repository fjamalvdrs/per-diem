"""
app.py  —  Per Diem Cost Estimator (v4)
=======================================
Streamlit app — overnight_lookup.json + rate_table_v4.json.
Estimates hotel + meals + local transport per technician per day.
Supplies/tools are always pass-through actuals and NOT included.
"""

import streamlit as st
import pandas as pd
import json
import os
from math import radians, sin, cos, sqrt, atan2
from datetime import date

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IRS_RATE = 0.725  # $/mile (IRS 2025 rate)

CANADIAN_PROVINCES = {"ON","BC","AB","QC","MB","SK","NS","NB","PE","NL","NT","YT","NU"}

STATES_LIST = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC",
    "AB","BC","MB","NB","NL","NS","NT","NU","ON","PE","QC","SK","YT",
]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Per Diem Estimator",
    page_icon="✈️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Global CSS — injected once at the top of main()
# Every text/bg color is explicit so Streamlit dark/light mode can't override.
# ---------------------------------------------------------------------------
_GLOBAL_CSS = """
<style>
/* ── Table base ─────────────────────────────────────── */
.pd-table {
    width: 100%;
    border-collapse: collapse;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 14px;
    margin-bottom: 4px;
    border-radius: 8px;
    overflow: hidden;
}
.pd-table th {
    background-color: #1e3a5f !important;
    color: #ffffff !important;
    padding: 10px 14px !important;
    text-align: left !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    border: none !important;
}
.pd-table td {
    background-color: #ffffff !important;
    color: #111827 !important;
    padding: 8px 14px !important;
    border: none !important;
    border-bottom: 1px solid #e5e7eb !important;
    vertical-align: middle !important;
}
.pd-table tr:nth-child(even) td {
    background-color: #f9fafb !important;
    color: #111827 !important;
}
.pd-table .total-row td {
    background-color: #eff6ff !important;
    color: #1e3a5f !important;
    font-weight: 700 !important;
    border-top: 2px solid #bfdbfe !important;
    border-bottom: none !important;
}
.pd-table .val-cell {
    text-align: right !important;
    font-variant-numeric: tabular-nums !important;
    white-space: nowrap !important;
}
.pd-table .detail-cell {
    color: #6b7280 !important;
    font-size: 12px !important;
}
.pd-table .total-row .detail-cell {
    color: #3b82f6 !important;
    font-size: 12px !important;
}

/* ── Route info card ────────────────────────────────── */
.route-card {
    background-color: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 8px;
    padding: 12px 18px;
    margin: 10px 0 14px 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 14px;
    color: #0c4a6e;
    display: flex;
    flex-wrap: wrap;
    gap: 6px 0;
}
.route-card .rc-item {
    display: inline-block;
    margin-right: 0;
}
.route-card .rc-sep {
    color: #7dd3fc;
    margin: 0 10px;
    font-weight: 300;
}
.route-card .rc-label {
    color: #64748b;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: block;
    margin-bottom: 1px;
}
.route-card .rc-val {
    color: #0c4a6e;
    font-weight: 600;
    font-size: 15px;
}

/* ── Trip type pill ─────────────────────────────────── */
.trip-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 999px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 8px;
}
.trip-pill-high    { background:#dcfce7; color:#166534; border:1px solid #86efac; }
.trip-pill-medium  { background:#fef9c3; color:#854d0e; border:1px solid #fde047; }
.trip-pill-low     { background:#f1f5f9; color:#475569; border:1px solid #cbd5e1; }

/* ── Confidence badge ───────────────────────────────── */
.conf-badge {
    border-radius: 4px;
    padding: 1px 7px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin-left: 6px;
    vertical-align: middle;
}
.conf-high   { background:#dcfce7 !important; color:#166534 !important; border:1px solid #86efac; }
.conf-medium { background:#fef9c3 !important; color:#854d0e !important; border:1px solid #fde047; }
.conf-low    { background:#ffedd5 !important; color:#9a3412 !important; border:1px solid #fdba74; }

/* ── Total box ──────────────────────────────────────── */
.total-box {
    background-color: #1e3a5f;
    border-radius: 10px;
    padding: 20px 24px;
    margin: 16px 0 8px 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.total-box .tb-label {
    color: #93c5fd;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.total-box .tb-amount {
    color: #ffffff;
    font-size: 34px;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 4px;
}
.total-box .tb-breakdown {
    color: #bfdbfe;
    font-size: 13px;
}
.total-box .tb-range {
    color: #7dd3fc;
    font-size: 12px;
    margin-top: 6px;
}

/* ── Section label ──────────────────────────────────── */
.section-label {
    color: #6b7280;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin: 18px 0 6px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #e5e7eb;
}
</style>
"""

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def get_hotel_geo_mult(geo_index: dict, cust_id=None, city=None, state=None) -> tuple:
    """
    Return (multiplier, display_label, gsa_rate) for a destination.
    multiplier > 1.0 = expensive area, < 1.0 = cheaper area.
    gsa_rate = raw GSA lodging per diem for the destination (used as hotel floor).
    """
    CANADIAN_PROVINCES = {"ON","BC","AB","QC","MB","SK","NS","NB","PE","NL","NT","YT","NU"}
    if state and state.upper() in CANADIAN_PROVINCES:
        return 1.0, "", None  # no GSA floor for international

    by_cust = geo_index.get("by_customer", {})
    if cust_id and str(cust_id) in by_cust:
        entry    = by_cust[str(cust_id)]
        mult     = float(entry.get("multiplier", 1.0))
        gsa_rate = entry.get("gsa_rate")
        src      = entry.get("source", "")
        if src in ("city_exact", "city_partial"):
            label = f"{entry.get('city','')}, {entry.get('state','')} · {mult:.2f}×"
        elif src == "state_avg":
            label = f"{entry.get('state','')} avg · {mult:.2f}×"
        else:
            label = ""
        return mult, label, gsa_rate

    by_city = geo_index.get("by_city_state", {})
    if city and state:
        key = f"{city.lower().strip()}_{state.upper().strip()}"
        if key in by_city:
            entry    = by_city[key]
            mult     = float(entry["multiplier"])
            gsa_rate = entry.get("gsa_rate")
            return mult, f"{city}, {state} · {mult:.2f}×", gsa_rate

    by_state = geo_index.get("by_state", {})
    if state and state.upper() in by_state:
        entry    = by_state[state.upper()]
        mult     = float(entry["multiplier"])
        gsa_rate = entry.get("avg_rate")
        return mult, f"{state} avg · {mult:.2f}×", gsa_rate

    # Unknown US location — floor at national average
    national_avg = geo_index.get("national_avg_lodging", 145.18)
    return 1.0, "", national_avg


def apply_geo_correction(cell: dict, geo_mult: float, geo_label: str = "",
                         gsa_rate: float = None) -> dict:
    """
    Return a new cell dict with hotel_rate adjusted for location and floored at GSA rate.

    Two adjustments applied in order:
      1. Geo-scale: hotel_base × geo_mult  (location cost adjustment)
      2. GSA floor: max(geo-scaled, gsa_rate)  (guards against company-card data gap)

    Only hotel is adjusted — meals and transport are not location-sensitive enough.
    """
    if not cell:
        return cell
    # Skip only when no geo data AND no GSA rate to floor against
    if abs(geo_mult - 1.0) < 0.001 and not gsa_rate:
        return cell

    c = dict(cell)
    hotel_base = c.get("hotel_rate") or 0
    hotel_hist = round(hotel_base * geo_mult, 2)      # geo-adjusted historical
    gsa_floor  = round(float(gsa_rate), 2) if (gsa_rate and gsa_rate > 0) else 0

    hotel_final      = max(hotel_hist, gsa_floor)
    used_gsa_floor   = bool(gsa_floor > 0 and hotel_final > hotel_hist + 0.50)
    delta            = hotel_final - hotel_base

    c["hotel_rate"]            = round(hotel_final, 2)
    c["hotel_rate_base"]       = round(hotel_base, 2)
    c["hotel_rate_historical"] = hotel_hist
    c["hotel_rate_gsa"]        = gsa_floor if gsa_floor > 0 else None
    c["hotel_used_gsa_floor"]  = used_gsa_floor
    c["hotel_geo_mult"]        = round(geo_mult, 4)
    c["hotel_geo_label"]       = geo_label
    c["total_rate"]            = round(
        hotel_final
        + (c.get("meals_rate") or 0)
        + (c.get("transport_rate") or 0),
        2,
    )
    # Shift p25/p75 by the full hotel delta (preserves width of range)
    if c.get("total_p25") is not None:
        c["total_p25"] = round(c["total_p25"] + delta, 2)
    if c.get("total_p75") is not None:
        c["total_p75"] = round(c["total_p75"] + delta, 2)
    return c


def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def distance_band(miles):
    if miles < 300:  return "Under 300"
    if miles < 500:  return "300-500"
    if miles < 750:  return "500-750"
    if miles < 1000: return "750-1000"
    if miles < 1500: return "1000-1500"
    if miles < 2000: return "1500-2000"
    return "2000+"

def season(month):
    if month in (3,4,5):    return "Spring"
    if month in (6,7,8):    return "Summer"
    if month in (9,10,11):  return "Fall"
    return "Winter"

def map_job_type(ui_label):
    if "Installation" in ui_label or "Startup" in ui_label:
        return "Installation"
    return "Service"

def _fmt(val, prefix="$"):
    if val is None: return "—"
    return f"{prefix}{val:,.0f}"

def _fine_band(miles):
    if miles is None: return None
    if miles < 100:   return "Under 100"
    if miles < 300:   return "100-300"
    return None

def rate_confidence(cell):
    if not cell: return "LOW"
    basis = cell.get("basis", "")
    n = cell.get("n_clean", cell.get("n", 0))
    if "Customer" in basis and cell.get("n", 0) >= 10: return "HIGH"
    if "Customer" in basis and cell.get("n", 0) >= 3:  return "MEDIUM"
    if n >= 30 and "global" not in basis.lower():       return "HIGH"
    if n >= 10:                                          return "MEDIUM"
    return "LOW"

def _conf_badge_html(level):
    css = {"HIGH": "conf-high", "MEDIUM": "conf-medium", "LOW": "conf-low"}
    return f'<span class="conf-badge {css.get(level,"conf-low")}">{level}</span>'

# ── Confidence-based billing guidance ────────────────────────────────────────
_BUFFER_RULES = {
    "HIGH":   {"pct": 0,  "action": "Use estimate directly — no buffer needed."},
    "MEDIUM": {"pct": 15, "action": "Add 15% buffer before invoicing — regional data, not customer-specific."},
    "LOW":    {"pct": 25, "action": "Add 25% buffer before invoicing — sparse data. Consider a custom quote."},
}
_CONF_COLORS = {
    "HIGH":   {"bg": "rgba(34,197,94,0.10)",  "border": "rgba(34,197,94,0.4)",  "icon": "🟢", "tc": "#86efac"},
    "MEDIUM": {"bg": "rgba(234,179,8,0.10)",  "border": "rgba(234,179,8,0.4)",  "icon": "🟡", "tc": "#fde047"},
    "LOW":    {"bg": "rgba(239,68,68,0.10)",  "border": "rgba(239,68,68,0.4)",  "icon": "🔴", "tc": "#f87171"},
}

def _basis_plain(cell, cust_name, band, seas, is_fly):
    """Translate the internal lookup basis code into a plain-English sentence."""
    if not cell:
        return "no data — estimate unavailable"
    basis = cell.get("basis", "")
    n     = cell.get("n", 0)
    mode  = "flying" if is_fly else "driving"
    if "Customer" in basis and "seasonal" in basis.lower():
        return f"{n} past trips to {cust_name}, {seas} season"
    if "Customer" in basis:
        return f"{n} past trips to {cust_name} (all seasons)"
    if "State" in basis:
        return f"{n} regional trips, {band} miles, {mode}"
    if "Season" in basis and "Distance" in basis.lower():
        return f"{n} trips, {band} miles {mode}, {seas} season"
    if "Distance" in basis and "global" not in basis.lower():
        return f"{n} trips, {band} miles {mode}"
    return f"{n} trips — broad average (limited data for this specific route)"

def render_confidence_guidance(cell, cust_display, daily_used, fee_used,
                               n_days, band, seas, is_fly):
    """
    Secondary block below the estimate box.
    Explains WHERE the confidence level comes from (the data source).
    The buffer amount itself is shown inside render_total_box.
    """
    conf       = rate_confidence(cell) if cell else "LOW"
    colors     = _CONF_COLORS[conf]
    cust_name  = cust_display.get("name", "this customer")
    basis_text = _basis_plain(cell, cust_name, band, seas, is_fly)

    st.markdown(
        f'<div style="background:{colors["bg"]};border:1px solid {colors["border"]};'
        f'border-radius:8px;padding:10px 16px;margin:4px 0 0 0;">'
        f'<span style="color:{colors["tc"]};font-size:11px;font-weight:700;'
        f'letter-spacing:0.05em;text-transform:uppercase;">{conf}</span>'
        f'<span style="color:#6b7280;font-size:11px;margin:0 6px;">·</span>'
        f'<span style="color:#d1d5db;font-size:12px;">Based on: {basis_text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_customers():
    df = pd.read_excel(os.path.join(BASE_DIR, "ContractsCustomersAddresses.xlsx"))
    df = df.dropna(subset=["latitude", "longitude", "CustomerName"])
    df["label"] = (
        df["CustomerName"].str.strip() + " — " +
        df["City"].fillna("").str.strip() + ", " +
        df["State"].fillna("").str.strip()
    )
    return df.sort_values("CustomerName").reset_index(drop=True)

@st.cache_data
def load_techs():
    df = pd.read_excel(os.path.join(BASE_DIR, "UserHomeAddress.xlsx"))
    df.columns = [c.strip() for c in df.columns]
    lat_col   = next((c for c in df.columns if "lat"      in c.lower()), None)
    lon_col   = next((c for c in df.columns if "lon"      in c.lower()), None)
    name_col  = next((c for c in df.columns if "username" in c.lower() or "name" in c.lower()), None)
    city_col  = next((c for c in df.columns if "city"     in c.lower()), None)
    state_col = next((c for c in df.columns if "state"    in c.lower()), None)
    df = df.dropna(subset=[lat_col, lon_col, name_col])
    city  = df[city_col].fillna("").str.strip()  if city_col  else ""
    state = df[state_col].fillna("").str.strip() if state_col else ""
    df["label"] = df[name_col].str.strip() + " (" + city + ", " + state + ")"
    df = df.rename(columns={lat_col:"Latitude", lon_col:"Longitude", name_col:"UserName"})
    return df.sort_values("UserName").reset_index(drop=True)

@st.cache_data
def load_rates():
    with open(os.path.join(BASE_DIR, "rate_table_v4.json")) as f:
        return json.load(f)

@st.cache_data
def load_classifier():
    with open(os.path.join(BASE_DIR, "overnight_lookup.json")) as f:
        return json.load(f)

@st.cache_data
def load_geo_index():
    path = os.path.join(BASE_DIR, "geo_index.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_master():
    path = os.path.join(BASE_DIR, "master_trips_imputed.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df["trip_start_date"] = pd.to_datetime(df["trip_start_date"], errors="coerce")
    return df


def _ensure_rate_col(df):
    """
    Ensure `daily_rate_expenses_final` exists — compute on-the-fly from raw columns
    if it was not pre-computed (e.g. deployed master_trips_imputed.csv may lack it).
    Returns the df (possibly a copy with the column added), or an empty DataFrame
    if the required source columns are also absent.
    """
    if df.empty:
        return df
    if "daily_rate_expenses_final" in df.columns:
        return df
    needed = ["exp_hotel", "exp_meals", "exp_local_transport", "total_trip_days"]
    if not all(c in df.columns for c in needed):
        return df  # can't compute — return as-is (callers check column presence)
    import numpy as np
    df = df.copy()
    days = df["total_trip_days"].replace(0, np.nan)
    df["daily_rate_expenses_final"] = (
        df["exp_hotel"].fillna(0)
        + df["exp_meals"].fillna(0)
        + df["exp_local_transport"].fillna(0)
    ) / days
    return df


# ---------------------------------------------------------------------------
# Explain modal helpers
# ---------------------------------------------------------------------------

def get_example_trips(master_df, basis, customer_id, cust_name,
                      band, seas, is_fly, dist_miles, cust_state=None):
    """Return (DataFrame of up to 25 matching historical trips, plain-English label)."""
    if master_df.empty:
        return pd.DataFrame(), ""

    master_df = _ensure_rate_col(master_df)
    rate_col  = "daily_rate_expenses_final"
    if rate_col not in master_df.columns:
        return pd.DataFrame(), ""

    df = master_df[
        (master_df["data_category"] == "E_complete") &
        (master_df["year"] >= 2022) &
        (master_df[rate_col].notna()) &
        (master_df[rate_col] > 10)
    ].copy()

    fine = _fine_band(dist_miles)

    def _fly_mask(sub):
        if "is_fly_trip" not in sub.columns:
            return sub
        flag = 1 if is_fly else 0
        return sub[sub["is_fly_trip"].fillna(0).astype(float).round().astype(int) == flag]

    if "Customer" in basis and customer_id:
        sub = df[df["CustomerIDAcu"].astype(str) == str(customer_id)]
        if "seasonal" in basis.lower():
            sub2 = sub[sub["season"] == seas]
            label = f"{cust_name} · {seas}"
            if len(sub2) >= 2:
                sub = sub2
        else:
            label = cust_name
        return sub.sort_values("trip_start_date", ascending=False).head(25), label

    if "State" in basis and cust_state:
        sub = _fly_mask(df[(df["State"] == cust_state) & (df["distance_band"] == band)])
        label = f"{cust_state} · {band} mi · {'fly' if is_fly else 'drive'}"
        return sub.sort_values("trip_start_date", ascending=False).head(25), label

    if fine and fine in basis:
        sub = _fly_mask(df)
        if fine == "Under 100":
            sub = sub[sub["distance_miles"] < 100]
        else:
            sub = sub[(sub["distance_miles"] >= 100) & (sub["distance_miles"] < 300)]
        if "Season" in basis:
            sub2 = sub[sub["season"] == seas]
            if len(sub2) >= 2:
                sub = sub2
        label = f"{fine} mi · {'fly' if is_fly else 'drive'}"
        return sub.sort_values("trip_start_date", ascending=False).head(25), label

    sub = _fly_mask(df[df["distance_band"] == band])
    if "Season" in basis:
        sub2 = sub[sub["season"] == seas]
        if len(sub2) >= 2:
            sub = sub2
    label = f"{band} mi · {'fly' if is_fly else 'drive'}"
    return sub.sort_values("trip_start_date", ascending=False).head(25), label


@st.dialog("How did we get this number?", width="large")
def explain_modal(cell_type, cell, af_cell, dr_cell, is_fly,
                  dist_miles, band, seas, mode_label, final_trip,
                  p_overnight, on_basis, on_confidence, on_n,
                  customer_id, cust_display, n_days, daily_used, fee_used,
                  master_df):

    cust_name  = cust_display.get("name",  "this customer")
    cust_state = cust_display.get("state", "")

    CONF_STYLE = {
        "HIGH":   ("#dcfce7", "#166534"),
        "MEDIUM": ("#fef9c3", "#854d0e"),
        "LOW":    ("#fff7ed", "#9a3412"),
    }

    # ── Inner helpers ─────────────────────────────────────────────────────────

    def _section(title):
        st.markdown(
            f'<p style="color:#6b7280;font-size:10px;font-weight:700;letter-spacing:0.08em;'
            f'text-transform:uppercase;margin:0 0 6px 0;">{title}</p>',
            unsafe_allow_html=True,
        )

    def _sep():
        st.markdown(
            '<div style="border-top:1px solid rgba(255,255,255,0.07);margin:14px 0 14px 0;"></div>',
            unsafe_allow_html=True,
        )

    def _card(bg, fg, title, body):
        st.markdown(
            f'<div style="background:{bg};border-radius:8px;padding:14px 18px;margin:0 0 12px 0;">'
            f'<div style="color:{fg};font-size:15px;font-weight:700;margin-bottom:5px;">{title}</div>'
            f'<div style="color:#1f2937;font-size:14px;line-height:1.65;">{body}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    def _cost_bars(components):
        total = sum(a for _, a, _ in components if a > 0)
        if total <= 0:
            return ""
        rows = ""
        for label, amount, color in components:
            if amount <= 0:
                continue
            pct = amount / total * 100
            rows += (
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:7px;">'
                f'<span style="color:#9ca3af;font-size:12px;width:110px;flex-shrink:0;">{label}</span>'
                f'<div style="flex:1;background:rgba(255,255,255,0.08);border-radius:3px;height:6px;overflow:hidden;">'
                f'<div style="background:{color};width:{pct:.1f}%;height:100%;border-radius:3px;"></div>'
                f'</div>'
                f'<span style="color:#e5e7eb;font-size:12px;width:54px;text-align:right;">${amount:,.0f}</span>'
                f'</div>'
            )
        rows += (
            f'<div style="display:flex;align-items:center;gap:10px;margin-top:6px;'
            f'padding-top:6px;border-top:1px solid rgba(255,255,255,0.1);">'
            f'<span style="color:#9ca3af;font-size:12px;width:110px;flex-shrink:0;">Total/day</span>'
            f'<div style="flex:1;"></div>'
            f'<span style="color:#ffffff;font-size:13px;font-weight:700;width:54px;text-align:right;">${total:,.0f}</span>'
            f'</div>'
        )
        return f'<div style="margin:8px 0 4px 0;">{rows}</div>'

    def _dist_strip(estimate, p25, p75, lo, hi):
        """Distribution strip showing where estimate sits vs historical range."""
        if not (p25 and p75 and estimate and lo < hi):
            return ""
        span   = hi - lo
        def _p(v): return max(1, min(97, (v - lo) / span * 100))
        p25p   = _p(p25);  p75p = _p(p75);  estp = _p(estimate)
        zone_w = max(p75p - p25p, 1)
        in_zone = p25 <= estimate <= p75
        dot_col = "#3b82f6" if in_zone else ("#22c55e" if estimate < p25 else "#f59e0b")
        verdict = (
            "Within the typical range — a solid, expected estimate."
            if in_zone else
            ("Below the typical range — cost-favorable." if estimate < p25
             else "Above the typical range — worth a closer look before quoting.")
        )
        verdict_col = "#93c5fd" if in_zone else ("#86efac" if estimate < p25 else "#fbbf24")
        return (
            f'<div style="background:rgba(255,255,255,0.04);border-radius:10px;'
            f'padding:14px 18px;margin:8px 0 14px 0;">'
            f'<p style="color:#d1d5db;font-size:12px;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.05em;margin:0 0 14px 0;">'
            f'Where does this estimate sit?</p>'
            f'<div style="position:relative;height:28px;margin-bottom:10px;">'
            f'<div style="position:absolute;top:12px;left:0;right:0;height:4px;'
            f'background:rgba(255,255,255,0.1);border-radius:2px;"></div>'
            f'<div style="position:absolute;top:8px;left:{p25p:.1f}%;width:{zone_w:.1f}%;'
            f'height:12px;background:rgba(59,130,246,0.3);border-radius:4px;'
            f'border:1px solid rgba(59,130,246,0.45);"></div>'
            f'<div style="position:absolute;top:4px;left:{estp:.1f}%;'
            f'transform:translateX(-50%);width:20px;height:20px;'
            f'background:{dot_col};border-radius:50%;'
            f'border:3px solid #ffffff;box-shadow:0 0 8px rgba(0,0,0,0.5);"></div>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
            f'<span style="color:#e5e7eb;font-size:11px;font-weight:600;">${lo:,.0f} low</span>'
            f'<span style="color:#93c5fd;font-size:11px;">typical  ${p25:,.0f} – ${p75:,.0f}</span>'
            f'<span style="color:#e5e7eb;font-size:11px;font-weight:600;">${hi:,.0f} high</span>'
            f'</div>'
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<div style="width:10px;height:10px;border-radius:50%;'
            f'background:{dot_col};flex-shrink:0;"></div>'
            f'<span style="color:{verdict_col};font-size:13px;font-weight:600;">'
            f'${estimate:,.0f}/day — {verdict}</span>'
            f'</div></div>'
        )

    def _trend_bars(trend_data):
        """Year-over-year avg daily rate bars."""
        if not trend_data or len(trend_data) < 2:
            return ""
        years = sorted(trend_data.keys())
        vals  = [trend_data[y] for y in years]
        mx    = max(vals) or 1
        rows  = ""
        for i, (yr, val) in enumerate(zip(years, vals)):
            bar_w = val / mx * 100
            if i == 0:
                chg_html = ""
            else:
                chg = (val - vals[i-1]) / max(vals[i-1], 1) * 100
                arrow  = "▲" if chg > 0 else "▼"
                c_col  = "#f87171" if chg > 5 else ("#86efac" if chg < -2 else "#fde68a")
                chg_html = (
                    f'<span style="color:{c_col};font-size:11px;'
                    f'margin-left:8px;">{arrow} {abs(chg):.0f}%</span>'
                )
            rows += (
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
                f'<span style="color:#9ca3af;font-size:12px;width:36px;">{yr}</span>'
                f'<div style="flex:1;background:rgba(255,255,255,0.08);border-radius:4px;'
                f'height:8px;overflow:hidden;">'
                f'<div style="background:#3b82f6;width:{bar_w:.1f}%;height:100%;'
                f'border-radius:4px;"></div></div>'
                f'<span style="color:#ffffff;font-size:12px;font-weight:600;'
                f'width:55px;text-align:right;">${val:,.0f}</span>'
                f'{chg_html}</div>'
            )
        total_chg = (vals[-1] - vals[0]) / max(vals[0], 1) * 100
        if abs(total_chg) < 3:
            summary = "Costs have been stable over this period."
            s_col   = "#d1d5db"
        elif total_chg > 0:
            summary = f"Costs have risen {total_chg:.0f}% since {years[0]}. This estimate reflects current prices."
            s_col   = "#fbbf24"
        else:
            summary = f"Costs have decreased {abs(total_chg):.0f}% since {years[0]}."
            s_col   = "#86efac"
        return (
            f'<div style="background:rgba(255,255,255,0.04);border-radius:10px;'
            f'padding:14px 18px;margin:12px 0 8px 0;">'
            f'<p style="color:#d1d5db;font-size:12px;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.05em;margin:0 0 12px 0;">'
            f'Cost trend — avg daily rate for similar trips</p>'
            f'{rows}'
            f'<p style="color:{s_col};font-size:12px;margin:8px 0 0 0;">{summary}</p>'
            f'</div>'
        )

    def _trip_calc(daily_rate, trip_fee, fee_label, current_days):
        if not daily_rate:
            return ""
        trip_fee = trip_fee or 0
        rows = ""
        for d in range(1, 6):
            total      = daily_rate * d + trip_fee
            is_current = (current_days is not None and d == int(current_days))
            bg_row     = "rgba(59,130,246,0.18)" if is_current else "transparent"
            border     = "border-left:3px solid #3b82f6;" if is_current else "border-left:3px solid transparent;"
            badge      = ('<span style="background:#3b82f6;color:#fff;font-size:10px;'
                          'padding:1px 6px;border-radius:999px;margin-left:6px;">selected</span>'
                          if is_current else "")
            day_word   = "day" if d == 1 else "days"
            fee_part   = f" + ${trip_fee:,.0f} {fee_label}" if trip_fee else ""
            rows += (
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;padding:8px 12px;border-radius:6px;'
                f'background:{bg_row};{border}margin-bottom:4px;">'
                f'<span style="color:#d1d5db;font-size:13px;min-width:70px;">'
                f'{d} {day_word}{badge}</span>'
                f'<span style="color:#6b7280;font-size:12px;flex:1;text-align:center;">'
                f'${daily_rate:,.0f} × {d}{fee_part}</span>'
                f'<span style="color:#ffffff;font-size:14px;font-weight:700;">'
                f'${total:,.0f}</span>'
                f'</div>'
            )
        return (
            f'<div style="background:rgba(255,255,255,0.04);border-radius:10px;'
            f'padding:14px 18px;margin:12px 0 8px 0;">'
            f'<p style="color:#d1d5db;font-size:12px;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.05em;margin:0 0 10px 0;">'
            f'What if the trip runs longer?</p>'
            f'{rows}</div>'
        )

    def _similar_all():
        """All E_complete trips matching the same lookup basis — for trend + distribution."""
        if master_df.empty:
            return pd.DataFrame()
        mdf = _ensure_rate_col(master_df)
        rate_col = "daily_rate_expenses_final"
        if rate_col not in mdf.columns:
            return pd.DataFrame()
        basis_str = cell.get("basis", "") if cell else ""
        try:
            base = mdf[
                (mdf["data_category"] == "E_complete") &
                (mdf[rate_col].notna()) &
                (mdf[rate_col] > 10)
            ]
            mode_val = 1 if is_fly else 0
            if "Customer" in basis_str and customer_id:
                sub = base[base["CustomerIDAcu"].astype(str) == str(customer_id)]
                if len(sub) >= 3:
                    return sub
            sub = base[base["distance_band"] == band]
            if "is_fly_trip" in sub.columns:
                sub = sub[sub["is_fly_trip"].fillna(0).astype(float).round().astype(int) == mode_val]
            return sub if len(sub) >= 3 else base
        except Exception:
            return pd.DataFrame()

    # ── Pre-compute shared data ────────────────────────────────────────────────
    similar_df = _similar_all()
    basis_str  = cell.get("basis", "") if cell else ""
    # Fall back to cell's total_rate if daily_used wasn't captured (e.g. ambiguous branch)
    daily_rate_est = daily_used or (cell.get("total_rate") if cell else None)

    # For distribution strip
    p25 = cell.get("total_p25") if cell else None
    p75 = cell.get("total_p75") if cell else None
    if p25 and p75 and not similar_df.empty and "daily_rate_expenses_final" in similar_df.columns:
        vals_all = similar_df["daily_rate_expenses_final"].dropna()
        vals_all = vals_all[vals_all > 10]
        dist_lo  = float(vals_all.quantile(0.05)) if len(vals_all) >= 5 else max(0, p25 - (p75-p25))
        dist_hi  = float(vals_all.quantile(0.95)) if len(vals_all) >= 5 else p75 + (p75-p25)
    else:
        dist_lo = dist_hi = None

    # For trend
    trend_data = {}
    if not similar_df.empty and "year" in similar_df.columns:
        tg = (similar_df[similar_df["year"].between(2022, 2025)]
              .groupby("year")["daily_rate_expenses_final"].mean())
        trend_data = {int(k): round(float(v), 2) for k, v in tg.items()}

    # Trip fee for calculator
    if is_fly and af_cell:
        trip_fee_val = af_cell.get("model_rate") or af_cell.get("median", 0)
        fee_label_str = "airfare"
    elif not is_fly and dr_cell:
        trip_fee_val = round(dist_miles * 2 * IRS_RATE, 0) if dist_miles else (dr_cell.get("model_rate") or dr_cell.get("mean", 0))
        fee_label_str = "drive"
    else:
        trip_fee_val  = 0
        fee_label_str = ""

    # ════════════════════════════════════════════════════════════════════════
    # 1 — TRIP TYPE
    # ════════════════════════════════════════════════════════════════════════
    # ── 1 · Trip Type ─────────────────────────────────────────────────────────
    _section("Trip Type")

    if final_trip == "day_trip":
        trip_label = "Day Trip";   prob_pct = int((1 - p_overnight) * 100)
    elif final_trip == "overnight":
        trip_label = "Overnight";  prob_pct = int(p_overnight * 100)
    else:
        trip_label = "Uncertain";  prob_pct = int(p_overnight * 100)

    if on_confidence == "HIGH" and "past trips" in on_basis:
        trip_body = (f"We checked the history for <strong>{cust_name}</strong>. "
                     f"<strong>{on_basis}</strong> — so we're treating this as a {trip_label.lower()} trip.")
    elif dist_miles >= 300:
        trip_body = (f"At <strong>{dist_miles:,.0f} miles</strong>, trips this distance are almost always overnight. "
                     f"<strong>{prob_pct}%</strong> of similar trips in our records required a hotel stay.")
    else:
        trip_body = (f"Based on <strong>{on_n} similar trips</strong> in the {band} mile range, "
                     f"<strong>{prob_pct}% were {trip_label.lower()} trips</strong>.")

    # ── 2 · Confidence (paired with trip type) ────────────────────────────────
    conf_level = rate_confidence(cell) if cell else "LOW"
    conf_details = {
        "HIGH": (
            "Strong estimate",
            f"Directly based on {cust_name}'s own trip history — most accurate picture available.",
        ),
        "MEDIUM": (
            "Reasonable estimate",
            f"No specific history for this route yet — using regional averages. Expect ±20–30% variance.",
        ),
        "LOW": (
            "Use with caution",
            f"Limited data for this route — broad averages only. Consider a detailed quote.",
        ),
    }
    conf_heading, conf_message = conf_details.get(conf_level, conf_details["LOW"])

    # Render Trip Type and Confidence side by side
    col_tt, col_cf = st.columns(2)
    with col_tt:
        bg, fg = CONF_STYLE.get(on_confidence, ("#f1f5f9", "#374151"))
        _card(bg, fg, f"{trip_label} · {prob_pct}% likely", trip_body)
    with col_cf:
        bg, fg = CONF_STYLE.get(conf_level, ("#f1f5f9", "#374151"))
        _card(bg, fg, f"{conf_level} · {conf_heading}", conf_message)

    # ── 3 · Daily Rate Breakdown ──────────────────────────────────────────────
    if cell:
        _sep()
        _section("Daily Rate Breakdown")

        n = cell.get("n", 0)
        if "Customer" in basis_str and "seasonal" in basis_str.lower():
            source = f"<strong>{n} past trips to {cust_name} in {seas}</strong>"
        elif "Customer" in basis_str:
            source = f"<strong>{n} past trips to {cust_name}</strong> (across all seasons)"
        elif "State" in basis_str:
            source = f"<strong>{n} trips in {cust_state}</strong>, {band} miles, {'flying' if is_fly else 'driving'}"
        elif "Under 100" in basis_str:
            source = f"<strong>{n} short trips under 100 miles</strong>, {'flying' if is_fly else 'driving'}, {seas}"
        elif "100-300" in basis_str:
            source = f"<strong>{n} trips in the 100–300 mile range</strong>, {'flying' if is_fly else 'driving'}, {seas}"
        elif "Season" in basis_str and "Distance" in basis_str:
            source = f"<strong>{n} trips in the {band} mile range</strong>, {'flying' if is_fly else 'driving'}, {seas}"
        elif "Distance" in basis_str:
            source = f"<strong>{n} trips in the {band} mile range</strong>, {'flying' if is_fly else 'driving'}"
        else:
            source = f"<strong>{n} trips</strong> (broad average — limited data for this specific route)"

        st.markdown(
            f'<p style="font-size:13px;color:#d1d5db;margin:0 0 6px 0;">'
            f'Averaged from {source}:</p>',
            unsafe_allow_html=True,
        )

        if cell_type == "overnight":
            hotel_r = cell.get("hotel_rate", 0) or 0
            meals_r = cell.get("meals_rate", 0) or 0
            trans_r = cell.get("transport_rate", 0) or 0
            st.markdown(_cost_bars([
                ("Hotel",           hotel_r, "#1d4ed8"),
                ("Meals",           meals_r, "#3b82f6"),
                ("Local Transport", trans_r, "#60a5fa"),
            ]), unsafe_allow_html=True)
            daily_rate_est = daily_rate_est or (hotel_r + meals_r + trans_r)
        else:
            meals_r = cell.get("meals_rate", 0) or 0
            trans_r = cell.get("transport_rate", 0) or 0
            st.markdown(_cost_bars([
                ("Meals",     meals_r, "#3b82f6"),
                ("Transport", trans_r, "#60a5fa"),
            ]), unsafe_allow_html=True)
            daily_rate_est = daily_rate_est or (meals_r + trans_r)

        if p25 and p75 and daily_rate_est and dist_lo is not None:
            st.markdown(
                _dist_strip(daily_rate_est, p25, p75, dist_lo, dist_hi),
                unsafe_allow_html=True,
            )

        total_std  = cell.get("total_std")
        total_mean = cell.get("total_mean") or cell.get("total_rate")
        if total_std and total_mean and total_mean > 0 and (total_std / total_mean) > 0.45:
            cv_pct = int(total_std / total_mean * 100)
            st.warning(
                f"**High variance:** Costs for this route vary significantly "
                f"(spread ±{cv_pct}%). The actual trip may run 20–40% above or "
                f"below this estimate — build in some buffer when quoting."
            )

    # ── 4 · Trip Fee ──────────────────────────────────────────────────────────
    _sep()
    _section("Trip Fee")

    if is_fly and af_cell:
        n_af = af_cell.get("n", 0)
        st.markdown(
            f'<p style="font-size:13px;color:#d1d5db;margin:0 0 2px 0;">'
            f'Based on <strong>{n_af} past flights</strong> in the <strong>{band} mile range</strong>. '
            f'Median round-trip airfare per technician: '
            f'<strong style="color:#93c5fd;">${trip_fee_val:,.0f}</strong>.</p>',
            unsafe_allow_html=True,
        )
    elif not is_fly and dr_cell:
        n_dr = dr_cell.get("n", 0)
        st.markdown(
            f'<p style="font-size:13px;color:#d1d5db;margin:0 0 2px 0;">'
            f'${IRS_RATE}/mi × <strong>{dist_miles:,.0f} mi</strong> × 2 (round trip) = '
            f'<strong style="color:#93c5fd;">${trip_fee_val:,.0f}</strong>. '
            f'Cross-checked against <strong>{n_dr} similar drive trips</strong>. '
            f'Note: distance is straight-line — actual road miles are typically 15–25% longer.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("No trip fee data available for this scenario.")

    # ── 5 · Historical Evidence ───────────────────────────────────────────────
    if cell:
        _sep()
        _section("Historical Evidence")

        if trend_data:
            st.markdown(_trend_bars(trend_data), unsafe_allow_html=True)

        examples, filter_desc = get_example_trips(
            master_df, basis_str, customer_id, cust_name,
            band, seas, is_fly, dist_miles, cust_state
        )
        if not examples.empty:
            show_cols = [c for c in [
                "Title", "CustomerName", "City", "State",
                "trip_start_date", "total_trip_days", "daily_rate_expenses_final",
            ] if c in examples.columns]
            disp = examples[show_cols].copy().rename(columns={
                "Title":                     "Ticket",
                "CustomerName":              "Customer",
                "City":                      "City",
                "State":                     "ST",
                "trip_start_date":           "Date",
                "total_trip_days":           "Days",
                "daily_rate_expenses_final": "Actual $/Day",
            })
            if "Date" in disp.columns:
                disp["Date"] = pd.to_datetime(disp["Date"], errors="coerce").dt.strftime("%b %Y")
            if "Days" in disp.columns:
                disp["Days"] = disp["Days"].apply(
                    lambda x: str(int(x)) if pd.notna(x) and float(x) > 0 else "—"
                )
            if "Actual $/Day" in disp.columns:
                disp["Actual $/Day"] = disp["Actual $/Day"].apply(
                    lambda x: f"${x:,.0f}" if pd.notna(x) and x > 10 else "—"
                )
            st.markdown(
                f'<p style="font-size:12px;color:#9ca3af;margin:6px 0 4px 0;">'
                f'Most recent matching trips ({filter_desc}):</p>',
                unsafe_allow_html=True,
            )
            st.dataframe(disp, use_container_width=True, hide_index=True)
        elif not trend_data:
            st.markdown(
                '<div style="background:rgba(255,255,255,0.04);border-radius:8px;'
                'padding:14px 18px;color:#9ca3af;font-size:13px;">'
                'No historical trip data available for this route in the training dataset. '
                'The estimate uses broader regional averages.'
                '</div>',
                unsafe_allow_html=True,
            )



# ---------------------------------------------------------------------------
# Rate lookups
# ---------------------------------------------------------------------------

def lookup_day_trip_rate(rates, band, seas):
    c = rates.get("day_trip_rates",{}).get("by_distance_season",{}).get(band,{}).get(seas)
    if c and c.get("n",0) >= 5: return dict(c, basis="Distance + Season")
    c = rates.get("day_trip_rates",{}).get("by_distance",{}).get(band)
    if c and c.get("n",0) >= 5: return dict(c, basis="Distance band")
    c = rates.get("day_trip_rates",{}).get("global")
    if c: return dict(c, basis="Global fallback")
    return None

def lookup_overnight_rate(rates, band, seas, is_fly, customer_id=None, state=None, dist_miles=None):
    mode_str = "1" if is_fly else "0"
    on = rates.get("overnight_rates", {})

    # Priority 1 — Customer history (N≥10: enough data to trust customer-specific patterns)
    if customer_id:
        c = on.get("by_customer_season",{}).get(str(customer_id),{}).get(seas)
        if c and c.get("n",0) >= 10: return dict(c, basis="Customer history (seasonal)")
        c = on.get("by_customer",{}).get(str(customer_id))
        if c and c.get("n",0) >= 10: return dict(c, basis="Customer history")

    # Priority 2 — State-level hybrid: hotel+meals from broad state cell (mode/distance
    # don't causally affect hotel or meals) + transport from mode-specific state cell
    # (mode DOES matter for local transport: fly=Uber/rental, drive=own car).
    if state:
        hm = on.get("by_state_season_only",{}).get(str(state),{}).get(seas)
        if not hm or hm.get("n",0) < 10:
            hm = on.get("by_state_only",{}).get(str(state))
        if hm and hm.get("n",0) >= 10:
            c = dict(hm)
            tr = on.get("by_state_distance_mode",{}).get(str(state),{}).get(band,{}).get(mode_str)
            if tr and tr.get("n",0) >= 5:
                c["transport_rate"] = tr.get("transport_rate", hm.get("transport_rate", 0))
                c["transport_n"]    = tr.get("transport_n", tr.get("n", 0))
                c["total_rate"]     = round(
                    (c.get("hotel_rate") or 0)
                    + (c.get("meals_rate") or 0)
                    + c["transport_rate"], 4
                )
                return dict(c, basis=f"State ({state}) · Mode transport")
            return dict(c, basis=f"State ({state})")

    # Priority 3–8 — Distance-band fallbacks (existing, unchanged)
    fine = _fine_band(dist_miles)
    if fine:
        c = on.get("fine_by_distance_season_mode",{}).get(fine,{}).get(seas,{}).get(mode_str)
        if c and c.get("n",0) >= 5: return dict(c, basis=f"Distance {fine} + Season + Mode")
        c = on.get("fine_by_distance_mode",{}).get(fine,{}).get(mode_str)
        if c and c.get("n",0) >= 5: return dict(c, basis=f"Distance {fine} + Mode")

    c = on.get("by_distance_season_mode",{}).get(band,{}).get(seas,{}).get(mode_str)
    if c and c.get("n",0) >= 5: return dict(c, basis="Distance + Season + Mode")
    c = on.get("by_distance_mode",{}).get(band,{}).get(mode_str)
    if c and c.get("n",0) >= 5: return dict(c, basis="Distance + Mode")
    c = on.get("by_distance",{}).get(band)
    if c and c.get("n",0) >= 5: return dict(c, basis="Distance band")
    c = on.get("global")
    if c: return dict(c, basis="Global fallback")
    return None

def lookup_airfare(rates, band, seas):
    af = rates.get("airfare_rates", {})
    c = af.get("by_distance_season",{}).get(band,{}).get(seas)
    if c and c.get("n",0) >= 3: return c
    c = af.get("by_distance",{}).get(band)
    if c and c.get("n",0) >= 3: return c
    return None

def lookup_drive(rates, band):
    return rates.get("drive_rates",{}).get("by_distance",{}).get(band)

def get_overnight_probability(classifier, customer_id, dist_miles, job_type):
    if customer_id:
        c = classifier.get("by_customer",{}).get(str(customer_id))
        if c and c["n"] >= 3:
            p, n = c["p_overnight"], c["n"]
            return p, f"{c['n_overnight']} of {n} past trips overnight", "HIGH", n

    band = distance_band(dist_miles) if dist_miles else None
    if band:
        dj = classifier.get("by_distance_jobtype",{}).get(band,{}).get(job_type)
        if dj and dj["n"] >= 5:
            p, n = dj["p_overnight"], dj["n"]
            conf = "HIGH" if dist_miles >= 300 and p > 0.90 else ("MEDIUM" if n >= 10 else "LOW")
            return p, f"{band} + {job_type} (N={n})", conf, n
        dd = classifier.get("by_distance_only",{}).get(band)
        if dd:
            p, n = dd["p_overnight"], dd["n"]
            conf = "HIGH" if dist_miles >= 300 and p > 0.90 else "LOW"
            return p, f"{band} only (N={n})", conf, n

    g = classifier.get("global", {})
    return g.get("p_overnight", 0.7), "Global average", "LOW", g.get("n", 0)

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _table_html(rows, header, conf_level=None):
    """
    rows: list of (label, value_str, detail_str)
    Produces a styled HTML table. Colors are explicit on every element.
    """
    badge = _conf_badge_html(conf_level) if conf_level else ""
    head = (
        f'<tr><th colspan="3" style="'
        f'background-color:#1e3a5f !important;color:#ffffff !important;'
        f'padding:10px 14px;text-align:left;font-weight:600;font-size:12px;'
        f'letter-spacing:0.04em;text-transform:uppercase;border:none;">'
        f'{header}{badge}</th></tr>'
    )
    body = []
    for i, (lbl, val, detail) in enumerate(rows):
        is_total = lbl.upper() == "TOTAL"
        if is_total:
            row_bg   = "#eff6ff"
            row_fg   = "#1e3a5f"
            det_fg   = "#3b82f6"
            fw       = "700"
            border_t = "border-top:2px solid #bfdbfe;"
        elif i % 2 == 1:
            row_bg   = "#f9fafb"
            row_fg   = "#111827"
            det_fg   = "#6b7280"
            fw       = "400"
            border_t = ""
        else:
            row_bg   = "#ffffff"
            row_fg   = "#111827"
            det_fg   = "#6b7280"
            fw       = "400"
            border_t = ""

        body.append(
            f'<tr>'
            f'<td style="background-color:{row_bg} !important;color:{row_fg} !important;'
            f'padding:8px 14px;border:none;border-bottom:1px solid #e5e7eb;{border_t}">'
            f'{lbl}</td>'
            f'<td style="background-color:{row_bg} !important;color:{row_fg} !important;'
            f'padding:8px 14px;border:none;border-bottom:1px solid #e5e7eb;{border_t}'
            f'text-align:right;font-weight:{fw};white-space:nowrap;font-variant-numeric:tabular-nums;">'
            f'{val}</td>'
            f'<td style="background-color:{row_bg} !important;color:{det_fg} !important;'
            f'padding:8px 14px;border:none;border-bottom:1px solid #e5e7eb;{border_t}font-size:12px;">'
            f'{detail}</td>'
            f'</tr>'
        )

    return (
        f'<div style="border-radius:8px;overflow:hidden;margin-bottom:12px;">'
        f'<table style="width:100%;border-collapse:collapse;font-family:-apple-system,BlinkMacSystemFont,'
        f"'Segoe UI',sans-serif;font-size:14px;background-color:#ffffff !important;\">"
        f"{head}{''.join(body)}</table></div>"
    )


def render_overnight_table(cell, dist_miles):
    if not cell:
        st.error("No rate data available for this scenario.")
        return None

    hotel_r  = cell.get("hotel_rate", 0) or 0
    meals_r  = cell.get("meals_rate", 0) or 0
    trans_r  = cell.get("transport_rate", 0) or 0
    total_r  = cell.get("total_rate") or (hotel_r + meals_r + trans_r)
    basis    = cell.get("basis", "—")
    n        = cell.get("n", 0)
    hotel_n  = cell.get("hotel_n", n)
    meals_n  = cell.get("meals_n", n)
    trans_n  = cell.get("transport_n", n)
    p25      = cell.get("total_p25")
    p75      = cell.get("total_p75")
    trans_med  = cell.get("transport_median")
    trans_mean = cell.get("transport_mean", trans_r)

    trans_note = "local transport mean"
    if trans_med is not None and trans_mean and trans_med < trans_mean * 0.5:
        trans_note = f"mean used (median ${trans_med:,.0f} suppressed by $0 filings)"

    # Hotel row note — shows whether GSA floor or historical data was used
    geo_label       = cell.get("hotel_geo_label", "")
    used_gsa_floor  = cell.get("hotel_used_gsa_floor", False)
    hotel_hist_disp = cell.get("hotel_rate_historical")
    gsa_disp        = cell.get("hotel_rate_gsa")

    if used_gsa_floor and gsa_disp:
        hotel_note = f"GSA floor ${gsa_disp:,.0f}"
        if geo_label:
            hotel_note += f" · {geo_label}"
        if hotel_hist_disp:
            hotel_note += f" (hist. ${hotel_hist_disp:,.0f})"
    elif geo_label:
        hotel_note = f"N={hotel_n} · {geo_label}"
    else:
        hotel_note = f"N={hotel_n}"

    range_str = f"  ·  range {_fmt(p25)}–{_fmt(p75)}/day" if (p25 and p75) else ""
    rows = [
        ("Hotel",           f"{_fmt(hotel_r)}/day", hotel_note),
        ("Meals",           f"{_fmt(meals_r)}/day", f"N={meals_n}"),
        ("Local Transport", f"{_fmt(trans_r)}/day", trans_note),
        ("TOTAL",           f"{_fmt(total_r)}/day", f"{basis}{range_str}"),
    ]
    conf = rate_confidence(cell)
    st.markdown(_table_html(rows, f"Daily Rate  ·  N={n} trips", conf), unsafe_allow_html=True)
    return total_r


def render_day_trip_table(cell, dist_miles):
    if not cell:
        st.error("No rate data available for this scenario.")
        return None

    meals_r = cell.get("meals_rate", 0) or 0
    trans_r = cell.get("transport_rate", 0) or 0
    total_r = cell.get("total_rate") or (meals_r + trans_r)
    basis   = cell.get("basis", "—")
    n       = cell.get("n", 0)
    meals_n = cell.get("meals_n", n)
    trans_n = cell.get("transport_n", n)
    p25     = cell.get("total_p25")
    p75     = cell.get("total_p75")

    irs_floor = (dist_miles * 2 * IRS_RATE) if dist_miles and dist_miles > 0 else 0
    trans_note = (
        f"IRS floor {_fmt(irs_floor)} ({dist_miles:.0f} mi × 2 × ${IRS_RATE}/mi)"
        if irs_floor > 0 else f"N={trans_n}"
    )
    range_str = f"  ·  range {_fmt(p25)}–{_fmt(p75)}/day" if (p25 and p75) else ""
    rows = [
        ("Meals",     f"{_fmt(meals_r)}/day", f"N={meals_n}"),
        ("Transport", f"{_fmt(trans_r)}/day", trans_note),
        ("TOTAL",     f"{_fmt(total_r)}/day", f"{basis}{range_str}"),
    ]
    conf = rate_confidence(cell)
    st.markdown(_table_html(rows, f"Day Trip Rate  ·  N={n} trips", conf), unsafe_allow_html=True)
    return total_r


def render_trip_fee(af_cell=None, dr_cell=None, is_fly=True, dist_miles=None):
    if is_fly and af_cell:
        fee = af_cell.get("model_rate") or af_cell.get("median") or af_cell.get("mean")
        n   = af_cell.get("n", 0)
        rows = [("Airfare", f"{_fmt(fee)}/trip", f"median of N={n} trips")]
        st.markdown(_table_html(rows, "One-Time Trip Fee"), unsafe_allow_html=True)
        return fee
    elif not is_fly and dr_cell:
        n = dr_cell.get("n", 0)
        if dist_miles and dist_miles > 0:
            fee    = round(dist_miles * 2 * IRS_RATE, 0)
            detail = f"${IRS_RATE}/mi × {dist_miles:,.0f} mi × 2 (round trip)  ·  N={n} trips"
        else:
            fee    = dr_cell.get("model_rate") or dr_cell.get("mean")
            detail = f"IRS mileage  ·  N={n} trips"
        rows = [("Drive Cost", f"{_fmt(fee)}/trip", detail)]
        st.markdown(_table_html(rows, "One-Time Trip Fee"), unsafe_allow_html=True)
        return fee
    return None


def render_total_box(daily_rate, trip_fee, fee_label, n_days, rate_cell=None, basis_text=None):
    if daily_rate is None:
        return

    trip_fee = trip_fee or 0
    has_days = n_days is not None

    if has_days:
        total      = daily_rate * n_days + trip_fee
        days_word  = "day" if n_days == 1 else "days"
        fee_str    = f" + {_fmt(trip_fee)} {fee_label}" if trip_fee else ""
        top_label  = "Estimated Total · per technician"
        amount_str = _fmt(total)
        sub_html   = (
            f'<div style="color:#bfdbfe;font-size:13px;">'
            f'{_fmt(daily_rate)}/day × {n_days} {days_word}{fee_str}</div>'
        )
    else:
        total      = daily_rate + trip_fee
        fee_str    = f" + {_fmt(trip_fee)} {fee_label}" if trip_fee else ""
        top_label  = "Estimated Cost · Day 1 · per technician"
        amount_str = _fmt(total)
        sub_html   = (
            f'<div style="color:#bfdbfe;font-size:13px;">{_fmt(daily_rate)}/day{fee_str}</div>'
            f'<div style="color:#93c5fd;font-size:11px;margin-top:3px;">'
            f'Enter total days (labor + travel) above for a full trip total</div>'
        )

    range_html = ""
    if rate_cell and has_days:
        p25 = rate_cell.get("total_p25")
        p75 = rate_cell.get("total_p75")
        if p25 and p75:
            lo = p25 * n_days + trip_fee
            hi = p75 * n_days + trip_fee
            range_html = (
                f'<div style="color:#7dd3fc;font-size:12px;margin-top:6px;">'
                f'Typical range&nbsp;&nbsp;{_fmt(lo)} – {_fmt(hi)}</div>'
            )

    # ── Confidence footer — always shows level, action, and basis ─────────────
    conf        = rate_confidence(rate_cell) if rate_cell else "LOW"
    rules       = _BUFFER_RULES[conf]
    colors      = _CONF_COLORS[conf]
    border_col  = {"HIGH": "#22c55e", "MEDIUM": "#eab308", "LOW": "#ef4444"}[conf]
    buf_pct     = rules["pct"]
    action_text = rules["action"]

    # Invoice line — only when days entered and buffer > 0
    if has_days and buf_pct > 0:
        buffered    = total * (1 + buf_pct / 100)
        invoice_line = (
            f'<div style="margin-top:6px;margin-bottom:2px;">'
            f'<span style="color:#9ca3af;font-size:11px;">Invoice at&nbsp;</span>'
            f'<span style="color:#ffffff;font-size:16px;font-weight:700;">{_fmt(buffered)}</span>'
            f'<span style="color:#9ca3af;font-size:11px;">&nbsp;(+{buf_pct}% buffer)</span>'
            f'</div>'
        )
    else:
        invoice_line = ""

    # Basis line — shown when basis_text provided
    basis_line = (
        f'<div style="color:#9ca3af;font-size:11px;margin-top:5px;">'
        f'Based on: {basis_text}</div>'
        if basis_text else ""
    )

    conf_footer = (
        f'<div style="margin-top:12px;padding-top:10px;'
        f'border-top:1px solid rgba(255,255,255,0.12);">'
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<span style="font-size:13px;">{colors["icon"]}</span>'
        f'<span style="color:{colors["tc"]};font-size:11px;font-weight:700;'
        f'letter-spacing:0.05em;text-transform:uppercase;">{conf} CONFIDENCE</span>'
        f'</div>'
        f'{invoice_line}'
        f'<div style="color:#d1d5db;font-size:12px;margin-top:4px;">{action_text}</div>'
        f'{basis_line}'
        f'</div>'
    )

    st.markdown(
        f'<div style="background-color:#1e3a5f;border-radius:10px;padding:20px 24px;'
        f'margin:12px 0 8px 0;border-left:4px solid {border_col};'
        f'font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',sans-serif;">'
        f'<div style="color:#93c5fd;font-size:11px;font-weight:600;letter-spacing:0.06em;'
        f'text-transform:uppercase;margin-bottom:4px;">{top_label}</div>'
        f'<div style="color:#ffffff;font-size:34px;font-weight:700;line-height:1.1;margin-bottom:4px;">'
        f'{amount_str}</div>'
        f'{sub_html}'
        f'{range_html}'
        f'{conf_footer}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _route_card_html(dist_miles, band, seas, mode_label):
    mode_icon = "✈" if "Flight" in mode_label else "🚗"
    return (
        f'<div style="background-color:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;'
        f'padding:12px 18px;margin:8px 0 14px 0;font-family:-apple-system,BlinkMacSystemFont,'
        f"'Segoe UI',sans-serif;font-size:14px;color:#0c4a6e;\">"
        f'<span style="font-weight:700;font-size:16px;color:#0369a1;">{dist_miles:,.0f} mi</span>'
        f'<span style="color:#7dd3fc;margin:0 10px;">·</span>'
        f'<span style="color:#374151;">Band: <strong style="color:#0c4a6e;">{band}</strong></span>'
        f'<span style="color:#7dd3fc;margin:0 10px;">·</span>'
        f'<span style="color:#374151;">Season: <strong style="color:#0c4a6e;">{seas}</strong></span>'
        f'<span style="color:#7dd3fc;margin:0 10px;">·</span>'
        f'<span style="color:#374151;">{mode_icon} <strong style="color:#0c4a6e;">{mode_label}</strong></span>'
        f'</div>'
    )


def _section_label(text):
    st.markdown(
        f'<div style="color:#9ca3af;font-size:11px;font-weight:600;letter-spacing:0.07em;'
        f'text-transform:uppercase;margin:20px 0 8px 0;padding-bottom:5px;'
        f'border-bottom:1px solid #e5e7eb;">{text}</div>',
        unsafe_allow_html=True,
    )


# (explainability panel moved to explain_modal dialog below)


# (export removed — use the explain modal dialog instead)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MONTHS = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]
JOB_TYPES = [
    "Diagnostic / Troubleshoot",
    "Repair / Preventive Maintenance",
    "Installation / Startup",
    "Rebuild / Overhaul",
]

# ---------------------------------------------------------------------------
# Estimate tab
# ---------------------------------------------------------------------------

def render_estimate_tab(customers, techs, rates, classifier, master_df, geo_index=None):
    meta    = rates.get("metadata", {})

    # ── STEP 1 — Trip Details ────────────────────────────────────────────────
    _section_label("Step 1 — Trip Details")

    col_cust, col_tech = st.columns(2)

    manual_mode = col_cust.checkbox(
        "New / unlisted customer",
        key="manual_mode",
        help="Customer not yet in the database — enter details manually.",
    )

    if manual_mode:
        manual_name  = col_cust.text_input("Site Name", placeholder="e.g. Acme Corp - Houston", key="manual_name")
        mc1, mc2     = col_cust.columns([3, 2])
        manual_city  = mc1.text_input("City", key="manual_city")
        manual_state = mc2.selectbox("State", STATES_LIST, key="manual_state")
        manual_dist  = col_cust.number_input(
            "One-way distance (miles)", min_value=1, value=300, step=10, key="manual_dist",
        )
        cust_label = "— manual —"
    else:
        cust_label = col_cust.selectbox(
            "Customer Site",
            options=["— select —"] + customers["label"].tolist(),
            key="customer_sel",
        )

    tech_label = col_tech.selectbox(
        "Technician",
        options=["— select —"] + techs["label"].tolist(),
        key="tech_sel",
    )

    col_month, col_jtype, col_days = st.columns(3)

    with col_month:
        month_name = st.selectbox("Month of Travel", MONTHS, index=3, key="month_sel")
        month_num  = MONTHS.index(month_name) + 1
        seas       = season(month_num)

    with col_jtype:
        job_ui            = st.selectbox("Job Type", JOB_TYPES, key="jobtype_sel")
        job_type_internal = map_job_type(job_ui)

    with col_days:
        n_days = st.number_input(
            "Days on Site (labor + travel)",
            min_value=1,
            value=None,
            placeholder="optional",
            step=1,
            key="n_days",
            help="Enter total trip duration including travel days (e.g. fly out Monday, work Tue–Thu, fly home Friday = 5 days).",
        )

    # ── Validate inputs ──────────────────────────────────────────────────────
    if tech_label == "— select —":
        st.info("Select a technician above to continue.")
        return
    if not manual_mode and cust_label == "— select —":
        st.info("Select a customer site above — or check 'New / unlisted customer' to enter manually.")
        return
    if manual_mode and not locals().get("manual_name", "").strip():
        st.info("Enter a site name to continue.")
        return

    # ── Resolve distance + customer identity ─────────────────────────────────
    tech_row = techs[techs["label"] == tech_label].iloc[0]

    if manual_mode:
        dist_miles   = float(manual_dist)
        band         = distance_band(dist_miles)
        customer_id  = None
        cust_state   = str(manual_state).strip()
        cust_display = {"name": manual_name, "city": manual_city, "state": manual_state}
        cust_row     = None
    else:
        cust_row    = customers[customers["label"] == cust_label].iloc[0]
        tech_lat    = float(tech_row["Latitude"])
        tech_lon    = float(tech_row["Longitude"])
        cust_lat    = float(cust_row["latitude"])
        cust_lon    = float(cust_row["longitude"])
        dist_miles  = haversine(tech_lat, tech_lon, cust_lat, cust_lon)
        band        = distance_band(dist_miles)
        customer_id = str(cust_row["CustomerIDAcu"])
        cust_state  = str(cust_row.get("State","")).strip() if pd.notna(cust_row.get("State")) else None
        cust_display = {
            "name":  str(cust_row.get("CustomerName","")).strip(),
            "city":  str(cust_row.get("City","")).strip(),
            "state": str(cust_row.get("State","")).strip(),
        }

    # ── International flag ───────────────────────────────────────────────────
    # Based on job site state/province, not the customer's registered Country
    # field (which reflects company nationality, not where the work happens).
    US_STATES = {"AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID",
                 "IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS",
                 "MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK",
                 "OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV",
                 "WI","WY","DC","PR","GU","VI"}
    is_intl = cust_state not in US_STATES and cust_state != ""
    if is_intl:
        st.warning(
            f"**International destination ({cust_state}).**  "
            "Distance and rates are US domestic. A custom quote is recommended."
        )

    # ── STEP 2 — Travel Mode + Trip Type ─────────────────────────────────────
    _section_label("Step 2 — Travel & Trip Type")

    col_mode, col_trip = st.columns([1, 1])

    with col_mode:
        auto_fly    = dist_miles >= 300
        mode_hint   = "✈  Flight recommended" if auto_fly else "🚗  Drive recommended"
        st.caption(f"{mode_hint} ({dist_miles:,.0f} mi)")
        travel_mode = st.radio(
            "Travel Mode",
            options=["Auto", "Flight", "Car"],
            index=0,
            horizontal=True,
            key="travel_mode",
        )
        is_fly     = auto_fly if travel_mode == "Auto" else (travel_mode == "Flight")
        mode_label = "Flight" if is_fly else "Car"

    with col_trip:
        if is_fly:
            # Flight trips are almost always overnight — default to overnight.
            # Classifier not run; override selector stays for rare edge cases.
            p_overnight   = 1.0
            on_basis      = "Flight trip — overnight assumed"
            on_confidence = "HIGH"
            on_n          = 0
            suggestion    = "overnight"
        else:
            # Car trips: run classifier to determine day vs overnight
            p_overnight, on_basis, on_confidence, on_n = get_overnight_probability(
                classifier, customer_id, dist_miles, job_type_internal
            )
            if on_confidence == "HIGH":
                suggestion = "overnight" if p_overnight >= 0.75 else "day_trip"
            elif on_confidence == "MEDIUM":
                if p_overnight >= 0.75:   suggestion = "overnight"
                elif p_overnight <= 0.25: suggestion = "day_trip"
                else:                     suggestion = "ambiguous"
            else:
                suggestion = "ambiguous"

        override = st.selectbox(
            "Trip Type",
            options=["Auto-detect", "Overnight", "Day Trip"],
            index=0,
            key="trip_override",
            help="Auto-detect uses historical patterns. Override if you know the trip type.",
        )
        final_trip = (
            suggestion if override == "Auto-detect"
            else ("overnight" if override == "Overnight" else "day_trip")
        )

        if is_fly and final_trip != "day_trip":
            # Flight + overnight (auto or manual) — clean static pill
            st.markdown(
                '<div style="background:#dcfce7;color:#166534;border:1px solid #86efac;'
                'display:inline-block;border-radius:999px;padding:4px 12px;'
                'font-size:13px;font-weight:600;margin-top:4px;">✈ Overnight assumed'
                '<span style="font-weight:400;margin-left:6px;font-size:11px;'
                'opacity:0.8;">HIGH</span></div>',
                unsafe_allow_html=True,
            )
            st.caption("Almost all flight trips are overnight. Override above if needed.")
        elif is_fly and final_trip == "day_trip":
            # Flight + day trip manual override — flag it
            st.markdown(
                '<div style="background:#fff7ed;color:#9a3412;border:1px solid #fdba74;'
                'display:inline-block;border-radius:999px;padding:4px 12px;'
                'font-size:13px;font-weight:600;margin-top:4px;">⚠ Day trip override'
                '<span style="font-weight:400;margin-left:6px;font-size:11px;'
                'opacity:0.8;">LOW</span></div>',
                unsafe_allow_html=True,
            )
            st.caption("Flying for a day trip is unusual — verify with ops.")
        else:
            # Car trip — show classifier result
            pill_pct  = p_overnight if final_trip != "day_trip" else (1 - p_overnight)
            pill_text = "overnight" if final_trip != "day_trip" else "day trip"
            pill_css  = (
                "background:#dcfce7;color:#166534;border:1px solid #86efac"
                if on_confidence == "HIGH"
                else (
                    "background:#fef9c3;color:#854d0e;border:1px solid #fde047"
                    if on_confidence == "MEDIUM"
                    else "background:#f1f5f9;color:#475569;border:1px solid #cbd5e1"
                )
            )
            st.markdown(
                f'<div style="{pill_css};display:inline-block;border-radius:999px;'
                f'padding:4px 12px;font-size:13px;font-weight:600;margin-top:4px;">'
                f'{pill_pct:.0%} likely {pill_text}'
                f'<span style="font-weight:400;margin-left:6px;font-size:11px;opacity:0.8;">'
                f'{on_confidence}</span></div>',
                unsafe_allow_html=True,
            )
            st.caption(on_basis)

    if final_trip == "day_trip" and is_fly:
        st.warning(
            "Day trip selected with flight mode — flying for a day trip is very unusual. "
            "Verify before using this estimate."
        )
    elif final_trip == "day_trip" and dist_miles >= 300:
        st.warning(
            f"Day trip selected at {dist_miles:,.0f} miles — trips this long are almost always overnight. "
            "Verify before using this estimate."
        )

    st.markdown(_route_card_html(dist_miles, band, seas, mode_label), unsafe_allow_html=True)

    # ── STEP 3 — Estimate ────────────────────────────────────────────────────
    _section_label("Step 3 — Estimate")

    day_cell = lookup_day_trip_rate(rates, band, seas)
    on_cell  = lookup_overnight_rate(rates, band, seas, is_fly, customer_id, cust_state, dist_miles)
    af_cell  = lookup_airfare(rates, band, seas) if is_fly else None
    dr_cell  = lookup_drive(rates, band)         if not is_fly else None

    # Apply geographic hotel-cost correction (GSA-based multiplier for destination)
    if geo_index and on_cell:
        cust_city = cust_display.get("city", "")
        geo_mult, geo_label, gsa_rate = get_hotel_geo_mult(
            geo_index, customer_id, cust_city, cust_state
        )
        on_cell = apply_geo_correction(on_cell, geo_mult, geo_label, gsa_rate)

    daily_used = fee_used = None
    _cust_name = cust_display.get("name", "this customer")

    def _overnight(cell, nd):
        daily   = render_overnight_table(cell, dist_miles)
        fee     = render_trip_fee(af_cell, dr_cell, is_fly, dist_miles)
        fee_lbl = "airfare" if is_fly else "drive"
        bt      = _basis_plain(cell, _cust_name, band, seas, is_fly)
        render_total_box(daily, fee, fee_lbl, nd, rate_cell=cell, basis_text=bt)
        return daily, fee

    def _day_trip(cell, nd):
        daily = render_day_trip_table(cell, dist_miles)
        if not is_fly:
            fee = render_trip_fee(None, dr_cell, is_fly=False, dist_miles=dist_miles)
        else:
            fee = None
            st.caption("Note: flying for a day trip is unusual — confirm with ops.")
        fee_lbl = "drive" if not is_fly else ""
        bt      = _basis_plain(cell, _cust_name, band, seas, is_fly)
        render_total_box(daily, fee if not is_fly else None, fee_lbl, nd, rate_cell=cell, basis_text=bt)
        return daily, fee

    if final_trip == "overnight":
        daily_used, fee_used = _overnight(on_cell, n_days)
        exp_cell_type, exp_cell = "overnight", on_cell

    elif final_trip == "day_trip":
        daily_used, fee_used = _day_trip(day_cell, n_days)
        exp_cell_type, exp_cell = "day_trip", day_cell

    else:  # ambiguous — side by side
        col_dt, col_on = st.columns(2)
        with col_dt:
            st.markdown(
                '<p style="color:#6b7280;font-size:12px;font-weight:600;'
                'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">If Day Trip</p>',
                unsafe_allow_html=True,
            )
            _day_trip(day_cell, n_days)
        with col_on:
            st.markdown(
                '<p style="color:#6b7280;font-size:12px;font-weight:600;'
                'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">If Overnight</p>',
                unsafe_allow_html=True,
            )
            daily_used, fee_used = _overnight(on_cell, n_days)
        st.caption("Select 'Overnight' or 'Day Trip' in the Trip Type field above to confirm.")
        exp_cell_type, exp_cell = "overnight", on_cell

    # ── Explain button + disclaimer ───────────────────────────────────────────
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1.5, 1, 1.5])
    with btn_col:
        if st.button("How did we get this number?", use_container_width=True):
            explain_modal(
                cell_type=exp_cell_type, cell=exp_cell,
                af_cell=af_cell, dr_cell=dr_cell, is_fly=is_fly,
                dist_miles=dist_miles, band=band, seas=seas,
                mode_label=mode_label, final_trip=final_trip,
                p_overnight=p_overnight, on_basis=on_basis,
                on_confidence=on_confidence, on_n=on_n,
                customer_id=customer_id, cust_display=cust_display,
                n_days=n_days, daily_used=daily_used, fee_used=fee_used,
                master_df=master_df,
            )

    st.markdown(
        '<p style="text-align:center;color:#f87171;font-size:13px;font-weight:600;margin-top:10px;">'
        'These figures exclude supplies, tools, and labor.</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)

    st.markdown(
        '<h2 style="margin-bottom:2px;color:#e5e7eb;">Per Diem Cost Estimator</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#6b7280;margin-top:0;font-size:14px;">'
        'Hotel · meals · local transport per technician per day. '
        'Supplies and tools are pass-through actuals and not included.</p>',
        unsafe_allow_html=True,
    )

    try:
        customers  = load_customers()
        techs      = load_techs()
        rates      = load_rates()
        classifier = load_classifier()
    except Exception as e:
        st.error(f"Failed to load data files: {e}")
        st.stop()

    master_df = load_master()
    geo_index = load_geo_index()

    render_estimate_tab(customers, techs, rates, classifier, master_df, geo_index)


if __name__ == "__main__":
    main()
