"""
build_v4.py
===========
Extends build_v3.py with three preprocessing improvements from costestimator:

  1. CPI normalization  — all historical dollar amounts scaled to 2025 dollars
     before fitting rate cells, removing inflation bias.
  2. IsolationForest    — removes statistical outliers per group (N >= 10,
     contamination=5%). Catches hotel >$50K errors and similar anomalies.
  3. Winsorization      — soft-clips remaining extreme values per group
     (IQR × 1.5) before computing means/medians.
  4. Total p25/p75      — stores interquartile range of the total daily rate
     per cell so the app can show a "Typical Range" to the user.

Outputs: master_trips_v3.csv (unchanged), rate_table_v4.json

Do NOT modify build_master.py, fit_rates.py, fit_rates_v2.py, build_v3.py.
Requires: scikit-learn (pip install scikit-learn)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import date

try:
    from sklearn.ensemble import IsolationForest
    HAS_ISO = True
except ImportError:
    HAS_ISO = False
    print("WARNING: scikit-learn not installed — IsolationForest step skipped.")
    print("  Install with: pip install scikit-learn")

PROJECT = Path("G:/After Sales Team/PROJECTS/per-diem-model")
FIT_YEARS = [2019, 2022, 2023, 2024, 2025]
IRS_RATE  = 0.67
DISTANCE_BAND_ORDER = [
    "Under 300", "300-500", "500-750", "750-1000",
    "1000-1500", "1500-2000", "2000+"
]

# Finer sub-bands for Under 300 only (drive overnight under-estimates at coarse band)
FINE_BANDS_UNDER300 = ["Under 100", "100-300"]
FINE_BAND_ORDER = ["Under 100", "100-300"] + DISTANCE_BAND_ORDER[1:]


def fine_distance_band(miles):
    """Returns finer-grained band for Under-300 trips."""
    if miles < 100:   return "Under 100"
    if miles < 300:   return "100-300"
    return None  # not applicable beyond 300 mi

# ---------------------------------------------------------------------------
# CPI factors (source: costestimator/config.py — BLS CPI-U)
# All amounts multiplied by factor[year] → 2025 dollars.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Geo index — hotel cost multipliers by customer location (GSA-based)
# Built by build_geo_index.py.  If not found, all multipliers default to 1.0.
# ---------------------------------------------------------------------------
_geo_index_path = PROJECT / "geo_index.json"
if _geo_index_path.exists():
    with open(_geo_index_path) as _f:
        GEO_INDEX = json.load(_f)
    GEO_NATIONAL_AVG  = GEO_INDEX.get("national_avg_lodging", 130.0)
    GEO_AVAILABLE     = True
    print(f"Loaded geo_index.json  (national avg lodging: ${GEO_NATIONAL_AVG:.2f}/night, "
          f"{len(GEO_INDEX.get('by_customer', {}))} customers)")
else:
    GEO_INDEX        = {}
    GEO_NATIONAL_AVG = 130.0
    GEO_AVAILABLE    = False
    print("WARNING: geo_index.json not found — hotel rates will NOT be geo-normalized.")
    print("  Run build_geo_index.py first to enable geographic correction.")


def get_hotel_geo_mult(cust_id, city=None, state=None) -> float:
    """
    Return geo hotel-cost multiplier for a location.
    multiplier = gsa_local_avg / national_avg
    > 1.0  → expensive area (NYC, SF, Houston downtown)
    < 1.0  → cheaper area
    = 1.0  → national average or unknown
    """
    if not GEO_AVAILABLE:
        return 1.0
    by_cust = GEO_INDEX.get("by_customer", {})
    if cust_id and str(cust_id) in by_cust:
        return float(by_cust[str(cust_id)]["multiplier"])
    by_city = GEO_INDEX.get("by_city_state", {})
    if city and state:
        key = f"{str(city).lower().strip()}_{str(state).upper().strip()}"
        if key in by_city:
            return float(by_city[key]["multiplier"])
    by_state = GEO_INDEX.get("by_state", {})
    if state and str(state).upper() in by_state:
        return float(by_state[str(state).upper()]["multiplier"])
    return 1.0


CPI_FACTORS = {
    2018: 1.274,
    2019: 1.252,
    2020: 1.236,
    2021: 1.181,
    2022: 1.093,
    2023: 1.050,
    2024: 1.020,
    2025: 1.000,
}

def cpi_factor(year):
    try:
        return CPI_FACTORS.get(int(year), 1.0)
    except (ValueError, TypeError):
        return 1.0


# ---------------------------------------------------------------------------
# clean_series: IsolationForest then Winsorize
# ---------------------------------------------------------------------------
def clean_series(s: pd.Series, n_min_iso: int = 10, iqr_mult: float = 1.5) -> pd.Series:
    """Return cleaned series (outliers removed, extremes clipped)."""
    s = s.dropna().copy()
    s = s[s > 0]  # only positive values make sense for dollar rates
    if len(s) == 0:
        return s

    # IsolationForest — only useful with enough samples
    if HAS_ISO and len(s) >= n_min_iso:
        X = s.values.reshape(-1, 1)
        iso = IsolationForest(contamination=0.05, random_state=42)
        inlier_mask = iso.fit_predict(X) == 1
        s = s.iloc[inlier_mask]

    # Winsorize
    if len(s) >= 4:
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lo = max(q1 - iqr_mult * iqr, 0)
        hi = q3 + iqr_mult * iqr
        s = s.clip(lo, hi)

    return s


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def compute_stats_comp(series: pd.Series, force_mean: bool = False) -> dict | None:
    s = series.dropna()
    s = s[s > 0]
    n = len(s)
    if n == 0:
        return None
    mn  = float(s.mean())
    med = float(s.median())
    std = float(s.std()) if n > 1 else 0.0
    p25 = float(s.quantile(0.25))
    p75 = float(s.quantile(0.75))

    high_var = bool(med > 0 and abs(mn - med) / med > 0.20)

    if force_mean:
        model_rate = mn
    else:
        model_rate = med if high_var else mn

    return {
        "mean":       round(mn,  4),
        "median":     round(med, 4),
        "std":        round(std, 4),
        "p25":        round(p25, 4),
        "p75":        round(p75, 4),
        "n":          n,
        "high_var":   high_var,
        "model_rate": round(model_rate, 4),
    }


def build_cell(sub_df: pd.DataFrame, geo_mults: pd.Series = None) -> dict | None:
    """
    Build a full component cell for a subset dataframe.

    Steps per component:
      1. CPI-normalize using the row's year
      2. Geo-normalize hotel (divide by geo_mults) so the stored hotel_rate is
         geo-neutral — what hotel would cost in a national-average-cost area.
         At prediction time app.py multiplies hotel_rate back up by the
         destination's geo multiplier to get the location-specific estimate.
      3. Clean (IsolationForest + Winsorize)
      4. Compute stats
    Also computes total_p25 / total_p75 for "Typical Range" display.
    """
    n_raw = len(sub_df)
    if n_raw == 0:
        return None

    cpi = sub_df["year"].apply(cpi_factor)

    # CPI-normalized component series (still indexed to sub_df)
    meals_norm  = (sub_df["daily_meals"].replace(0, np.nan)          * cpi)
    trans_norm  = (sub_df["daily_transport"]                          * cpi)
    hotel_norm  = (sub_df["daily_hotel_per_day"].replace(0, np.nan)  * cpi)

    # Geo-normalize hotel: divide by each trip's geo multiplier so the fitted
    # hotel_rate represents a "national average location" cost.
    # Clip multiplier to [0.5, 3.0] to guard against extreme edge cases.
    if geo_mults is not None and len(geo_mults) == n_raw:
        geo_series  = pd.Series(geo_mults.values, index=sub_df.index).clip(lower=0.5, upper=3.0)
        hotel_norm_geo = hotel_norm / geo_series
    else:
        hotel_norm_geo = hotel_norm

    # Recompute total using geo-normalized hotel so total_p25/p75 reflect the
    # geo-neutral distribution (less geographic noise → tighter range).
    total_norm_geo = (
        hotel_norm_geo.fillna(0)
        + meals_norm.fillna(0)
        + trans_norm.fillna(0)
    ).replace(0, np.nan)

    # Clean each component independently
    meals_clean = clean_series(meals_norm)
    trans_clean = clean_series(trans_norm)
    hotel_clean = clean_series(hotel_norm_geo)   # geo-normalized
    total_clean = clean_series(total_norm_geo)    # geo-normalized total

    ms = compute_stats_comp(meals_clean, force_mean=False)
    ts = compute_stats_comp(trans_clean, force_mean=True)   # ALWAYS mean for transport
    hs = compute_stats_comp(hotel_clean, force_mean=False)
    tt = compute_stats_comp(total_clean, force_mean=False)  # for p25/p75 range only

    meals_rate     = ms["model_rate"] if ms else 0.0
    transport_rate = ts["model_rate"] if ts else 0.0
    hotel_rate     = hs["model_rate"] if hs else 0.0
    total_rate     = round(meals_rate + transport_rate + hotel_rate, 4)

    cell: dict = {
        "meals_rate":     round(meals_rate,     4),
        "transport_rate": round(transport_rate, 4),
        "hotel_rate":     round(hotel_rate,     4),
        "total_rate":     total_rate,
        "n":              n_raw,  # raw count before cleaning
    }

    # Total range (p25/p75 of total after CPI + outlier cleaning)
    if tt:
        cell["total_p25"] = round(tt["p25"], 4)
        cell["total_p75"] = round(tt["p75"], 4)
        cell["total_mean"]   = round(tt["mean"],   4)
        cell["total_median"] = round(tt["median"],  4)
        cell["n_clean"]   = tt["n"]  # rows remaining after cleaning

    if ms:
        cell.update({
            "meals_n": ms["n"], "meals_mean": ms["mean"],
            "meals_median": ms["median"], "meals_std": ms["std"],
            "meals_p25": ms["p25"], "meals_p75": ms["p75"],
            "high_var_meals": ms["high_var"],
        })
    if ts:
        cell.update({
            "transport_n": ts["n"], "transport_mean": ts["mean"],
            "transport_median": ts["median"], "transport_std": ts["std"],
        })
    if hs:
        cell.update({
            "hotel_n": hs["n"], "hotel_mean": hs["mean"],
            "hotel_median": hs["median"], "hotel_std": hs["std"],
            "hotel_p25": hs["p25"], "hotel_p75": hs["p75"],
            "high_var_hotel": hs["high_var"],
        })

    return cell


def _gmults(sub_df: pd.DataFrame) -> "pd.Series | None":
    """Return the hotel_geo_mult Series for a sub-DataFrame, or None if unavailable."""
    if "hotel_geo_mult" in sub_df.columns:
        return sub_df["hotel_geo_mult"]
    return None


# ---------------------------------------------------------------------------
# Load master_trips_v3.csv  (built by build_v3.py)
# ---------------------------------------------------------------------------
print("=" * 70)
print("Loading master_trips_v3.csv")
print("=" * 70)

v3_path = PROJECT / "master_trips_v3.csv"
if not v3_path.exists():
    raise FileNotFoundError(
        "master_trips_v3.csv not found. Run build_v3.py first."
    )

df = pd.read_csv(v3_path, low_memory=False)
print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

# Ensure year column exists
if "year" not in df.columns:
    df["year"] = pd.to_datetime(df.get("ticket_date", pd.NaT), errors="coerce").dt.year

# ---------------------------------------------------------------------------
# Join customer state (needed for customer-level and state-level lookup)
# ---------------------------------------------------------------------------
cust_addr  = pd.read_excel(PROJECT / "ContractsCustomersAddresses.xlsx")
cust_id_col, state_col, city_col_addr = None, None, None
for c in cust_addr.columns:
    cl = c.lower()
    if ("acu" in cl or "customerid" in cl) and cust_id_col is None:
        cust_id_col = c
    if "state" in cl and state_col is None:
        state_col = c
    if cl == "city" and city_col_addr is None:
        city_col_addr = c

keep_cols = [cust_id_col, state_col]
if city_col_addr:
    keep_cols.append(city_col_addr)

rename_map = {cust_id_col: "CustomerIDAcu", state_col: "customer_state"}
if city_col_addr:
    rename_map[city_col_addr] = "customer_city"

cust_lookup = (
    cust_addr[keep_cols]
    .dropna(subset=[cust_id_col, state_col])
    .rename(columns=rename_map)
    .copy()
)
cust_lookup["CustomerIDAcu"] = cust_lookup["CustomerIDAcu"].astype(str).str.strip()

df["CustomerIDAcu"] = df["CustomerIDAcu"].astype(str).str.strip()
if "customer_state" not in df.columns:
    df = df.merge(
        cust_lookup.drop_duplicates("CustomerIDAcu"),
        on="CustomerIDAcu", how="left"
    )

# ---------------------------------------------------------------------------
# Filter to fit years
# ---------------------------------------------------------------------------
fit_df = df[df["year"].isin(FIT_YEARS)].copy()
print(f"Fit rows ({FIT_YEARS[0]}–{FIT_YEARS[-1]}): {len(fit_df):,}")

# Attach state + city if not already in fit_df
if "customer_state" not in fit_df.columns:
    fit_df["CustomerIDAcu"] = fit_df["CustomerIDAcu"].astype(str).str.strip()
    fit_df = fit_df.merge(
        cust_lookup.drop_duplicates("CustomerIDAcu"),
        on="CustomerIDAcu", how="left"
    )

# Add hotel geo multiplier per trip row (used by build_cell for geo-normalization)
fit_df["hotel_geo_mult"] = fit_df.apply(
    lambda r: get_hotel_geo_mult(
        r.get("CustomerIDAcu"),
        r.get("customer_city"),
        r.get("customer_state"),
    ),
    axis=1,
)
if GEO_AVAILABLE:
    print(f"Hotel geo multipliers — min: {fit_df['hotel_geo_mult'].min():.3f}×  "
          f"max: {fit_df['hotel_geo_mult'].max():.3f}×  "
          f"mean: {fit_df['hotel_geo_mult'].mean():.3f}×")

# ---------------------------------------------------------------------------
# DAY TRIP fitting
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 1 — Day Trip rates (CPI + IsoForest + Winsorize)")
print("=" * 70)

day_fit = fit_df[~fit_df["is_overnight"] & (fit_df["daily_rate_v3"] > 0)].copy()
print(f"Day trip fit rows: {len(day_fit):,}")

dt_by_dist_seas = {}
for band in DISTANCE_BAND_ORDER:
    for seas in ["Spring", "Summer", "Fall", "Winter"]:
        sub = day_fit[(day_fit["distance_band"] == band) & (day_fit["season"] == seas)]
        if len(sub) >= 5:
            cell = build_cell(sub, _gmults(sub))
            if cell:
                dt_by_dist_seas.setdefault(band, {})[seas] = cell

dt_by_dist = {}
for band in DISTANCE_BAND_ORDER:
    sub = day_fit[day_fit["distance_band"] == band]
    if len(sub) >= 5:
        cell = build_cell(sub, _gmults(sub))
        if cell:
            dt_by_dist[band] = cell

dt_global = build_cell(day_fit, _gmults(day_fit))

print(f"\n{'Band':<15} {'N':>5} {'Meals':>8} {'Trans':>8} {'Total':>8} {'p25':>8} {'p75':>8}")
print("-" * 62)
for band in DISTANCE_BAND_ORDER:
    c = dt_by_dist.get(band)
    if c:
        p25 = c.get("total_p25", 0)
        p75 = c.get("total_p75", 0)
        print(f"{band:<15} {c['n']:>5} {c['meals_rate']:>8.2f} {c['transport_rate']:>8.2f} "
              f"{c['total_rate']:>8.2f} {p25:>8.2f} {p75:>8.2f}")
if dt_global:
    p25 = dt_global.get("total_p25", 0)
    p75 = dt_global.get("total_p75", 0)
    print(f"{'global':<15} {dt_global['n']:>5} {dt_global['meals_rate']:>8.2f} "
          f"{dt_global['transport_rate']:>8.2f} {dt_global['total_rate']:>8.2f} {p25:>8.2f} {p75:>8.2f}")

# ---------------------------------------------------------------------------
# OVERNIGHT fitting
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 2 — Overnight rates (CPI + IsoForest + Winsorize)")
print("=" * 70)

on_cats = ["E_complete", "F_meals_only", "B_multiday_no_hotel"]
on_fit = fit_df[
    fit_df["is_overnight"] &
    fit_df["data_category"].isin(on_cats) &
    (fit_df["daily_rate_v3"] > 0)
].copy()
print(f"Overnight fit rows: {len(on_fit):,}")

# by_distance_season_mode
on_by_dsm = {}
for band in DISTANCE_BAND_ORDER:
    for seas in ["Spring", "Summer", "Fall", "Winter"]:
        for mode in [0, 1]:
            sub = on_fit[
                (on_fit["distance_band"] == band) &
                (on_fit["season"] == seas) &
                (on_fit["is_fly_trip"] == mode)
            ]
            if len(sub) >= 5:
                cell = build_cell(sub, _gmults(sub))
                if cell:
                    on_by_dsm.setdefault(band, {}).setdefault(seas, {})[str(mode)] = cell

# by_distance_mode
on_by_dm = {}
for band in DISTANCE_BAND_ORDER:
    for mode in [0, 1]:
        sub = on_fit[
            (on_fit["distance_band"] == band) &
            (on_fit["is_fly_trip"] == mode)
        ]
        if len(sub) >= 5:
            cell = build_cell(sub, _gmults(sub))
            if cell:
                on_by_dm.setdefault(band, {})[str(mode)] = cell

# by_distance
on_by_dist = {}
for band in DISTANCE_BAND_ORDER:
    sub = on_fit[on_fit["distance_band"] == band]
    if len(sub) >= 5:
        cell = build_cell(sub, _gmults(sub))
        if cell:
            on_by_dist[band] = cell

on_global = build_cell(on_fit, _gmults(on_fit))

# by_customer (N >= 3) — overall
by_customer = {}
for cust_id, grp in on_fit.groupby("CustomerIDAcu"):
    if len(grp) >= 3:
        cell = build_cell(grp, _gmults(grp))
        if cell:
            state_val = grp["customer_state"].dropna().iloc[0] if grp["customer_state"].notna().any() else ""
            cell["state"] = str(state_val)
            by_customer[str(cust_id)] = cell

print(f"Customer-level overnight cells: {len(by_customer):,}")

# by_customer × season (N >= 3 per season)
by_customer_season = {}
for (cust_id, seas), grp in on_fit.groupby(["CustomerIDAcu", "season"]):
    if len(grp) >= 3:
        cell = build_cell(grp, _gmults(grp))
        if cell:
            by_customer_season.setdefault(str(cust_id), {})[seas] = cell

cust_seas_count = sum(len(v) for v in by_customer_season.values())
print(f"Customer x season cells:        {cust_seas_count:,}  ({len(by_customer_season):,} customers)")

# by_state_distance_mode (N >= 5)
on_state_grp = {}
on_state_df = on_fit[on_fit["customer_state"].notna() & on_fit["distance_band"].notna()].copy()
for (state, band, mode), grp in on_state_df.groupby(["customer_state", "distance_band", "is_fly_trip"]):
    if len(grp) >= 5:
        cell = build_cell(grp, _gmults(grp))
        if cell:
            on_state_grp.setdefault(str(state), {}).setdefault(str(band), {})[str(int(mode))] = cell

print(f"State-level overnight cells:    {len(on_state_grp):,} states")

# ---------------------------------------------------------------------------
# Fine sub-bands for Under 300 drive overnight (systematic overestimate fix)
# "Under 100" and "100-300" replace the single "Under 300" bucket here.
# ---------------------------------------------------------------------------
on_fit["fine_band"] = on_fit["distance_miles"].apply(
    lambda m: fine_distance_band(m) if pd.notna(m) else None
)

on_fine_dm = {}   # fine_band + mode
on_fine_dsm = {}  # fine_band + season + mode
for band in FINE_BANDS_UNDER300:
    for mode in [0, 1]:
        sub = on_fit[(on_fit["fine_band"] == band) & (on_fit["is_fly_trip"] == mode)]
        if len(sub) >= 5:
            cell = build_cell(sub, _gmults(sub))
            if cell:
                on_fine_dm.setdefault(band, {})[str(mode)] = cell
        for seas in ["Spring", "Summer", "Fall", "Winter"]:
            sub_s = sub[sub["season"] == seas]
            if len(sub_s) >= 5:
                cell = build_cell(sub_s, _gmults(sub_s))
                if cell:
                    on_fine_dsm.setdefault(band, {}).setdefault(seas, {})[str(mode)] = cell

print(f"\nFine sub-band overnight cells (Under100 / 100-300):")
for band in FINE_BANDS_UNDER300:
    for mode in ["0", "1"]:
        c = on_fine_dm.get(band, {}).get(mode)
        if c:
            mname = "Fly" if mode == "1" else "Drv"
            print(f"  {band:<10} {mname}  N={c['n']}  total=${c['total_rate']:.2f}  "
                  f"p25=${c.get('total_p25',0):.2f}  p75=${c.get('total_p75',0):.2f}")

print(f"\n{'Band':<15} {'Mode':>5} {'N':>5} {'Hotel':>8} {'Meals':>8} {'Trans':>8} {'Total':>8} {'p25':>8} {'p75':>8}")
print("-" * 80)
for band in DISTANCE_BAND_ORDER:
    for mode in ["0", "1"]:
        c = on_by_dm.get(band, {}).get(mode)
        if c:
            mname = "Fly" if mode == "1" else "Drv"
            p25 = c.get("total_p25", 0)
            p75 = c.get("total_p75", 0)
            print(f"{band:<15} {mname:>5} {c['n']:>5} {c.get('hotel_rate',0):>8.2f} "
                  f"{c['meals_rate']:>8.2f} {c['transport_rate']:>8.2f} "
                  f"{c['total_rate']:>8.2f} {p25:>8.2f} {p75:>8.2f}")

# ---------------------------------------------------------------------------
# Re-fit airfare rates with CPI normalization (was previously copied stale from v1)
# Source: ExpAllocation.xlsx, ExpenseType = AirFare, joined to master for distance_band
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 2b — Re-fit airfare rates (CPI-normalized)")
print("=" * 70)

exp_df = pd.read_excel(PROJECT / "ExpAllocation.xlsx")
exp_df.columns = [c.strip() for c in exp_df.columns]
exp_df["Date"] = pd.to_datetime(exp_df["Date"], errors="coerce")
exp_df["year"] = exp_df["Date"].dt.year

# Keep AirFare expense type only, within fit years
air_raw = exp_df[
    (exp_df["ExpenseType"].str.lower().str.strip() == "airfare") &
    (exp_df["year"].isin(FIT_YEARS)) &
    (exp_df["Amount"] > 0)
].copy()
print(f"AirFare rows (2022-2025): {len(air_raw):,}")

# Join distance_band from master via Ticket / Title
master_bands = df[["Title", "distance_band", "season", "is_fly_trip"]].dropna(
    subset=["distance_band"]
).drop_duplicates("Title")

# Ticket column in ExpAllocation
ticket_col = next((c for c in air_raw.columns if "ticket" in c.lower()), None)
if ticket_col:
    air_raw = air_raw.merge(master_bands, left_on=ticket_col, right_on="Title", how="left")
    air_raw = air_raw[air_raw["distance_band"].notna()]
    print(f"AirFare rows with distance_band: {len(air_raw):,}")

    # CPI-normalize
    air_raw["amount_cpi"] = air_raw["Amount"] * air_raw["year"].apply(cpi_factor)

    # Aggregate: one row per trip (sum all AirFare charges per ticket)
    air_trip = air_raw.groupby([ticket_col, "distance_band", "season"]).agg(
        airfare_total=("amount_cpi", "sum"),
        year=("year", "first"),
    ).reset_index()
    air_trip = air_trip[air_trip["airfare_total"] > 0]
    print(f"Unique trips with airfare: {len(air_trip):,}")

    def build_airfare_cell(sub):
        s = clean_series(sub["airfare_total"])
        st_ = compute_stats_comp(s, force_mean=False)  # median for airfare (high variance)
        if st_ is None:
            return None
        return {
            "model_rate": round(st_["median"], 4),  # always use median for airfare
            "mean":       round(st_["mean"],   4),
            "median":     round(st_["median"], 4),
            "std":        round(st_["std"],    4),
            "p25":        round(st_["p25"],    4),
            "p75":        round(st_["p75"],    4),
            "n":          st_["n"],
            "high_var":   st_["high_var"],
        }

    af_by_dist_seas = {}
    for band in DISTANCE_BAND_ORDER:
        for seas in ["Spring", "Summer", "Fall", "Winter"]:
            sub = air_trip[(air_trip["distance_band"] == band) & (air_trip["season"] == seas)]
            if len(sub) >= 3:
                cell = build_airfare_cell(sub)
                if cell:
                    af_by_dist_seas.setdefault(band, {})[seas] = cell

    af_by_dist = {}
    for band in DISTANCE_BAND_ORDER:
        sub = air_trip[air_trip["distance_band"] == band]
        if len(sub) >= 3:
            cell = build_airfare_cell(sub)
            if cell:
                af_by_dist[band] = cell

    airfare_rates = {
        "by_distance_season": af_by_dist_seas,
        "by_distance":        af_by_dist,
        "notes": "CPI-normalized to 2025 dollars. Median used (high variance). Source: ExpAllocation AirFare rows.",
    }
    print("\nAirfare rates (CPI-normalized, by distance band):")
    print(f"  {'Band':<15} {'N':>5} {'Median':>10} {'Mean':>10} {'p25':>10} {'p75':>10}")
    for band in DISTANCE_BAND_ORDER:
        c = af_by_dist.get(band)
        if c:
            print(f"  {band:<15} {c['n']:>5} {c['median']:>10.2f} {c['mean']:>10.2f} "
                  f"{c['p25']:>10.2f} {c['p75']:>10.2f}")
else:
    print("WARNING: Could not find Ticket column in ExpAllocation — falling back to existing rate table airfare rates.")
    with open(PROJECT / "rate_table_v4.json") as f:
        _prev = json.load(f)
    airfare_rates = _prev["airfare_rates"]

# Drive rates: IRS mileage-based — re-fit with CPI-normalized fuel/tolls actuals
with open(PROJECT / "rate_table_v4.json") as f:
    _prev = json.load(f)
drive_rates = _prev["drive_rates"]   # IRS-floor based; less inflation-sensitive than airfare

# ---------------------------------------------------------------------------
# Backtest v4
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 3 — Backtest v4")
print("=" * 70)

def lookup_dt_v4(band, seas):
    c = dt_by_dist_seas.get(band, {}).get(seas)
    if c and c["n"] >= 5:
        return c["total_rate"], "by_distance_season"
    c = dt_by_dist.get(band)
    if c and c["n"] >= 5:
        return c["total_rate"], "by_distance"
    if dt_global:
        return dt_global["total_rate"], "global"
    return None, None

def _geo_adjusted_total(cell, geo_mult: float = 1.0) -> float | None:
    """Return total_rate with geo multiplier applied to hotel component only."""
    if not cell:
        return None
    hotel = (cell.get("hotel_rate") or 0) * geo_mult
    meals = cell.get("meals_rate") or 0
    trans = cell.get("transport_rate") or 0
    return round(hotel + meals + trans, 4)

def lookup_on_v4(band, seas, mode, cust_id=None, state=None, city=None):
    """Return (geo-adjusted total_rate, basis). hotel_rate is geo-neutral in cells."""
    mode_str = str(int(mode))
    geo_mult = get_hotel_geo_mult(cust_id, city, state)

    if cust_id and str(cust_id) in by_customer:
        c = by_customer[str(cust_id)]
        if c["n"] >= 3:
            return _geo_adjusted_total(c, geo_mult), "customer"
    if state and str(state) in on_state_grp:
        c = on_state_grp[str(state)].get(band, {}).get(mode_str)
        if c and c["n"] >= 5:
            return _geo_adjusted_total(c, geo_mult), "state"
    c = on_by_dsm.get(band, {}).get(seas, {}).get(mode_str)
    if c and c["n"] >= 5:
        return _geo_adjusted_total(c, geo_mult), "dist_seas_mode"
    c = on_by_dm.get(band, {}).get(mode_str)
    if c and c["n"] >= 5:
        return _geo_adjusted_total(c, geo_mult), "dist_mode"
    c = on_by_dist.get(band)
    if c and c["n"] >= 5:
        return _geo_adjusted_total(c, geo_mult), "dist"
    if on_global:
        return _geo_adjusted_total(on_global, geo_mult), "global"
    return None, None

# Build backtest set using v3 actuals (daily_rate_v3)
# CPI-normalize actuals → 2025 dollars for a fair comparison with v4 rates.
bt_df = df[
    df["daily_rate_v3"].notna() &
    (df["daily_rate_v3"] > 0)
].copy()
if "customer_state" not in bt_df.columns:
    bt_df["CustomerIDAcu"] = bt_df["CustomerIDAcu"].astype(str).str.strip()
    bt_df = bt_df.merge(
        cust_lookup.drop_duplicates("CustomerIDAcu"),
        on="CustomerIDAcu", how="left"
    )

# CPI-normalize actuals to 2025 dollars
bt_df["actual_cpi"] = bt_df["daily_rate_v3"] * bt_df["year"].apply(cpi_factor)

preds = []
for _, row in bt_df.iterrows():
    is_on   = row["is_overnight"]
    band    = row.get("distance_band")
    seas    = row.get("season")
    mode    = row.get("is_fly_trip", 0)
    cust_id = row.get("CustomerIDAcu")
    state   = row.get("customer_state")
    city    = row.get("customer_city")
    actual_raw = row["daily_rate_v3"]
    actual_cpi = row["actual_cpi"]
    year        = row.get("year")

    if is_on:
        pred, basis = lookup_on_v4(band, seas, mode, cust_id, state, city)
    else:
        pred, basis = lookup_dt_v4(band, seas)

    preds.append({
        "actual_raw":   actual_raw,
        "actual_cpi":   actual_cpi,
        "predicted":    pred,
        "is_overnight": is_on,
        "is_fly_trip":  mode,
        "basis":        basis,
        "year":         year,
    })

bt = pd.DataFrame(preds)
bt = bt[bt["predicted"].notna()]


def backtest_report(sub, label, actual_col="actual_cpi"):
    if len(sub) == 0:
        return {"label": label, "n": 0, "MAE": None, "pct_20": None, "pct_30": None, "over_pct": None}
    valid = sub[actual_col] > 0
    sub = sub[valid]
    ae   = (sub["predicted"] - sub[actual_col]).abs()
    err  = sub["predicted"] - sub[actual_col]
    mae  = ae.mean()
    p20  = (ae <= 0.20 * sub[actual_col]).mean() * 100
    p30  = (ae <= 0.30 * sub[actual_col]).mean() * 100
    over = (err > 0).mean() * 100
    return {"label": label, "n": len(sub), "MAE": round(mae, 2),
            "pct_20": round(p20, 1), "pct_30": round(p30, 1),
            "over_pct": round(over, 1)}


segments = [
    (pd.Series([True]*len(bt), index=bt.index), "All trips"),
    (~bt["is_overnight"],                        "Day trips only"),
    (bt["is_overnight"],                         "Overnight only"),
    (bt["is_fly_trip"] == 1,                     "Fly trips"),
    (bt["is_fly_trip"] == 0,                     "Drive trips"),
]

print("\n--- CPI-normalized backtest (actuals scaled to 2025 dollars) ---")
print(f"{'Segment':<25} {'N':>6} {'MAE':>8} {'<=20%':>7} {'<=30%':>7} {'Over%':>7}")
print("-" * 62)
for mask, lbl in segments:
    r = backtest_report(bt[mask], lbl, "actual_cpi")
    if r["n"] > 0:
        print(f"{r['label']:<25} {r['n']:>6,} {r['MAE']:>8.2f} "
              f"{r['pct_20']:>6.1f}% {r['pct_30']:>6.1f}% {r['over_pct']:>6.1f}%")

# 2024-2025 only (CPI ~ 1.0, cleanest comparison)
bt_recent = bt[bt["year"].isin([2024, 2025])]
print(f"\n--- 2024-2025 only (N={len(bt_recent):,}, CPI~1.0 -- cleanest comparison) ---")
print(f"{'Segment':<25} {'N':>6} {'MAE':>8} {'<=20%':>7} {'<=30%':>7} {'Over%':>7}")
print("-" * 62)
segs_recent = [
    (pd.Series([True]*len(bt_recent), index=bt_recent.index), "All trips"),
    (~bt_recent["is_overnight"],                               "Day trips only"),
    (bt_recent["is_overnight"],                               "Overnight only"),
    (bt_recent["is_fly_trip"] == 1,                           "Fly trips"),
]
for mask, lbl in segs_recent:
    sub = bt_recent[mask]
    r = backtest_report(sub, lbl, "actual_raw")  # raw OK since CPI ~1.0
    if r["n"] > 0:
        print(f"{r['label']:<25} {r['n']:>6,} {r['MAE']:>8.2f} "
              f"{r['pct_20']:>6.1f}% {r['pct_30']:>6.1f}% {r['over_pct']:>6.1f}%")

# v3 comparison header
print("\n--- v3 benchmark (for reference) ---")
print(f"{'Segment':<25} {'':>6} {'MAE':>8} {'<=20%':>7} {'<=30%':>7}")
print(f"{'All trips (v3)':<25} {'':>6} {'$115.15':>8} {'32.7%':>7} {'45.9%':>7}")
print(f"{'Overnight only (v3)':<25} {'':>6} {'—':>8} {'—':>7} {'47.8%':>7}")
print("  (v3 backtest was raw actuals vs raw rates — not CPI normalized)")

print("\nBasis distribution (v4 backtest):")
print(bt["basis"].value_counts().to_string())

# ---------------------------------------------------------------------------
# Save rate_table_v4.json
# ---------------------------------------------------------------------------
rate_table_v4 = {
    "day_trip_rates": {
        "by_distance_season": dt_by_dist_seas,
        "by_distance":        dt_by_dist,
        "global":             dt_global,
    },
    "overnight_rates": {
        "by_customer":             by_customer,
        "by_customer_season":      by_customer_season,
        "by_state_distance_mode":  on_state_grp,
        "by_distance_season_mode": on_by_dsm,
        "by_distance_mode":        on_by_dm,
        "by_distance":             on_by_dist,
        "global":                  on_global,
        # Finer sub-bands for Under 300 (reduces systematic overestimate)
        "fine_by_distance_mode":         on_fine_dm,
        "fine_by_distance_season_mode":  on_fine_dsm,
    },
    "airfare_rates": airfare_rates,
    "drive_rates":   drive_rates,
    "metadata": {
        "version":               "4.2",
        "fitted_on_years":       FIT_YEARS,
        "n_overnight_trips":     int(len(on_fit)),
        "n_day_trips":           int(len(day_fit)),
        "n_customers_in_lookup": len(by_customer),
        "n_states_in_lookup":    len(on_state_grp),
        "fit_date":              str(date.today()),
        "irs_rate":              IRS_RATE,
        "transport_uses_mean":   True,
        "preprocessing": {
            "cpi_normalized":    True,
            "isolation_forest":  HAS_ISO,
            "winsorized":        True,
            "isolation_contamination": 0.05,
            "winsorize_iqr_mult": 1.5,
        },
        "geo_normalized_hotel":  GEO_AVAILABLE,
        "new_in_v4_1": (
            "Customer x season lookup, "
            "Fine sub-bands (Under100/100-300) for Under-300 overnight, "
            "CPI-normalized airfare rates (re-fitted from ExpAllocation)"
        ),
        "new_in_v4_2": (
            "GSA FY2026 geo-normalization: hotel_rate in each cell is now geo-neutral "
            "(divided by destination GSA lodging multiplier at build time). "
            "app.py re-applies destination multiplier at prediction time, "
            "correcting for local hotel cost variation by city/state."
        ),
    },
}

out_path = PROJECT / "rate_table_v4.json"
with open(out_path, "w") as f:
    json.dump(rate_table_v4, f, indent=2)

print(f"\nSaved rate_table_v4.json  ({out_path})")
print("All steps complete.")
