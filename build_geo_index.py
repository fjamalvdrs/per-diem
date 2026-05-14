"""
build_geo_index.py
==================
Downloads GSA FY2026 per diem lodging + M&IE rates and builds geo_index.json.

geo_index.json maps customer locations to a hotel cost multiplier relative
to the national average, so build_v4.py can geo-normalize hotel costs before
fitting rate cells — and app.py can re-apply the destination's local cost
level at prediction time.

Also stores gsa_meals (M&IE rate) per location so app.py can apply
max(historical_meals, gsa_meals) as a floor — same pattern as hotel.

Usage:
    python build_geo_index.py [--api-key YOUR_KEY]

If no API key is provided, uses DEMO_KEY (30 req/hr per IP — fine for a
one-time build). Get a free production key at: https://api.data.gov/signup/

Output files:
    gsa_rates_fy2026.csv   — raw GSA per diem table (all cities/states)
    geo_index.json         — compiled multipliers by customer / city-state / state
"""

import argparse
import json
import statistics
import requests
import pandas as pd
from pathlib import Path
from datetime import date

PROJECT = Path("G:/After Sales Team/PROJECTS/per-diem-model")

# GSA standard lodging rate for all locations without a specific city listing.
# FY2026: $110/night.
GSA_STANDARD_RATE = 110

# GSA standard M&IE rate for all locations without a specific city listing.
# FY2025: $68/day.
GSA_STANDARD_MEALS = 68

CANADIAN_PROVINCES = {
    "ON", "BC", "AB", "QC", "MB", "SK", "NS", "NB",
    "PE", "NL", "NT", "YT", "NU",
}


# ---------------------------------------------------------------------------
# Step 1 — Fetch GSA FY2026 rates
# ---------------------------------------------------------------------------

US_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC",
]

def fetch_gsa_rates(api_key: str, year: int = 2025) -> pd.DataFrame:
    """
    Fetch GSA per diem lodging + M&IE rates for all 50 US states + DC by making
    one request per state (the state/all endpoint is broken in the v2 API).
    Returns a flat DataFrame: state, city, county, is_standard, avg_lodging, meals.
    avg_lodging is the average of the 12 monthly lodging values.
    meals is the GSA M&IE rate (single value per city, no monthly variation).
    AK and HI return empty from the API — filled with standard rates.
    """
    base = "https://api.gsa.gov/travel/perdiem/v2/rates/state/{state}/year/{year}"
    print(f"Fetching GSA FY{year} per diem rates (one request per state) ...")

    rows = []
    for state in US_STATES:
        url  = base.format(state=state, year=year)
        resp = requests.get(url, params={"api_key": api_key}, timeout=20)
        if resp.status_code != 200 or not resp.text.strip():
            print(f"  SKIP {state}: HTTP {resp.status_code}")
            continue
        data         = resp.json()
        state_blocks = data.get("rates", [])
        if not state_blocks:
            # AK and HI return empty — add a standard-rate placeholder
            rows.append({
                "state":       state,
                "city":        "Standard Rate",
                "county":      "",
                "is_standard": True,
                "avg_lodging": GSA_STANDARD_RATE,
                "meals":       GSA_STANDARD_MEALS,
            })
            print(f"  {state}: no city data — using standard rate")
            continue
        # Per-state call returns one block in rates[]
        for entry in state_blocks[0].get("rate", []):
            city        = str(entry.get("city",   "")).strip()
            county      = str(entry.get("county", "")).strip()
            std_raw     = entry.get("standardRate", "false")
            is_standard = str(std_raw).lower() == "true"
            months      = entry.get("months", {}).get("month", [])
            if months:
                vals        = [m.get("value", 0) for m in months if m.get("value", 0) > 0]
                avg_lodging = round(sum(vals) / len(vals), 2) if vals else GSA_STANDARD_RATE
            else:
                avg_lodging = GSA_STANDARD_RATE
            meals = int(entry.get("meals", GSA_STANDARD_MEALS) or GSA_STANDARD_MEALS)
            rows.append({
                "state":       state,
                "city":        city,
                "county":      county,
                "is_standard": is_standard,
                "avg_lodging": avg_lodging,
                "meals":       meals,
            })

    df = pd.DataFrame(rows)
    print(f"  Fetched {len(df):,} rate entries across {df['state'].nunique()} states")
    non_std = df[~df["is_standard"]]
    print(f"  Non-standard (city-specific) entries: {len(non_std):,}")
    print(f"  Lodging rate range: ${df['avg_lodging'].min():.0f} – ${df['avg_lodging'].max():.0f}/night")
    print(f"  M&IE rate range:    ${df['meals'].min():.0f} – ${df['meals'].max():.0f}/day")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Build lookup structures
# ---------------------------------------------------------------------------

def build_city_state_lookup(gsa_df: pd.DataFrame) -> dict:
    """
    city+state key (lowercase) → {"lodging": avg_lodging, "meals": meals}.
    Includes both city name and county name keys for broader matching.
    """
    lookup = {}
    for _, row in gsa_df[~gsa_df["is_standard"]].iterrows():
        state  = row["state"].upper()
        entry  = {"lodging": row["avg_lodging"], "meals": int(row.get("meals", GSA_STANDARD_MEALS) or GSA_STANDARD_MEALS)}
        # City key
        if row["city"]:
            key = f"{row['city'].lower().strip()}_{state}"
            if key not in lookup:
                lookup[key] = entry
        # County key (county seat often listed by county name)
        if row["county"]:
            key2 = f"{row['county'].lower().strip()}_{state}"
            if key2 not in lookup:
                lookup[key2] = entry
    return lookup


def build_state_avg(gsa_df: pd.DataFrame) -> dict:
    """state → {"lodging": avg_lodging, "meals": avg_meals} of all city-specific entries."""
    avgs = {}
    for state, grp in gsa_df[~gsa_df["is_standard"]].groupby("state"):
        avgs[str(state).upper()] = {
            "lodging": round(float(grp["avg_lodging"].mean()), 2),
            "meals":   round(float(grp["meals"].mean()), 2),
        }
    return avgs


def match_location(city: str, state: str,
                   city_state_lookup: dict,
                   state_avg: dict,
                   national_avg: float) -> tuple[float, int, str]:
    """
    Return (gsa_lodging, gsa_meals, source_label) for a given city + state.
    Source priority: exact city → partial city → state average → national average.
    """
    national_meals = GSA_STANDARD_MEALS

    if city and state:
        state_up  = state.upper().strip()
        city_low  = city.lower().strip()
        # Exact match
        exact_key = f"{city_low}_{state_up}"
        if exact_key in city_state_lookup:
            entry = city_state_lookup[exact_key]
            return entry["lodging"], entry["meals"], "city_exact"
        # Partial match (city name is a substring of a GSA entry key)
        for k, v in city_state_lookup.items():
            if k.endswith(f"_{state_up}") and city_low in k:
                return v["lodging"], v["meals"], "city_partial"

    if state and state.upper() in state_avg:
        entry = state_avg[state.upper()]
        return entry["lodging"], round(entry["meals"]), "state_avg"

    return national_avg, national_meals, "national_avg"


# ---------------------------------------------------------------------------
# Step 3 — Map customers to multipliers
# ---------------------------------------------------------------------------

def map_customers(cust_df: pd.DataFrame,
                  cust_id_col: str,
                  city_col: str | None,
                  state_col: str | None,
                  city_state_lookup: dict,
                  state_avg: dict,
                  national_avg: float) -> dict:
    """
    Returns by_customer dict: { CustomerIDAcu → {multiplier, city, state, gsa_rate, source} }
    """
    by_customer = {}
    source_counts: dict[str, int] = {}

    for _, row in cust_df.iterrows():
        cust_id = str(row.get(cust_id_col, "")).strip()
        if not cust_id or cust_id.lower() == "nan":
            continue

        city  = str(row.get(city_col,  "")).strip() if city_col  else ""
        state = str(row.get(state_col, "")).strip() if state_col else ""

        # Canadian customers — no US GSA data; treat as international (mult = 1.0)
        if state.upper() in CANADIAN_PROVINCES:
            by_customer[cust_id] = {
                "multiplier": 1.0,
                "city":       city,
                "state":      state,
                "gsa_rate":   national_avg,
                "gsa_meals":  GSA_STANDARD_MEALS,
                "source":     "international",
            }
            source_counts["international"] = source_counts.get("international", 0) + 1
            continue

        gsa_rate, gsa_meals, source = match_location(
            city, state, city_state_lookup, state_avg, national_avg
        )
        multiplier = round(gsa_rate / national_avg, 6)

        by_customer[cust_id] = {
            "multiplier": multiplier,
            "city":       city,
            "state":      state,
            "gsa_rate":   gsa_rate,
            "gsa_meals":  gsa_meals,
            "source":     source,
        }
        source_counts[source] = source_counts.get(source, 0) + 1

    print(f"\n  Customer geo multiplier sources:")
    for src, cnt in sorted(source_counts.items()):
        print(f"    {src:<20}: {cnt:>4} customers")

    return by_customer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build geo_index.json from GSA FY2026 per diem rates."
    )
    parser.add_argument(
        "--api-key", default="DEMO_KEY",
        help="api.data.gov API key (free at https://api.data.gov/signup/). "
             "Defaults to DEMO_KEY (30 req/hr — fine for one-time use).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Building geo_index.json from GSA FY2026 Per Diem Rates")
    print("=" * 70)

    # 1. Fetch
    gsa_df = fetch_gsa_rates(args.api_key, year=2025)
    csv_path = PROJECT / "gsa_rates_fy2026.csv"
    gsa_df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path.name}")

    # 2. Build lookups
    city_state_lookup = build_city_state_lookup(gsa_df)
    state_avg         = build_state_avg(gsa_df)

    # National average: mean of all city-specific (non-standard-rate) lodging values.
    # This is the denominator for all multipliers — "average-cost US city".
    non_standard_rates = gsa_df[~gsa_df["is_standard"]]["avg_lodging"].tolist()
    national_avg       = round(sum(non_standard_rates) / len(non_standard_rates), 2)

    print(f"\n  National avg GSA lodging (non-standard cities): ${national_avg:.2f}/night")
    print(f"  GSA standard rate (unlisted locations):         ${GSA_STANDARD_RATE}/night")

    # 3. Load customer addresses
    print()
    print("=" * 70)
    print("Mapping customers to GSA rates")
    print("=" * 70)

    cust_df = pd.read_excel(PROJECT / "ContractsCustomersAddresses.xlsx")
    print(f"  Loaded {len(cust_df):,} customer rows, columns: {list(cust_df.columns)}")

    # Detect column names robustly
    cust_id_col = next(
        (c for c in cust_df.columns
         if "acu" in c.lower() or "customerid" in c.lower()), None
    )
    city_col = next(
        (c for c in cust_df.columns if c.lower() == "city"), None
    )
    state_col = next(
        (c for c in cust_df.columns if "state" in c.lower()), None
    )
    zip_col = next(
        (c for c in cust_df.columns if "zip" in c.lower()), None
    )

    print(f"  Columns found — ID: {cust_id_col}, City: {city_col}, "
          f"State: {state_col}, ZIP: {zip_col}")

    if not cust_id_col:
        raise ValueError("Cannot find customer ID column in ContractsCustomersAddresses.xlsx")

    # 4. Map each customer
    by_customer = map_customers(
        cust_df, cust_id_col, city_col, state_col,
        city_state_lookup, state_avg, national_avg,
    )
    print(f"\n  Total customers mapped: {len(by_customer):,}")

    # 5. Summary stats on multipliers (US only)
    us_mults = [
        v["multiplier"] for v in by_customer.values()
        if v.get("source") != "international"
    ]
    if us_mults:
        print(f"\n  US customer multiplier stats:")
        print(f"    Min    : {min(us_mults):.3f}×")
        print(f"    Max    : {max(us_mults):.3f}×")
        print(f"    Mean   : {statistics.mean(us_mults):.3f}×")
        print(f"    Median : {statistics.median(us_mults):.3f}×")
        print(f"    Stdev  : {statistics.stdev(us_mults):.3f}")

    # Top 10 highest and lowest for sanity check
    sorted_by_mult = sorted(
        [(k, v) for k, v in by_customer.items() if v.get("source") != "international"],
        key=lambda x: x[1]["multiplier"], reverse=True,
    )
    print("\n  Top 5 highest-cost locations:")
    for cid, info in sorted_by_mult[:5]:
        print(f"    {cid:<10} {info['city']:<20} {info['state']}  "
              f"${info['gsa_rate']:.0f}/night  {info['multiplier']:.2f}×")
    print("  Top 5 lowest-cost locations:")
    for cid, info in sorted_by_mult[-5:]:
        print(f"    {cid:<10} {info['city']:<20} {info['state']}  "
              f"${info['gsa_rate']:.0f}/night  {info['multiplier']:.2f}×")

    # 6. Build city_state and state entries (for manual customer lookups in app.py)
    by_city_state = {
        k: {
            "multiplier": round(v["lodging"] / national_avg, 6),
            "gsa_rate":   v["lodging"],
            "gsa_meals":  v["meals"],
        }
        for k, v in city_state_lookup.items()
    }
    by_state_entry = {
        st: {
            "multiplier": round(v["lodging"] / national_avg, 6),
            "avg_rate":   v["lodging"],
            "gsa_meals":  round(v["meals"]),
        }
        for st, v in state_avg.items()
    }

    # 7. Save geo_index.json
    print()
    print("=" * 70)
    print("Saving geo_index.json")
    print("=" * 70)

    geo_index = {
        "national_avg_lodging": national_avg,
        "standard_rate":        GSA_STANDARD_RATE,
        "by_customer":          by_customer,
        "by_city_state":        by_city_state,
        "by_state":             by_state_entry,
        "metadata": {
            "source":               "GSA FY2026 Per Diem Rates",
            "api_endpoint":         "https://api.gsa.gov/travel/perdiem/v2/rates/state/all/year/2026",
            "fiscal_year":          2026,
            "built_date":           str(date.today()),
            "n_customers":          len(by_customer),
            "n_city_state_entries": len(by_city_state),
            "n_states":             len(by_state_entry),
            "national_avg_lodging": national_avg,
            "note": (
                "multiplier = gsa_local_avg / national_avg. "
                "Apply to hotel component only (meals/transport not geo-corrected). "
                "Canadian customers and unlisted locations default to 1.0."
            ),
        },
    }

    out_path = PROJECT / "geo_index.json"
    with open(out_path, "w") as f:
        json.dump(geo_index, f, indent=2)

    print(f"  Saved geo_index.json")
    print(f"    Customers  : {len(by_customer):,}")
    print(f"    City-state : {len(by_city_state):,}")
    print(f"    States     : {len(by_state_entry):,}")
    print()
    print("Next step: run build_v4.py to regenerate rate_table_v4.json")
    print("with geo-normalized hotel rates.")


if __name__ == "__main__":
    main()
