"""
Census data preprocessor. Run once to build the lookup table:

    python scripts/build_census_index.py

Writes:
  data/census_lookup.json — structured census data for fast lookups

Rerun when census data files change.
"""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
CENSUS_DIR = ROOT / "Census Data"

# Primary data source with state-level historical data
MAIN_FILE = CENSUS_DIR / "Summary Spreadsheets of KA Population" / "Summary of KA Population 1910-2020" / "Korean-American Population 1910 - 2020 Koreans and All.xlsx"

# 2020 Census with county-level data
COUNTY_FILE_2020 = CENSUS_DIR / "2020 Census" / "2020 DHCA Korean Alone or In Combination.xlsx"

# State name normalization
STATE_ABBREV = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
    "puerto rico": "PR",
}

# Reverse mapping
ABBREV_TO_STATE = {v: k for k, v in STATE_ABBREV.items()}


def normalize_state(name: str) -> str | None:
    """Convert state name to abbreviation."""
    if not name:
        return None
    name_lower = name.lower().strip()
    if name_lower in STATE_ABBREV:
        return STATE_ABBREV[name_lower]
    if name_lower.upper() in ABBREV_TO_STATE:
        return name_lower.upper()
    return None


def safe_int(val) -> int | None:
    """Convert value to int, handling NaN and strings."""
    if pd.isna(val):
        return None
    if isinstance(val, str):
        val = val.replace(",", "").strip()
        if val == "-" or val == "":
            return None
        try:
            return int(float(val))
        except ValueError:
            return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def load_state_historical_data() -> dict:
    """Load state-level data from 1910-2020."""
    if not MAIN_FILE.exists():
        print(f"Warning: Main census file not found: {MAIN_FILE}")
        return {}

    print(f"Reading {MAIN_FILE.name}...")

    # Read the "By State All 1910-2020" sheet
    try:
        df = pd.read_excel(MAIN_FILE, sheet_name="By State All 1910-2020", header=None)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return {}

    # Find the header row (contains year columns like 1910, 1920, etc.)
    header_row = None
    for i, row in df.iterrows():
        row_str = " ".join(str(v) for v in row.values if pd.notna(v))
        if "1910" in row_str and "2020" in row_str:
            header_row = i
            break

    if header_row is None:
        print("Could not find header row with year columns")
        return {}

    # Set headers and skip to data
    df.columns = df.iloc[header_row]
    df = df.iloc[header_row + 1:].reset_index(drop=True)

    # Find the state column (first column with state names)
    state_col = df.columns[0]

    # Extract year columns
    year_cols = {}
    for col in df.columns:
        col_str = str(col).strip()
        # Handle various formats: 1910, 1950.0, 2000*, 2010*
        # Extract 4-digit year from start of string
        match = re.match(r"^(19\d{2}|20[0-2]\d)", col_str.replace(".0", ""))
        if match:
            year = int(match.group(1))
            # Skip comparison columns like "2010 vs 2020"
            if " vs " not in col_str:
                year_cols[year] = col

    state_data = {}
    for _, row in df.iterrows():
        state_name = str(row[state_col]).strip() if pd.notna(row[state_col]) else ""
        state_abbrev = normalize_state(state_name)

        if not state_abbrev:
            continue

        state_data[state_abbrev] = {}
        for year, col in year_cols.items():
            val = safe_int(row[col])
            if val is not None:
                state_data[state_abbrev][str(year)] = {"korean_alone": val}

    print(f"  Loaded {len(state_data)} states with historical data")
    return state_data


def load_county_data_2020() -> dict:
    """Load county-level data from 2020 census."""
    if not COUNTY_FILE_2020.exists():
        print(f"Warning: County census file not found: {COUNTY_FILE_2020}")
        return {}

    print(f"Reading {COUNTY_FILE_2020.name}...")

    try:
        df = pd.read_excel(COUNTY_FILE_2020, sheet_name="County")
    except Exception as e:
        print(f"Error reading county data: {e}")
        return {}

    county_data = {}
    for _, row in df.iterrows():
        state = str(row.get("State", "")).strip()
        county_raw = str(row.get("County", "")).strip()
        total = safe_int(row.get("Total"))

        if not state or not county_raw:
            continue

        # Clean county name (remove " County, State" suffix)
        county = re.sub(r"\s+County,.*$", "", county_raw).strip()
        # Also try removing just ", State"
        if county == county_raw:
            county = re.sub(r",\s*\w+(\s+\w+)?$", "", county_raw).strip()

        state_abbrev = normalize_state(state) or state.upper()

        if state_abbrev not in county_data:
            county_data[state_abbrev] = {}

        if total is not None:
            county_data[state_abbrev][county] = {
                "2020": {
                    "korean_combined": total,
                }
            }

    total_counties = sum(len(counties) for counties in county_data.values())
    print(f"  Loaded {total_counties} counties across {len(county_data)} states")
    return county_data


def compute_national_totals(state_data: dict) -> dict:
    """Compute national totals from state data."""
    national = {}

    # Get all years present in the data
    all_years = set()
    for state_info in state_data.values():
        all_years.update(state_info.keys())

    for year in sorted(all_years):
        total = 0
        for state_info in state_data.values():
            if year in state_info and state_info[year].get("korean_alone"):
                total += state_info[year]["korean_alone"]
        if total > 0:
            national[year] = {"korean_alone": total}

    return national


def main():
    DATA.mkdir(exist_ok=True)

    # Load data from various sources
    state_data = load_state_historical_data()
    county_data = load_county_data_2020()
    national_data = compute_national_totals(state_data)

    # Build the final lookup structure
    census_lookup = {
        "national": national_data,
        "state": state_data,
        "county": county_data,
    }

    # Save to JSON
    output_path = DATA / "census_lookup.json"
    output_path.write_text(json.dumps(census_lookup, indent=2, ensure_ascii=False))

    print(f"\nDone. Saved to {output_path}")
    print(f"  National: {len(national_data)} years")
    print(f"  States: {len(state_data)} states")
    print(f"  Counties: {sum(len(c) for c in county_data.values())} counties")


if __name__ == "__main__":
    main()
