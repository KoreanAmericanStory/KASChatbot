"""
Census tools for the KAS chatbot.

Claude calls these as tools, so any phrasing works ("How many Koreans…", "KA population",
"grew fastest", "Queens vs Bergen County"). The tools return exact figures, pre-calculated
changes and the source caveats, so the model never has to guess numbers or do arithmetic.
"""

import json
import re
from pathlib import Path

DATA = Path(__file__).parent / "data"
_raw = json.loads((DATA / "census_lookup.json").read_text(encoding="utf-8"))

NATIONAL = {y: v["korean_alone"] for y, v in _raw.get("national", {}).items() if v.get("korean_alone")}
STATES = {st: {y: v["korean_alone"] for y, v in yrs.items() if v.get("korean_alone")} for st, yrs in _raw.get("state", {}).items()}
COUNTIES = {st: {name: v["2020"]["korean_combined"] for name, v in cs.items() if v.get("2020", {}).get("korean_combined")}
            for st, cs in _raw.get("county", {}).items()}
NOTES = _raw.get("notes", {"general": "", "by_year": {}})
ALL_YEARS = ["1910", "1920", "1930", "1940", "1950", "1960", "1970", "1980", "1990", "2000", "2010", "2020"]

STATE_NAMES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
    "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC", "washington dc": "DC",
    "washington d.c.": "DC", "dc": "DC", "d.c.": "DC",
}
ABBR_TO_NAME = {v: k.title() for k, v in STATE_NAMES.items() if len(k) > 3}
ABBR_TO_NAME["DC"] = "Washington, D.C."

NATIONAL_ALIASES = {"us", "u.s.", "usa", "u.s.a.", "united states", "the united states", "america", "national",
                    "nationwide", "nation", "the us", "the u.s.", "country", "whole country", "all states", "total"}

# Cities and neighborhoods → the 2020 counties that contain them (counties are the finest level in the data).
CITY_TO_COUNTIES = {
    "new york city": ("NY", ["New York", "Kings", "Queens", "Bronx", "Richmond"], "New York City (all five boroughs)"),
    "nyc": ("NY", ["New York", "Kings", "Queens", "Bronx", "Richmond"], "New York City (all five boroughs)"),
    "manhattan": ("NY", ["New York"], "Manhattan (New York County)"),
    "brooklyn": ("NY", ["Kings"], "Brooklyn (Kings County)"),
    "queens": ("NY", ["Queens"], "Queens (Queens County)"),
    "flushing": ("NY", ["Queens"], "Queens County, which includes Flushing"),
    "bronx": ("NY", ["Bronx"], "the Bronx (Bronx County)"),
    "the bronx": ("NY", ["Bronx"], "the Bronx (Bronx County)"),
    "staten island": ("NY", ["Richmond"], "Staten Island (Richmond County)"),
    "los angeles": ("CA", ["Los Angeles"], "Los Angeles County"),
    "la": ("CA", ["Los Angeles"], "Los Angeles County"),
    "l.a.": ("CA", ["Los Angeles"], "Los Angeles County"),
    "koreatown": ("CA", ["Los Angeles"], "Los Angeles County, which includes LA's Koreatown"),
    "orange county": ("CA", ["Orange"], "Orange County, CA"),
    "san francisco": ("CA", ["San Francisco"], "San Francisco County"),
    "bay area": ("CA", ["San Francisco", "Alameda", "Santa Clara", "San Mateo", "Contra Costa", "Marin", "Solano", "Napa", "Sonoma"], "the nine-county Bay Area"),
    "san jose": ("CA", ["Santa Clara"], "Santa Clara County, which includes San Jose"),
    "san diego": ("CA", ["San Diego"], "San Diego County"),
    "chicago": ("IL", ["Cook"], "Cook County, which includes Chicago"),
    "seattle": ("WA", ["King"], "King County, which includes Seattle"),
    "houston": ("TX", ["Harris"], "Harris County, which includes Houston"),
    "dallas": ("TX", ["Dallas"], "Dallas County"),
    "atlanta": ("GA", ["Fulton", "DeKalb", "Gwinnett", "Cobb"], "the core Atlanta counties (Fulton, DeKalb, Gwinnett, Cobb)"),
    "duluth": ("GA", ["Gwinnett"], "Gwinnett County, which includes Duluth"),
    "philadelphia": ("PA", ["Philadelphia"], "Philadelphia County"),
    "boston": ("MA", ["Suffolk"], "Suffolk County, which includes Boston"),
    "honolulu": ("HI", ["Honolulu"], "Honolulu County"),
    "fort lee": ("NJ", ["Bergen"], "Bergen County, which includes Fort Lee and Palisades Park"),
    "palisades park": ("NJ", ["Bergen"], "Bergen County, which includes Fort Lee and Palisades Park"),
    "annandale": ("VA", ["Fairfax"], "Fairfax County, which includes Annandale"),
    "northern virginia": ("VA", ["Fairfax", "Loudoun", "Prince William", "Arlington", "Alexandria city"], "Northern Virginia (Fairfax, Loudoun, Prince William, Arlington, Alexandria)"),
    "baltimore": ("MD", ["Baltimore city"], "Baltimore city"),
    "detroit": ("MI", ["Wayne"], "Wayne County, which includes Detroit"),
    "minneapolis": ("MN", ["Hennepin"], "Hennepin County, which includes Minneapolis"),
    "denver": ("CO", ["Denver"], "Denver County"),
    "las vegas": ("NV", ["Clark"], "Clark County, which includes Las Vegas"),
    "anchorage": ("AK", ["Anchorage Municipality"], "Anchorage"),
}

COUNTY_SUFFIX_RE = re.compile(r"\s+(county|parish|borough|municipality|census area|city and borough)$", re.I)


def _fmt(n):
    return f"{n:,}"


def _clean(name: str) -> str:
    s = name.strip().lower().replace("’", "'")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^(the )?(state of |commonwealth of )", "", s)
    s = re.sub(r"\s+state$", "", s)
    return s.strip(" .,")


def resolve_place(name: str):
    """Return ('national', None, label) | ('state', ST, label) | ('counties', (ST, [names]), label) | ('error', msg, None)."""
    raw = name
    s = _clean(name)
    if s in NATIONAL_ALIASES:
        return ("national", None, "the United States")
    if s in STATE_NAMES:
        st = STATE_NAMES[s]
        return ("state", st, ABBR_TO_NAME.get(st, st))
    if s.upper() in STATES and len(s) == 2:
        return ("state", s.upper(), ABBR_TO_NAME.get(s.upper(), s.upper()))
    if s in CITY_TO_COUNTIES:
        st, names, label = CITY_TO_COUNTIES[s]
        return ("counties", (st, names), label)

    # "X County, ST" / "X County, State" / "X, ST" / "X County"
    state_hint = None
    m = re.match(r"^(.*?),\s*([a-z .]+)$", s)
    if m:
        s, st_part = m.group(1).strip(), m.group(2).strip()
        state_hint = STATE_NAMES.get(st_part) or (st_part.upper() if st_part.upper() in STATES else None)
    if s in CITY_TO_COUNTIES and (state_hint is None or CITY_TO_COUNTIES[s][0] == state_hint):
        st, names, label = CITY_TO_COUNTIES[s]
        return ("counties", (st, names), label)
    base = COUNTY_SUFFIX_RE.sub("", s).strip()
    matches = []
    for st, cs in COUNTIES.items():
        if state_hint and st != state_hint:
            continue
        for cname in cs:
            cbase = COUNTY_SUFFIX_RE.sub("", cname.lower()).strip()
            if cbase == base or cname.lower() == s:
                matches.append((st, cname))
    if len(matches) == 1:
        st, cname = matches[0]
        return ("counties", (st, [cname]), f"{cname}{'' if COUNTY_SUFFIX_RE.search(cname) or cname.endswith('city') else ' County'}, {st}")
    if len(matches) > 1:
        opts = ", ".join(f"{c} ({st})" for st, c in matches[:12])
        return ("error", f"'{raw}' matches several counties: {opts}. Ask which state, or call again with 'County, ST'.", None)
    return ("error", f"'{raw}' was not found. The data covers the U.S., the 50 states + D.C., and (for 2020 only) counties. "
                     f"Cities are only available through the county that contains them.", None)


def _year_notes(years):
    out = [f"{y}: {NOTES['by_year'][y]}" for y in years if y in NOTES.get("by_year", {})]
    return out


def _norm_years(years):
    out = []
    for y in years or []:
        m = re.search(r"(1[89]\d\d|20[0-2]\d)", str(y))
        if m:
            d = str(int(m.group(1)) // 10 * 10)
            if d not in out:
                out.append(d)
    return sorted(out)


def _series_text(label, series: dict, years: list[str]):
    """Figures for one place, plus changes between the requested (or all) years."""
    lines = []
    wanted = years or sorted(series)
    have = [y for y in wanted if y in series]
    missing = [y for y in wanted if y not in series]
    if have:
        lines.append(f"{label}: " + "; ".join(f"{y}: {_fmt(series[y])}" for y in have))
    if missing:
        lines.append(f"{label}: no figure in the census table for {', '.join(missing)} "
                     f"(years with data: {', '.join(sorted(series)) or 'none'}).")
    if len(have) >= 2:
        steps = []
        for a, b in zip(have, have[1:]):
            d = series[b] - series[a]
            pct = f", {d / series[a] * 100:+.1f}%" if series[a] >= 100 else " (percent not meaningful: very small starting number)"
            steps.append(f"{a}→{b}: {d:+,}{pct}")
        lines.append(f"{label} change between consecutive listed years: " + "; ".join(steps))
        a, b = have[0], have[-1]
        if len(have) > 2:
            d = series[b] - series[a]
            pct = f" ({d / series[a] * 100:+.1f}%, {series[b] / series[a]:.1f}x)" if series[a] >= 100 else ""
            lines.append(f"{label} overall {a}→{b}: {d:+,}{pct}")
        elif series[a] >= 100:
            lines.append(f"{label} {a}→{b} is {series[b] / series[a]:.1f}x.")
    return lines, have


def census_lookup(places: list[str], years: list[str] | None = None) -> str:
    years = _norm_years(years)
    lines, used_years = [], set()
    for p in places or ["United States"]:
        kind, key, label = resolve_place(p)
        if kind == "error":
            lines.append(key)
            continue
        if kind == "national":
            l, have = _series_text("United States", NATIONAL, years)
        elif kind == "state":
            l, have = _series_text(label, STATES.get(key, {}), years)
        else:
            st, names = key
            found = {n: COUNTIES.get(st, {}).get(n) for n in names}
            total = sum(v for v in found.values() if v)
            parts = [f"{n}: {_fmt(v)}" for n, v in found.items() if v]
            absent = [n for n, v in found.items() if not v]
            l = [f"{label}, 2020: {_fmt(total)}" + (f" ({'; '.join(parts)})" if len(names) > 1 else "")]
            if absent:
                l.append(f"Not in the county table: {', '.join(absent)}.")
            if years and years != ["2020"]:
                l.append("County and city figures exist only for 2020; earlier years are only available for states and the U.S.")
            have = ["2020"]
            if "which includes" in label or "counties" in label or "Bay Area" in label or "Virginia" in label:
                l.append("Note: city/area figures are for the whole county or counties that contain it.")
        lines.extend(l)
        used_years.update(have)
    notes = _year_notes(sorted(used_years | set(years)))
    return "\n".join(lines + ["", "Source notes: " + NOTES.get("general", "")] + notes)


def census_rank(level: str = "state", year: str = "2020", by: str = "population", start_year: str | None = None,
                top_n: int = 10, ascending: bool = False, state: str | None = None) -> str:
    level = (level or "state").lower()
    top_n = max(1, min(int(top_n or 10), 60))
    if level.startswith("count"):
        pool = {}
        st_filter = None
        if state:
            kind, key, _ = resolve_place(state)
            st_filter = key if kind == "state" else None
        for st, cs in COUNTIES.items():
            if st_filter and st != st_filter:
                continue
            for n, v in cs.items():
                pool[f"{n}, {st}"] = v
        rows = sorted(pool.items(), key=lambda kv: kv[1], reverse=not ascending)[:top_n]
        head = f"Counties{' in ' + ABBR_TO_NAME.get(st_filter, st_filter) if st_filter else ''} ranked by Korean American population, 2020:"
        body = [f"{i}. {n}: {_fmt(v)}" for i, (n, v) in enumerate(rows, 1)]
        return "\n".join([head] + body + ["", "Source notes: " + NOTES.get("general", "")] + _year_notes(["2020"]))

    year = (_norm_years([year]) or ["2020"])[0]
    by = (by or "population").lower()
    if by in ("growth", "change", "growth_number", "growth_percent", "percent", "percent_growth"):
        use_pct = "percent" in by
        start = (_norm_years([start_year]) or ["2010"])[0]
        rows = []
        for st, s in STATES.items():
            if start in s and year in s:
                d = s[year] - s[start]
                pct = d / s[start] * 100 if s[start] else float("inf")
                rows.append((st, s[start], s[year], d, pct))
        rows.sort(key=(lambda r: r[4]) if use_pct else (lambda r: r[3]), reverse=not ascending)
        head = f"States ranked by {'percent' if use_pct else 'numeric'} growth in Korean American population, {start}→{year}:" if rows else "No data."
        body = [f"{i}. {ABBR_TO_NAME.get(st, st)}: {_fmt(a)} → {_fmt(b)} ({d:+,}, {p:+.1f}%)" for i, (st, a, b, d, p) in enumerate(rows[:top_n], 1)]
        return "\n".join([head] + body + ["", "Source notes: " + NOTES.get("general", "")] + _year_notes([start, year]))

    rows = sorted(((st, s[year]) for st, s in STATES.items() if year in s), key=lambda r: r[1], reverse=not ascending)
    total = NATIONAL.get(year)
    head = f"States ranked by Korean American population, {year}" + (f" (U.S. total {_fmt(total)}):" if total else ":")
    body = [f"{i}. {ABBR_TO_NAME.get(st, st)}: {_fmt(v)}" + (f" ({v / total * 100:.1f}% of all Korean Americans)" if total else "")
            for i, (st, v) in enumerate(rows[:top_n], 1)]
    if not rows:
        body = [f"No state figures for {year}."]
    return "\n".join([head] + body + ["", "Source notes: " + NOTES.get("general", "")] + _year_notes([year]))


CENSUS_OVERVIEW = (
    "Census data available: Korean American population for the U.S. and every state for 1910, 1920, 1930, 1940, "
    "1950 (Hawaii only), 1970, 1980, 1990, 2000, 2010 and 2020 (no 1960), and for about 1,600 counties in 2020 only. "
    "No data on total population, income, age, language or other characteristics."
)
