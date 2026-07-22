"""
KAS Archive Assistant backend.

Local dev (from project root):
    uvicorn main:app --reload

Serves /api/chat and the demo page at http://localhost:8000.
Deployed on Vercel via its Python runtime (auto-detects `app`).
"""

import json
import os
import re
from pathlib import Path

import anthropic
import numpy as np
import voyageai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
DATA = ROOT / "data"

load_dotenv(ROOT / ".env")

RECORDS_PATH = DATA / "records.json"
CHUNKS_PATH = DATA / "chunks.json"
CHUNK_EMB_PATH = DATA / "chunk_embeddings.npy"
CENSUS_PATH = DATA / "census_lookup.json"

for p in (RECORDS_PATH, CHUNKS_PATH, CHUNK_EMB_PATH):
    if not p.exists():
        raise SystemExit(
            f"Missing {p.name}. Run: python scripts/build_index.py"
        )

records = json.loads(RECORDS_PATH.read_text(encoding="utf-8"))
chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
chunk_embeddings = np.load(CHUNK_EMB_PATH)
norm_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)

# Load census data if available
census_data = None
if CENSUS_PATH.exists():
    census_data = json.loads(CENSUS_PATH.read_text(encoding="utf-8"))

voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MODEL = "claude-haiku-4-5"
TOP_K = 6
OVER_K = 40

SYSTEM_PROMPT = """You are a guide to the Korean American Story (KAS) Legacy Project — a video archive of oral-history interviews with Korean Americans, plus census data on Korean American demographics.

Guidelines:
- Be warm but professional — no effusive phrases like "Great question!" or "I love that!"
- Keep responses SHORT: 2-4 sentences max. Brevity is essential.
- Cite interviews as [1], [2], etc. Don't describe every citation — just mention 2-3 highlights and let visitors explore.
- If interviews don't match well, briefly suggest related topics.
- For greetings, respond in one short sentence and offer to help.
- When census data is provided, use those exact numbers. Mention the census year when citing statistics.
- NEVER end with a question. No "What interests you?", "Would you like to know more?", etc. Each response is standalone with no memory of previous messages.

Tone: A knowledgeable museum guide — warm, professional, concise."""


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


VERSION_TAG_RE = re.compile(r"\s*\((full|edited|short|trailer)\)\s*", re.I)


def base_title(title: str) -> str:
    return VERSION_TAG_RE.sub(" ", title).strip().lower()


def is_edited(title: str) -> bool:
    return "(edited)" in title.lower()


def display_title(title: str) -> str:
    return VERSION_TAG_RE.sub(" ", title).strip()


# The CSV has separate rows for the (Full) and (Edited) cuts of many interviews.
# Their embeddings land close together, so both versions used to surface as two
# near-identical citations. Collapse them to one card per interview and prefer
# the Edited cut for visitors.
PREFERRED_RID_BY_BASE: dict[str, int] = {}
for r in records:
    bt = base_title(r["title"])
    if bt not in PREFERRED_RID_BY_BASE or is_edited(r["title"]):
        PREFERRED_RID_BY_BASE[bt] = r["id"]

# Build index of chunk indices by record ID (may have multiple chunks per record now)
CHUNK_INDICES_BY_RID: dict[int, list[int]] = {}
for i, c in enumerate(chunks):
    rid = c["record_id"]
    if rid not in CHUNK_INDICES_BY_RID:
        CHUNK_INDICES_BY_RID[rid] = []
    CHUNK_INDICES_BY_RID[rid].append(i)


# --- Census data lookup ---

STATE_NAMES = {
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
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC", "d.c.": "DC",
    "puerto rico": "PR",
}

CENSUS_PATTERNS = [
    r'\b(population|how many|census|number of)\b.*\b(korean)\b',
    r'\b(korean)\b.*\b(population|how many|number|census)\b',
    r'\b(19[0-9]{2}|20[0-2][0-9])\b.*\b(korean|population)\b',
    r'\b(korean)\b.*\b(19[0-9]{2}|20[0-2][0-9])\b',
    r'\b(korean)\b.*\b(population)\b.*\b(changed|over time|trend|growth)\b',
    r'\bwhich states\b.*\b(most|largest)\b.*\bkorean\b',
]

# Patterns for meta questions about census data availability
META_CENSUS_PATTERNS = [
    r'\b(what|which|tell me about)\b.*(census|demographic|population)\s*(data|information|stats)',
    r'\b(census|demographic|population)\s*(data|information|stats).*(have|available|contain)',
    r'\bwhat.*(data|information).*(have|available)\b.*\b(census|demographic|population)\b',
    r'\b(do you have|is there).*(census|demographic|population)',
]

CENSUS_SUMMARY = """The archive includes U.S. Census data on Korean American population from 1910 to 2020:

- **National totals** for each decade from 1910-2020
- **State-level data** for all 50 states + DC across census years
- **County-level data** for 2020 (over 1,500 counties)

Example questions you can ask:
- "How many Korean Americans were in California in 1990?"
- "What was the Korean American population in 2020?"
- "How has the Korean population in Texas changed over time?"
- "How many Korean Americans lived in Los Angeles County in 2020?"
"""


def is_meta_census_query(query: str) -> bool:
    """Detect if query is asking about what census data is available."""
    query_lower = query.lower()
    return any(re.search(p, query_lower) for p in META_CENSUS_PATTERNS)


def is_census_query(query: str) -> bool:
    """Detect if query is asking about census/population data."""
    if not census_data:
        return False
    query_lower = query.lower()
    return any(re.search(p, query_lower) for p in CENSUS_PATTERNS)


def parse_years_from_query(query: str) -> list[str]:
    """Extract all years from query, returning list of census decade years."""
    query_lower = query.lower()
    years = []

    # Handle decades like "1990s" -> 1990
    decade_matches = re.findall(r'\b(19[0-9]0|20[0-2]0)s\b', query_lower)
    years.extend(decade_matches)

    # Handle specific years
    year_matches = re.findall(r'\b(19[0-9]{2}|20[0-2][0-9])\b', query_lower)
    for year in year_matches:
        # Round to nearest census year (decades)
        year_int = int(year)
        census_year = str((year_int // 10) * 10)
        if census_year not in years:
            years.append(census_year)

    return sorted(set(years))


def parse_year_from_query(query: str) -> str | None:
    """Extract a single year from query (first one found)."""
    years = parse_years_from_query(query)
    return years[0] if years else None


def parse_location_from_query(query: str) -> tuple[str | None, str | None]:
    """Extract state and optionally county from query. Returns (state_abbrev, county_name)."""
    query_lower = query.lower()

    # Check for state names
    state_abbrev = None
    for state_name, abbrev in STATE_NAMES.items():
        if state_name in query_lower:
            state_abbrev = abbrev
            break

    # Also check for abbreviations (but exclude common English words)
    ABBREV_EXCLUSIONS = {"in", "or", "ok", "me", "hi", "oh"}  # IN, OR, OK, ME, HI, OH
    if not state_abbrev:
        for abbrev in STATE_NAMES.values():
            abbrev_lower = abbrev.lower()
            if abbrev_lower in ABBREV_EXCLUSIONS:
                continue  # Skip ambiguous abbreviations
            if re.search(rf'\b{abbrev_lower}\b', query_lower):
                state_abbrev = abbrev
                break

    # Check for common cities/counties (map to state + county)
    # Use word boundaries to avoid false matches like "la" in "population"
    city_mappings = [
        (r"\blos angeles\b", "CA", "Los Angeles"),
        (r"\bl\.?a\.?\b", "CA", "Los Angeles"),  # LA or L.A.
        (r"\bnew york city\b", "NY", "New York"),
        (r"\bnyc\b", "NY", "New York"),
        (r"\bchicago\b", "IL", "Cook"),
        (r"\bsan francisco\b", "CA", "San Francisco"),
        (r"\bseattle\b", "WA", "King"),
        (r"\bhouston\b", "TX", "Harris"),
        (r"\bdallas\b", "TX", "Dallas"),
        (r"\batlanta\b", "GA", "Fulton"),
    ]

    for pattern, state, county in city_mappings:
        if re.search(pattern, query_lower):
            return (state, county)

    # Check for nationwide/national/us - but NOT "Korean Americans" which contains "america"
    national_patterns = [r"\bnationwide\b", r"\bnational\b", r"\bunited states\b", r"\bu\.s\.\b", r"\bthe country\b", r"\bin america\b"]
    if any(re.search(p, query_lower) for p in national_patterns):
        return ("US", None)

    return (state_abbrev, None)


def lookup_census(query: str) -> str | None:
    """Look up census data based on query. Returns formatted string or None."""
    if not census_data:
        return None

    years_found = parse_years_from_query(query)
    year = years_found[0] if len(years_found) == 1 else None
    state_abbrev, county = parse_location_from_query(query)
    query_lower = query.lower()

    # Check if this is a comparison or trend query
    is_comparison = len(years_found) >= 2
    is_trend_query = is_comparison or any(kw in query_lower for kw in ["over time", "changed", "trend", "growth", "history", "between"])

    results = []

    # If asking about a specific county
    if county and state_abbrev and state_abbrev != "US":
        county_data = census_data.get("county", {}).get(state_abbrev, {}).get(county, {})
        if county_data:
            for yr, data in county_data.items():
                if year and yr != year:
                    continue
                pop = data.get("korean_combined") or data.get("korean_alone")
                if pop:
                    results.append(f"In {yr}, there were {pop:,} Korean Americans in {county} County, {state_abbrev}.")

    # If asking about a state
    elif state_abbrev and state_abbrev != "US":
        state_info = census_data.get("state", {}).get(state_abbrev, {})
        if state_info:
            if year and year in state_info:
                pop = state_info[year].get("korean_alone")
                if pop:
                    results.append(f"According to the {year} census, there were {pop:,} Korean Americans in {state_abbrev}.")
            elif not year:
                # Show most recent data
                years = sorted(state_info.keys(), reverse=True)
                if years:
                    latest = years[0]
                    pop = state_info[latest].get("korean_alone")
                    if pop:
                        results.append(f"According to the {latest} census, there were {pop:,} Korean Americans in {state_abbrev}.")

    # If asking about national data or top states
    elif state_abbrev == "US" or not state_abbrev:
        query_lower = query.lower()

        # Check if asking about top states
        if re.search(r'\b(which|what|top)\b.*\bstates?\b.*\b(most|largest|highest)\b', query_lower) or \
           re.search(r'\b(most|largest)\b.*\bkorean\b.*\bstates?\b', query_lower):
            state_data = census_data.get("state", {})
            # Get 2020 data for all states and sort
            state_pops = []
            for st, years in state_data.items():
                if "2020" in years:
                    pop = years["2020"].get("korean_alone", 0)
                    if pop:
                        state_pops.append((st, pop))
            state_pops.sort(key=lambda x: -x[1])
            top_5 = state_pops[:5]
            if top_5:
                top_str = ", ".join([f"{st} ({pop:,})" for st, pop in top_5])
                results.append(f"Top 5 states by Korean American population (2020): {top_str}.")
            return " ".join(results) if results else None

        national = census_data.get("national", {})

        if is_comparison and len(years_found) >= 2:
            # Compare two specific years
            y1, y2 = years_found[0], years_found[-1]
            pop1 = national.get(y1, {}).get("korean_alone")
            pop2 = national.get(y2, {}).get("korean_alone")
            if pop1 and pop2:
                change = pop2 - pop1
                pct_change = (change / pop1) * 100 if pop1 > 0 else 0
                results.append(f"Korean American population: {pop1:,} in {y1} → {pop2:,} in {y2} (a change of {change:+,}, or {pct_change:+.1f}%).")
            elif pop1:
                results.append(f"According to the {y1} census, there were {pop1:,} Korean Americans in the United States.")
            elif pop2:
                results.append(f"According to the {y2} census, there were {pop2:,} Korean Americans in the United States.")
        elif is_trend_query and national:
            # Return trend data with multiple years
            all_years = sorted(national.keys())
            trend_parts = []
            for yr in all_years:
                pop = national[yr].get("korean_alone")
                if pop:
                    trend_parts.append(f"{pop:,} in {yr}")
            if trend_parts:
                results.append(f"Korean American population over time: {', '.join(trend_parts)}.")
        elif year and year in national:
            pop = national[year].get("korean_alone")
            if pop:
                results.append(f"According to the {year} census, there were {pop:,} Korean Americans in the United States.")
        elif not year and national:
            # Show most recent
            all_years = sorted(national.keys(), reverse=True)
            if all_years:
                latest = all_years[0]
                pop = national[latest].get("korean_alone")
                if pop:
                    results.append(f"According to the {latest} census, there were {pop:,} Korean Americans in the United States.")

    return " ".join(results) if results else None


def retrieve(query: str, k: int = TOP_K):
    """Chunk-level retrieval, deduped to one card per interview (Edited preferred).

    With transcripts, each interview may have multiple chunks. We keep the best
    (highest-scoring) chunk per interview for context, while still deduping
    across Full/Edited versions.
    """
    result = voyage.embed([query], model="voyage-3-large", input_type="query")
    q = np.array(result.embeddings[0], dtype=np.float32)
    q /= np.linalg.norm(q)
    scores = norm_embeddings @ q
    top_idx = np.argsort(-scores)[:OVER_K]

    # Track best chunk index and score per base title
    best_chunk_by_base: dict[str, tuple[int, float]] = {}
    for idx in top_idx:
        idx = int(idx)
        chunk = chunks[idx]
        bt = base_title(records[chunk["record_id"]]["title"])
        score = float(scores[idx])
        if bt not in best_chunk_by_base or score > best_chunk_by_base[bt][1]:
            best_chunk_by_base[bt] = (idx, score)

    # Rank by score and take top k
    ranked = sorted(best_chunk_by_base.items(), key=lambda kv: -kv[1][1])[:k]

    out = []
    for bt, (best_chunk_idx, _) in ranked:
        # Use preferred record (Edited version if available)
        rid = PREFERRED_RID_BY_BASE[bt]
        best_chunk = chunks[best_chunk_idx]

        # If the best chunk belongs to a different version (Full vs Edited),
        # find the equivalent chunk in the preferred record, or use the best one
        if best_chunk["record_id"] != rid:
            # Try to find a chunk at similar timestamp in preferred record
            preferred_chunks = CHUNK_INDICES_BY_RID.get(rid, [])
            if preferred_chunks:
                # Use first chunk of preferred record as fallback
                best_chunk = chunks[preferred_chunks[0]]
            # else: keep the best_chunk from the other version

        out.append((records[rid], best_chunk))
    return out


def format_timestamp(seconds: int) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def build_context(hits):
    lines = []
    for i, (record, chunk) in enumerate(hits, 1):
        meta = []
        if record.get("interviewee"):
            meta.append(f"Interviewee: {record['interviewee']}")
        if record.get("date_recorded"):
            meta.append(f"Recorded: {record['date_recorded']}")
        header = f"[{i}] {display_title(record['title'])}"
        if chunk["has_transcript"]:
            # Show transcript excerpt with timestamp
            timestamp = format_timestamp(chunk["start_seconds"])
            # Extract just the transcript text (skip the header we added)
            text = chunk["text"]
            # The chunk text format is: "Title\nInterviewee: ...\n\nactual transcript"
            # Extract the actual transcript part after the double newline
            if "\n\n" in text:
                text = text.split("\n\n", 1)[1]
            body = f"[{timestamp}] \"{text[:600]}\""
        else:
            body = f"Description: {record['description'][:400]}"
        lines.append(header + "\n" + "\n".join(meta) + "\n" + body)
    return "\n\n".join(lines)


CITE_RE = re.compile(r"\[(\d+)\]")


def cited_indices(answer: str) -> set[int]:
    return {int(m.group(1)) for m in CITE_RE.finditer(answer)}


def citation_payload(hits, cited: set[int]):
    out = []
    for i, (record, chunk) in enumerate(hits, 1):
        if i not in cited:
            continue
        vid = record.get("youtube_video_id")
        url = record.get("youtube_url")
        thumbnail = None
        start = 0
        if vid:
            thumbnail = f"https://i.ytimg.com/vi/{vid}/mqdefault.jpg"
            if chunk["has_transcript"] and chunk["start_seconds"] > 0:
                start = int(chunk["start_seconds"])
                url = f"https://www.youtube.com/watch?v={vid}&t={start}s"
        out.append({
            "index": i,
            "title": display_title(record["title"]),
            "interviewee": record.get("interviewee") or None,
            "date": record.get("date_recorded") or None,
            "youtube_url": url,
            "thumbnail_url": thumbnail,
            "start_seconds": start,
        })
    return out


app = FastAPI(title="KAS Archive Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/chat")
def chat(req: ChatRequest):
    # Check for meta census query first (asking about what data is available)
    census_context = None
    census_only = False  # If True, skip interview retrieval

    if is_meta_census_query(req.message):
        census_context = CENSUS_SUMMARY
        census_only = True
    elif is_census_query(req.message):
        census_context = lookup_census(req.message)
        # If we got census data, this is a census-only query
        if census_context:
            census_only = True

    # Only retrieve interviews if this isn't a census-only query
    hits = [] if census_only else retrieve(req.message)

    # Build context
    context_parts = []
    if census_context:
        context_parts.append(f"Census Data:\n{census_context}")
    if hits:
        context_parts.append(f"Retrieved interviews:\n\n{build_context(hits)}")

    user_turn = (
        "\n\n".join(context_parts) + "\n\n"
        f"---\nVisitor's question: {req.message}"
    )

    messages = [{"role": "user", "content": user_turn}]

    try:
        resp = claude.messages.create(
            model=MODEL,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {e.message}")

    answer = "".join(b.text for b in resp.content if b.type == "text").strip()
    cited = cited_indices(answer)
    return {"answer": answer, "citations": citation_payload(hits, cited)}


app.mount("/widget", StaticFiles(directory=ROOT / "widget"), name="widget")
app.mount("/demo", StaticFiles(directory=ROOT / "demo"), name="demo")


@app.get("/")
def root():
    return FileResponse(ROOT / "demo" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "records": len(records), "chunks": len(chunks)}
