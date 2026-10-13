"""
KAS Archive Assistant backend.

Local dev (from project root):
    uvicorn main:app --reload

Serves /api/chat and the demo page at http://localhost:8000.
Deployed on Vercel via its Python runtime (auto-detects `app`).
"""

import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import anthropic
import numpy as np

import census
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
chunk_embeddings = np.load(CHUNK_EMB_PATH).astype(np.float32)  # index is stored as float16
norm_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)


voyage = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MODEL = "claude-haiku-4-5"
TOP_K = 6                      # interviews shown to the answer model
VECTOR_K = 60                  # candidate passages from meaning-based search
KEYWORD_K = 60                 # candidate passages from exact-word search
RERANK_MODEL = "rerank-2.5"    # Voyage re-ranker: reads question + passage together
RERANK_DOC_CHARS = 2000
FULL_TRANSCRIPT_INTERVIEWS = 3     # top N interviews are read in full
FULL_TRANSCRIPT_MAX_CHARS = 24000  # ~6k tokens; longer transcripts are windowed around the best moment
EXCERPTS_PER_INTERVIEW = 2         # the rest get their best few moments
EXCERPT_CHARS = 1500

SYSTEM_PROMPT = """You are the guide to the Korean American Story (KAS) archive: a video archive of oral-history interviews and programs with Korean Americans (Legacy Project, Letters to My Hometown, K-Pod, ROAR Story Slam, Not Your Average, Through the Fog and more), plus U.S. Census data on the Korean American population.

You have tools. Use them; never answer factual questions from your own general knowledge.
- search_interviews: for anything about people, experiences, stories, topics, places, eras or episodes. Write a focused search query in your own words that captures exactly what the visitor wants (names, places, eras, themes, qualifiers). For multi-part or comparison questions, search separately for each part. If results don't fit, search again with different wording (at most 3 searches).
- census_lookup / census_rank: the ONLY source for population numbers. "Koreans", "Korean Americans", "KA", "Korean people" all mean the Korean American population. Use census_lookup for specific places and years (it also gives changes and percentages, so never calculate yourself) and census_rank for "which state/county has the most / grew fastest". Cities are only available through the county that contains them (2020 only); say so when relevant.
- Questions that need both (e.g. a number plus related stories): use both.

Answering rules:
- Read results closely and answer the visitor's exact question, respecting qualifiers (e.g. "didn't", "as a child", "before the war", "in Chicago", "second-generation", "adoptees", "women").
- Only cite interviews that genuinely fit. Cite them as [n] using the numbers in the search results. If nothing fits well, say so in one sentence and name the closest matches.
- Give totals by default. Only mention the Korean-alone vs. multiracial split when the visitor asks about it.
- Report census figures exactly as given, with the census year and place. If a source note affects the answer (e.g. 1950 counts Hawaii only, 1930 leaves out Hawaii, 2000-2020 totals include mixed-heritage Korean Americans, no 1960 data), mention it briefly. If the data doesn't cover something (a year, a city, income, age...), say so plainly instead of estimating. Never take population numbers from interviews.
- Never invent details about a person; base every statement on tool results.
- Be warm but professional; no effusive phrases like "Great question!".
- Keep it SHORT: 2-4 sentences. For several census figures, a compact list is fine. Don't describe every citation; highlight 2-3.
- For greetings, reply in one short sentence and offer to help (no tools needed).
- When a visitor names a person, interview, episode or topic, search and give a short overview right away. Never reply by asking what they want to know about it.
- Every interview you mention must carry its [n] citation.
- Don't end with an offer or question ("Would you like to know more?"). The only exception: if a place is ambiguous (e.g. "Montgomery County"), ask which one.

Tone: A knowledgeable museum guide — warm, professional, concise.

""" + census.CENSUS_OVERVIEW


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


def chunk_body(chunk: dict) -> str:
    """Transcript/description text without the "Title / Interviewee" header."""
    text = chunk["text"]
    return text.split("\n\n", 1)[1] if "\n\n" in text else text


# --- Exact-word (BM25) index: catches names, places and terms meaning-based search can miss ---

TOKEN_RE = re.compile(r"[a-z0-9\uac00-\ud7a3]+")
STOPWORDS = set("""a an and are as at be but by did do does for from had has have he her his how i in is it its
me my of on or our she so that the their them they this to was we were what when where which who whom why
with you your about any anyone are there tell show find interview interviews story stories someone people
talk talks talked talking video videos""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS]


def _build_keyword_index():
    k1, b = 1.5, 0.75
    tfs = [Counter(tokenize(c["text"])) for c in chunks]
    lengths = np.array([sum(tf.values()) for tf in tfs], dtype=np.float32)
    avg = float(lengths.mean()) if len(lengths) else 1.0
    postings: dict[str, list[tuple[int, int]]] = {}
    for i, tf in enumerate(tfs):
        for term, n in tf.items():
            postings.setdefault(term, []).append((i, n))
    n_docs = len(chunks)
    index = {}
    for term, plist in postings.items():
        idx = np.fromiter((i for i, _ in plist), dtype=np.int32, count=len(plist))
        tf = np.fromiter((n for _, n in plist), dtype=np.float32, count=len(plist))
        idf = math.log(1 + (n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
        weight = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[idx] / avg))
        index[term] = (idx, weight.astype(np.float32))
    return index


KEYWORD_INDEX = _build_keyword_index()


def keyword_scores(query: str) -> np.ndarray:
    scores = np.zeros(len(chunks), dtype=np.float32)
    for term in set(tokenize(query)):
        if term in KEYWORD_INDEX:
            idx, weight = KEYWORD_INDEX[term]
            np.add.at(scores, idx, weight)
    return scores


# --- Census figures live in census.py and are reached through Claude tool calls (see TOOLS below) ---


def retrieve(query: str, k: int = TOP_K):
    """Find the interviews that best answer the question.

    1. Gather candidate passages two ways: by meaning (Voyage embeddings) and by exact words (BM25).
    2. Re-rank all candidates with Voyage's re-ranker, which reads the question and each passage
       together and so understands qualifiers and negations much better than similarity alone.
    3. Group passages by interview (Full/Edited cuts count as one) and keep the top k interviews,
       remembering each interview's best-matching moments.
    """
    result = voyage.embed([query], model="voyage-3-large", input_type="query")
    q = np.array(result.embeddings[0], dtype=np.float32)
    q /= np.linalg.norm(q)
    vec = norm_embeddings @ q
    kw = keyword_scores(query)

    cand = list(np.argsort(-vec)[:VECTOR_K])
    if kw.max() > 0:
        for i in np.argsort(-kw)[:KEYWORD_K]:
            if kw[i] > 0:
                cand.append(i)
    cand = list(dict.fromkeys(int(i) for i in cand))

    try:
        rr = voyage.rerank(
            query=query,
            documents=[chunks[i]["text"][:RERANK_DOC_CHARS] for i in cand],
            model=RERANK_MODEL,
            truncation=True,
        )
        scored = [(cand[r.index], float(r.relevance_score)) for r in rr.results]
    except Exception as e:  # never let the re-ranker take the chatbot down
        print(f"rerank failed, falling back to vector order: {e}")
        scored = [(i, float(vec[i])) for i in cand]
    scored.sort(key=lambda x: -x[1])

    by_base: dict[str, list[tuple[int, float]]] = {}
    order: list[str] = []
    for idx, score in scored:
        bt = base_title(records[chunks[idx]["record_id"]]["title"])
        if bt not in by_base:
            by_base[bt] = []
            order.append(bt)
        by_base[bt].append((idx, score))

    hits = []
    for bt in order[:k]:
        passages = [i for i, _ in by_base[bt]]
        best = chunks[passages[0]]
        own = records[best["record_id"]]
        preferred = records[PREFERRED_RID_BY_BASE[bt]]
        # Link the video that the matching transcript actually belongs to, so timestamps are right.
        if own.get("youtube_video_id") or not preferred.get("youtube_video_id"):
            link_record, timestamps_ok = own, True
        else:
            link_record, timestamps_ok = preferred, False
        hits.append({
            "record": link_record,
            "transcript_rid": best["record_id"],
            "best_chunk": best,
            "passages": passages,
            "timestamps_ok": timestamps_ok,
            "has_transcript": bool(best["has_transcript"]),
        })
    return hits


def format_timestamp(seconds: int) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def full_transcript(rid: int, best_idx: int) -> str:
    """The whole transcript with timestamps; if very long, a window centred on the best moment."""
    idxs = CHUNK_INDICES_BY_RID.get(rid, [])
    pieces = [f"[{format_timestamp(chunks[i]['start_seconds'])}] {chunk_body(chunks[i])}" for i in idxs]
    if sum(len(p) for p in pieces) <= FULL_TRANSCRIPT_MAX_CHARS:
        return "\n".join(pieces)
    centre = idxs.index(best_idx) if best_idx in idxs else 0
    lo = hi = centre
    total = len(pieces[centre])
    while True:
        grew = False
        if lo > 0 and total + len(pieces[lo - 1]) <= FULL_TRANSCRIPT_MAX_CHARS:
            lo -= 1; total += len(pieces[lo]); grew = True
        if hi < len(pieces) - 1 and total + len(pieces[hi + 1]) <= FULL_TRANSCRIPT_MAX_CHARS:
            hi += 1; total += len(pieces[hi]); grew = True
        if not grew:
            break
    text = "\n".join(pieces[lo:hi + 1])
    if lo > 0:
        text = "[…earlier part of the interview omitted…]\n" + text
    if hi < len(pieces) - 1:
        text += "\n[…later part of the interview omitted…]"
    return text


def build_context(hits, start: int = 1, full_n: int = FULL_TRANSCRIPT_INTERVIEWS):
    blocks = []
    for n, h in enumerate(hits, start):
        record = h["record"]
        lines = [f"[{n}] {display_title(record['title'])}"]
        if record.get("interviewee"):
            lines.append(f"Interviewee: {record['interviewee']}")
        if record.get("date_recorded"):
            lines.append(f"Recorded: {record['date_recorded']}")
        if record.get("description"):
            lines.append(f"Description: {record['description'][:700]}")
        if h["has_transcript"]:
            if n - start < full_n:
                lines.append("Full transcript:\n" + full_transcript(h["transcript_rid"], h["passages"][0]))
            else:
                lines.append("Most relevant moments:")
                for i in h["passages"][:EXCERPTS_PER_INTERVIEW]:
                    c = chunks[i]
                    lines.append(f"[{format_timestamp(c['start_seconds'])}] \"{chunk_body(c)[:EXCERPT_CHARS]}\"")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


CITE_RE = re.compile(r"\[(\d+)\]")


def cited_indices(answer: str) -> set[int]:
    return {int(m.group(1)) for m in CITE_RE.finditer(answer)}


def citation_payload(hits, cited: set[int]):
    out = []
    for n, h in enumerate(hits, 1):
        if n not in cited:
            continue
        record, chunk = h["record"], h["best_chunk"]
        vid = record.get("youtube_video_id")
        url = record.get("youtube_url")
        thumbnail = None
        start = 0
        if vid:
            thumbnail = f"https://i.ytimg.com/vi/{vid}/mqdefault.jpg"
            if h["timestamps_ok"] and chunk["has_transcript"] and chunk["start_seconds"] > 0:
                start = int(chunk["start_seconds"])
                url = f"https://www.youtube.com/watch?v={vid}&t={start}s"
        out.append({
            "index": n,
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


TOOLS = [
    {
        "name": "search_interviews",
        "description": "Search the KAS video archive (transcripts and descriptions of oral-history interviews and programs). "
                       "Returns the best-matching interviews, numbered for citation, with full transcripts for the top matches "
                       "of the first search and the most relevant moments for the rest.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A focused search query in your own words, e.g. 'adoptee who chose not to search for birth family' or 'fled Pyongyang as a child during the Korean War'."}},
            "required": ["query"],
        },
    },
    {
        "name": "census_lookup",
        "description": "Exact Korean American population figures from the U.S. Census for places and years, with pre-calculated "
                       "changes/percentages and source caveats. Totals for 2000-2020 already include multiracial Korean Americans. "
                       "Set breakdown=true ONLY when the visitor asks about multiracial/mixed-heritage vs. Korean-only numbers; it then "
                       "adds that split for 2000, 2010 and 2020. Places: 'United States', any state, D.C., a Census region or division "
                       "('South', 'Midwest', 'New England', 'Pacific'), 'East Coast', 'Pacific Northwest', or (2020 only) a county "
                       "like 'Bergen County, NJ' or a major city/area like 'New York City', 'Queens', 'Los Angeles', 'Chicago', 'Bay Area'. "
                       "Omit years to get every year available.",
        "input_schema": {
            "type": "object",
            "properties": {
                "places": {"type": "array", "items": {"type": "string"}, "description": "One or more places."},
                "years": {"type": "array", "items": {"type": "string"}, "description": "Census years (decades 1910-2020). Optional."},
                "breakdown": {"type": "boolean", "description": "True only if the visitor asks about multiracial / mixed-heritage / Korean-only counts."},
            },
            "required": ["places"],
        },
    },
    {
        "name": "census_rank",
        "description": "Rank states (any census year), Census regions/divisions, or counties (2020 only) by Korean American population; "
                       "rank states by growth between two census years (by='growth' numeric, by='growth_percent' percent); or rank "
                       "states by multiracial Korean Americans in 2000/2010/2020 (by='multiracial' count, by='multiracial_percent' share).",
        "input_schema": {
            "type": "object",
            "properties": {
                "level": {"type": "string", "enum": ["state", "county", "region", "division"]},
                "year": {"type": "string", "description": "Census year, default 2020 (end year for growth)."},
                "by": {"type": "string", "enum": ["population", "growth", "growth_percent", "multiracial", "multiracial_percent"]},
                "start_year": {"type": "string", "description": "Start year for growth rankings."},
                "top_n": {"type": "integer", "description": "How many to list (default 10)."},
                "ascending": {"type": "boolean", "description": "True to list the smallest first."},
                "state": {"type": "string", "description": "For county rankings: limit to one state."},
            },
        },
    },
]

MAX_TOOL_ROUNDS = 4

# Small talk is the only thing answered without looking anything up.
SMALL_TALK_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|hiya|greetings|thanks|thank you|thx|ok|okay|cool|great|nice|bye|goodbye|"
    r"good (morning|afternoon|evening)|who are you|what are you|what can you do|how does this work|help)"
    r"[\s!.?,]*(there|again|so much|a lot)?[\s!.?]*$", re.I)


class ToolSession:
    """Runs tool calls for one question and keeps every interview shown, so citations line up."""

    def __init__(self):
        self.hits: list[dict] = []
        self.seen: dict[str, int] = {}
        self.searches = 0

    def run(self, name: str, args: dict) -> str:
        try:
            if name == "search_interviews":
                return self.search(str(args.get("query", "")).strip())
            if name == "census_lookup":
                return census.census_lookup(args.get("places") or ["United States"], args.get("years"), bool(args.get("breakdown")))
            if name == "census_rank":
                return census.census_rank(**{k: v for k, v in args.items()
                                             if k in ("level", "year", "by", "start_year", "top_n", "ascending", "state")})
            return f"Unknown tool {name}."
        except Exception as e:  # report tool problems to the model instead of failing the request
            print(f"tool {name} failed: {e!r}")
            return f"The {name} tool hit an error ({e}). Answer with what you have, or say the information is unavailable."

    def search(self, query: str) -> str:
        if not query:
            return "Empty query."
        self.searches += 1
        if self.searches > 3:
            return "Search limit reached for this question; answer with the results you already have."
        found = retrieve(query)
        new, repeats = [], []
        for h in found:
            bt = base_title(h["record"]["title"])
            if bt in self.seen:
                repeats.append(f"[{self.seen[bt]}] {display_title(h['record']['title'])}")
                continue
            new.append(h)
        start = len(self.hits) + 1
        for i, h in enumerate(new):
            self.seen[base_title(h["record"]["title"])] = start + i
        self.hits.extend(new)
        full_n = FULL_TRANSCRIPT_INTERVIEWS if self.searches == 1 else 0
        text = build_context(new, start=start, full_n=full_n) if new else "No new interviews found."
        if repeats:
            text += "\n\nAlso matched (already listed above): " + "; ".join(repeats)
        return text


@app.post("/api/chat")
def chat(req: ChatRequest):
    session = ToolSession()
    messages = [{"role": "user", "content": req.message}]
    answer = ""
    try:
        for round_no in range(MAX_TOOL_ROUNDS + 1):
            final_round = round_no == MAX_TOOL_ROUNDS
            if final_round:
                choice = {"type": "none"}
            elif round_no == 0 and not SMALL_TALK_RE.match(req.message):
                choice = {"type": "any"}  # always look things up before answering a real question
            else:
                choice = {"type": "auto"}
            resp = claude.messages.create(
                model=MODEL,
                max_tokens=700,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                tool_choice=choice,
                messages=messages,
            )
            if resp.stop_reason != "tool_use":
                answer = "".join(b.text for b in resp.content if b.type == "text").strip()
                break
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if block.type == "tool_use":
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": session.run(block.name, block.input or {})})
            messages.append({"role": "user", "content": results})
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {e.message}")

    if not answer:
        answer = "Sorry, I couldn't put together an answer to that. Please try rephrasing your question."
    cited = cited_indices(answer) | named_but_uncited(answer, session.hits)
    return {"answer": answer, "citations": citation_payload(session.hits, cited)}


def named_but_uncited(answer: str, hits: list[dict]) -> set[int]:
    """Safety net: if the answer names an interviewee but forgot the [n], still show that video card."""
    low = answer.lower()
    extra = set()
    for n, h in enumerate(hits, 1):
        r = h["record"]
        names = {(r.get("interviewee") or "").strip(), display_title(r["title"]).split("|")[0].split(" - ")[-1].strip()}
        for name in names:
            if len(name) >= 6 and name.lower() in low:
                extra.add(n)
                break
    return extra


app.mount("/widget", StaticFiles(directory=ROOT / "widget"), name="widget")
app.mount("/demo", StaticFiles(directory=ROOT / "demo"), name="demo")


@app.get("/")
def root():
    return FileResponse(ROOT / "demo" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "records": len(records), "chunks": len(chunks)}
