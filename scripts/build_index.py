"""
One-time indexer. Run this once after setting keys in .env:

    python scripts/build_index.py

Writes:
  data/youtube_videos.json   — cached list of channel videos
  data/records.json          — interview metadata (from CSV + YouTube match)
  data/chunks.json           — searchable chunks (multiple per interview if transcript available)
  data/chunk_embeddings.npy  — Voyage embeddings aligned with chunks.json

Rerun when the CSV, channel, or transcripts change.
"""

import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import voyageai
from dotenv import load_dotenv
from googleapiclient.discovery import build
from rapidfuzz import fuzz, process

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
CSV_PATH = ROOT / "AI Chat Bot - KAS Legacy Project Metadata - LP Metadata_10122026.csv"
YT_CACHE = DATA / "youtube_videos.json"
TRANSCRIPT_PATH = DATA / "10202026_Master KAS Transcripts.txt"

EMBED_BATCH = 128

load_dotenv(ROOT / ".env")
VOYAGE_KEY = os.environ.get("VOYAGE_API_KEY", "")
YT_KEY = os.environ.get("YOUTUBE_API_KEY", "")
YT_HANDLE = os.environ.get("KAS_YOUTUBE_HANDLE", "KoreanAmericanStory")

if not VOYAGE_KEY:
    sys.exit("VOYAGE_API_KEY missing from .env")
if not YT_KEY:
    sys.exit("YOUTUBE_API_KEY missing from .env")


def normalize_title(s: str) -> str:
    s = re.sub(r"\s*\((full|edited|short|trailer)\)\s*", " ", s, flags=re.I)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def parse_srt_timestamp(ts: str) -> int:
    """Parse SRT timestamp like 00:05:23,404 into total seconds."""
    # Remove milliseconds if present
    ts = ts.split(",")[0]
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + int(s)
    return 0


def extract_title_from_filename(filename: str) -> str:
    """Extract interview title from transcript filename.

    Handles various formats:
    - LP_20250411_Charlotte Koh_ENG.txt -> Charlotte Koh
    - KPOD_01_20190510_Jin Soon Choi.txt -> Jin Soon Choi
    - GALA_2019_Performance_Claire Choi_ENG.txt -> GALA 2019 Performance Claire Choi
    - Letters to my Hometown_20260402_Kim Rogers Family_ENG.txt -> Letters to my Hometown Kim Rogers Family
    """
    # Remove .txt extension
    name = filename.replace(".txt", "")
    # Remove language suffix like _ENG, _KOR
    name = re.sub(r"_(?:ENG|KOR|eng|kor)$", "", name)

    # Handle LP (Legacy Project) format: LP_YYYYMMDD_Name
    lp_match = re.match(r"LP_\d{8}_(.+)", name, re.I)
    if lp_match:
        return lp_match.group(1).replace("_", " ").strip()

    # Handle KPOD format: KPOD_NN_YYYYMMDD_Name
    kpod_match = re.match(r"KPOD_\d+_\d{8}_(.+)", name, re.I)
    if kpod_match:
        return kpod_match.group(1).replace("_", " ").strip()

    # Handle Letters to my Hometown format: Letters to my Hometown_YYYYMMDD_Name
    letters_match = re.match(r"(Letters to my Hometown)_\d{8}_(.+)", name, re.I)
    if letters_match:
        return f"{letters_match.group(1)} {letters_match.group(2)}".replace("_", " ").strip()

    # Handle other formats with embedded dates: remove _YYYYMMDD_ patterns
    name = re.sub(r"_\d{8}_", "_", name)

    # Replace underscores with spaces
    name = name.replace("_", " ")
    return name.strip()


# ---------------------------------------------------------------------------
# Matching helpers (Oct 2026). The original matcher used fuzz.WRatio >= 75, which
# treats any shared word ("Kim", "Lee", "Family") as a match, so many transcripts
# and YouTube links were attached to the wrong interview. Matching is now done on
# the person's name only, within the same series, and must be a close match.
# ---------------------------------------------------------------------------
SERIES_RULES = [  # (series group, regex matched against a filename or Series value)
    ("kpod", r"^kpod|k-?pod"),
    ("nya", r"not your average"),
    ("roar", r"roar"),
    ("gala", r"gala|trailblazer"),
    # Only the "Queer Joy in Conversation" episodes are their own series. The individual
    # "Legacy Project Queer Joy" interviews are Legacy Project videos and must match lp rows.
    ("qj", r"queer joy in conversation|^queer joy_"),
    ("sfa", r"six feet apart"),
    ("perilla", r"perilla"),
    ("ttf", r"^ttf|through the fog"),
    ("naya", r"^naya|\bnaya\b"),
    # Letters to My Hometown is its own series: a person's Legacy Project interview and their
    # family's Letters to My Hometown video are separate pieces and must never be mixed up.
    ("ltmh", r"letters? (to|from) my hometown|^ltmh"),
    ("lp", r"legacy|^lp_"),
]

SERIES_WORDS = re.compile(
    r"\b(legacy project|letters? to my hometown( phase \d+)?|not your average|k ?pod|kpod|ep \d+|"
    r"roar story slam( all stars| chicago| dc| la)?|kasgala\d*|gala|six feet apart|day in the life|"
    r"perilla diaries|queer joy( in conversation)?|through the fog|ttf|naya|korean american story|"
    r"lp|ltmh|full|edited|edit|short|trailer|part \d+|eng|kor|video)\b"
)


def series_group(text: str) -> str:
    t = (text or "").strip().lower()
    for group, pat in SERIES_RULES:
        if re.search(pat, t):
            return group
    return "lp"  # blank Series values in the original CSV are Legacy Project


def series_of_filename(filename: str) -> str:
    return series_group(filename)


def name_core(title: str) -> str:
    """Reduce a title to just the person/topic name for matching."""
    s = normalize_title(title)
    s = SERIES_WORDS.sub(" ", s)
    s = re.sub(r"\b\d+\b", " ", s)          # dates, episode numbers, years
    s = re.sub(r"\b(dr|rev|reverend|pastor|mr|mrs|ms|ft|aka|the|mp4|srt|txt)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def names_match(a: str, b: str) -> float:
    """Score 0-100: how confidently two name cores refer to the same person."""
    if not a or not b:
        return 0
    if a == b or a.replace(" ", "") == b.replace(" ", ""):
        return 100
    return max(fuzz.token_sort_ratio(a, b), fuzz.ratio(a.replace(" ", ""), b.replace(" ", "")))


NAME_CUTOFF = 88


def parse_transcripts(transcript_path: Path) -> dict[str, list[dict]]:
    """Parse master transcript file (SRT format) into segments by interview.

    File format:
    === FILENAME.txt ===
    1
    00:00:05,404 --> 00:00:07,071
    - I hate my grandma.
    2
    00:00:08,339 --> 00:00:12,006
    Next line of text...

    Returns: list of {"file", "title", "series", "core", "chunks": [{"start_seconds", "text"}, ...]}
    (one entry per transcript file; nothing is overwritten when two files share a name)
    """
    if not transcript_path.exists():
        return []

    content = transcript_path.read_text(encoding="utf-8")
    interviews = []

    # Split by interview headers: === FILENAME.txt ===
    header_pattern = re.compile(r"^===\s*(.+?)\s*===$", re.MULTILINE)

    # Find all headers and their positions
    headers = list(header_pattern.finditer(content))
    if not headers:
        print("  No interview headers found in transcript file")
        return []

    # SRT timestamp pattern: 00:00:05,404 --> 00:00:07,071
    srt_ts_pattern = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})")

    for i, match in enumerate(headers):
        filename = match.group(1).strip()
        title = extract_title_from_filename(filename)
        start_pos = match.end()
        end_pos = headers[i + 1].start() if i + 1 < len(headers) else len(content)
        section = content[start_pos:end_pos].strip()

        if not section:
            continue

        # Parse SRT entries: find timestamps and collect text until next timestamp
        lines_data = []
        ts_matches = list(srt_ts_pattern.finditer(section))

        for j, ts_match in enumerate(ts_matches):
            start_ts = ts_match.group(1)
            start_seconds = parse_srt_timestamp(start_ts)

            # Text starts after the timestamp line, ends at next entry number or next timestamp
            text_start = ts_match.end()
            if j + 1 < len(ts_matches):
                # Find where next entry begins (line number before timestamp)
                text_end = ts_matches[j + 1].start()
                # Back up to skip the entry number line
                text_section = section[text_start:text_end]
                # Remove trailing entry number (digit on its own line at end)
                text_section = re.sub(r"\n\d+\s*$", "", text_section)
            else:
                text_section = section[text_start:]

            # Clean up the text
            text = text_section.strip()
            # Remove leading dash/hyphen speaker indicators
            text = re.sub(r"^-\s*", "", text)
            # Collapse whitespace
            text = " ".join(text.split())

            if text and not text.startswith("(") and not text.endswith(")"):
                # Skip pure stage directions like "(applause)"
                lines_data.append({
                    "start_seconds": start_seconds,
                    "text": text,
                })
            elif text:
                # Include stage directions but still add them
                lines_data.append({
                    "start_seconds": start_seconds,
                    "text": text,
                })

        if not lines_data:
            continue

        # Group lines into ~60-90 second chunks
        chunks = []
        current_chunk_start = lines_data[0]["start_seconds"]
        current_texts = []

        for line in lines_data:
            current_texts.append(line["text"])
            elapsed = line["start_seconds"] - current_chunk_start
            word_count = sum(len(t.split()) for t in current_texts)

            # Chunk when: 60+ seconds elapsed OR 300+ words accumulated
            if elapsed >= 60 or word_count >= 300:
                chunks.append({
                    "start_seconds": current_chunk_start,
                    "text": " ".join(current_texts),
                })
                current_texts = []
                current_chunk_start = line["start_seconds"]

        # Add remaining content as final chunk
        if current_texts:
            chunks.append({
                "start_seconds": current_chunk_start,
                "text": " ".join(current_texts),
            })

        interviews.append({
            "file": filename,
            "title": title,
            "series": series_of_filename(filename),
            "core": name_core(title),
            "chunks": chunks,
        })

    return interviews


def primary_record(owners: list[dict]) -> dict:
    """Pick ONE row to hold a transcript, so it is stored once instead of once per
    Full/Edited row (that duplication doubled the index size). Prefers a row with a
    YouTube link, then the Edited version (usually the public video)."""
    return sorted(owners, key=lambda r: (not r.get("youtube_video_id"),
                                         "(edited)" not in r["title"].lower(), r["id"]))[0]


def match_transcripts_to_records(records: list[dict], transcripts: list[dict]) -> dict[int, list[dict]]:
    """Attach each transcript to the interview record(s) it belongs to.

    A transcript matches a record only when both are in the same series and the
    person's name matches closely (NAME_CUTOFF). It attaches to every record with
    that same name in that series (e.g. the Full and Edited rows). If two transcripts
    belong to the same record (e.g. a 2015 and a 2023 interview of the same person),
    their chunks are combined rather than one overwriting the other.

    Returns: {record_id: transcript_chunks}
    """
    if not transcripts:
        return {}

    by_series: dict[str, list[dict]] = {}
    for r in records:
        if not r.get("transcript_files"):
            by_series.setdefault(r["series_group"], []).append(r)

    # 1. Explicit assignments from the "Transcript File" column
    by_file: dict[str, list[dict]] = {}
    for r in records:
        for f in r.get("transcript_files", []):
            by_file.setdefault(f.lower(), []).append(r)
    matched: dict[int, list[dict]] = {}
    unmatched = []
    explicit = 0
    remaining = []
    for t in transcripts:
        owners = by_file.get(t["file"].lower())
        if owners:
            explicit += 1
            matched.setdefault(primary_record(owners)["id"], []).extend(t["chunks"])
        else:
            remaining.append(t)
    listed = {f for f in by_file}
    missing = listed - {t["file"].lower() for t in transcripts}
    if missing:
        print(f"  Warning: {len(missing)} file(s) named in the Transcript File column are not in the transcript file:")
        for f in sorted(missing):
            print(f"    - {f}")
    print(f"  {explicit} transcript(s) attached via the Transcript File column")

    # 2. Automatic matching for anything not assigned explicitly
    for t in remaining:
        candidates = by_series.get(t["series"], [])
        scored = [(names_match(t["core"], r["core"]), r) for r in candidates]
        best = max((sc for sc, _ in scored), default=0)
        if best < NAME_CUTOFF:
            unmatched.append(t["file"])
            continue
        winners = [r for sc, r in scored if sc == best]
        # Only attach to records that share one name (Full/Edited/Part rows of one interview)
        core = winners[0]["core"]
        owners = [r for r in winners if r["core"] == core]
        matched.setdefault(primary_record(owners)["id"], []).extend(t["chunks"])

    if unmatched:
        print(f"  Warning: {len(unmatched)} transcript(s) have no matching row in the CSV (they will not be searchable):")
        for f in unmatched:
            print(f"    - {f}")
    return matched


def extract_interviewee(contributor: str) -> str:
    if not contributor:
        return ""
    for part in contributor.split(";"):
        if "interviewee" in part.lower():
            return re.sub(r",?\s*interviewee\s*$", "", part, flags=re.I).strip()
    return ""


def fetch_all_videos():
    if YT_CACHE.exists():
        print(f"Using cached YouTube list: {YT_CACHE}")
        return json.loads(YT_CACHE.read_text())

    try:
        return _fetch_videos_from_youtube()
    except Exception as e:
        backup = DATA / "youtube_videos_saved.json"
        if backup.exists():
            print(f"  WARNING: couldn't fetch the YouTube list ({str(e)[:150]})")
            print(f"  Using the saved list instead ({backup.name}). Videos published after it was saved won't get links.")
            return json.loads(backup.read_text())
        raise


def _fetch_videos_from_youtube():
    print(f"Fetching videos from @{YT_HANDLE}...")
    yt = build("youtube", "v3", developerKey=YT_KEY)
    ch = yt.channels().list(part="contentDetails", forHandle=YT_HANDLE).execute()
    if not ch.get("items"):
        sys.exit(f"Channel @{YT_HANDLE} not found")
    uploads = ch["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    videos = []
    page_token = None
    while True:
        resp = yt.playlistItems().list(
            playlistId=uploads, part="snippet", maxResults=50, pageToken=page_token,
        ).execute()
        for item in resp["items"]:
            videos.append({
                "id": item["snippet"]["resourceId"]["videoId"],
                "title": item["snippet"]["title"],
                # descriptions usually name the guest even when the title doesn't
                "description": item["snippet"].get("description", ""),
            })
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    # --- Mark YouTube Shorts so they are never linked -------------------------------
    # 1) The channel's own Shorts playlist ("UUSH" + channel id) lists exactly its Shorts.
    shorts = set()
    shorts_list_ok = False
    try:
        shorts_pl = "UUSH" + uploads[2:]
        token = None
        while True:
            resp = yt.playlistItems().list(
                playlistId=shorts_pl, part="snippet", maxResults=50, pageToken=token,
            ).execute()
            shorts.update(i["snippet"]["resourceId"]["videoId"] for i in resp["items"])
            token = resp.get("nextPageToken")
            if not token:
                break
        shorts_list_ok = True
    except Exception as e:
        print(f"  Note: couldn't read the channel's Shorts list ({str(e)[:100]}); using video length instead")
    # 2) Video length for every upload (also a backup if the Shorts list isn't available)
    ids = [v["id"] for v in videos]
    durations = {}
    privacy = {}
    for i in range(0, len(ids), 50):
        resp = yt.videos().list(part="contentDetails,status", id=",".join(ids[i:i + 50])).execute()
        for it in resp["items"]:
            durations[it["id"]] = iso_duration_seconds(it["contentDetails"].get("duration", ""))
            privacy[it["id"]] = it.get("status", {}).get("privacyStatus", "")
    for v in videos:
        # unlisted / private videos must never be shown in chat results
        v["privacy"] = privacy.get(v["id"], "unknown")
        v["duration_seconds"] = durations.get(v["id"])
        d = v["duration_seconds"]
        # Shorts are vertical clips up to 3 minutes. Anything 60s or under is always treated as a
        # Short; 61-180s counts only if the Shorts list couldn't be read.
        v["is_short"] = (v["id"] in shorts) or (d is not None and d <= 60) or \
                        (not shorts_list_ok and d is not None and d <= 180)

    YT_CACHE.write_text(json.dumps(videos, indent=2))
    n_short = sum(v["is_short"] for v in videos)
    n_hidden = sum(v["privacy"] != "public" for v in videos)
    print(f"Fetched {len(videos)} videos ({n_short} Shorts and {n_hidden} unlisted/private videos "
          f"will never be linked), cached to {YT_CACHE}")
    return videos


def iso_duration_seconds(s: str):
    """'PT1H2M3S' -> 3723 seconds."""
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s or "")
    if not m:
        return None
    d, h, mi, se = (int(x) if x else 0 for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + se


def load_records():
    print(f"Reading {CSV_PATH.name}...")
    records = []
    with CSV_PATH.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = (row.get("Title") or "").strip()
            if not title:
                continue
            series = (row.get("Series") or "").strip()
            records.append({
                "id": len(records),
                "title": title,
                "description": (row.get("Description") or "").strip(),
                "series": series,
                "series_group": series_group(series),
                "core": name_core(title),
                "contributor": (row.get("Contributor") or "").strip(),
                "creator": (row.get("Creator") or "").strip(),
                "date_recorded": (row.get("Date Recorded") or "").strip(),
                "keywords": (row.get("Keywords") or "").strip(),
                "interviewee": extract_interviewee(row.get("Contributor") or ""),
                # Optional column: paste a YouTube link here to override automatic matching
                "youtube_override": (row.get("YouTube URL") or "").strip(),
                # Optional column: transcript file name(s) from the master transcript file,
                # separated by " | ". When filled, these transcripts attach to this row directly.
                "transcript_files": [x.strip() for x in (row.get("Transcript File") or "").split("|") if x.strip()],
            })
    print(f"Loaded {len(records)} records")
    return records


YT_ID = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([\w-]{11})")


def match_youtube(records, videos):
    """Link each record to its YouTube video.

    1. If the CSV row has a "YouTube URL", that link is used as-is.
    2. Otherwise a video is linked only if its title contains the person's name
       (allowing small spelling differences). Ties prefer videos from the same series.
       If nothing clearly matches, the record gets no link rather than a wrong one.
    """
    # Never link a Short, or a video that isn't public (unlisted/private)
    # Shorts are never linked. Unlisted/private videos are still MATCHED, so that an interview whose
    # video isn't public can be hidden from the chatbot entirely (see "hidden" below).
    shorts = {v["id"] for v in videos if v.get("is_short")}
    nonpublic = {v["id"] for v in videos if v.get("privacy", "public") != "public"}
    durations = {v["id"]: v.get("duration_seconds") for v in videos}
    all_ids = {v["id"] for v in videos}
    if not any("is_short" in v for v in videos):
        print("  WARNING: this YouTube list has no Shorts/visibility info - re-fetch it so they can be excluded")
    videos = [v for v in videos if v["id"] not in shorts]
    public_ids = {v["id"] for v in videos} - nonpublic
    vids = []
    for v in videos:
        norm = normalize_title(v["title"])
        vids.append((v, set(norm.split()), norm.replace(" ", ""), series_group(v["title"]) if
                     any(re.search(p, v["title"].lower()) for _, p in SERIES_RULES) else None))

    matched = overridden = 0
    desc_links = []
    desc_norm = [normalize_title(v.get("description", "")) for v in videos]
    _cache = {}

    def desc_mentions(name):
        if name not in _cache:
            pat = re.compile(r"(^| )" + re.escape(name) + r"( |$)")
            _cache[name] = sum(1 for d in desc_norm if d and pat.search(d))
        return _cache[name]

    for r in records:
        r["youtube_video_id"] = None
        r["youtube_url"] = None
        if r.get("youtube_override", "").lower() == "none":
            continue  # "none" in the YouTube URL column = deliberately no video link
        if r.get("youtube_override", "").lower() == "hide":
            r["hidden"] = "marked 'hide' in the YouTube URL column"
            continue
        if r.get("youtube_override"):
            # one or more links separated by " | " - Shorts are skipped; the longest regular video wins
            ids = [m.group(1) for m in YT_ID.finditer(r["youtube_override"])]
            shorts.update(re.findall(r"/shorts/([\w-]{11})", r["youtube_override"]))  # a /shorts/ link is a Short
            # only public, regular videos from the channel's list (an ID that isn't in the list is
            # most likely unlisted or private, so it's skipped too)
            ok = [i for i in ids if i not in shorts and i in public_ids]
            # several regular videos listed (e.g. full episode + clip): use the longest one
            pick = max(ok, key=lambda i: (durations.get(i) or 0, -ok.index(i))) if ok else None
            if pick:
                r["youtube_video_id"] = pick
                r["youtube_url"] = f"https://www.youtube.com/watch?v={pick}"
                overridden += 1
                continue
            regular = [i for i in ids if i not in shorts]
            if regular:
                # its only (non-Short) videos are unlisted/private or not on the channel: hide the interview
                r["hidden"] = "its video in the YouTube URL column is unlisted/private (or not on the channel)"
                continue
            if ids:
                print(f"  Note: the YouTube URL column for '{r['title']}' only has Shorts - searching for a regular video instead")
                # fall through to automatic matching below
        name = r["core"]
        tokens = name.split()
        if not tokens:
            continue
        best, best_score = None, 0
        for v, vtokens, vflat, vseries in vids:
            vnorm = normalize_title(v["title"])
            vwords = vnorm.split()
            if re.search(r"(^| )" + re.escape(name) + r"( |$)", vnorm) or name.replace(" ", "") in vwords:
                score = 100   # the full name appears as a phrase in the video title
            else:
                # tolerate a small typo (e.g. Ashely/Ashley, Jungo/Jungdo), but the name must still
                # appear as consecutive words, so "Samuel Lee" never matches "Samuel Sang Gook Lee"
                n = len(tokens)
                score = 0
                if n >= 2:
                    for i in range(len(vwords) - n + 1):
                        window = vwords[i:i + n]
                        # every word must match; short words (Jun/Jung, Son/One) must match exactly
                        if all(a == w or (len(a) >= 4 and len(w) >= 4 and a[0] == w[0] and fuzz.ratio(a, w) >= 80)
                               for a, w in zip(tokens, window)):
                            score = 90
                            break
            if not score:
                continue
            if vseries is not None and vseries != r["series_group"]:
                continue  # never link a video from a different series (e.g. a ROAR video to a Through the Fog row)
            if vseries == r["series_group"]:
                score += 5
            if v["id"] in nonpublic:
                score -= 50   # a public video of the same person always wins over an unlisted one
            if score > best_score:
                best, best_score = v, score
        via_description = False
        # Names found in many descriptions are crew credits (editors, producers), not guests
        if not best and len(name) >= 5 and desc_mentions(name) <= 8:
            # Most KAS video titles no longer include the guest's name, so every row is also
            # checked against video DESCRIPTIONS. A name in the title still wins; otherwise the
            # video whose description names the person most often and earliest is used.
            # Only same-series videos (by title label) or videos with no series label qualify,
            # so e.g. a Through the Fog row can never link to a ROAR video.
            pat = re.compile(r"(^| )" + re.escape(name) + r"( |$)")
            for v, vtokens, vflat, vseries in vids:
                if (vseries is not None and vseries != r["series_group"]) or not v.get("description"):
                    continue
                dnorm = normalize_title(v["description"])
                hits = len(pat.findall(dnorm)) or (1 if name.replace(" ", "") in dnorm.split() else 0)
                if not hits:
                    continue
                first = dnorm.find(name) if name in dnorm else len(dnorm)
                # named often and early = the guest, not a host or a credit line
                score = hits * 10 + max(0, 300 - first) / 30 + (5 if vseries == r["series_group"] else 0)
                if v["id"] in nonpublic:
                    score *= 0.5   # prefer a public video; an unlisted-only match still hides the row
                if score > best_score:
                    best, best_score = v, score
            via_description = best is not None
        if best and best["id"] in nonpublic:
            r["hidden"] = f"its video is {best.get('privacy', 'not public')}: {best['title'][:60]}"
            continue
        if best:
            r["youtube_video_id"] = best["id"]
            r["youtube_url"] = f"https://www.youtube.com/watch?v={best['id']}"
            matched += 1
            if via_description:
                desc_links.append((r["title"], best["title"]))
    print(f"Matched {matched + overridden}/{len(records)} records to YouTube videos "
          f"({overridden} from the YouTube URL column, {len(desc_links)} via video descriptions)")
    if desc_links:
        print("  Linked via the video description - please spot-check these:")
        for row, vid in sorted(set(desc_links)):
            print(f"    {row}  ->  {vid}")


def build_description_chunk(record: dict) -> str:
    """Build a text chunk from record metadata (fallback when no transcript)."""
    parts = [record["title"]]
    if record.get("interviewee"):
        parts.append(f"Interviewee: {record['interviewee']}")
    if record.get("description"):
        parts.append(record["description"])
    if record.get("keywords"):
        parts.append(f"Keywords: {record['keywords']}")
    return "\n".join(parts)


def build_chunks(records: list[dict], transcripts: dict[int, list[dict]]):
    """Build searchable chunks from records, using transcripts when available.

    Records with transcripts get multiple chunks (one per ~60-90 second segment).
    Records without transcripts get a single chunk from their description.
    """
    chunks = []
    transcript_matches = 0
    transcript_chunks_created = 0

    for r in records:
        if r["id"] in transcripts:
            transcript_matches += 1
            # Create multiple chunks from transcript segments
            for segment in transcripts[r["id"]]:
                # Include title and interviewee in each chunk for context
                header_parts = [r["title"]]
                if r.get("interviewee"):
                    header_parts.append(f"Interviewee: {r['interviewee']}")
                header = "\n".join(header_parts)

                chunk = {
                    "record_id": r["id"],
                    "start_seconds": segment["start_seconds"],
                    "text": f"{header}\n\n{segment['text']}",
                    "has_transcript": True,
                }
                if segment.get("timed") is False:
                    chunk["timed"] = False  # full-recording transcript: times don't match the edited video
                chunks.append(chunk)
                transcript_chunks_created += 1
        else:
            # Fallback to description-only chunk
            chunks.append({
                "record_id": r["id"],
                "start_seconds": 0,
                "text": build_description_chunk(r),
                "has_transcript": False,
            })

    print(f"Created {len(chunks)} chunks ({transcript_matches} interviews with transcripts, {transcript_chunks_created} transcript chunks)")
    return chunks


def embed_chunks(chunks):
    import time
    vo = voyageai.Client(api_key=VOYAGE_KEY)

    # Check for partial progress file
    partial_path = DATA / "chunk_embeddings_partial.npy"
    progress_path = DATA / "embed_progress.txt"
    start_idx = 0
    embeddings = []

    if partial_path.exists() and progress_path.exists():
        start_idx = int(progress_path.read_text().strip())
        embeddings = list(np.load(partial_path))
        print(f"Resuming from batch {start_idx // 8 + 1} ({start_idx} chunks already done)")

    print(f"Embedding {len(chunks)} chunks with voyage-3-large (rate-limited mode)...")

    # With payment method: use normal batch sizes
    batch_size = 64
    delay_seconds = 1
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    remaining_batches = (len(chunks) - start_idx + batch_size - 1) // batch_size
    print(f"  {remaining_batches} batches remaining, ~{remaining_batches * delay_seconds // 60} minutes estimated")

    for i in range(start_idx, len(chunks), batch_size):
        batch = [c["text"] for c in chunks[i:i + batch_size]]
        batch_num = i // batch_size + 1

        for attempt in range(5):
            try:
                result = vo.embed(batch, model="voyage-3-large", input_type="document")
                embeddings.extend(result.embeddings)
                break
            except Exception as e:
                err_str = str(e).lower()
                if "rate" in err_str or "limit" in err_str:
                    wait = 90 * (attempt + 1)  # Longer waits for rate limits
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                elif "timeout" in err_str or "timed out" in err_str:
                    wait = 30 * (attempt + 1)
                    print(f"  Timeout, waiting {wait}s and retrying...")
                    time.sleep(wait)
                else:
                    raise
        else:
            # Save progress before failing
            np.save(partial_path, np.array(embeddings, dtype=np.float32))
            progress_path.write_text(str(i))
            print(f"  Saved progress at {i} chunks. Re-run to resume.")
            raise RuntimeError(f"Failed after 5 attempts at chunk {i}")

        # Save progress every 50 batches
        if batch_num % 50 == 0:
            np.save(partial_path, np.array(embeddings, dtype=np.float32))
            progress_path.write_text(str(i + batch_size))
            print(f"  [Progress saved]")

        print(f"  {batch_num}/{total_batches} batches ({min(i + batch_size, len(chunks))}/{len(chunks)} chunks)")

        if i + batch_size < len(chunks):
            time.sleep(delay_seconds)

    # Clean up progress files on success
    if partial_path.exists():
        partial_path.unlink()
    if progress_path.exists():
        progress_path.unlink()

    return np.array(embeddings, dtype=np.float32)


def embed_reusing_previous(chunks):
    """Only send new or changed passages to Voyage; reuse vectors from the previous build for the rest."""
    old_chunks_path, old_emb_path = DATA / "chunks.json", DATA / "chunk_embeddings.npy"
    known = {}
    if old_chunks_path.exists() and old_emb_path.exists():
        try:
            old_chunks = json.loads(old_chunks_path.read_text(encoding="utf-8"))
            old_emb = np.load(old_emb_path).astype(np.float32)
            if len(old_chunks) == len(old_emb):
                known = {c["text"]: old_emb[i] for i, c in enumerate(old_chunks)}
        except Exception as e:
            print(f"  (couldn't read the previous index, embedding everything: {e})")
    todo = [c for c in chunks if c["text"] not in known]
    print(f"Reusing {len(chunks) - len(todo)} passages from the previous build; embedding {len(todo)} new/changed passages")
    if todo:
        fresh = embed_chunks(todo)
        for c, v in zip(todo, fresh):
            known[c["text"]] = v
    return np.array([known[c["text"]] for c in chunks], dtype=np.float32)


def main():
    DATA.mkdir(exist_ok=True)
    videos = fetch_all_videos()
    records = load_records()
    match_youtube(records, videos)

    # Interviews whose video is unlisted/private are left out of the chatbot completely
    hidden = [r for r in records if r.get("hidden")]
    hidden_files = {f.lower() for r in hidden for f in r.get("transcript_files", [])}
    if hidden:
        print(f"Leaving out {len(hidden)} row(s) whose video is not public:")
        for r in hidden:
            print(f"    {r['title']}  ({r['hidden']})")
    records = [r for r in records if not r.get("hidden")]
    for i, r in enumerate(records):
        r["id"] = i

    # Parse transcripts and match to records
    print(f"Looking for transcripts at {TRANSCRIPT_PATH}...")
    raw_transcripts = parse_transcripts(TRANSCRIPT_PATH)
    if raw_transcripts:
        print(f"  Found {len(raw_transcripts)} transcript(s) in transcript file")
        raw_transcripts = [t for t in raw_transcripts if t["file"].lower() not in hidden_files]
        # Transcripts of full, unedited recordings (from zzFull Videos) are named ZZFULL_...; their
        # timestamps don't line up with the edited YouTube videos, so links go to the start of the video.
        for t in raw_transcripts:
            if t["file"].upper().startswith("ZZFULL_"):
                for c in t["chunks"]:
                    c["timed"] = False
        transcripts = match_transcripts_to_records(records, raw_transcripts)
        print(f"  {len(transcripts)} record(s) now have transcript text")
    else:
        print("  No transcripts found (file missing or empty)")
        transcripts = {}

    chunks = build_chunks(records, transcripts)
    embeddings = embed_reusing_previous(chunks)

    (DATA / "records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    (DATA / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
    # float16 halves the file size (keeps it under GitHub's 25 MB web-upload limit);
    # main.py converts back to float32 when loading. Search quality is unaffected.
    np.save(DATA / "chunk_embeddings.npy", embeddings.astype(np.float16))
    print(f"\nDone. {len(records)} records, {len(chunks)} chunks, embeddings shape {embeddings.shape}")


if __name__ == "__main__":
    main()
