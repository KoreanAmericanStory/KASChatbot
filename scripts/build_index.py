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
CSV_PATH = ROOT / "AI Chat Bot - KAS Legacy Project Metadata - LP Metadata_10212025.csv"
YT_CACHE = DATA / "youtube_videos.json"
TRANSCRIPT_PATH = DATA / "06082026_Master KAS Transcripts.txt"

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

    Returns: {normalized_title: [{"start_seconds": int, "text": str}, ...]}
    """
    if not transcript_path.exists():
        return {}

    content = transcript_path.read_text(encoding="utf-8")
    interviews = {}

    # Split by interview headers: === FILENAME.txt ===
    header_pattern = re.compile(r"^===\s*(.+?)\s*===$", re.MULTILINE)

    # Find all headers and their positions
    headers = list(header_pattern.finditer(content))
    if not headers:
        print("  No interview headers found in transcript file")
        return {}

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

        title_key = normalize_title(title)
        interviews[title_key] = chunks

    return interviews


def match_transcripts_to_records(records: list[dict], transcripts: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Fuzzy-match transcript titles to CSV record titles.

    Returns: {normalized_record_title: transcript_chunks}
    """
    if not transcripts:
        return {}

    record_titles = [normalize_title(r["title"]) for r in records]
    matched = {}
    unmatched_transcripts = []

    for transcript_title, chunks in transcripts.items():
        # Try exact match first
        if transcript_title in record_titles:
            matched[transcript_title] = chunks
            continue

        # Fuzzy match
        hit = process.extractOne(
            transcript_title,
            record_titles,
            scorer=fuzz.WRatio,
            score_cutoff=75,
        )
        if hit:
            matched_title, _, _ = hit
            matched[matched_title] = chunks
        else:
            unmatched_transcripts.append(transcript_title)

    if unmatched_transcripts:
        print(f"  Warning: {len(unmatched_transcripts)} transcript(s) could not be matched to records")

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
            })
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    YT_CACHE.write_text(json.dumps(videos, indent=2))
    print(f"Fetched {len(videos)} videos, cached to {YT_CACHE}")
    return videos


def load_records():
    print(f"Reading {CSV_PATH.name}...")
    records = []
    with CSV_PATH.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = (row.get("Title") or "").strip()
            if not title:
                continue
            records.append({
                "id": len(records),
                "title": title,
                "description": (row.get("Description") or "").strip(),
                "series": (row.get("Series") or "").strip(),
                "contributor": (row.get("Contributor") or "").strip(),
                "creator": (row.get("Creator") or "").strip(),
                "date_recorded": (row.get("Date Recorded") or "").strip(),
                "keywords": (row.get("Keywords") or "").strip(),
                "interviewee": extract_interviewee(row.get("Contributor") or ""),
            })
    print(f"Loaded {len(records)} records")
    return records


def match_youtube(records, videos):
    yt_titles_norm = [normalize_title(v["title"]) for v in videos]
    matched = 0
    for r in records:
        hit = process.extractOne(
            normalize_title(r["title"]),
            yt_titles_norm,
            scorer=fuzz.WRatio,
            score_cutoff=72,
        )
        if hit:
            _, _, idx = hit
            r["youtube_video_id"] = videos[idx]["id"]
            r["youtube_url"] = f"https://www.youtube.com/watch?v={videos[idx]['id']}"
            matched += 1
        else:
            r["youtube_video_id"] = None
            r["youtube_url"] = None
    print(f"Matched {matched}/{len(records)} records to YouTube videos")


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


def build_chunks(records: list[dict], transcripts: dict[str, list[dict]]):
    """Build searchable chunks from records, using transcripts when available.

    Records with transcripts get multiple chunks (one per ~60-90 second segment).
    Records without transcripts get a single chunk from their description.
    """
    chunks = []
    transcript_matches = 0
    transcript_chunks_created = 0

    for r in records:
        title_key = normalize_title(r["title"])

        if title_key in transcripts:
            transcript_matches += 1
            # Create multiple chunks from transcript segments
            for segment in transcripts[title_key]:
                # Include title and interviewee in each chunk for context
                header_parts = [r["title"]]
                if r.get("interviewee"):
                    header_parts.append(f"Interviewee: {r['interviewee']}")
                header = "\n".join(header_parts)

                chunks.append({
                    "record_id": r["id"],
                    "start_seconds": segment["start_seconds"],
                    "text": f"{header}\n\n{segment['text']}",
                    "has_transcript": True,
                })
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


def main():
    DATA.mkdir(exist_ok=True)
    videos = fetch_all_videos()
    records = load_records()
    match_youtube(records, videos)

    # Parse transcripts and match to records
    print(f"Looking for transcripts at {TRANSCRIPT_PATH}...")
    raw_transcripts = parse_transcripts(TRANSCRIPT_PATH)
    if raw_transcripts:
        print(f"  Found {len(raw_transcripts)} interview(s) in transcript file")
        transcripts = match_transcripts_to_records(records, raw_transcripts)
        print(f"  Matched {len(transcripts)} transcript(s) to records")
    else:
        print("  No transcripts found (file missing or empty)")
        transcripts = {}

    chunks = build_chunks(records, transcripts)
    embeddings = embed_chunks(chunks)

    (DATA / "records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    (DATA / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
    np.save(DATA / "chunk_embeddings.npy", embeddings)
    print(f"\nDone. {len(records)} records, {len(chunks)} chunks, embeddings shape {embeddings.shape}")


if __name__ == "__main__":
    main()
