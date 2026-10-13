"""
Accuracy test for the KAS chatbot. Asks every question in accuracy_questions.json and checks the answers.

    python3 tests/run_accuracy_test.py                       # tests the live chatbot
    python3 tests/run_accuracy_test.py https://<test-link>   # tests a preview/test link

Uses only Python's standard library. Each run costs roughly 50 questions x ~1.5 cents.
Writes accuracy_report.txt next to this file.
"""
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
URL = (sys.argv[1] if len(sys.argv) > 1 else "https://kas-chatbot-gamma.vercel.app").rstrip("/")
QUESTIONS = json.loads((HERE / "accuracy_questions.json").read_text(encoding="utf-8"))


def ask(q):
    req = urllib.request.Request(f"{URL}/api/chat", data=json.dumps({"message": q}).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read()), time.time() - t
        except Exception as e:
            err = e
            time.sleep(3)
    return {"answer": f"ERROR: {err}", "citations": []}, time.time() - t


def digits(s):
    return re.sub(r"(?<=\d),(?=\d)", "", s)


def check(t, out):
    ans = out.get("answer", "")
    flat = digits(ans)
    titles = [c.get("title", "") for c in out.get("citations", [])]
    problems = []
    if ans.startswith("ERROR"):
        return [ans]
    for n in t.get("numbers", []):
        if not re.search(rf"(?<!\d){n}(?!\d)", flat):
            problems.append(f"missing number {int(n):,}")
    for n in t.get("forbid_numbers", []):
        if re.search(rf"(?<!\d){n}(?!\d)", flat):
            problems.append(f"should not say {int(n):,}")
    for w in t.get("words", []):
        if w.lower() not in ans.lower():
            problems.append(f"missing '{w}'")
    if t.get("any_words") and not any(w.lower() in ans.lower() for w in t["any_words"]):
        problems.append("expected one of: " + ", ".join(repr(w.strip()) for w in t["any_words"]))
    for w in t.get("forbid_words", []):
        if w.lower() in ans.lower() or any(w.lower() in x.lower() for x in titles):
            problems.append(f"must not mention '{w}'")
    if t.get("cited_any") and not any(n.lower() in x.lower() for n in t["cited_any"] for x in titles):
        problems.append("expected a video of: " + ", ".join(t["cited_any"]))
    for n in t.get("cited_none", []):
        bad = [x for x in titles if n.lower() in x.lower()]
        if bad:
            problems.append(f"should not link: {bad[0]}")
    if "min_citations" in t and len(titles) < t["min_citations"]:
        problems.append(f"expected at least {t['min_citations']} video link(s)")
    if "max_citations" in t and len(titles) > t["max_citations"]:
        problems.append(f"expected at most {t['max_citations']} video link(s)")
    return problems


def main():
    print(f"Testing {URL} with {len(QUESTIONS)} questions (about 2-4 minutes)...\n")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda t: (t, *ask(t["q"])), QUESTIONS))
    lines, failed = [], 0
    for t, out, secs in results:
        problems = check(t, out)
        failed += bool(problems)
        status = "PASS" if not problems else "FAIL"
        lines.append(f"[{status}] {t['q']}  ({secs:.1f}s)")
        if problems:
            lines.append("       problems: " + "; ".join(problems))
            lines.append("       answer:   " + out.get("answer", "").replace("\n", " ")[:400])
            lines.append("       videos:   " + "; ".join(c.get("title", "") for c in out.get("citations", [])))
    secs = sorted(s for _, _, s in results)
    summary = (f"\n{len(QUESTIONS) - failed} of {len(QUESTIONS)} passed. "
               f"Typical answer time {secs[len(secs) // 2]:.1f}s, slowest {secs[-1]:.1f}s.\nTested: {URL}")
    report = "\n".join(lines) + "\n" + summary + "\n"
    print(report)
    (HERE / "accuracy_report.txt").write_text(report, encoding="utf-8")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
