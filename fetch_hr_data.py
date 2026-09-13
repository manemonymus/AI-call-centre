"""
Fetch a real HR policy Q&A dataset and route each pair to an HR department.

Source: strova-ai/hr-policies-qa-dataset on Hugging Face (644 real HR-policy
question/answer pairs). Pulled over the public datasets-server API — no Kaggle
or Hugging Face login required.

Each Q&A is classified into one of four HR departments by keyword so that each
department can be given its OWN knowledge base (see ingest.py). The source
answers name a placeholder company ("Kreeda Labs"); we rewrite that to your
COMPANY_NAME so answers sound like your org.

Run once:
    python fetch_hr_data.py
Writes hr_faq.csv (columns: question, answer, department). Re-runnable; the CSV
is what ingest.py loads into ChromaDB.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DATASET = "strova-ai/hr-policies-qa-dataset"
ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset={ds}&config=default&split=train&offset={off}&length={length}"
)
OUT_CSV = Path(__file__).parent / "hr_faq.csv"
COMPANY_NAME = os.getenv("COMPANY_NAME", "Hearthstone")
SOURCE_COMPANY = "Kreeda Labs"

# Department routing, tuned to what this corpus actually contains (a corporate
# conduct + policy + leave/attendance handbook). Keyword scoring is the offline
# fallback; the LLM classifier below is preferred. "general" is the catch-all.
DEPARTMENT_KEYWORDS: dict[str, list[str]] = {
    "leave": [
        "leave", "compensatory off", "comp off", "vacation", "pto", "holiday",
        "overtime", "working hours", "attendance", "absence", "time off",
        "sick", "shift", "day off", "work-life",
    ],
    "conduct": [
        "conduct", "gift", "entertainment", "bribery", "corruption", "harassment",
        "ethic", "conflict of interest", "integrity", "misconduct", "discrimination",
        "dress code", "behaviour", "behavior", "respect",
    ],
    "compliance": [
        "violation", "violate", "disciplinary", "discipline", "report", "reporting",
        "complaint", "accountability", "compliance", "investigat", "whistle",
        "penalty", "consequence", "terminat", "sanction", "audit",
    ],
}
DEFAULT_DEPARTMENT = "general"


DEPARTMENTS = ["leave", "conduct", "compliance", "general"]
DEPARTMENT_DESC = (
    "leave = leave, compensatory/comp off, vacation/PTO, holidays, overtime, working hours, "
    "attendance, shifts, work-life balance; "
    "conduct = code of conduct, gifts & entertainment, anti-bribery/corruption, harassment, "
    "conflicts of interest, ethics, dress code, workplace behavior; "
    "compliance = reporting violations, disciplinary action, investigations, accountability, "
    "consequences/penalties for breaking policy, who to contact to report an issue; "
    "general = how policies are reviewed/updated/communicated, policy scope, and any general "
    "HR policy question that doesn't clearly fit the others (this is the catch-all)."
)


def classify_keyword(question: str, answer: str) -> str:
    """Fast offline fallback: keyword scoring on the question text."""
    text = question.lower()
    best, best_score = DEFAULT_DEPARTMENT, 0
    for dept, keywords in DEPARTMENT_KEYWORDS.items():
        score = sum(text.count(kw) for kw in keywords)
        if score > best_score:
            best, best_score = dept, score
    return best


# --- LLM classifier (accurate; one-time offline batch) -----------------------
def _llm_chat(prompt: str) -> str | None:
    """Call whichever LLM is available: Anthropic if keyed, else local Ollama.

    Returns the response text, or None if no LLM is reachable.
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except Exception as e:
            print(f"  (Anthropic classify failed: {e}; trying Ollama)")

    try:
        body = json.dumps(
            {
                "model": os.getenv("OLLAMA_MODEL", "qwen2.5"),
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0},
            }
        ).encode()
        req = Request(
            os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
            + "/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=120) as resp:
            return json.load(resp)["message"]["content"]
    except Exception:
        return None


def classify_batch(items: list[tuple[str, str]], size: int = 25) -> list[str]:
    """Classify (question, answer) pairs into departments using an LLM, with a
    per-item keyword fallback whenever the LLM is unavailable or unparseable."""
    labels: list[str] = []
    use_llm = _llm_chat("Reply with the single word: ready") is not None
    if not use_llm:
        print("  No LLM reachable — using keyword classification fallback.")
        return [classify_keyword(q, a) for q, a in items]

    for start in range(0, len(items), size):
        chunk = items[start : start + size]
        numbered = "\n".join(f"{i}. {q}" for i, (q, _) in enumerate(chunk))
        prompt = (
            "You route employee questions to the right HR department.\n"
            f"Departments: {DEPARTMENT_DESC}\n\n"
            "Classify each numbered question into exactly one department id "
            f"({', '.join(DEPARTMENTS)}). Reply with ONLY a JSON object mapping "
            'the number (as a string) to the department id, e.g. {"0":"payroll"}.\n\n'
            f"{numbered}"
        )
        raw = _llm_chat(prompt) or ""
        mapping = _parse_label_json(raw)
        for i, (q, a) in enumerate(chunk):
            dept = mapping.get(str(i))
            labels.append(dept if dept in DEPARTMENTS else classify_keyword(q, a))
        print(f"  classified {min(start + size, len(items))}/{len(items)}")
        time.sleep(0.1)
    return labels


def _parse_label_json(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return {str(k): str(v).strip().lower() for k, v in json.loads(raw[start : end + 1]).items()}
    except (ValueError, AttributeError):
        return {}


def _extract_qa(row: dict) -> tuple[str, str] | None:
    """Pull (question, answer) from a dataset row's chat messages."""
    messages = row.get("messages") or []
    question = answer = ""
    for msg in messages:
        role, content = msg.get("role"), (msg.get("content") or "").strip()
        if role == "user" and not question:
            question = content
        elif role == "assistant" and not answer:
            answer = content
    if question and answer:
        return question, answer
    return None


def _normalize(text: str) -> str:
    return text.replace(SOURCE_COMPANY, COMPANY_NAME)


def fetch_rows(page: int = 100) -> list[dict]:
    rows, offset = [], 0
    while True:
        url = ROWS_URL.format(ds=DATASET, off=offset, length=page)
        req = Request(url, headers={"User-Agent": "hr-helpline-ingest/1.0"})
        with urlopen(req, timeout=60) as resp:
            payload = json.load(resp)
        batch = payload.get("rows", [])
        if not batch:
            break
        rows.extend(r["row"] for r in batch)
        total = payload.get("num_rows_total")
        offset += page
        print(f"  fetched {len(rows)}" + (f"/{total}" if total else "") + " rows")
        if total and len(rows) >= total:
            break
        if len(batch) < page:
            break
        time.sleep(0.3)  # be polite to the public API
    return rows


def main() -> None:
    print(f"Fetching {DATASET} from Hugging Face datasets-server ...")
    try:
        rows = fetch_rows()
    except Exception as e:
        print(f"ERROR fetching dataset: {e}", file=sys.stderr)
        sys.exit(1)

    # Dedupe and collect Q&A pairs first.
    pairs: list[tuple[str, str]] = []
    seen = set()
    for row in rows:
        qa = _extract_qa(row)
        if not qa or qa[0] in seen:
            continue
        seen.add(qa[0])
        pairs.append(qa)

    print(f"Classifying {len(pairs)} Q&A pairs into departments ...")
    departments = classify_batch(pairs)

    counts: dict[str, int] = {}
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer", "department"])
        for (question, answer), dept in zip(pairs, departments):
            writer.writerow([_normalize(question), _normalize(answer), dept])
            counts[dept] = counts.get(dept, 0) + 1

    print(f"\nWrote {len(pairs)} Q&A pairs to {OUT_CSV.name}")
    for dept in sorted(counts):
        print(f"  {dept:10s} {counts[dept]}")


if __name__ == "__main__":
    main()
