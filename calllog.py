"""
Per-call logging: a JSONL event stream plus a final conversation transcript.

Every call gets one file in call_logs/ (override with CALL_LOG_DIR). Events
include the caller's transcribed lines, language switches, department
transfers, tool calls, and call start/end — the raw material for QA,
dispute evidence, and later cost/latency analysis.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from loguru import logger

LOG_DIR = Path(os.getenv("CALL_LOG_DIR", Path(__file__).parent / "call_logs"))


class CallLogger:
    """Appends timestamped JSON events for a single call."""

    def __init__(self, channel: str = "local"):
        self.call_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.path = LOG_DIR / f"{self.call_id}.jsonl"
        self.event("call_start", channel=channel)
        logger.info(f"Call log: {self.path}")

    def event(self, event_type: str, **data) -> None:
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "type": event_type,
            **data,
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            logger.warning(f"Call log write failed: {e}")

    def user_line(self, text: str, language: str | None = None) -> None:
        self.event("user", text=text, language=language)

    def dump_transcript(self, context) -> None:
        """Write the full LLM conversation context at call end."""
        try:
            messages = context.get_messages(truncate_large_values=True)
            self.event("transcript", messages=json.loads(json.dumps(messages, default=str)))
        except Exception as e:  # transcripts are best-effort; never break call teardown
            logger.warning(f"Transcript dump failed: {e}")

    def end(self) -> None:
        self.event("call_end")
