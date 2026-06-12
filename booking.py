"""
Real (local, free) booking and ticketing storage backed by SQLite.

Replaces the prototype's fake "just confirm it" bookings with actual
persistence: availability is computed from business hours minus booked
slots, double-booking is impossible, and escalations become tickets a
human can work through later.

The database file is created next to this module on first use
(override with the CALLCENTER_DB environment variable).
"""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(os.getenv("CALLCENTER_DB", Path(__file__).parent / "callcenter.db"))

# Business hours: Mon-Fri, hourly slots starting 8:00 through 16:00.
BUSINESS_DAYS = {0, 1, 2, 3, 4}  # Monday=0 .. Friday=4
FIRST_SLOT_HOUR = 8
LAST_SLOT_HOUR = 16

_SCHEMA = """
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phone TEXT NOT NULL,
    customer_name TEXT NOT NULL DEFAULT '',
    service TEXT NOT NULL DEFAULT '',
    slot TEXT NOT NULL,              -- "YYYY-MM-DD HH:MM"
    address TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'booked',  -- booked | canceled
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phone TEXT NOT NULL DEFAULT '',
    customer_name TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL,
    urgency TEXT NOT NULL DEFAULT 'normal',
    language TEXT NOT NULL DEFAULT 'en',
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL
);
"""


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_phone(phone: str) -> str:
    """Digits only, last 10 — so '858-555-0198' and '+1 8585550198' match."""
    return re.sub(r"\D", "", phone or "")[-10:]


def _slot_label(slot: datetime) -> str:
    hour = slot.strftime("%I %p").lstrip("0")
    return f"{slot.strftime('%A, %B')} {slot.day}, {slot.year} at {hour}"


def _parse_slot(slot_str: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(slot_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _booked_slots(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT slot FROM appointments WHERE status = 'booked'").fetchall()
    return {row["slot"] for row in rows}


def _candidate_slots(start: datetime, days_ahead: int) -> list[datetime]:
    """All business-hour slots from `start` forward, in order."""
    slots = []
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    for offset in range(days_ahead + 1):
        current = day + timedelta(days=offset)
        if current.weekday() not in BUSINESS_DAYS:
            continue
        for hour in range(FIRST_SLOT_HOUR, LAST_SLOT_HOUR + 1):
            slot = current.replace(hour=hour)
            if slot > start:
                slots.append(slot)
    return slots


def date_problem(date_str: str) -> str | None:
    """Why a requested date can't work, or None if it's bookable in principle."""
    try:
        day = datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except ValueError:
        return "That date wasn't understood — use YYYY-MM-DD format."
    today = datetime.now()
    if day.date() < today.date():
        return f"That date already passed — today is {today:%A, %Y-%m-%d}."
    if day.weekday() not in BUSINESS_DAYS:
        return "We only book Monday through Friday."
    return None


def available_slots(
    date_str: str | None = None, days_ahead: int = 7, limit: int = 6
) -> list[dict]:
    """Open slots, soonest first. Pass a YYYY-MM-DD date to narrow to one day."""
    now = datetime.now()
    with _conn() as conn:
        booked = _booked_slots(conn)

    if date_str:
        try:
            day = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        except ValueError:
            return []
        start = max(now, day.replace(hour=0, minute=0))
        candidates = [
            s for s in _candidate_slots(start, days_ahead=0) if s.date() == day.date()
        ]
    else:
        candidates = _candidate_slots(now, days_ahead)

    open_slots = [s for s in candidates if s.strftime("%Y-%m-%d %H:%M") not in booked]
    return [
        {"slot": s.strftime("%Y-%m-%d %H:%M"), "label": _slot_label(s)}
        for s in open_slots[:limit]
    ]


def book(
    phone: str,
    customer_name: str,
    service: str,
    slot_str: str,
    address: str = "",
    notes: str = "",
) -> dict:
    """Book a slot. Returns {'booked': True, ...} or {'booked': False, 'reason', 'alternatives'}."""
    phone = normalize_phone(phone)
    if len(phone) < 7:
        return {
            "booked": False,
            "reason": "need a valid phone number for the booking",
            "alternatives": [],
        }
    slot = _parse_slot(slot_str)
    if slot is None:
        return {
            "booked": False,
            "reason": "Could not understand that date and time. Use YYYY-MM-DD HH:MM.",
            "alternatives": available_slots(),
        }
    problems = []
    if slot <= datetime.now():
        problems.append("that time is in the past")
    if slot.weekday() not in BUSINESS_DAYS:
        problems.append("we only book Monday through Friday")
    if not (FIRST_SLOT_HOUR <= slot.hour <= LAST_SLOT_HOUR) or slot.minute != 0:
        problems.append(
            f"appointments start on the hour between {FIRST_SLOT_HOUR} AM and {LAST_SLOT_HOUR - 12} PM"
        )
    if problems:
        return {
            "booked": False,
            "reason": "; ".join(problems),
            "alternatives": available_slots(),
        }

    slot_key = slot.strftime("%Y-%m-%d %H:%M")
    with _conn() as conn:
        if slot_key in _booked_slots(conn):
            return {
                "booked": False,
                "reason": "that time was just taken",
                "alternatives": available_slots(),
            }
        cursor = conn.execute(
            "INSERT INTO appointments (phone, customer_name, service, slot, address, notes,"
            " status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'booked', ?, ?)",
            (phone, customer_name, service, slot_key, address, notes, _now(), _now()),
        )
        return {
            "booked": True,
            "appointment_id": cursor.lastrowid,
            "slot": slot_key,
            "label": _slot_label(slot),
            "service": service,
            "customer_name": customer_name,
        }


def upcoming_for_phone(phone: str) -> list[dict]:
    """Future booked appointments for a phone number, soonest first."""
    now_key = datetime.now().strftime("%Y-%m-%d %H:%M")
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM appointments WHERE phone = ? AND status = 'booked' AND slot >= ?"
            " ORDER BY slot",
            (normalize_phone(phone), now_key),
        ).fetchall()
    results = []
    for row in rows:
        record = dict(row)
        slot = _parse_slot(record["slot"])
        record["label"] = _slot_label(slot) if slot else record["slot"]
        results.append(record)
    return results


def cancel(phone: str, appointment_id: int | None = None) -> dict:
    """Cancel by id, or the caller's next upcoming appointment."""
    with _conn() as conn:
        if appointment_id is not None:
            row = conn.execute(
                "SELECT * FROM appointments WHERE id = ? AND status = 'booked'",
                (appointment_id,),
            ).fetchone()
        else:
            now_key = datetime.now().strftime("%Y-%m-%d %H:%M")
            row = conn.execute(
                "SELECT * FROM appointments WHERE phone = ? AND status = 'booked'"
                " AND slot >= ? ORDER BY slot LIMIT 1",
                (normalize_phone(phone), now_key),
            ).fetchone()
        if row is None:
            return {"canceled": False, "reason": "no upcoming appointment found"}
        conn.execute(
            "UPDATE appointments SET status = 'canceled', updated_at = ? WHERE id = ?",
            (_now(), row["id"]),
        )
        slot = _parse_slot(row["slot"])
        return {
            "canceled": True,
            "appointment_id": row["id"],
            "slot": row["slot"],
            "label": _slot_label(slot) if slot else row["slot"],
            "service": row["service"],
        }


def create_ticket(
    summary: str,
    phone: str = "",
    customer_name: str = "",
    urgency: str = "normal",
    language: str = "en",
) -> dict:
    """File an escalation ticket for a human to follow up on."""
    urgency = str(urgency or "").strip().lower()
    urgency = {"high": "urgent", "critical": "urgent", "emergency": "urgent", "medium": "normal"}.get(urgency, urgency)
    if urgency not in ("low", "normal", "urgent"):
        urgency = "normal"
    with _conn() as conn:
        cursor = conn.execute(
            "INSERT INTO tickets (phone, customer_name, summary, urgency, language, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (normalize_phone(phone), customer_name, summary, urgency, language, _now()),
        )
        return {"ticket_id": cursor.lastrowid, "urgency": urgency}
