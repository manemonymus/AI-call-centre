"""
Shared multi-agent flow for the call center (used by both bot.py and server.py).

Agents are Pipecat Flows nodes: Router -> Billing | Scheduling | Customer
Service, plus an Escalation node (human callback tickets) and an End node.
Each department has its own AI voice per language; transfers switch the
active voice and speak a greeting in the caller's current language.

What's real here (no faking):
  - Bookings hit a local SQLite calendar (booking.py) with availability and
    double-booking protection.
  - Escalations create tickets a human can work later.
  - The bot is instructed to never invent prices/policies (RAG-grounded),
    to read bookings back before committing, and to answer truthfully that
    it is an AI.

Expected flow_manager.state entries (set by the caller before initialize()):
    state["voices"] = services.VoiceDirectory
    state["log"]    = calllog.CallLogger (optional)
"""

from __future__ import annotations

import csv
import os
import re
from datetime import datetime
from pathlib import Path

from loguru import logger

from pipecat.frames.frames import ManuallySwitchServiceFrame, TTSSpeakFrame
from pipecat_flows import FlowArgs, FlowManager, FlowsFunctionSchema, NodeConfig

import booking

# RAG knowledge base — enabled once you run: python ingest.py
# Set ENABLE_RAG=false to switch it off. Falls back gracefully if missing.
SEARCH_KB_TOOL = None
if os.getenv("ENABLE_RAG", "true").lower() == "true":
    try:
        from rag import SEARCH_KB_TOOL
    except ImportError:
        logger.warning("RAG disabled: chromadb not installed or rag.py not found.")

COMPANY_NAME = os.getenv("COMPANY_NAME", "Hearthstone Home Services")
CUSTOMERS_CSV = Path(__file__).parent / "customers.csv"

# ---------------------------------------------------------------------------
# Opening greeting (spoken by the deterministic TTSSpeakFrame at call start).
# Includes the AI + recording disclosure: California B.O.T. Act / Utah AIPA
# safe harbors, all-party-consent recording states, EU AI Act Art. 50.
# The Spanish-language disclosure is replayed by LanguageRouter the first
# time a caller switches to Spanish.
# ---------------------------------------------------------------------------
OPENING_GREETING = (
    f"Thanks for calling {COMPANY_NAME}! Just so you know, I'm an automated "
    "A.I. assistant, and this call may be recorded and transcribed by automated "
    "systems. You can speak English or Spanish. How can I help you today?"
)

GREETINGS: dict[str, dict[str, str]] = {
    "billing": {
        "en": "You've reached billing! To get started, what's the best phone number on your account?",
        "es": "¡Le atiende el departamento de facturación! Para empezar, ¿cuál es el número de teléfono de su cuenta?",
    },
    "scheduling": {
        "en": "You've reached scheduling! I can help you book, reschedule, or cancel a visit. What's the best phone number on your account?",
        "es": "¡Le atiende el departamento de citas! Puedo ayudarle a reservar, cambiar o cancelar una visita. ¿Cuál es el número de teléfono de su cuenta?",
    },
    "customer_service": {
        "en": "You've reached our customer care team! I'm here to help — what's the best phone number on your account?",
        "es": "¡Le atiende nuestro equipo de atención al cliente! Estoy aquí para ayudarle. ¿Cuál es el número de teléfono de su cuenta?",
    },
    "escalation": {
        "en": "I'm sorry for the trouble. I'll take down your details and have a member of our team call you back. Could I get your name and the best number to reach you?",
        "es": "Lamento las molestias. Tomaré sus datos para que un miembro de nuestro equipo le devuelva la llamada. ¿Me puede dar su nombre y el mejor número para contactarle?",
    },
}

# ---------------------------------------------------------------------------
# Shared voice style + guardrails, prepended to every agent's persona.
# ---------------------------------------------------------------------------
VOICE_STYLE = (
    "You are on a live phone call, so keep every reply to one or two short, "
    "natural sentences. Use contractions. Ask only one question at a time. Never "
    "read out symbols, bullet points, or formatting. Do not mention tools, "
    "lookups, or internal steps. Respond in the language the caller is currently "
    "speaking — English or Spanish. If the caller asks whether you are a robot, "
    "an AI, or a human, answer truthfully that you are an AI assistant. Never "
    "state a price, fee, policy, or promise that you have not retrieved from the "
    "knowledge base or the caller's record — if you don't know, say a team "
    "member will follow up. If the caller is angry, asks for a human, or you "
    "cannot help after two attempts, use the route_to_human function."
)


# ---------------------------------------------------------------------------
# Customer lookup (customers.csv).
# ---------------------------------------------------------------------------
def _digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def _load_customers() -> list[dict]:
    if not CUSTOMERS_CSV.exists():
        logger.warning(f"No customer file at {CUSTOMERS_CSV}; lookups will miss.")
        return []
    with open(CUSTOMERS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


CUSTOMERS = _load_customers()


def _find_customer(phone: str) -> dict | None:
    query = _digits(phone)
    if not query:
        return None
    for row in CUSTOMERS:
        stored = _digits(row.get("phone_number", ""))
        if not stored:
            continue
        if (
            stored[-10:] == query[-10:]
            or stored.endswith(query)
            or query.endswith(stored)
        ):
            return row
    return None


async def lookup_customer(args: FlowArgs, flow_manager: FlowManager):
    phone = str(args.get("phone_number", ""))
    record = _find_customer(phone)
    if record:
        result = {
            "found": True,
            "customer_name": record.get("customer_name", ""),
            "service_address": record.get("service_address", ""),
            "service_history": record.get("service_history", ""),
            "last_call_notes": record.get("last_call_notes", ""),
        }
        flow_manager.state["customer"] = result
        flow_manager.state["phone"] = booking.normalize_phone(phone)
        logger.info(f"lookup_customer: matched {result['customer_name']}")
    else:
        result = {"found": False}
        flow_manager.state["phone"] = booking.normalize_phone(phone)
    _log(flow_manager, "tool", name="lookup_customer", found=result["found"])
    return result, None


LOOKUP_TOOL = FlowsFunctionSchema(
    name="lookup_customer",
    description=(
        "Look up an existing customer in the database by phone number. Call this "
        "once, as early as possible, using the phone number on the caller's "
        "account. Returns the customer's name, address, service history, and last "
        "call notes, or found=false if there is no matching record."
    ),
    properties={
        "phone_number": {
            "type": "string",
            "description": "The caller's phone number, digits only or formatted.",
        }
    },
    required=["phone_number"],
    handler=lookup_customer,
)


# ---------------------------------------------------------------------------
# Booking tools (real SQLite calendar — see booking.py).
# ---------------------------------------------------------------------------
async def check_availability(args: FlowArgs, flow_manager: FlowManager):
    date = args.get("date") or None
    if date:
        problem = booking.date_problem(str(date))
        if problem:
            _log(flow_manager, "tool", name="check_availability", date=date, problem=problem)
            return {
                "available": False,
                "note": problem,
                "alternatives": booking.available_slots(),
            }, None
    slots = booking.available_slots(date_str=date)
    _log(flow_manager, "tool", name="check_availability", date=date, count=len(slots))
    if not slots:
        return {
            "available": False,
            "note": "No open slots on that day." if date else "No open slots in that window.",
            "alternatives": booking.available_slots() if date else [],
        }, None
    return {"available": True, "slots": slots}, None


CHECK_AVAILABILITY_TOOL = FlowsFunctionSchema(
    name="check_availability",
    description=(
        "Get open appointment slots. Call this before offering any times — only "
        "offer times this function returns. Optionally pass a specific date."
    ),
    properties={
        "date": {
            "type": "string",
            "description": "Optional specific day to check, formatted YYYY-MM-DD.",
        }
    },
    required=[],
    handler=check_availability,
)


async def book_appointment(args: FlowArgs, flow_manager: FlowManager):
    result = booking.book(
        phone=str(args.get("phone_number") or flow_manager.state.get("phone", "")),
        customer_name=str(args.get("customer_name", "")),
        service=str(args.get("service", "")),
        slot_str=str(args.get("slot", "")),
        address=str(args.get("address", "")),
        notes=str(args.get("notes", "")),
    )
    _log(flow_manager, "tool", name="book_appointment", **result)
    return result, None


BOOK_TOOL = FlowsFunctionSchema(
    name="book_appointment",
    description=(
        "Book a technician visit in the real calendar. Only call this AFTER you "
        "have read the day, time, and service back to the caller and they have "
        "explicitly confirmed. Use a slot returned by check_availability."
    ),
    properties={
        "phone_number": {"type": "string", "description": "Caller's phone number."},
        "customer_name": {"type": "string", "description": "Caller's name."},
        "service": {
            "type": "string",
            "description": "Short description of the service needed, e.g. 'HVAC tune-up'.",
        },
        "slot": {
            "type": "string",
            "description": "The chosen slot, formatted YYYY-MM-DD HH:MM (24-hour).",
        },
        "address": {"type": "string", "description": "Service address, if known."},
        "notes": {"type": "string", "description": "Any extra notes."},
    },
    required=["phone_number", "customer_name", "service", "slot"],
    handler=book_appointment,
)


async def list_appointments(args: FlowArgs, flow_manager: FlowManager):
    phone = str(args.get("phone_number") or flow_manager.state.get("phone", ""))
    appointments = [
        {
            "appointment_id": a["id"],
            "slot": a["slot"],
            "label": a["label"],
            "service": a["service"],
        }
        for a in booking.upcoming_for_phone(phone)
    ]
    _log(flow_manager, "tool", name="list_appointments", count=len(appointments))
    return {"appointments": appointments}, None


LIST_APPOINTMENTS_TOOL = FlowsFunctionSchema(
    name="list_appointments",
    description=(
        "List the caller's upcoming appointments with their ids. Use this before "
        "canceling or rescheduling so you act on the right appointment."
    ),
    properties={
        "phone_number": {"type": "string", "description": "Caller's phone number."}
    },
    required=["phone_number"],
    handler=list_appointments,
)


async def cancel_appointment(args: FlowArgs, flow_manager: FlowManager):
    phone = str(args.get("phone_number") or flow_manager.state.get("phone", ""))
    appointment_id = args.get("appointment_id")
    try:
        appointment_id = int(appointment_id) if appointment_id is not None else None
    except (TypeError, ValueError):
        appointment_id = None
    result = booking.cancel(phone, appointment_id)
    _log(flow_manager, "tool", name="cancel_appointment", **result)
    return result, None


CANCEL_TOOL = FlowsFunctionSchema(
    name="cancel_appointment",
    description=(
        "Cancel an appointment. Confirm with the caller before calling this. If "
        "the caller has more than one upcoming appointment, call "
        "list_appointments first and pass the right appointment_id; without an "
        "id this cancels their soonest appointment."
    ),
    properties={
        "phone_number": {"type": "string", "description": "Caller's phone number."},
        "appointment_id": {
            "type": "integer",
            "description": "The id from list_appointments, when the caller has several.",
        },
    },
    required=["phone_number"],
    handler=cancel_appointment,
)


# ---------------------------------------------------------------------------
# Escalation: take a message, create a real ticket.
# ---------------------------------------------------------------------------
async def create_ticket(args: FlowArgs, flow_manager: FlowManager):
    voices = flow_manager.state.get("voices")
    result = booking.create_ticket(
        summary=str(args.get("summary", "")),
        phone=str(args.get("phone_number") or flow_manager.state.get("phone", "")),
        customer_name=str(args.get("customer_name", "")),
        urgency=str(args.get("urgency", "normal")),
        language=voices.language if voices else "en",
    )
    _log(flow_manager, "tool", name="create_ticket", **result)
    return {
        "ticket_created": True,
        "ticket_number": result["ticket_id"],
        "urgency": result["urgency"],
        "note": "Tell the caller their ticket number and that the team will call back within one business day.",
    }, None


CREATE_TICKET_TOOL = FlowsFunctionSchema(
    name="create_ticket",
    description=(
        "File a callback ticket for a human team member. Call this once you have "
        "the caller's name, phone number, and a one-sentence summary of the issue."
    ),
    properties={
        "customer_name": {"type": "string", "description": "Caller's name."},
        "phone_number": {"type": "string", "description": "Best callback number."},
        "summary": {
            "type": "string",
            "description": "One-sentence summary of the issue for the human team.",
        },
        "urgency": {
            "type": "string",
            "enum": ["low", "normal", "urgent"],
            "description": "How urgent the follow-up is.",
        },
    },
    required=["customer_name", "phone_number", "summary"],
    handler=create_ticket,
)


# ---------------------------------------------------------------------------
# Transfers: switch the department voice, greet in the caller's language,
# then hand the conversation to the department node.
# ---------------------------------------------------------------------------
def _log(flow_manager: FlowManager, event_type: str, **data) -> None:
    log = flow_manager.state.get("log")
    if log:
        log.event(event_type, **data)


async def _switch_voice_and_greet(
    flow_manager: FlowManager, department: str, greeting_key: str
) -> None:
    voices = flow_manager.state.get("voices")
    frames = []
    language = "en"
    if voices:
        language = voices.language
        frames.append(
            ManuallySwitchServiceFrame(service=voices.set_department(department))
        )
    greeting = GREETINGS.get(greeting_key, {}).get(language)
    if greeting:
        frames.append(TTSSpeakFrame(greeting))
    if frames:
        await flow_manager.worker.queue_frames(frames)


def _make_transfer(department: str, node_factory, greeting_key: str | None = None):
    greeting_key = greeting_key or department

    # The voice switch + greeting run as a "function" pre-action on the new
    # node rather than directly in this handler: a function action executes
    # only when the pipeline has drained everything queued before it, so the
    # new voice can never overlap audio the old voice is still speaking.
    async def switch_and_greet(action: dict, flow_manager: FlowManager) -> None:
        await _switch_voice_and_greet(flow_manager, department, greeting_key)

    async def handler(args: FlowArgs, flow_manager: FlowManager):
        _log(flow_manager, "transfer", department=greeting_key)
        node = node_factory()
        node["pre_actions"] = [
            {"type": "function", "handler": switch_and_greet}
        ] + list(node.get("pre_actions", []))
        return {"status": "transferring", "to": greeting_key}, node

    return handler


def _transfer_tool(name: str, description: str, handler) -> FlowsFunctionSchema:
    return FlowsFunctionSchema(
        name=name, description=description, properties={}, required=[], handler=handler
    )


# Defined below the node factories they reference (resolved at call time).
ROUTE_BILLING = _transfer_tool(
    "route_to_billing",
    "Transfer the caller to billing: invoices, payments, refunds, charges, "
    "balances, or pricing on a past job.",
    _make_transfer("billing", lambda: create_billing_node()),
)
ROUTE_SCHEDULING = _transfer_tool(
    "route_to_scheduling",
    "Transfer the caller to scheduling: booking, rescheduling, or canceling a "
    "technician visit.",
    _make_transfer("scheduling", lambda: create_scheduling_node()),
)
ROUTE_CS = _transfer_tool(
    "route_to_customer_service",
    "Transfer the caller to customer service: problems, complaints, follow-ups, "
    "bad experiences, or general questions.",
    _make_transfer("customer_service", lambda: create_cs_node()),
)
ROUTE_HUMAN = _transfer_tool(
    "route_to_human",
    "Use when the caller asks for a human or supervisor, is upset, or you cannot "
    "help after two attempts. Takes a message for a human callback.",
    _make_transfer(
        "customer_service", lambda: create_escalation_node(), greeting_key="escalation"
    ),
)


async def _end_call(args: FlowArgs, flow_manager: FlowManager):
    _log(flow_manager, "end_requested")
    return {"status": "ending"}, create_end_node()


END_CALL = _transfer_tool(
    "end_call",
    "End the call politely. Use only when the caller says they are done and has "
    "no other questions.",
    _end_call,
)

GLOBAL_FUNCTIONS = [ROUTE_HUMAN, END_CALL]


def _rag_tools() -> list:
    return [SEARCH_KB_TOOL] if SEARCH_KB_TOOL else []


# ---------------------------------------------------------------------------
# Nodes.
# ---------------------------------------------------------------------------
def create_router_node() -> NodeConfig:
    return {
        "name": "router",
        "role_message": (
            f"You are the virtual receptionist for {COMPANY_NAME}, a home services "
            f"company. You are the first person every caller reaches. {VOICE_STYLE}"
        ),
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "The caller has already been greeted. Based on what they say, "
                    "decide which department they need and call the matching "
                    "transfer function immediately, WITHOUT saying anything first — "
                    "the department announces itself when the transfer lands. "
                    "Billing covers invoices, payments, refunds, charges, balances, "
                    "and pricing on a past job. Scheduling covers booking, "
                    "rescheduling, canceling, appointments, and technician visits. "
                    "Customer service covers problems, complaints, follow-ups, bad "
                    "experiences, and general questions. If it's unclear, ask one "
                    "quick question; if it's still unclear, use customer service. "
                    "Don't try to solve the request yourself. If the caller speaks "
                    "Spanish, reply in Spanish."
                ),
            }
        ],
        "functions": [ROUTE_BILLING, ROUTE_SCHEDULING, ROUTE_CS],
        "respond_immediately": False,
    }


def create_billing_node() -> NodeConfig:
    return {
        "name": "billing",
        "role_message": (
            f"You are a billing specialist at {COMPANY_NAME}. You help callers "
            f"understand charges, invoices, payments, and balances. You sound calm, "
            f"professional, and clear. {VOICE_STYLE}"
        ),
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "The caller has just been greeted and asked for their phone "
                    "number. Once you have it, call lookup_customer. Use their name "
                    "and history naturally. Only state amounts, dates, or charges "
                    "you actually find in their record or the knowledge base; if "
                    "you don't have a detail, say you'll look into it rather than "
                    "guessing. Never invent prices or promise refunds — refund "
                    "requests go to route_to_human. If the caller needs scheduling "
                    "or has a service problem, call the matching transfer function "
                    "without announcing it — the department greets them itself."
                ),
            }
        ],
        "functions": [LOOKUP_TOOL, ROUTE_SCHEDULING, ROUTE_CS] + _rag_tools(),
        "respond_immediately": False,
    }


def create_scheduling_node() -> NodeConfig:
    return {
        "name": "scheduling",
        "role_message": (
            f"You are a scheduling coordinator at {COMPANY_NAME}. You help callers "
            f"book, reschedule, or cancel technician visits. You are warm, upbeat, "
            f"and efficient. {VOICE_STYLE}"
        ),
        "task_messages": [
            {
                "role": "system",
                "content": (
                    f"Today is {datetime.now():%A, %B %d, %Y} "
                    f"({datetime.now():%Y-%m-%d}). "
                    "The caller has just been greeted and asked for their phone "
                    "number. Once you have it, call lookup_customer. To book: call "
                    "check_availability and offer two or three of the returned "
                    "times conversationally — never offer a time it didn't return. "
                    "Before booking, read the full details back (day, time, "
                    "service) and get an explicit yes; only then call "
                    "book_appointment, and confirm using what it returns. To "
                    "reschedule or cancel: call list_appointments first; if there "
                    "is more than one, confirm which and pass its appointment_id "
                    "to cancel_appointment; for a reschedule, then book the new "
                    "time the same careful way. If a booking fails, offer the "
                    "alternatives it returns. If the caller needs billing or has a "
                    "complaint, call the matching transfer function without "
                    "announcing it — the department greets them itself."
                ),
            }
        ],
        "functions": [
            LOOKUP_TOOL,
            CHECK_AVAILABILITY_TOOL,
            BOOK_TOOL,
            LIST_APPOINTMENTS_TOOL,
            CANCEL_TOOL,
            ROUTE_BILLING,
            ROUTE_CS,
        ]
        + _rag_tools(),
        "respond_immediately": False,
    }


def create_cs_node() -> NodeConfig:
    return {
        "name": "customer_service",
        "role_message": (
            f"You are a customer care specialist at {COMPANY_NAME}. You handle "
            f"general questions, complaints, and follow-up support. You are "
            f"empathetic, calm, and helpful, and you acknowledge how the caller "
            f"feels before jumping to solutions. {VOICE_STYLE}"
        ),
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "The caller has just been greeted and asked for their phone "
                    "number. Once you have it, call lookup_customer and pay "
                    "attention to their service history and last call notes. Get a "
                    "brief, clear picture of the issue with a short follow-up "
                    "question if needed. For complaints, acknowledge and apologize "
                    "before next steps. "
                    + (
                        "Use search_knowledge_base to answer policy or service "
                        "questions. "
                        if SEARCH_KB_TOOL
                        else "For policy questions you can't answer from the "
                        "caller's record, say a team member will follow up. "
                    )
                    + "Don't promise refunds, credits, or outcomes you can't "
                    "confirm — those go to route_to_human, which files a callback "
                    "ticket. If the caller needs billing or to schedule a visit, "
                    "call the matching transfer function without announcing it — "
                    "the department greets them itself."
                ),
            }
        ],
        "functions": [LOOKUP_TOOL, ROUTE_BILLING, ROUTE_SCHEDULING] + _rag_tools(),
        "respond_immediately": False,
    }


def create_escalation_node() -> NodeConfig:
    return {
        "name": "escalation",
        "role_message": (
            f"You take messages for the human team at {COMPANY_NAME}. You are "
            f"patient and reassuring. {VOICE_STYLE}"
        ),
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "The caller has just been told you'll take their details for a "
                    "human callback. Collect three things, one at a time if needed: "
                    "their name, the best callback number, and a one-sentence "
                    "summary of the issue. Then call create_ticket, tell them their "
                    "ticket number, and say a team member will call back within one "
                    "business day. Don't argue, don't promise outcomes. When "
                    "they're done, use end_call."
                ),
            }
        ],
        "functions": [CREATE_TICKET_TOOL],
        "respond_immediately": False,
    }


def create_end_node() -> NodeConfig:
    return {
        "name": "end",
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "The conversation is over. Say one short, warm goodbye sentence "
                    "in the language the caller has been speaking. Do not ask "
                    "anything else."
                ),
            }
        ],
        "functions": [],
        "respond_immediately": True,
        "post_actions": [{"type": "end_conversation"}],
    }
