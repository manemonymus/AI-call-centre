"""
Shared multi-agent flow for the HR employee helpline (bot.py and server.py).

Employees call in; a receptionist routes them to one of four HR departments,
each with its OWN AI voice and its OWN knowledge base (per-department RAG):

    Reception ─┬─ Leave & Time-Off      (hr_leave KB)
               ├─ Conduct & Ethics      (hr_conduct KB)
               ├─ Compliance & Reporting(hr_compliance KB)
               └─ HR Policy & General    (hr_general KB)
    + Escalation (human HR callback ticket)  + End

Each department answers ONLY from its own knowledge base (rag.make_search_tool),
so departments don't cross-contaminate. The data is real HR policy Q&A — build
it with `python fetch_hr_data.py` then `python ingest.py`.

What's grounded/real:
  - Answers come from the department's RAG knowledge base; agents are told
    never to invent policies, numbers, or rules not retrieved.
  - Escalations create real callback tickets (booking.create_ticket).
  - Employee lookup (employees.csv) lets agents greet by name and skip
    re-asking once identity is known.

Expected flow_manager.state entries (set before initialize()):
    state["voices"] = services.VoiceDirectory
    state["log"]    = calllog.CallLogger (optional)
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

from loguru import logger

from pipecat.frames.frames import ManuallySwitchServiceFrame, TTSSpeakFrame
from pipecat_flows import FlowArgs, FlowManager, FlowsFunctionSchema, NodeConfig

import booking

# Per-department knowledge-base search tools. Disabled gracefully if chromadb
# isn't installed; set ENABLE_RAG=false to turn off.
_RAG = os.getenv("ENABLE_RAG", "true").lower() == "true"
if _RAG:
    try:
        from rag import make_search_tool
    except ImportError:
        logger.warning("RAG disabled: chromadb not installed or rag.py not found.")
        _RAG = False
if not _RAG:
    def make_search_tool(_department):  # type: ignore
        return None

COMPANY_NAME = os.getenv("COMPANY_NAME", "Hearthstone")
EMPLOYEES_CSV = Path(__file__).parent / "employees.csv"

# ---------------------------------------------------------------------------
# Opening greeting — spoken at call start by a deterministic TTSSpeakFrame.
# Includes the AI + recording disclosure (California B.O.T. Act / Utah AIPA
# safe harbors, all-party recording consent, EU AI Act Art. 50). The Spanish
# disclosure is replayed by LanguageRouter the first time a caller switches.
# ---------------------------------------------------------------------------
OPENING_GREETING = (
    f"Thanks for calling the {COMPANY_NAME} HR help line! Just so you know, I'm "
    "an automated A.I. assistant, and this call may be recorded and transcribed "
    "by automated systems. You can speak English or Spanish. What can I help you "
    "with today?"
)

# First-contact greetings ask for the employee ID / phone; once we know the
# caller, the *_known variants greet by name and skip re-asking.
GREETINGS: dict[str, dict[str, str]] = {
    "leave": {
        "en": "You've reached the leave and time-off team! What's your employee ID or the phone number on file?",
        "es": "¡Le atiende el equipo de permisos y ausencias! ¿Cuál es su número de empleado o el teléfono registrado?",
        "en_known": "You've reached the leave and time-off team! What would you like to know{name}?",
        "es_known": "¡Le atiende el equipo de permisos y ausencias! ¿Qué desea saber{name}?",
    },
    "conduct": {
        "en": "You've reached the conduct and ethics team! What's your employee ID or the phone number on file?",
        "es": "¡Le atiende el equipo de conducta y ética! ¿Cuál es su número de empleado o el teléfono registrado?",
        "en_known": "You've reached the conduct and ethics team! How can I help{name}?",
        "es_known": "¡Le atiende el equipo de conducta y ética! ¿En qué puedo ayudarle{name}?",
    },
    "compliance": {
        "en": "You've reached compliance and reporting! What's your employee ID or the phone number on file?",
        "es": "¡Le atiende el equipo de cumplimiento! ¿Cuál es su número de empleado o el teléfono registrado?",
        "en_known": "You've reached compliance and reporting! What can I do for you{name}?",
        "es_known": "¡Le atiende el equipo de cumplimiento! ¿Qué puedo hacer por usted{name}?",
    },
    "general": {
        "en": "You've reached general HR! What's your employee ID or the phone number on file?",
        "es": "¡Le atiende recursos humanos! ¿Cuál es su número de empleado o el teléfono registrado?",
        "en_known": "You've reached general HR! What's your question{name}?",
        "es_known": "¡Le atiende recursos humanos! ¿Cuál es su pregunta{name}?",
    },
    "escalation": {
        "en": "I'll take down your details and have someone from HR call you back. What should they know?",
        "es": "Tomaré sus datos para que alguien de recursos humanos le devuelva la llamada. ¿Qué deben saber?",
        "en_known": "I'll have someone from HR call you back{name}. What should they know?",
        "es_known": "Haré que alguien de recursos humanos le devuelva la llamada{name}. ¿Qué deben saber?",
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
    "ask for information the caller already gave earlier in this call — their "
    "name, employee ID, and looked-up details stay valid across department "
    "transfers. You handle HR policy questions ONLY: never give legal, medical, "
    "or financial advice, and never state a policy, number, deadline, or rule "
    "that you did not get from search_knowledge_base or the caller's record. If "
    "you don't find it, say an HR specialist will follow up. For personal "
    "disputes, complaints about a person, or anything sensitive the knowledge "
    "base can't resolve, use route_to_human."
)


# ---------------------------------------------------------------------------
# Employee lookup (employees.csv).
# ---------------------------------------------------------------------------
def _digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def _load_employees() -> list[dict]:
    if not EMPLOYEES_CSV.exists():
        logger.warning(f"No employee file at {EMPLOYEES_CSV}; lookups will miss.")
        return []
    with open(EMPLOYEES_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


EMPLOYEES = _load_employees()


def _find_employee(identifier: str) -> dict | None:
    """Match by employee ID (e.g. E1042) or by phone number (format-agnostic)."""
    ident = (identifier or "").strip().lower()
    if ident:
        for row in EMPLOYEES:
            if row.get("employee_id", "").strip().lower() == ident:
                return row
    query = _digits(identifier)
    if not query:
        return None
    for row in EMPLOYEES:
        stored = _digits(row.get("phone_number", ""))
        if stored and (
            stored[-10:] == query[-10:]
            or stored.endswith(query)
            or query.endswith(stored)
        ):
            return row
    return None


async def lookup_employee(args: FlowArgs, flow_manager: FlowManager):
    identifier = str(args.get("identifier", ""))
    record = _find_employee(identifier)
    if record:
        result = {
            "found": True,
            "employee_name": record.get("employee_name", ""),
            "employee_id": record.get("employee_id", ""),
            "job_title": record.get("job_title", ""),
            "department": record.get("department", ""),
            "hire_date": record.get("hire_date", ""),
            "manager": record.get("manager", ""),
        }
        flow_manager.state["employee"] = result
        flow_manager.state["phone"] = booking.normalize_phone(
            record.get("phone_number", "")
        )
        logger.info(f"lookup_employee: matched {result['employee_name']}")
    else:
        result = {"found": False}
    _log(flow_manager, "tool", name="lookup_employee", found=result["found"])
    return result, None


LOOKUP_TOOL = FlowsFunctionSchema(
    name="lookup_employee",
    description=(
        "Look up the caller in the employee directory by their employee ID "
        "(e.g. E1042) or phone number. Call this once, early, to greet them by "
        "name and tailor answers. Returns name, title, department, hire date, "
        "and manager, or found=false if there's no match."
    ),
    properties={
        "identifier": {
            "type": "string",
            "description": "The caller's employee ID or phone number.",
        }
    },
    required=["identifier"],
    handler=lookup_employee,
)


# ---------------------------------------------------------------------------
# Escalation: take a message, create a real HR callback ticket.
# ---------------------------------------------------------------------------
async def create_ticket(args: FlowArgs, flow_manager: FlowManager):
    voices = flow_manager.state.get("voices")
    employee = flow_manager.state.get("employee") or {}
    name = args.get("employee_name") or employee.get("employee_name", "")
    result = booking.create_ticket(
        summary=str(args.get("summary", "")),
        phone=str(args.get("phone_number") or flow_manager.state.get("phone", "")),
        customer_name=str(name),
        urgency=str(args.get("urgency", "normal")),
        language=voices.language if voices else "en",
    )
    _log(flow_manager, "tool", name="create_ticket", **result)
    return {
        "ticket_created": True,
        "ticket_number": result["ticket_id"],
        "urgency": result["urgency"],
        "note": "Tell the caller their ticket number and that HR will call back within one business day.",
    }, None


CREATE_TICKET_TOOL = FlowsFunctionSchema(
    name="create_ticket",
    description=(
        "File a callback ticket for a human HR representative. Call this once you "
        "have the caller's name, a callback number, and a one-sentence summary."
    ),
    properties={
        "employee_name": {"type": "string", "description": "Caller's name."},
        "phone_number": {"type": "string", "description": "Best callback number."},
        "summary": {
            "type": "string",
            "description": "One-sentence summary of the issue for the HR team.",
        },
        "urgency": {
            "type": "string",
            "enum": ["low", "normal", "urgent"],
            "description": "How urgent the follow-up is.",
        },
    },
    required=["employee_name", "phone_number", "summary"],
    handler=create_ticket,
)


# ---------------------------------------------------------------------------
# Logging + transfer helpers.
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
        service = voices.set_department(department)
        frames.append(ManuallySwitchServiceFrame(service=service))
        _log(
            flow_manager,
            "voice_switch",
            department=department,
            voice=voices.voice_map.get((department, language), "?"),
        )

    # Once we know who's calling, greet by name and don't re-ask for their ID.
    employee = flow_manager.state.get("employee") or {}
    known = bool(employee.get("found") or flow_manager.state.get("phone"))
    variant = f"{language}_known" if known else language
    greeting = GREETINGS.get(greeting_key, {}).get(variant)
    if greeting:
        first_name = str(employee.get("employee_name", "")).split(" ")[0]
        name = f", {first_name}" if first_name else ""
        greeting = greeting.format(name=name)
        frames.append(TTSSpeakFrame(greeting))
    if frames:
        await flow_manager.worker.queue_frames(frames)


def _make_transfer(department: str, node_factory, greeting_key: str | None = None):
    greeting_key = greeting_key or department

    # The voice switch + greeting run as a "function" pre-action on the new
    # node so they execute only after in-flight audio has drained — the new
    # voice can never overlap the previous one.
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


ROUTE_LEAVE = _transfer_tool(
    "route_to_leave",
    "Transfer to the leave and time-off team: PTO, vacation, compensatory off, "
    "overtime, working hours, holidays, and attendance.",
    _make_transfer("leave", lambda: create_leave_node()),
)
ROUTE_CONDUCT = _transfer_tool(
    "route_to_conduct",
    "Transfer to conduct and ethics: code of conduct, gifts and entertainment, "
    "anti-bribery, harassment, conflicts of interest, and ethical behavior.",
    _make_transfer("conduct", lambda: create_conduct_node()),
)
ROUTE_COMPLIANCE = _transfer_tool(
    "route_to_compliance",
    "Transfer to compliance and reporting: how to report a violation, "
    "disciplinary processes, investigations, and consequences for breaking policy.",
    _make_transfer("compliance", lambda: create_compliance_node()),
)
ROUTE_GENERAL = _transfer_tool(
    "route_to_general",
    "Transfer to general HR: how policies are reviewed, updated, or communicated, "
    "and any HR policy question that doesn't fit the other teams.",
    _make_transfer("general", lambda: create_general_node()),
)
ROUTE_HUMAN = _transfer_tool(
    "route_to_human",
    "Use when the caller asks for a human, is upset, raises a personal dispute or "
    "complaint about a specific person, or you can't help after two attempts. "
    "Takes a message for a human HR callback.",
    _make_transfer(
        "general", lambda: create_escalation_node(), greeting_key="escalation"
    ),
)


async def _end_call(args: FlowArgs, flow_manager: FlowManager):
    _log(flow_manager, "end_requested")
    return {"status": "ending"}, create_end_node()


END_CALL = _transfer_tool(
    "end_call",
    "End the call politely. Use only when the caller says they're done and has no "
    "other questions.",
    _end_call,
)

GLOBAL_FUNCTIONS = [ROUTE_HUMAN, END_CALL]

# Every department can hand off to any sibling department.
_SIBLING_ROUTES = {
    "leave": [ROUTE_CONDUCT, ROUTE_COMPLIANCE, ROUTE_GENERAL],
    "conduct": [ROUTE_LEAVE, ROUTE_COMPLIANCE, ROUTE_GENERAL],
    "compliance": [ROUTE_LEAVE, ROUTE_CONDUCT, ROUTE_GENERAL],
    "general": [ROUTE_LEAVE, ROUTE_CONDUCT, ROUTE_COMPLIANCE],
}


def _department_functions(department: str) -> list:
    """Lookup + this department's own scoped KB search + sibling transfers."""
    tools = [LOOKUP_TOOL]
    search = make_search_tool(department)
    if search:
        tools.append(search)
    tools.extend(_SIBLING_ROUTES[department])
    return tools


_DEPARTMENT_TASK = (
    "If this conversation does not already contain the caller's record, they "
    "were just asked for their employee ID or phone number — once you have it, "
    "call lookup_employee (once per call). If their record is already here, do "
    "NOT ask again. For any specific policy question, call search_knowledge_base "
    "and answer ONLY from what it returns — never guess a rule, number, or "
    "deadline. If it finds nothing, say an HR specialist will follow up. If the "
    "caller's question belongs to another team, call the matching transfer "
    "function without announcing it — that team greets them itself."
)


# ---------------------------------------------------------------------------
# Department nodes.
# ---------------------------------------------------------------------------
def create_router_node() -> NodeConfig:
    return {
        "name": "router",
        "role_message": (
            f"You are the virtual HR receptionist for {COMPANY_NAME}. You are the "
            f"first person every employee reaches when they call HR. {VOICE_STYLE}"
        ),
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "The caller has already been greeted. Based on what they say, "
                    "decide which HR team they need and call the matching transfer "
                    "function immediately, WITHOUT saying anything first — the team "
                    "announces itself when the transfer lands. Leave & time-off: "
                    "PTO, vacation, compensatory off, overtime, working hours, "
                    "attendance. Conduct & ethics: code of conduct, gifts, "
                    "anti-bribery, harassment, conflicts of interest. Compliance & "
                    "reporting: reporting a violation, disciplinary action, "
                    "investigations, consequences. General HR: how policies are "
                    "reviewed or communicated, and anything else. If it's unclear, "
                    "ask one quick question; if still unclear, use general HR. "
                    "Don't try to answer the question yourself."
                ),
            }
        ],
        "functions": [ROUTE_LEAVE, ROUTE_CONDUCT, ROUTE_COMPLIANCE, ROUTE_GENERAL],
        "respond_immediately": False,
    }


def _department_node(name: str, persona: str) -> NodeConfig:
    return {
        "name": name,
        "role_message": f"{persona} {VOICE_STYLE}",
        "task_messages": [{"role": "system", "content": _DEPARTMENT_TASK}],
        "functions": _department_functions(name),
        "respond_immediately": False,
    }


def create_leave_node() -> NodeConfig:
    return _department_node(
        "leave",
        f"You are a leave and time-off specialist at {COMPANY_NAME}. You help "
        "employees understand PTO, vacation, compensatory off, overtime, working "
        "hours, and attendance policy. You are warm and clear.",
    )


def create_conduct_node() -> NodeConfig:
    return _department_node(
        "conduct",
        f"You are a workplace conduct and ethics specialist at {COMPANY_NAME}. "
        "You explain the code of conduct, gifts and entertainment rules, "
        "anti-bribery policy, harassment policy, and conflicts of interest. You "
        "are professional and even-handed.",
    )


def create_compliance_node() -> NodeConfig:
    return _department_node(
        "compliance",
        f"You are a compliance and reporting specialist at {COMPANY_NAME}. You "
        "explain how to report a concern, what disciplinary processes look like, "
        "and the consequences of policy violations. You are calm, precise, and "
        "non-judgmental.",
    )


def create_general_node() -> NodeConfig:
    return _department_node(
        "general",
        f"You are a general HR specialist at {COMPANY_NAME}. You handle questions "
        "about how policies are reviewed, updated, and communicated, and any HR "
        "policy question that doesn't fit another team. You are friendly and helpful.",
    )


def create_escalation_node() -> NodeConfig:
    return {
        "name": "escalation",
        "role_message": (
            f"You take messages for the human HR team at {COMPANY_NAME}. You are "
            f"patient and reassuring. {VOICE_STYLE}"
        ),
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "The caller has just been told you'll take their details for an "
                    "HR callback. Collect three things, one at a time if needed: "
                    "their name, the best callback number, and a one-sentence "
                    "summary of the issue. Then call create_ticket, tell them their "
                    "ticket number, and say HR will call back within one business "
                    "day. Don't argue or promise outcomes. When done, use end_call."
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
