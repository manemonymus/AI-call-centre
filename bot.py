#!/usr/bin/env python3
"""
Local-only multi-agent AI call center demo — Pipecat + Pipecat Flows.

Everything runs on your machine. No phone line, no cloud APIs, no API keys:

    Microphone + speakers  ->  LocalAudioTransport
    Speech-to-text         ->  Whisper (faster-whisper), runs locally
    LLM brains             ->  a local model served by Ollama
    Text-to-speech         ->  Piper, runs locally (voice auto-downloaded)

Four "agents" are implemented as Flow nodes:

    Router  ->  Billing | Scheduling | Customer Service

A department "transfer" is just a node transition. The conversation/context
carries across the handoff, so the caller never has to repeat themselves.

The single tool, lookup_customer, reads customers.csv by phone number.

Verified against pipecat-ai==1.3.0 and pipecat-ai-flows==1.2.0.
"""

import asyncio
import csv
import re
from pathlib import Path

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import Model, WhisperSTTService
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat_flows import FlowArgs, FlowManager, FlowsFunctionSchema, NodeConfig

# ---------------------------------------------------------------------------
# Configuration — tweak for your machine.
# ---------------------------------------------------------------------------
COMPANY_NAME = "Hearthstone Home Services"

# Any tool-calling Ollama model: llama3.2, qwen2.5, llama3.1, etc.
# Pull it first, e.g.:  ollama pull llama3.2
OLLAMA_MODEL = "qwen2.5"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# TINY / BASE / SMALL / MEDIUM / LARGE — bigger is more accurate but slower.
WHISPER_MODEL = Model.BASE

# A Piper voice id. Auto-downloaded into the working directory on first run.
# Browse voices at https://github.com/rhasspy/piper (e.g. en_US-amy-medium).
PIPER_VOICE = "en_US-lessac-medium"

CUSTOMERS_CSV = Path(__file__).parent / "customers.csv"


# ---------------------------------------------------------------------------
# The one tool: look a customer up in the local CSV by phone number.
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
    """Tolerant phone match: ignores formatting and country-code differences."""
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
    """lookup_customer handler. Returns (result, next_node); None == stay here."""
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
        flow_manager.state["customer"] = result  # remember across handoffs
        logger.info(f"lookup_customer: matched {result['customer_name']}")
    else:
        result = {"found": False}
        logger.info(f"lookup_customer: no record for '{phone}'")
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
# Shared voice style, prepended to every agent's persona.
# ---------------------------------------------------------------------------
VOICE_STYLE = (
    "You are on a live phone call, so keep every reply to one or two short, "
    "natural sentences. Use contractions. Ask only one question at a time. Never "
    "read out symbols, bullet points, or formatting. Do not mention tools, "
    "lookups, internal steps, or that this is a demo."
)


# ---------------------------------------------------------------------------
# Transfer handlers (these return the next node, which performs the handoff).
# ---------------------------------------------------------------------------
async def route_to_billing(args: FlowArgs, flow_manager: FlowManager):
    return {"status": "transferring", "to": "billing"}, create_billing_node()


async def route_to_scheduling(args: FlowArgs, flow_manager: FlowManager):
    return {"status": "transferring", "to": "scheduling"}, create_scheduling_node()


async def route_to_customer_service(args: FlowArgs, flow_manager: FlowManager):
    return {"status": "transferring", "to": "customer_service"}, create_cs_node()


def _transfer_tool(name: str, description: str, handler) -> FlowsFunctionSchema:
    return FlowsFunctionSchema(
        name=name, description=description, properties={}, required=[], handler=handler
    )


ROUTE_BILLING = _transfer_tool(
    "route_to_billing",
    "Transfer the caller to billing: invoices, payments, refunds, charges, "
    "balances, or pricing on a past job.",
    route_to_billing,
)
ROUTE_SCHEDULING = _transfer_tool(
    "route_to_scheduling",
    "Transfer the caller to scheduling: booking, rescheduling, or canceling a "
    "technician visit.",
    route_to_scheduling,
)
ROUTE_CS = _transfer_tool(
    "route_to_customer_service",
    "Transfer the caller to customer service: problems, complaints, follow-ups, "
    "bad experiences, or general questions.",
    route_to_customer_service,
)


# ---------------------------------------------------------------------------
# The four agents, defined as Flow nodes.
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
                    "decide which department they need, briefly tell them you're "
                    "connecting them, then call the matching transfer function. "
                    "Billing covers invoices, payments, refunds, charges, balances, "
                    "and pricing on a past job. Scheduling covers booking, "
                    "rescheduling, canceling, appointments, and technician visits. "
                    "Customer service covers problems, complaints, follow-ups, bad "
                    "experiences, and general questions. If it's unclear, ask one "
                    "quick question; if it's still unclear, use customer service. "
                    "Don't try to solve the request yourself."
                ),
            }
        ],
        "functions": [ROUTE_BILLING, ROUTE_SCHEDULING, ROUTE_CS],
        # Wait for the caller to speak; the greeting is spoken separately. This
        # also stops the model from firing a transfer before the caller says
        # anything.
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
                    "If you don't already know the caller's phone number from the "
                    "conversation, ask for the best number on their account, then "
                    "call lookup_customer with it. Use their name and history "
                    "naturally. Only state amounts, dates, or charges you actually "
                    "find in their record; if you don't have a detail, say you'll "
                    "look into it rather than guessing. Never invent prices or "
                    "promise refunds. If the caller actually needs scheduling or has "
                    "a service problem, briefly say you'll connect them and call the "
                    "matching transfer function."
                ),
            }
        ],
        "functions": [LOOKUP_TOOL, ROUTE_SCHEDULING, ROUTE_CS],
        "respond_immediately": True,
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
                    "If you don't already know the caller's phone number, ask for "
                    "the number on their account, then call lookup_customer. For "
                    "booking, offer general availability conversationally, for "
                    "example 'we can get a technician out tomorrow morning or "
                    "afternoon' or 'our next opening is tomorrow at 10 AM or 2 PM', "
                    "let them choose, then confirm confidently as if it's booked. "
                    "For a reschedule, reference their existing appointment if you "
                    "can see one and confirm the new time. For a cancellation, "
                    "confirm it's canceled and offer to rebook. There is no real "
                    "calendar — just confirm directly with the caller. If the caller "
                    "needs billing or has a complaint, say you'll connect them and "
                    "call the matching transfer function."
                ),
            }
        ],
        "functions": [LOOKUP_TOOL, ROUTE_BILLING, ROUTE_CS],
        "respond_immediately": True,
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
                    "If you don't already know the caller's phone number, ask for "
                    "the number on their account, then call lookup_customer and pay "
                    "attention to their service history and last call notes. Get a "
                    "brief, clear picture of the issue with a short follow-up "
                    "question if needed. For complaints, acknowledge and apologize "
                    "before next steps. Don't promise refunds, credits, or outcomes "
                    "you can't confirm — it's fine to say a specialist will follow "
                    "up. If the caller actually needs billing or to schedule a "
                    "visit, say you'll connect them and call the matching transfer "
                    "function."
                ),
            }
        ],
        "functions": [LOOKUP_TOOL, ROUTE_BILLING, ROUTE_SCHEDULING],
        "respond_immediately": True,
    }


# ---------------------------------------------------------------------------
# Build the pipeline and run.
# ---------------------------------------------------------------------------
async def main() -> None:
    logger.info("Local AI call center starting. Speak into your mic; Ctrl+C to quit.")

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    # device="cpu" avoids needing CUDA/cuBLAS DLLs (Whisper's "auto" default
    # tries the GPU); int8 keeps CPU inference reasonably fast.
    stt = WhisperSTTService(model=WHISPER_MODEL, device="cpu", compute_type="int8")
    llm = OLLamaLLMService(
        settings=OLLamaLLMService.Settings(model=OLLAMA_MODEL),
        base_url=OLLAMA_BASE_URL,
    )
    tts = PiperTTSService(settings=PiperTTSService.Settings(voice=PIPER_VOICE))

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            VADProcessor(vad_analyzer=SileroVADAnalyzer()),  # detects speech / interruptions
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    worker = PipelineWorker(pipeline)

    flow_manager = FlowManager(
        llm=llm,
        context_aggregator=context_aggregator,
        worker=worker,
        transport=transport,
    )

    # Set up the router. It waits for the caller to speak, then routes.
    await flow_manager.initialize(create_router_node())

    # Speak a fixed greeting. Queued here, it buffers and plays the moment the
    # pipeline starts (the same mechanism verified by speak_test.py). We do this
    # rather than have the local model generate the greeting, which it tended to
    # skip in favor of immediately calling a transfer function.
    await worker.queue_frames(
        [TTSSpeakFrame(f"Thanks for calling {COMPANY_NAME}! How can I help you today?")]
    )

    runner = PipelineRunner(handle_sigint=True)
    await runner.run(worker)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye.")
