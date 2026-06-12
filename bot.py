#!/usr/bin/env python3
"""
Local-only multi-agent AI call center — Pipecat + Pipecat Flows.

Everything runs on your machine. No phone line, no cloud APIs, no API keys:

    Microphone + speakers  ->  LocalAudioTransport
    Speech-to-text         ->  Whisper (faster-whisper), runs locally,
                               auto-detects English vs Spanish per utterance
    LLM brains             ->  a local model served by Ollama
    Text-to-speech         ->  Piper, runs locally — a DIFFERENT voice per
                               department, with Spanish voices when the
                               caller speaks Spanish

Five "agents" are implemented as Flow nodes (see flows.py):

    Router -> Billing | Scheduling | Customer Service | Escalation

A department "transfer" switches the active TTS voice and carries the
conversation context across, so the caller never repeats themselves.
Bookings are real (SQLite calendar in booking.py), escalations create
callback tickets, and every call writes a transcript to call_logs/.

First run downloads the extra Piper voices (~300 MB total). To swap any
component for a cloud service later, see the env vars in services.py.

Verified against pipecat-ai==1.3.0 and pipecat-ai-flows==1.2.0.
"""

import asyncio

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
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat_flows import FlowManager

from calllog import CallLogger
from flows import GLOBAL_FUNCTIONS, OPENING_GREETING, create_router_node
from services import LanguageRouter, VoiceDirectory, build_llm, build_stt, build_tts


async def main() -> None:
    logger.info("Local AI call center starting. Speak into your mic; Ctrl+C to quit.")

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    call_logger = CallLogger(channel="local")
    stt = build_stt()
    llm = build_llm()
    voices = VoiceDirectory()  # downloads any missing Piper voices on first run
    tts = build_tts(voices)

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            VADProcessor(vad_analyzer=SileroVADAnalyzer()),  # speech/interruptions
            stt,
            LanguageRouter(voices, call_logger),  # EN<->ES voice + prompt switching
            context_aggregator.user(),
            llm,
            tts,  # ServiceSwitcher over all department/language voices
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
        global_functions=GLOBAL_FUNCTIONS,  # route_to_human + end_call everywhere
    )
    flow_manager.state["voices"] = voices
    flow_manager.state["log"] = call_logger

    # Set up the router. It waits for the caller to speak, then routes.
    await flow_manager.initialize(create_router_node())

    # Deterministic spoken greeting with the AI + recording disclosure. Queued
    # here, it buffers and plays the moment the pipeline starts.
    await worker.queue_frames([TTSSpeakFrame(OPENING_GREETING)])

    runner = PipelineRunner(handle_sigint=True)
    try:
        await runner.run(worker)
    finally:
        call_logger.dump_transcript(context)
        call_logger.end()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye.")
