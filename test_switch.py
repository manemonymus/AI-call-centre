"""Runtime diagnosis: does the voice actually switch on a department transfer?

Runs the real pipeline (minus audio transports) and traces which Piper
instance synthesizes each utterance.
"""

import asyncio
import warnings

warnings.filterwarnings("ignore")

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat_flows import FlowManager

import flows
from calllog import CallLogger
from services import LanguageRouter, VoiceDirectory, build_llm, build_tts

SYNTH_LOG = []


def sniff_logs(message):
    text = message.record["message"]
    if "Generating TTS" in text:
        SYNTH_LOG.append(text)


async def main():
    logger.add(sniff_logs, level="DEBUG")

    voices = VoiceDirectory()
    tts = build_tts(voices)
    llm = build_llm()  # ollama; never invoked in this test
    log = CallLogger(channel="switch-test")

    context = LLMContext()
    ca = LLMContextAggregatorPair(context)
    pipeline = Pipeline(
        [
            VADProcessor(vad_analyzer=SileroVADAnalyzer()),
            LanguageRouter(voices, log),
            ca.user(),
            llm,
            tts,
            ca.assistant(),
        ]
    )
    worker = PipelineWorker(pipeline)
    fm = FlowManager(
        llm=llm, context_aggregator=ca, worker=worker,
        global_functions=flows.GLOBAL_FUNCTIONS,
    )
    fm.state["voices"] = voices
    fm.state["log"] = log
    await fm.initialize(flows.create_router_node())

    runner = PipelineRunner(handle_sigint=False)
    run_task = asyncio.create_task(runner.run(worker))

    # Voice 1: the router greeting
    await worker.queue_frames([TTSSpeakFrame("Hello from the router voice.")])
    await asyncio.sleep(4)

    # Simulate the billing transfer exactly as the LLM tool call would do it
    result, node = await flows.ROUTE_BILLING.handler({}, fm)
    await fm.set_node_from_config(node)
    await asyncio.sleep(6)

    # One more line — should still be the billing voice
    await worker.queue_frames([TTSSpeakFrame("Still the billing voice?")])
    await asyncio.sleep(4)

    await worker.queue_frames([EndFrame()])
    await asyncio.wait_for(run_task, timeout=15)

    print("\n=== SYNTHESIS TRACE ===")
    for line in SYNTH_LOG:
        print(" ", line[:120])
    router_svc = voices.service_for("router", "en")
    billing_svc = voices.service_for("billing", "en")
    print(f"\nrouter voice service:  {router_svc}")
    print(f"billing voice service: {billing_svc}")
    print(f"directory now points at: {voices.current_service()} (dept={voices.department})")


if __name__ == "__main__":
    asyncio.run(main())
