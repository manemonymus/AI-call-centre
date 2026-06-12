"""
Production phone server for the Hearthstone AI Call Center.

Callers dial your Twilio number → Twilio calls /webhook → your server returns
TwiML → Twilio streams audio to /ws → this bot handles the call.

The same pipeline as bot.py runs (see flows.py / services.py), just with a
phone transport instead of a microphone: per-department voices, English/
Spanish auto-detection, real SQLite bookings, escalation tickets, RAG, and
per-call transcripts in call_logs/.

Prerequisites
─────────────
1. Twilio account + phone number (twilio.com, ~$1.15/month for the number)
2. ngrok (ngrok.com) to expose this server while developing:
       ngrok http 8000
   (ngrok is for development only — use real hosting for production.)
3. In your Twilio console, set the phone number's "A Call Comes In" webhook to:
       https://<your-ngrok-id>.ngrok.io/webhook   (POST)
4. Environment variables (export them in the shell that runs this server —
   note there is no .env loader here):
       PUBLIC_HOSTNAME=<your-ngrok-id>.ngrok.io   (no https://)
       TWILIO_ACCOUNT_SID=ACxxxxxxxxx             (optional; without them the bot
       TWILIO_AUTH_TOKEN=xxxxxxxxx                 can't hang up — caller must)
       OLLAMA_MODEL=qwen2.5                       (or llama3.2, llama3.1, etc.)
   Optional provider switches (see services.py): LLM_PROVIDER, STT_PROVIDER,
   WHISPER_MODEL, VOICE_<DEPT>_<LANG>, ENABLE_RAG.

Run
───
    python server.py

Note on concurrency: each call builds its own pipeline (Whisper + all Piper
voices load per call, ~1 GB RAM each). The free local stack is for one or
two simultaneous calls — switch STT/LLM/TTS to hosted providers before real
traffic (see PLAN.md).
"""

import asyncio
import os

import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
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
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat_flows import FlowManager

from calllog import CallLogger
from flows import GLOBAL_FUNCTIONS, OPENING_GREETING, create_router_node
from services import LanguageRouter, VoiceDirectory, build_llm, build_stt, build_tts

PUBLIC_HOSTNAME = os.getenv("PUBLIC_HOSTNAME", "YOUR_NGROK_ID.ngrok.io")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
PORT = int(os.getenv("PORT", "8000"))


# ---------------------------------------------------------------------------
# Phone bot — runs once per incoming call.
# ---------------------------------------------------------------------------
async def phone_bot(transport: FastAPIWebsocketTransport, call_logger: CallLogger) -> None:
    stt = build_stt()
    llm = build_llm()
    voices = VoiceDirectory()
    tts = build_tts(voices)

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            VADProcessor(vad_analyzer=SileroVADAnalyzer()),
            stt,
            LanguageRouter(voices, call_logger),
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
        global_functions=GLOBAL_FUNCTIONS,
    )
    flow_manager.state["voices"] = voices
    flow_manager.state["log"] = call_logger

    await flow_manager.initialize(create_router_node())

    # Deterministic spoken greeting with the AI + recording disclosure.
    await worker.queue_frames([TTSSpeakFrame(OPENING_GREETING)])

    # handle_sigint=False so concurrent calls don't fight over the signal handler.
    try:
        await PipelineRunner(handle_sigint=False).run(worker)
    finally:
        call_logger.dump_transcript(context)
        call_logger.end()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(title="Hearthstone AI Call Center")


@app.post("/webhook")
async def twilio_webhook(request: Request) -> Response:
    """Twilio calls this POST endpoint when a call comes in.
    We return TwiML instructing Twilio to stream audio to our WebSocket.
    """
    hostname = PUBLIC_HOSTNAME.removeprefix("https://").removeprefix("http://")
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="wss://{hostname}/ws"/>'
        "</Connect>"
        "</Response>"
    )
    logger.info(f"Incoming call — streaming to wss://{hostname}/ws")
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Twilio Media Streams connects here. One call = one WebSocket connection."""
    await websocket.accept()
    call_logger = None
    try:
        transport_type, call_data = await parse_telephony_websocket(websocket)
        if transport_type != "twilio":
            logger.warning(f"Unexpected transport type: {transport_type}")
            await websocket.close()
            return

        stream_sid = call_data["stream_id"]
        call_sid = call_data.get("call_id")
        logger.info(f"Call started — stream_sid={stream_sid}")
        call_logger = CallLogger(channel="twilio")
        call_logger.event("twilio_call", stream_sid=stream_sid, call_sid=call_sid)

        # Auto hang-up (ending the call via Twilio's REST API on EndFrame)
        # requires the call SID and REST credentials; the serializer raises at
        # construction if it's enabled without them, so only enable it when we
        # have everything.
        can_hang_up = bool(call_sid and TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN)
        if not can_hang_up:
            logger.warning(
                "TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN not set — auto hang-up "
                "disabled; calls end when the caller hangs up."
            )
        serializer = TwilioFrameSerializer(
            stream_sid=stream_sid,
            call_sid=call_sid,
            account_sid=TWILIO_ACCOUNT_SID,
            auth_token=TWILIO_AUTH_TOKEN,
            params=TwilioFrameSerializer.InputParams(auto_hang_up=can_hang_up),
        )

        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                serializer=serializer,
                audio_in_enabled=True,
                audio_out_enabled=True,
            ),
        )

        await phone_bot(transport, call_logger)
        logger.info(f"Call ended — stream_sid={stream_sid}")

    except Exception as e:
        logger.exception(f"Error during call: {e}")
        if call_logger:
            call_logger.event("error", message=str(e))
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info(f"Starting server on port {PORT}. Point Twilio webhook to:")
    logger.info(f"  https://{PUBLIC_HOSTNAME}/webhook")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
