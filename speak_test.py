"""
Minimal audio-OUTPUT test — no LLM, no microphone.

Speaks one fixed sentence through Piper and your speakers, then exits.
  - If you HEAR it  -> audio output works; the only remaining issue is the
                       greeting logic, which I'll fix.
  - If you DON'T    -> it's an audio output / device / volume problem, and
                       that's what we'll chase next.

Run in your activated venv:   python speak_test.py
"""

import asyncio

from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineWorker
from pipecat.services.piper.tts import PiperTTSService
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

PIPER_VOICE = "en_US-lessac-medium"


async def main():
    transport = LocalAudioTransport(LocalAudioTransportParams(audio_out_enabled=True))
    tts = PiperTTSService(settings=PiperTTSService.Settings(voice=PIPER_VOICE))

    worker = PipelineWorker(Pipeline([tts, transport.output()]))

    # Queue the sentence, then end. These buffer until the pipeline starts,
    # so the audio plays once everything is live, then the script exits.
    await worker.queue_frames(
        [
            TTSSpeakFrame(
                "Hello! If you can hear this sentence, your audio output is "
                "working correctly."
            ),
            EndFrame(),
        ]
    )

    print(">>> You should hear a spoken sentence in a moment. <<<", flush=True)
    await PipelineRunner(handle_sigint=True).run(worker)
    print(">>> Done. Did you hear it? <<<", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
