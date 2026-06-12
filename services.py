"""
Service construction for the call center: multi-voice TTS, language-detecting
STT, and env-switchable providers.

Everything defaults to the free local stack (Whisper + Ollama + Piper). Each
component can be flipped to a hosted provider later with environment
variables, without touching the flow logic:

    LLM_PROVIDER = ollama (default) | openai | anthropic | google
    STT_PROVIDER = whisper (default) | deepgram
    TTS_PROVIDER = piper (default)            # cloud TTS: see README

Multi-voice design (free): Piper can't change voice after construction, so we
load one PiperTTSService per distinct voice and wrap them in Pipecat's
ServiceSwitcher. Department transfers and language switches push a
ManuallySwitchServiceFrame to activate the right voice.

Language design (free): the stock WhisperSTTService forces English and throws
away faster-whisper's detected language. DetectingWhisperSTT transcribes with
auto-detection and tags each TranscriptionFrame with the detected language
when confidence is high; LanguageRouter watches those tags and switches the
TTS voice + instructs the LLM when the caller changes language.
"""

from __future__ import annotations

import asyncio
import os

import numpy as np
from loguru import logger

# Load API keys etc. from a .env file in the project directory, if present.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMMessagesAppendFrame,
    ManuallySwitchServiceFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.service_switcher import (
    ServiceSwitcher,
    ServiceSwitcherStrategyManual,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.settings import assert_given
from pipecat.services.whisper.stt import Model, WhisperSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from calllog import CallLogger

# ---------------------------------------------------------------------------
# Voice casting: (department, language) -> Piper voice id.
# Override any entry with e.g. VOICE_BILLING_EN=en_US-joe-medium.
# Voices auto-download (~60 MB each) on first run.
# ---------------------------------------------------------------------------
DEPARTMENTS = ("router", "billing", "scheduling", "customer_service")
LANGUAGES = ("en", "es")

# Cast for maximum audible contrast on the common transfer paths: the router
# is male, so billing and customer care are female; scheduling (male) still
# contrasts with the router's timbre. Adjacent departments alternate gender.
DEFAULT_VOICES: dict[tuple[str, str], str] = {
    ("router", "en"): "en_US-lessac-medium",        # male, neutral
    ("billing", "en"): "en_US-hfc_female-medium",   # female, warm
    ("scheduling", "en"): "en_US-ryan-medium",      # male, deeper
    ("customer_service", "en"): "en_US-amy-medium", # female, brighter
    ("router", "es"): "es_ES-davefx-medium",        # male
    ("billing", "es"): "es_MX-claude-high",         # female
    ("scheduling", "es"): "es_ES-davefx-medium",    # male
    ("customer_service", "es"): "es_MX-claude-high",# female
}


def _voice_map() -> dict[tuple[str, str], str]:
    voices = {}
    for (dept, lang), default in DEFAULT_VOICES.items():
        voices[(dept, lang)] = os.getenv(f"VOICE_{dept.upper()}_{lang.upper()}", default)
    return voices


class VoiceDirectory:
    """Tracks the active department + language and owns one TTS service per voice."""

    def __init__(self):
        self.voice_map = _voice_map()
        self.department = "router"
        self.language = "en"
        self._services: dict[str, PiperTTSService] = {}

        # The router's English voice must be first: ServiceSwitcher activates
        # the first service in the list, and every call starts there.
        ordered_keys = [("router", "en")] + [
            k for k in self.voice_map if k != ("router", "en")
        ]
        for key in ordered_keys:
            voice_id = self.voice_map[key]
            if voice_id not in self._services:
                logger.info(f"Loading Piper voice '{voice_id}' for {key}")
                self._services[voice_id] = PiperTTSService(
                    settings=PiperTTSService.Settings(voice=voice_id)
                )

    @property
    def services(self) -> list[PiperTTSService]:
        return list(self._services.values())

    def service_for(self, department: str, language: str) -> PiperTTSService:
        voice_id = self.voice_map.get(
            (department, language), self.voice_map[("router", "en")]
        )
        return self._services[voice_id]

    def current_service(self) -> PiperTTSService:
        return self.service_for(self.department, self.language)

    def set_department(self, department: str) -> PiperTTSService:
        if department in DEPARTMENTS:
            self.department = department
        return self.current_service()

    def set_language(self, language: str) -> PiperTTSService:
        if language in LANGUAGES:
            self.language = language
        return self.current_service()


def build_tts(voices: VoiceDirectory) -> ServiceSwitcher:
    """One switcher over all loaded voices; flows push switch frames into it."""
    return ServiceSwitcher(
        services=voices.services, strategy_type=ServiceSwitcherStrategyManual
    )


# ---------------------------------------------------------------------------
# STT with real language detection.
# ---------------------------------------------------------------------------
_WHISPER_TO_PIPECAT = {"en": Language.EN, "es": Language.ES}


class DetectingWhisperSTT(WhisperSTTService):
    """WhisperSTTService that auto-detects language and tags transcriptions.

    The stock service passes a fixed language to faster-whisper and discards
    the TranscriptionInfo containing the detected language. We transcribe
    with language=None (auto-detect) and attach the detected language to the
    TranscriptionFrame when (a) it's one we support and (b) detection
    confidence clears a threshold — short utterances like "ok" are too easy
    to misdetect.
    """

    def __init__(self, *, min_language_confidence: float = 0.7, **kwargs):
        kwargs.setdefault(
            "settings",
            WhisperSTTService.Settings(
                model=Model.SMALL.value, language=None, no_speech_prob=0.4
            ),
        )
        super().__init__(**kwargs)
        self._min_language_confidence = min_language_confidence

    async def run_stt(self, audio: bytes):
        if not self._model:
            yield ErrorFrame("Whisper model not available")
            return

        await self.start_processing_metrics()

        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        language = assert_given(self._settings.language)

        def _transcribe():
            segments, info = self._model.transcribe(audio_float, language=language)
            return list(segments), info  # consume the generator off the event loop

        segments, info = await asyncio.to_thread(_transcribe)

        no_speech_prob_threshold = assert_given(self._settings.no_speech_prob)
        text = ""
        for segment in segments:
            if (
                no_speech_prob_threshold is not None
                and segment.no_speech_prob < no_speech_prob_threshold
            ):
                text += f"{segment.text} "

        await self.stop_processing_metrics()

        detected: Language | None = None
        detected_code = getattr(info, "language", None)
        confidence = getattr(info, "language_probability", 0.0) or 0.0
        if detected_code in _WHISPER_TO_PIPECAT and confidence >= self._min_language_confidence:
            detected = _WHISPER_TO_PIPECAT[detected_code]

        if text:
            await self._handle_transcription(text, True, detected)
            logger.debug(
                f"Transcription [{detected_code} p={confidence:.2f}]: [{text}]"
            )
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), detected)


# ---------------------------------------------------------------------------
# Language router: reacts to detected-language changes mid-call.
# ---------------------------------------------------------------------------
LANGUAGE_NOTES = {
    "es": (
        "NOTICE: The caller is now speaking Spanish. From this point on, respond "
        "ONLY in natural, conversational Spanish. Keep the same persona and rules."
    ),
    "en": (
        "NOTICE: The caller is now speaking English. From this point on, respond "
        "ONLY in English. Keep the same persona and rules."
    ),
}

# Spoken once, the first time a caller switches to Spanish: the legally
# meaningful AI/recording disclosure, in the caller's language.
DISCLOSURE_ES = (
    "Un momento. Le atiende un asistente automatizado de inteligencia artificial, "
    "y esta llamada puede ser grabada y transcrita por sistemas automatizados."
)

# Switching on a two-word utterance ("sí", "ok") is how calls flap between
# languages; require a minimum amount of evidence.
MIN_SWITCH_CHARS = 10


class LanguageRouter(FrameProcessor):
    """Watches TranscriptionFrames; on a confident language change, switches the
    active TTS voice and instructs the LLM to answer in the new language."""

    def __init__(self, voices: VoiceDirectory, call_logger: CallLogger | None = None, **kwargs):
        super().__init__(**kwargs)
        self._voices = voices
        self._log = call_logger
        self._spoke_es_disclosure = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            lang = None
            if frame.language == Language.ES:
                lang = "es"
            elif frame.language == Language.EN:
                lang = "en"

            if self._log:
                self._log.user_line(frame.text, language=lang)

            if (
                lang
                and lang != self._voices.language
                and len(frame.text.strip()) >= MIN_SWITCH_CHARS
            ):
                service = self._voices.set_language(lang)
                logger.info(f"Caller language switched to '{lang}'")
                if self._log:
                    self._log.event("language_switch", language=lang)

                await self.push_frame(ManuallySwitchServiceFrame(service=service))
                if lang == "es" and not self._spoke_es_disclosure:
                    self._spoke_es_disclosure = True
                    await self.push_frame(TTSSpeakFrame(DISCLOSURE_ES))
                await self.push_frame(
                    LLMMessagesAppendFrame(
                        messages=[{"role": "system", "content": LANGUAGE_NOTES[lang]}]
                    )
                )

        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Env-switchable factories for STT and LLM.
# ---------------------------------------------------------------------------
def build_stt():
    provider = os.getenv("STT_PROVIDER", "whisper").lower()
    if provider == "whisper":
        model = os.getenv("WHISPER_MODEL", Model.SMALL.value)
        return DetectingWhisperSTT(
            settings=WhisperSTTService.Settings(
                model=model, language=None, no_speech_prob=0.4
            ),
            device=os.getenv("WHISPER_DEVICE", "cpu"),
            compute_type=os.getenv("WHISPER_COMPUTE", "int8"),
            min_language_confidence=float(os.getenv("LANG_CONFIDENCE", "0.7")),
        )
    if provider == "deepgram":
        try:
            from pipecat.services.deepgram.stt import DeepgramSTTService
        except ImportError as e:
            raise ImportError(
                "Deepgram STT needs: pip install 'pipecat-ai[deepgram]' and DEEPGRAM_API_KEY."
            ) from e
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError(
                "STT_PROVIDER=deepgram but DEEPGRAM_API_KEY is not set. "
                "Get a key (with free credit) at console.deepgram.com and put it in .env"
            )
        # language="multi" enables code-switching across 10 languages (incl.
        # Spanish) and tags each TranscriptionFrame with the detected language
        # — LanguageRouter consumes those tags exactly like local Whisper's.
        # mip_opt_out keeps caller audio out of Deepgram's training data
        # (privacy-first default; opting out forfeits their discounted rates).
        return DeepgramSTTService(
            api_key=api_key,
            mip_opt_out=os.getenv("DEEPGRAM_MIP_OPT_OUT", "true").lower() == "true",
            settings=DeepgramSTTService.Settings(
                model=os.getenv("DEEPGRAM_MODEL", "nova-3-general"),
                language="multi",
                smart_format=True,
            ),
        )
    raise ValueError(f"Unknown STT_PROVIDER '{provider}'")


def build_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "ollama":
        from pipecat.services.ollama.llm import OLLamaLLMService

        return OLLamaLLMService(
            settings=OLLamaLLMService.Settings(model=os.getenv("OLLAMA_MODEL", "qwen2.5")),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        )
    if provider == "openai":
        from pipecat.services.openai.llm import OpenAILLMService

        return OpenAILLMService(
            settings=OpenAILLMService.Settings(model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini")),
        )
    if provider == "anthropic":
        try:
            from pipecat.services.anthropic.llm import AnthropicLLMService
        except ImportError as e:
            raise ImportError(
                "Anthropic LLM needs: pip install 'pipecat-ai[anthropic]' and ANTHROPIC_API_KEY."
            ) from e
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set. "
                "Get a key at console.anthropic.com and put it in .env"
            )
        return AnthropicLLMService(
            api_key=api_key,
            settings=AnthropicLLMService.Settings(
                model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
            ),
        )
    if provider == "google":
        try:
            from pipecat.services.google.llm import GoogleLLMService
        except ImportError as e:
            raise ImportError(
                "Google LLM needs: pip install 'pipecat-ai[google]' and GOOGLE_API_KEY."
            ) from e
        return GoogleLLMService(
            api_key=os.environ["GOOGLE_API_KEY"],
            settings=GoogleLLMService.Settings(
                model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
            ),
        )
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'")
