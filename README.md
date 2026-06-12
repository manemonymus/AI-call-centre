# AI Call Center — Pipecat + Pipecat Flows

A multi-agent AI call center for a home-services company that runs **entirely
free on your machine** — no cloud APIs, no keys — and is wired so each
component can be flipped to a production cloud service with an environment
variable (see `PLAN.md` for the full production roadmap).

```
  Microphone / speakers   ->  LocalAudioTransport   (bot.py)
  Real phone calls        ->  Twilio Media Streams  (server.py)
  Speech-to-text          ->  Whisper (faster-whisper), local —
                              auto-detects English vs Spanish per utterance
  LLM brains              ->  a local model served by Ollama
  Text-to-speech          ->  Piper, local — a DIFFERENT voice per
                              department, with Spanish voices when the
                              caller speaks Spanish
```

Five agents, built as Pipecat **Flow nodes** (shared in `flows.py`):

```
        ┌─────────┐
        │ Router  │  greets (with AI + recording disclosure), routes
        └────┬────┘
   ┌─────────┼─────────────┬──────────────┐
   ▼         ▼             ▼              ▼
Billing   Scheduling   Customer Svc   Escalation (human callback tickets)
 (Ryan)     (Amy)      (HFC female)    
```

A department "transfer" is a node transition **plus a voice switch** — the
conversation and the customer you've looked up carry across the handoff, so
callers never repeat themselves. Speak Spanish at any point and the bot
switches its voice and replies in Spanish (and replays the legal disclosure
in Spanish, once).

## What's real (not simulated)

- **Bookings** hit a local SQLite calendar (`booking.py`): availability is
  computed from business hours minus booked slots, double-booking is
  impossible, cancellations free the slot.
- **Escalations** create tickets in the same database for a human to work.
- **Every call** writes a JSONL event log + full transcript to `call_logs/`.
- **RAG**: `python ingest.py` loads `home_services_faq.csv` into ChromaDB;
  agents ground policy/pricing answers in it and are instructed to never
  invent prices or promise refunds.
- **Compliance greeting**: every call opens with the AI + recording
  disclosure (California B.O.T. Act / Utah AIPA safe harbors, all-party
  recording consent), and the bot truthfully answers "are you a robot?".

---

## Prerequisites

**1. Python 3.10+**

**2. Two system libraries** (Pipecat's mic input and Piper's speech need them):

- macOS: `brew install portaudio espeak-ng`
- Debian/Ubuntu: `sudo apt-get install portaudio19-dev espeak-ng`
- Windows: included with the pip packages — nothing to install.

**3. Ollama**, with a tool-calling model pulled. Install it from
<https://ollama.com>, then:

```bash
ollama pull qwen2.5      # or: llama3.1, mistral-nemo — must support tools
ollama pull nomic-embed-text   # for the RAG knowledge base
```

Make sure Ollama is running (`ollama serve`, or just launch the app).

## Setup

```bash
pip install -r requirements.txt
python ingest.py          # optional: load the FAQ knowledge base
```

## Run (microphone demo)

```bash
python bot.py
```

The **first run downloads** the Whisper model and six Piper voices (~400 MB
total, one time). After that, the assistant greets you and you can start
talking — in English or Spanish. Press **Ctrl+C** to quit.

## Run (real phone number via Twilio)

```bash
ngrok http 8000                          # dev only — see PLAN.md for hosting
PUBLIC_HOSTNAME=<id>.ngrok.io python server.py
```

Point your Twilio number's "A Call Comes In" webhook at
`https://<id>.ngrok.io/webhook`. Twilio costs ~$1.15/mo for the number plus
~$0.0125/min inbound; everything else stays free and local.

---

## Try it

When an agent asks for "the number on your account," say one of these — each
is wired to a sample record in `customers.csv`:

| Phone          | Customer     | Good for testing                              |
| -------------- | ------------ | --------------------------------------------- |
| `415-555-0142` | Maria Lopez  | billing / warranty follow-up                  |
| `415-555-0188` | James Chen   | a complaint (AC still not cooling) → service  |
| `415-555-0173` | Aisha Patel  | scheduling / maintenance-plan interest        |

Things to exercise:

- **Booking**: "I need a technician for my AC" → scheduling offers *real*
  open slots, reads the booking back, and writes it to `callcenter.db`.
- **Spanish**: say "Hola, necesito ayuda con mi factura" — the bot switches
  voice and language mid-call.
- **Escalation**: "I want to talk to a human" → the bot takes your details
  and files a callback ticket (check the `tickets` table).
- **Honesty**: "Are you a robot?" → truthful yes, as required by law in
  several states.
- **Hang up politely**: "That's all, thanks" → the bot says goodbye and ends
  the call.

---

## Configuration (all via environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `COMPANY_NAME` | Hearthstone Home Services | Used throughout the scripts |
| `LLM_PROVIDER` | `ollama` | `ollama` \| `openai` \| `anthropic` \| `google` |
| `OLLAMA_MODEL` | `qwen2.5` | Any tool-capable Ollama model |
| `STT_PROVIDER` | `whisper` | `whisper` (local) \| `deepgram` (cloud, multilingual) |
| `WHISPER_MODEL` | `small` | tiny/base/small/medium — bigger = more accurate |
| `LANG_CONFIDENCE` | `0.7` | Min confidence before a language switch |
| `VOICE_<DEPT>_<LANG>` | see `services.py` | e.g. `VOICE_BILLING_EN=en_US-joe-medium` |
| `ENABLE_RAG` | `true` | Knowledge-base lookups on/off |
| `CALLCENTER_DB` | `callcenter.db` | Bookings + tickets SQLite file |
| `CALL_LOG_DIR` | `call_logs/` | Per-call JSONL logs |

Agent personalities and routing rules are the `role_message` /
`task_messages` strings in `flows.py` — edit those to change behavior.

### Peeking at the data

```bash
python -c "import booking; print(booking.available_slots())"
sqlite3 callcenter.db "SELECT * FROM appointments; SELECT * FROM tickets;"
```

---

## Notes & caveats

- **Tool-calling reliability varies by model.** Routing, lookups, and
  bookings depend on the local model issuing function calls. `qwen2.5` and
  `llama3.1` are solid; very small models will misroute or skip read-backs.
- **Latency is yours to own.** A local CPU stack runs seconds per turn, not
  the <1s of production voice agents. That's the main thing the paid cloud
  swap buys (see `PLAN.md` §2).
- **Spanish voice quality**: Piper's community Spanish voices are serviceable
  but noticeably below its best English voices — and far below commercial
  TTS. Don't judge the multilingual design by Piper's Spanish.
- **Language detection needs a real sentence.** Switching triggers on a
  confident detection of 10+ characters, so "sí" alone won't flip it —
  that's deliberate (prevents language flapping).
- **One or two calls at a time.** Each call loads its own Whisper + six Piper
  models (~1 GB RAM). Real concurrency means hosted STT/TTS/LLM.
- **Versions are pinned** (`pipecat-ai==1.3.0`, `pipecat-ai-flows==1.2.0`)
  because Pipecat's API moves between releases; verified against exactly
  these.

## Troubleshooting

- **PortAudio / "no default output device"** → install `portaudio`, and check
  your OS mic permissions for the terminal.
- **espeak-ng error from Piper** → install `espeak-ng`.
- **"Connection refused" on `localhost:11434`** → Ollama isn't running, or the
  model isn't pulled. Run `ollama serve` and `ollama pull qwen2.5`.
- **Agent won't transfer or look up** → switch to a tool-capable model; try
  `qwen2.5`.
- **Garbled transcription** → set `WHISPER_MODEL=medium`, reduce background
  noise.
- **Bot answers Spanish in English** → speak a full sentence; check the call
  log in `call_logs/` for `language_switch` events and detection confidence.
