# Local AI Call Center — Pipecat + Pipecat Flows

A fully **local** multi-agent voice demo for a home-services company. You talk
into your microphone; a virtual receptionist greets you, figures out what you
need, and hands you to the right specialist. Nothing leaves your machine — no
phone line, no cloud APIs, no keys

```
  Microphone / speakers   ->  LocalAudioTransport
  Speech-to-text          ->  Whisper (faster-whisper), local
  LLM brains              ->  a local model served by Ollama
  Text-to-speech          ->  Piper, local (voice auto-downloaded)
```

Four agents, built as Pipecat **Flow nodes**:

```
        ┌─────────┐
        │ Router  │  greets, identifies intent, transfers
        └────┬────┘
   ┌─────────┼─────────────┐
   ▼         ▼             ▼
Billing   Scheduling   Customer Service
```

A department "transfer" is just a node transition — the conversation and the
customer you've looked up carry across the handoff, so callers never repeat
themselves. The one tool, `lookup_customer`, reads `customers.csv` by phone
number.

---

## Prerequisites

**1. Python 3.10+**

**2. Two system libraries** (Pipecat's mic input and Piper's speech need them):

- macOS: `brew install portaudio espeak-ng`
- Debian/Ubuntu: `sudo apt-get install portaudio19-dev espeak-ng`

**3. Ollama**, with a tool-calling model pulled. Install it from
<https://ollama.com>, then:

```bash
ollama pull llama3.1     # or: qwen2.5, mistral-nemo — must support tools
```

Make sure Ollama is running (`ollama serve`, or just launch the app).

---

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python bot.py
```

The **first run downloads** the Whisper model and the Piper voice (a few
hundred MB total, one time). After that, the assistant greets you and you can
start talking. Press **Ctrl+C** to quit.

---

## Try it

When an agent asks for "the number on your account," say one of these — each
is wired to a sample record in `customers.csv`:

| Phone          | Customer     | Good for testing                              |
| -------------- | ------------ | --------------------------------------------- |
| `415-555-0142` | Maria Lopez  | billing / warranty follow-up                  |
| `415-555-0188` | James Chen   | a complaint (AC still not cooling) → service  |
| `415-555-0173` | Aisha Patel  | scheduling / maintenance-plan interest        |

Any **other** number exercises the new-customer path (no record found).

A sample call:

> **Assistant:** "Thanks for calling Hearthstone Home Services, how can I help?"
> **You:** "I think I was overcharged on my last invoice."
> **Assistant:** "No problem, let me get you over to billing." *(transfers)*
> **Assistant (billing):** "Happy to help — what's the best phone number on your account?"
> **You:** "Four one five, five five five, oh one four two."
> **Assistant:** "Thanks, Maria — let me pull that up…"

You can also ask to be moved ("actually, can I also reschedule a visit?") and
the billing agent will hand you to scheduling.

---

## Configuration

Knobs live at the top of `bot.py`:

- `COMPANY_NAME` — used throughout the agents' scripts.
- `OLLAMA_MODEL` — any tool-capable Ollama model. Bigger = better routing.
- `WHISPER_MODEL` — `Model.TINY/BASE/SMALL/MEDIUM/LARGE`. `BASE` is the default
  speed/accuracy balance; bump it up if transcription struggles.
- `PIPER_VOICE` — any Piper voice id (browse them at
  <https://github.com/rhasspy/piper>).

The agent personalities and routing rules are the `task_messages` /
`role_message` strings in the `create_*_node()` functions — edit those to
change behavior.

---

## Notes & caveats

- **Tool-calling reliability varies by model.** The routing and lookups depend
  on the local model issuing function calls. `llama3.1` and `qwen2.5` are
  solid; very small models may route or look up inconsistently.
- **Latency is yours to own.** Response time depends on your CPU/GPU; the first
  reply also includes model warm-up. A GPU helps a lot.
- **Scheduling is simulated.** There's no real calendar — the scheduling agent
  confirms times conversationally. Swap in a real calendar tool for production.
- **Versions are pinned** (`pipecat-ai==1.3.0`, `pipecat-ai-flows==1.2.0`)
  because Pipecat's API changes between releases; this demo was verified
  against exactly these.
- **Mic/speaker only.** There's no telephony here. Going to real inbound phone
  calls means replacing `LocalAudioTransport` with a telephony transport
  (e.g. Twilio/Telnyx over a websocket) and a phone number + SIP trunk, which
  carries per-minute cost.

## Troubleshooting

- **PortAudio / "no default output device"** → install `portaudio`, and check
  your OS mic permissions for the terminal.
- **espeak-ng error from Piper** → install `espeak-ng`.
- **"Connection refused" on `localhost:11434`** → Ollama isn't running, or the
  model isn't pulled. Run `ollama serve` and `ollama pull llama3.1`.
- **Agent won't transfer or look up** → switch to a tool-capable model; try
  `qwen2.5`.
- **Garbled transcription** → raise `WHISPER_MODEL` to `Model.SMALL` or
  `Model.MEDIUM`, and reduce background noise.
