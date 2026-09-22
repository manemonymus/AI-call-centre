# AI HR Helpline — Pipecat + Pipecat Flows

A multi-agent AI **HR help line** for employees that runs **entirely free on
your machine** — no cloud APIs, no keys — with a **pluggable architecture** where each component (speech-to-text, LLM, text-to-speech) can be swapped between free local models (Whisper, Ollama, Piper) and production cloud services (Deepgram, Claude, Cartesia) via environment variables. See `PLAN.md` for the full production roadmap and cost breakdown.

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

Employees call in; a receptionist routes them to one of four HR teams, each
with its **own AI voice** and its **own knowledge base** (per-department RAG):

```
            ┌────────────┐
            │ Reception  │  greets (AI + recording disclosure), routes
            └─────┬──────┘
   ┌──────────────┼───────────────┬───────────────┐
   ▼              ▼               ▼               ▼
Leave &        Conduct &      Compliance &    HR Policy &
Time-Off       Ethics         Reporting       General
(hr_leave KB)  (hr_conduct)   (hr_compliance) (hr_general)
   └──────────────┴───────────────┴───────────────┘
                     + Escalation (human HR callback ticket)
```

Each department answers **only from its own knowledge base** — the conduct team
physically can't read the leave team's documents. A "transfer" is a node
transition **plus a voice switch**; the conversation and the employee you've
looked up carry across, so callers never repeat themselves. Speak Spanish at
any point and the bot switches its voice and replies in Spanish (replaying the
legal disclosure in Spanish, once).

## What's real (not simulated)

- **Per-department RAG over real HR data**: `python fetch_hr_data.py` pulls a
  real HR-policy Q&A dataset (`strova-ai/hr-policies-qa-dataset`, ~640 pairs)
  from Hugging Face — no login needed — and an LLM sorts each Q&A into the
  right department. `python ingest.py` loads each department's share into its
  own ChromaDB collection (`hr_leave`, `hr_conduct`, `hr_compliance`,
  `hr_general`). Agents answer only from their department's collection and are
  told never to invent a policy, number, or deadline.
- **Employee lookup** (`employees.csv`) by employee ID or phone, so agents
  greet by name and don't re-ask once identity is known.
- **Escalations** create real HR callback tickets in SQLite (`booking.py`).
- **Every call** writes a JSONL event log + full transcript to `call_logs/`.
- **Compliance greeting**: every call opens with the AI + recording
  disclosure (California B.O.T. Act / Utah AIPA safe harbors, all-party
  recording consent), and the bot truthfully answers "are you a robot?".

---

## Architecture: Cascaded Pipeline (not speech-to-speech)

This system uses a **cascaded STT → LLM → TTS pipeline** rather than a
speech-to-speech model. Why?

- **Per-department voices mid-call**: Speech-to-speech APIs (OpenAI Realtime,
  Gemini Live) lock the voice at session start and can't change it mid-call.
  Here, each department has its own voice identity, and transfers flip the voice
  instantly.
- **Deterministic routing & RAG**: The LLM is a deterministic agent (Claude with
  function calls, or Ollama with tools), not a `<generalist audio model>`. This
  keeps your control over what knowledge base is searched, what transfers are
  allowed, and what fallbacks fire.
- **Cost**: Cascaded pipeline at ~$0.05/min (production stack) vs. speech-to-speech at $0.18–0.46/min (OpenAI Realtime is 10–50x pricier).

**Local dev stack** (free) runs on your machine:

```
Whisper (faster-whisper)  ──>  Ollama (qwen2.5)  ──>  Piper (local voices)
     ~5-10s latency              ~5-10s per turn         ~200ms
```

**Production stack** (low-cost cloud) has <1.5s end-to-end latency:

```
Deepgram (streaming)  ──>  Claude Haiku 4.5  ──>  Cartesia Sonic 3.5
  ~600ms                    ~600–900ms              ~100ms
```

Both stacks share the same Python agent code, routing logic, and RAG
(`services.py` abstracts the provider swap).

---

## Tech stack

Python voice-AI app built on **Pipecat**, with every heavy component
swappable between a free local model and a hosted cloud service via an
environment variable.

| Layer | Free (local) | Production (cloud) | Shared |
|---|---|---|---|
| Voice pipeline | — | — | **Pipecat** 1.3.0 + **Pipecat Flows** 1.2.0 |
| Speech-to-text | **faster-whisper** (CPU, ~5s) | **Deepgram Nova-3** (~600ms) | swappable via `STT_PROVIDER` |
| LLM | **Ollama qwen2.5** (~5s/turn) | **Claude Haiku 4.5** (~600ms) | swappable via `LLM_PROVIDER`; tool-calling routing |
| Text-to-speech | **Piper** (local, ~200ms) | **Cartesia Sonic 3.5** (~100ms) | per-department voices via `ServiceSwitcher` |
| RAG | **ChromaDB** (one collection per dept) + **Ollama `nomic-embed-text`** |  | same across both stacks |
| Knowledge data | real HR Q&A from **Hugging Face**, LLM-classified into 4 departments |  | same across both stacks |
| Telephony | — | **Twilio** Media Streams | **FastAPI**/**Uvicorn** (`server.py`); **ngrok** for dev tunneling |
| Turn-taking | **Silero VAD** |  | shared |
| Storage | **SQLite** + **JSONL** transcripts |  | shared |
| Config | **python-dotenv** |  | per-stack secrets in `.env` |
| Logging | **loguru** |  | per-call event logs in `call_logs/` |

**Data flow of a single call:**

```
 Caller ──phone──> Twilio Media Streams ──WS──> FastAPI (server.py)
                                                     │
                                                     ▼
   Silero VAD ─> STT (Deepgram / Whisper) ─> Pipecat Flows agent
                                                     │
                       ┌─────────────────────────────┼───────────────┐
                       ▼                             ▼               ▼
              LanguageRouter (EN/ES)      per-department RAG    LLM (Claude / Ollama)
              switches voice + prompt      (ChromaDB lookup)     grounded answer
                       │                                             │
                       └──────────────> TTS (Piper) ─> audio ────────┘
                                                     │
                          SQLite tickets  +  JSONL call log  (side effects)
```

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
python fetch_hr_data.py   # download + classify the real HR dataset -> hr_faq.csv
python ingest.py          # load it into per-department ChromaDB collections
```

`fetch_hr_data.py` uses whatever LLM is available to classify each Q&A into a
department (Claude if `ANTHROPIC_API_KEY` is set, else local Ollama, else a
keyword fallback). `ingest.py` needs Ollama running with `nomic-embed-text` for
embeddings (free, local) regardless of which chat model you use.

## Run (microphone demo)

```bash
python bot.py
```

The **first run downloads** the Whisper model and six Piper voices (~400 MB
total, one time). After that, the assistant greets you and you can start
talking — in English or Spanish. Press **Ctrl+C** to quit.

## Upgrade the ears + brain (Deepgram + Claude)

The free local stack trades accuracy and speed for $0. When Whisper mishears
you or Ollama takes 10+ seconds to answer, switch to the hosted stack — same
code, two API keys:

```bash
cp .env.example .env     # then edit .env:
#   STT_PROVIDER=deepgram   + DEEPGRAM_API_KEY   (console.deepgram.com — free credit on signup)
#   LLM_PROVIDER=anthropic  + ANTHROPIC_API_KEY  (console.anthropic.com)
python bot.py
```

What it buys: phone-grade streaming transcription with proper Spanish
code-switching (Deepgram Nova-3 multilingual, ~$0.006/min), ~1s responses and
far more reliable routing/RAG tool calls (Claude Haiku 4.5,
~$0.005–0.015/min). Caller audio is kept out of Deepgram's training data by
default (`DEEPGRAM_MIP_OPT_OUT=true`). TTS stays on free local Piper either
way; mix and match providers freely.

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

When an agent asks for your employee ID or phone, use one of these sample
records in `employees.csv` (so it greets you by name):

| Employee ID | Phone          | Name         |
| ----------- | -------------- | ------------ |
| `E1042`     | `415-555-0142` | Maria Lopez  |
| `E1088`     | `415-555-0188` | James Chen   |
| `E1073`     | `415-555-0173` | Aisha Patel  |

Things to exercise (each routes to a different team + voice and answers from
that team's own knowledge base):

- **Leave**: "How long do I have to use my compensatory off?"
- **Conduct**: "Can I accept a gift from a supplier?"
- **Compliance**: "How do I report a policy violation?"
- **General**: "How often is the company policy reviewed?"
- **Per-department isolation**: ask the conduct team a leave question — it
  won't have a good answer, because it only searches its own knowledge base.
- **Memory across transfers**: give your ID once, then ask something for
  another team — it transfers without re-asking who you are.
- **Spanish**: "Hola, ¿cuántos días tengo para usar mi tiempo compensatorio?"
  — the bot switches voice and language mid-call.
- **Escalation**: "I need to talk to a real person about my manager" → it
  takes your details and files an HR callback ticket (`tickets` table).
- **Honesty**: "Are you a robot?" → truthful yes, as required by law in
  several states.
- **Hang up politely**: "That's all, thanks" → goodbye and the call ends.

---

## Configuration (all via environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `COMPANY_NAME` | Hearthstone | Used throughout the scripts (and to fix the dataset's placeholder company) |
| `LLM_PROVIDER` | `ollama` | `ollama` \| `openai` \| `anthropic` \| `google` |
| `OLLAMA_MODEL` | `qwen2.5` | Any tool-capable Ollama model |
| `STT_PROVIDER` | `whisper` | `whisper` (local) \| `deepgram` (cloud, multilingual) |
| `WHISPER_MODEL` | `small` | tiny/base/small/medium — bigger = more accurate |
| `LANG_CONFIDENCE` | `0.7` | Min confidence before a language switch |
| `VOICE_<DEPT>_<LANG>` | see `services.py` | dept = `ROUTER`/`LEAVE`/`CONDUCT`/`COMPLIANCE`/`GENERAL`, e.g. `VOICE_LEAVE_EN=en_US-amy-medium` |
| `ENABLE_RAG` | `true` | Per-department knowledge-base lookups on/off |
| `CALLCENTER_DB` | `callcenter.db` | HR callback-tickets SQLite file |
| `CALL_LOG_DIR` | `call_logs/` | Per-call JSONL logs |

Agent personalities and routing rules are the `role_message` /
`task_messages` strings in `flows.py` — edit those to change behavior. The
departments and their data live in `fetch_hr_data.py` (classification) and
`rag.py` (collections).

### Peeking at the data

```bash
sqlite3 callcenter.db "SELECT * FROM tickets;"          # HR callback tickets
python -c "import rag; [print(d, rag.get_collection(d).count()) for d in rag.DEPARTMENTS]"
```

---

## Notes & caveats

- **Tool-calling reliability varies by model.** Routing, employee lookups, and
  knowledge-base searches depend on the local model issuing function calls.
  `qwen2.5` and `llama3.1` are solid; very small models will misroute or forget
  to search the knowledge base (and then risk guessing — Claude is far better).
- **RAG is only as good as the data.** The shipped knowledge base is a real but
  conduct/policy-heavy public dataset, so the Leave/Conduct/Compliance/General
  split is uneven and answers reflect that dataset's company, not yours. Swap in
  your own `hr_faq.csv` (columns: question, answer, department) and re-ingest.
- **Latency is yours to own.** A local CPU stack runs seconds per turn, not
  the <1s of production voice agents. That's the main thing the paid cloud
  swap buys (see `PLAN.md` §2).
- **Spanish voice quality**: Piper's community Spanish voices are serviceable
  but noticeably below its best English voices — and far below commercial
  TTS. Don't judge the multilingual design by Piper's Spanish.
- **Language detection needs a real sentence.** Switching triggers on a
  confident detection of 10+ characters, so "sí" alone won't flip it —
  that's deliberate (prevents language flapping).
- **One or two calls at a time.** Each call loads its own Whisper + seven Piper
  voices (~1 GB RAM). Real concurrency means hosted STT/TTS/LLM.
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
- **Agent says it can't find anything** → the knowledge base is empty; run
  `python fetch_hr_data.py` then `python ingest.py`, and make sure Ollama has
  `nomic-embed-text` pulled (embeddings need it).
- **Garbled transcription** → set `WHISPER_MODEL=medium`, reduce background
  noise.
- **Bot answers Spanish in English** → speak a full sentence; check the call
  log in `call_logs/` for `language_switch` events and detection confidence.
