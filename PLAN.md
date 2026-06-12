# AI Call Center — Production Plan

*Researched and written June 2026. All prices verified against vendor pricing pages in June 2026 — re-check at decision time; several vendors changed prices in the last 12 months.*

> **Build status (June 12, 2026):** everything from this plan that's free has been built — per-department voices (Piper + ServiceSwitcher), English/Spanish auto-detection with mid-call voice/language switching, the compliance greeting, real SQLite bookings + escalation tickets, guardrail prompts, per-call transcripts, and env-switchable cloud providers (`services.py`). What remains is what costs money: hosted STT/LLM/TTS for production latency/quality, real hosting, and Twilio per-minute usage. See README.md.

## 1. Where you are today

The prototype already has the right skeleton — this matters, because the architecture decision is the one that's expensive to reverse:

| Already built | Status for production |
|---|---|
| Pipecat + Pipecat Flows multi-agent (Router → Billing / Scheduling / CS, context carries across transfers) | **Keep.** This is exactly the architecture production voice agents use. |
| Twilio number + Media Streams + FastAPI server (`server.py`) | **Keep.** Just replace ngrok with real hosting. |
| ChromaDB RAG + ingestion (`ingest.py`, `rag.py`) | **Keep**, improve content + grounding rules. |
| faster-whisper STT on CPU | **Replace** — batch model, no streaming, no reliable mid-call language detection, hallucinates on silence/hold music (~1% of transcriptions). |
| Ollama LLM on CPU | **Replace** (keep for dev) — peer-reviewed 2026 benchmark: Ollama hits a concurrency wall ~10 users with 54–122s first-token times and 13–30% timeout rates under load. CPU inference is 5–10s/turn vs. the ~800ms target. |
| Piper TTS, one English voice | **Replace** (keep for dev) — single voice, English-only voice; Spanish text through it produces garbage audio. Quality ceiling far below 2026 commercial TTS. |
| Fake bookings ("there is no real calendar") | **Replace** with a real calendar API before launch. |

**The one-sentence verdict:** the code you wrote survives; the three local models don't. Production voice agents need streaming cloud STT/TTS and a fast hosted LLM — the local stack structurally cannot hit conversational latency or handle concurrent calls.

## 2. Target architecture

**Cascaded pipeline (STT → LLM → TTS), not speech-to-speech.** This was verified, not assumed:

- OpenAI Realtime (speech-to-speech) **cannot change voice mid-session** once the assistant has spoken (documented `cannot_update_voice` error) — architecturally incompatible with "different AI voice per department." Gemini Live has the same session-level voice model.
- Speech-to-speech audio is 10–50x the LLM cost of cascaded ($0.18–0.46/min uncached for gpt-realtime-2 vs. ~$0.005–0.015/min for a small hosted LLM).
- Cascaded keeps your deterministic Flows routing, RAG injection, and guardrails.

### Component choices

| Component | Pick | Price | Why |
|---|---|---|---|
| Telephony | **Twilio** (keep) | $1.15/mo number; $0.0085/min inbound + $0.004/min Media Streams = **$0.0125/min** | Already integrated. Switch to Telnyx (~$0.0055/min) later if volume justifies the port. |
| Hosting | **Pipecat Cloud** (Daily) | **$0.01/min** active, no monthly minimum | Runs your exact Pipecat code. Solves the hardest infra problem (one process per call session) and bundles Krisp noise cancellation (free to 10k min/mo). Alternative: $7–25/mo VPS for the MVP phase only. |
| STT | **Deepgram Nova-3 Multilingual** (`language=multi`) | **$0.0058/min** (promo; ~$0.0092 list) | Streams 8kHz mulaw natively; returns **per-word language tags** — the exact signal to flip TTS voice + prompt language when a caller switches to Spanish. 10 languages incl. ES. Later: upgrade to **Flux Multilingual** ($0.0078/min, GA Apr 2026) which adds model-based turn detection (<400ms end-of-turn) in the same connection. Budget fallback: AssemblyAI Universal-Streaming Multilingual at $0.0025/min. ⚠️ Deepgram trains on your audio **by default** — set `mip_opt_out=true` (forfeits promo discount). |
| LLM | **Claude Haiku 4.5** for department agents | $1/$5 per MTok ≈ **$0.015/min** uncached, ~$0.005–0.007 with prompt caching | Strongest small-model tool-calling for routing/transfers (benchmarked on tau2-bench customer-service scenarios). TTFT ~0.6–0.9s. Optionally Gemini 2.5 Flash-Lite (~$0.0015/min, TTFT ~0.42s) for the simple router node. Avoid GPT-5.4-mini default settings (reasoning mode can balloon TTFT to tens of seconds — must pin reasoning to minimal). |
| TTS | **Cartesia Sonic 3.5** primary; ElevenLabs Flash v2.5 premium alternative | Cartesia ≈ **$0.009/call-min** ($39/mo Startup = 1.25M chars, unlimited voice slots); ElevenLabs ≈ $0.015/call-min ($0.05/1k chars) | Both Pipecat-native, stream 8kHz mulaw, sub-100ms claimed TTFB, 40+/32 languages. **Per-department voices = a `voice_id` per flow node — trivial, no extra cost with stock voices.** Audition each voice in BOTH English and Spanish (many "multilingual" voices carry an English accent into Spanish). |
| Turn detection | **Pipecat smart-turn v3** | Free (BSD-2) | 8M params, ~12ms CPU inference, multilingual incl. Spanish. Eliminates 200–400ms of fixed VAD wait. |
| Calendar | **Cal.com API** as function tools | Free tier | check-availability + create-booking tools; syncs Google/Outlook. Replaces the fake bookings. |
| Payments (if ever needed) | **Twilio `<Pay>`** | $0.15/transaction | DTMF masking keeps card numbers out of your STT/LLM/recordings entirely — never let the AI hear a card number (PCI-DSS). |

### Multilingual design (the part you asked about)

The LLM is the easy part — every candidate model speaks Spanish fluently; it's one system-prompt line ("always respond in the caller's language"). The real work:

1. **Detection**: Deepgram's per-word language tags tell you the caller's language in real time, including mid-call switches.
2. **Voice routing**: on language change, swap the TTS voice (or rely on multilingual voices that speak both) and the prompt language. Each department keeps its distinct voice identity per language.
3. **Cold start**: you don't know the language until the caller speaks — make the greeting bilingual ("Thanks for calling Hearthstone… Para español, simplemente hable en español").
4. **Compliance**: the AI/recording disclosure must be delivered in the caller's language to be effective (see §6).
5. **Fallback**: code-switching support covers ~10 languages (Deepgram). For unsupported languages, route to a human/voicemail rather than mis-transcribing silently.

## 3. Roadmap & timeline

Assumes one developer, near-full-time. Double the calendar time if part-time. Industry pattern (verified postmortem): "demo in an afternoon, production in ~5 months" — 2 months to MVP, 3 months hardening.

### Phase 0 — Cloud swap & real hosting (Weeks 1–2)
- Swap `WhisperSTTService` → `DeepgramSTTService` (language=multi), `OLLamaLLMService` → Anthropic/Google service, `PiperTTSService` → `CartesiaTTSService`. All are built-in Pipecat services — each swap is a few lines.
- Upgrade pipecat-ai beyond 1.3.0 (needed for Flux and newest services); re-test Flows transitions and the Twilio serializer.
- Deploy to Pipecat Cloud (or a $7–25/mo VPS for now); kill ngrok; point the Twilio webhook at the stable endpoint.
- Add the compliance greeting (see §6) — it's a one-line change; do it now.
- **Exit criteria:** real phone call, <1.5s response latency, English only, one voice.

### Phase 1 — Multi-voice + multilingual MVP (Weeks 3–5)
- Assign a distinct Cartesia/ElevenLabs voice per flow node (receptionist + 3 departments). Pre-warm TTS connections per voice so the handoff greeting is instant.
- Wire language detection → TTS voice + prompt-language switching; bilingual greeting.
- Add smart-turn v3 endpointing; verify barge-in correctly sends Twilio's `clear` message and tracks `mark` acks (otherwise transcripts include audio the caller never heard).
- Test with real Spanish and Spanglish callers — accents over 8kHz phone audio are where STT accuracy collapses (~96% → <80% on heavy accents).
- **Exit criteria:** caller speaks Spanish mid-call → agent answers in Spanish, correct department voice, no language flapping.

### Phase 2 — Real business logic & safety (Weeks 6–9)
- Cal.com integration: real availability + booking tools, schema-validate every tool call, **read back date/time/name and confirm before committing**.
- RAG hardening: ingest real company docs; prompt rule — never state a price/policy not retrieved from the knowledge base (Air Canada liability, §6).
- Human escalation: warm transfer via Twilio Conference keyed by CallSid (AI briefs the human, then drops); explicit triggers (caller asks for human, repeated low-confidence STT, repeated tool failures, anger signals). After-hours voicemail via TwiML `<Record>`.
- Customer data: move customers.csv → a real DB/CRM.
- Call recording ($0.0025/min + $0.0005/min storage) + transcript + per-turn latency + per-call cost logging.
- **Exit criteria:** a real booking lands on a real calendar; an angry caller reaches a human; nothing invented by the bot.

### Phase 3 — Hardening & scale (Months 3–6)
- Reconnect logic for vendor WebSockets (the #1 reported operational pain: STT sockets silently dying after 60–70s of silence; ~1-in-50 dropped connections reported).
- Load test at 2x expected peak (published failure mode: 800ms turns at 10 concurrent calls → 3s at 500).
- Prompt-cache optimization (stable shared prefix; department deltas late in the prompt — node transitions otherwise invalidate the cache every transfer).
- Scripted regression test calls before each deploy; defer Coval/Hamming eval platforms (enterprise-priced) until volume justifies.
- Monitor: escalation rate (one team hit a useless 92%), per-call cost, P50/P95 latency, language-detection accuracy.
- Concurrency planning: TTS plan tiers are the hidden cap (Cartesia Startup = 5 concurrent; ElevenLabs Flash: Free 4 / Creator 10 / Pro 20). Hitting the cap = 429 errors = agent goes mute mid-call.

**Total: ~2 months to a credible production MVP, ~4–6 months to genuinely hardened.**

## 4. Costs

### Per call-minute (recommended DIY stack)

| Item | $/min |
|---|---|
| Twilio inbound + Media Streams | 0.0125 |
| Pipecat Cloud hosting | 0.0100 |
| Deepgram Nova-3 multilingual STT | 0.0058 (promo) |
| Claude Haiku 4.5 (cached) | ~0.0050–0.0150 |
| Cartesia TTS | ~0.0090 |
| Recording + storage | 0.0030 |
| **Total** | **≈ $0.045–0.06/min** (~$0.25–0.30 per 5-min call) |

### Monthly scenarios (DIY stack)

| Volume | Variable | Fixed (number + Cartesia Startup + misc) | Total |
|---|---|---|---|
| Dev/testing (~200 min) | ~$10 | ~$45 | **~$55/mo** |
| 1,000 min/mo (~7 calls/day) | ~$50 | ~$45 | **~$95/mo** |
| 10,000 min/mo (~70 calls/day) | ~$500 | ~$280 (Cartesia Scale for concurrency) | **~$780/mo** |

### Comparison: managed platforms (if you'd rather not own the code)

| Platform | Realistic all-in | Notes |
|---|---|---|
| ElevenLabs Agents | ~$0.09–0.12/min | Best feature match: native auto language detection w/ mid-call voice switch, per-agent voices, built-in RAG, Twilio transfer. $0.08/min + LLM at cost. Deepest lock-in. |
| Vapi | ~$0.10–0.20/min | $0.05/min platform + pass-throughs ($0 with own API keys). Squads map to your router→departments. |
| Retell | ~$0.10–0.17/min | 20 concurrent free; built-in knowledge bases. |
| Bland | $0.11–0.14/min | Only true all-in price; better rates need $299–499/mo; no BYO-LLM. |

DIY at ~$0.05/min is roughly half the cheapest managed option, and you keep zero lock-in — but you own latency tuning, reconnects, and on-call. If maintaining the code stops being fun, prototype ElevenLabs Agents on the $22/mo Creator tier before committing either way.

### Build-phase cost
Mostly your time. Cash outlay during dev: <$60/mo. No GPU needed — skip self-hosted inference entirely until you sustain >4,000–5,000 min/mo (the break-even vs. a ~$200/mo GPU server) or compliance forces data residency.

## 5. How realistic is this?

**Technically: very.** This product category matured fast — every component is now an off-the-shelf, Pipecat-native API, and your prototype already implements the hard architectural parts (multi-agent flows, telephony transport, RAG). A working multilingual multi-voice MVP in ~5–8 weeks of focused work is a reasonable expectation, not optimism.

**The honest caveats:**
1. **The local-models dream doesn't survive contact with production.** CPU Whisper+Ollama+Piper is 5–10s/turn vs. the <800ms target, caps at ~1–2 concurrent calls, and can't do mid-call language detection. Keep it as your free dev environment; budget ~5¢/min for production.
2. **The last 20% takes 80% of the time.** Accents, barge-in edge cases, hallucinated commitments, silent WebSocket drops, load behavior — published postmortems consistently show hardening takes longer than building (latency tuning from 10s → 2.5s, accuracy 50% → 80% over 3 months for one documented team).
3. **Industry median latency is 1.4–1.7s, not 800ms.** You'll likely launch closer to the median and tune down. That's acceptable for a home-services line; just don't promise "indistinguishable from human" on day one.
4. **Concurrency costs money in steps, not smoothly** — TTS plan tiers, Pipecat Cloud instances, Twilio channels. Size plans before marketing the number.

## 6. Legal & compliance (US-focused; not legal advice)

**Do from day one (free):**
- **First-utterance disclosure on 100% of calls**, localized per language: *"You've reached [Company]. I'm an automated AI assistant, and this call may be recorded and transcribed by automated systems. How can I help?"* One sentence satisfies: California B.O.T. Act safe harbor, Utah AI Policy Act safe harbor (fines up to $2,500/violation), all-party-consent recording states (~11–13 states; never geo-target — play it always), EU AI Act Article 50 (applies **Aug 2, 2026**), and the pending FCC AI-disclosure rule.
- Program the bot to truthfully answer "are you a robot?" — Utah makes this mandatory on request.
- **Stay inbound-only.** The moment the AI makes outbound calls (callbacks, reminders), FCC ruling 24-17 makes AI voices "artificial" under TCPA → prior express written consent required, $500–1,500 statutory damages per call, uncapped class exposure. Outbound is a separate future project with a consent database.

**Live litigation risk to design around:** CIPA third-party-wiretap class actions ($5,000/violation) are actively proceeding against AI voice vendors answering business lines (Ambriz v. Google, Feb 2025; ConverseNow/Domino's, Aug 2025). Mitigations: the upfront disclosure, vendor DPAs barring use of call data for the vendor's own benefit, and `mip_opt_out=true` on every Deepgram request (it trains on your callers' audio by default otherwise).

**Liability for what the bot says:** Moffatt v. Air Canada (2024) — the company is liable for its bot's promises; the "chatbot is a separate entity" defense failed. Architecture is the defense: RAG-grounded answers only, no prices/policies from parametric memory, human escalation for refunds/disputes, transcripts retained (e.g., 90 days) as evidence.

**Payments:** never let card numbers transit STT/LLM/recordings (instant PCI-DSS scope). Use Twilio `<Pay>` DTMF masking ($0.15/transaction); add an interrupt rule if a caller starts reading digits aloud.

## 7. Top risks summary

| Risk | Mitigation |
|---|---|
| Latency blows the conversational budget | Streaming everything; co-locate server with vendors; smart-turn v3; prompt caching; measure per-turn |
| STT fails on accents/noise over 8kHz audio | Telephony-tuned multilingual STT; Krisp (free via Pipecat Cloud); confirmation loops on low confidence; test with real callers |
| Bot invents prices/bookings | RAG-only answers; schema-validated tools; read-back confirmation; post-call reconciliation vs. backend |
| Concurrency caps hit mid-growth | Track concurrent peaks; upgrade TTS tier before marketing pushes |
| Silent vendor WebSocket failures | Reconnect logic + per-call health metrics from day one |
| Recording/AI-disclosure lawsuits | Mandatory localized greeting; vendor DPAs; data-retention policy |
| Vendor price churn (3 repricings in the last year among these vendors) | Re-verify at decision time; everything here is swappable (Pipecat-native alternatives exist per component) |
