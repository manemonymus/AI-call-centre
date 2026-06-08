"""
Isolate why bot.py stalls: talk to Ollama's OpenAI-compatible endpoint directly
(no Pipecat involved) in all four combinations of tools x streaming.

Run inside your activated venv:   python ollama_check.py

It prints a label BEFORE each attempt. If one attempt just sits there with no
output, that's the combination that hangs — press Ctrl+C and tell me which
label it stopped on, plus whatever printed above it.
"""

from openai import OpenAI

MODEL = "llama3.2"  # match OLLAMA_MODEL in bot.py
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

tools = [
    {
        "type": "function",
        "function": {
            "name": "route_to_billing",
            "description": "Transfer the caller to the billing department.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
]

messages = [
    {"role": "system", "content": "You are a phone receptionist. Greet the caller in one short sentence."}
]


def attempt(label, *, use_tools, stream):
    print(f"\n=== {label} ===", flush=True)
    kwargs = {"model": MODEL, "messages": messages}
    if use_tools:
        kwargs["tools"] = tools
    try:
        if stream:
            got = False
            for chunk in client.chat.completions.create(stream=True, **kwargs):
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    print(delta.content, end="", flush=True)
                    got = True
                if getattr(delta, "tool_calls", None):
                    print(f"[tool_call chunk: {delta.tool_calls}]", end="", flush=True)
                    got = True
            print("\n-> done streaming" + ("" if got else "  <-- NOTHING was received"), flush=True)
        else:
            msg = client.chat.completions.create(**kwargs).choices[0].message
            print("content:", repr(msg.content), flush=True)
            print("tool_calls:", msg.tool_calls, flush=True)
    except KeyboardInterrupt:
        print("\n-> interrupted (this is the combination that hangs)", flush=True)
        raise
    except Exception as e:
        print(f"-> ERROR: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    attempt("1) no tools, no streaming", use_tools=False, stream=False)
    attempt("2) no tools, streaming", use_tools=False, stream=True)
    attempt("3) WITH tools, no streaming", use_tools=True, stream=False)
    attempt("4) WITH tools, streaming  (this is what bot.py does)", use_tools=True, stream=True)
    print("\nAll four attempts finished — if you got here, none of them hung.", flush=True)
