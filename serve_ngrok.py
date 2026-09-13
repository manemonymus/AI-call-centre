"""
One-command ngrok launcher for test phone calls.

Starts ngrok, reads the public URL from ngrok's local API, prints the exact
webhook URL to paste into your Twilio number, and launches the call-center
server with PUBLIC_HOSTNAME already wired — so you never copy/paste the
changing ngrok hostname by hand.

Prereqs (one time):
    1. pip install -r requirements.txt   (done)
    2. Install ngrok from https://ngrok.com/download and authenticate:
           ngrok config add-authtoken <your-token>
    3. A Twilio phone number (twilio.com).

Run:
    python serve_ngrok.py

Then copy the printed webhook URL into your Twilio number's
"A call comes in" (Voice) webhook (HTTP POST) and call the number.
Press Ctrl+C to stop; this also shuts ngrok down.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

PORT = int(os.getenv("PORT", "8000"))
NGROK_API = "http://127.0.0.1:4040/api/tunnels"


def _public_host_from_ngrok() -> str | None:
    """Return the https public hostname (no scheme) from ngrok's local API."""
    try:
        with urlopen(NGROK_API, timeout=2) as resp:
            tunnels = json.load(resp).get("tunnels", [])
    except (URLError, OSError, ValueError):
        return None
    # Prefer the https tunnel.
    for tunnel in sorted(tunnels, key=lambda t: t.get("proto") != "https"):
        url = tunnel.get("public_url", "")
        if url.startswith("https://"):
            return url.removeprefix("https://")
    return None


def main() -> None:
    if not shutil.which("ngrok"):
        sys.exit(
            "ngrok not found on PATH. Install it from https://ngrok.com/download "
            "and run `ngrok config add-authtoken <token>` once, then retry."
        )

    print(f"Starting ngrok on port {PORT} ...")
    ngrok = subprocess.Popen(
        ["ngrok", "http", str(PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        host = None
        for _ in range(20):  # ~10s for ngrok to come up
            time.sleep(0.5)
            host = _public_host_from_ngrok()
            if host:
                break
        if not host:
            sys.exit(
                "Could not read ngrok's public URL from http://127.0.0.1:4040 . "
                "Is ngrok authenticated? Try running `ngrok http 8000` manually."
            )

        os.environ["PUBLIC_HOSTNAME"] = host
        bar = "=" * 64
        print(
            f"\n{bar}\n"
            f"  ngrok is live.  Paste this into your Twilio number's\n"
            f"  Voice 'A call comes in' webhook (HTTP POST):\n\n"
            f"      https://{host}/webhook\n\n"
            f"  Then call the number. Ctrl+C here to stop.\n"
            f"{bar}\n"
        )

        # Import AFTER setting PUBLIC_HOSTNAME so server.py picks it up, and
        # only now (so a missing dependency doesn't leave ngrok orphaned early).
        import uvicorn

        import server

        uvicorn.run(server.app, host="0.0.0.0", port=PORT)
    finally:
        ngrok.terminate()
        print("\nngrok stopped.")


if __name__ == "__main__":
    main()
