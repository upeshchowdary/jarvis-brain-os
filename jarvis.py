"""JARVIS launcher.

    python jarvis.py              # API + console (http://127.0.0.1:8000/console)
    python jarvis.py --voice      # + hands-free voice ("hey jarvis …")
    python jarvis.py --tray       # + system-tray icon and Win+J hotkey for the console
"""

from __future__ import annotations

import argparse
import asyncio
import threading
import webbrowser

import uvicorn

from core.config import ensure_api_token, settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JARVIS")
    parser.add_argument("--voice", action="store_true", help="start the hands-free voice loop")
    parser.add_argument("--tray", action="store_true", help="show a system-tray icon (needs pystray + pillow)")
    parser.add_argument("--open", action="store_true", help="open the console in the browser")
    args = parser.parse_args()

    token = ensure_api_token()
    url = f"http://{settings.HOST}:{settings.PORT}/console"
    print(f"JARVIS → {url}\nAPI token (also in .env): {token[:6]}…  — paste it into the console once.")

    config = uvicorn.Config("app.main:app", host=settings.HOST, port=settings.PORT, log_level="warning")
    server = uvicorn.Server(config)

    async def run() -> None:
        if args.voice:
            async def start_voice() -> None:
                await asyncio.sleep(1.5)
                from voice.voice_manager import voice_manager

                if not await voice_manager.start():
                    print("Voice could not start (microphone?). Text and console still work.")
                else:
                    print("🎙  Voice online — say “Hey JARVIS …”")

            asyncio.get_running_loop().create_task(start_voice())
        await server.serve()

    if args.open:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    if args.tray:
        from jarvis_tray import run_tray

        threading.Thread(target=lambda: asyncio.run(run()), daemon=True).start()
        run_tray(url, on_quit=lambda: setattr(server, "should_exit", True))
    else:
        asyncio.run(run())


if __name__ == "__main__":
    main()
