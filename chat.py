"""Interactive terminal chat for JARVIS (streams replies token by token).

Usage:  python chat.py            (server must be running: python -m app.main)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import httpx

from core.config import ensure_api_token, settings

BASE_URL = os.environ.get("JARVIS_URL", f"http://{settings.HOST}:{settings.PORT}")
HEADERS = {"Authorization": f"Bearer {settings.JARVIS_API_TOKEN or ensure_api_token()}"}


async def check_server(client: httpx.AsyncClient) -> bool:
    try:
        res = await client.get(f"{BASE_URL}/health", timeout=3.0)
        data = res.json()
        print(f"Connected to {data.get('app_name')} — model {data.get('active_provider')}/{data.get('active_model')}")
        return True
    except Exception:
        print("Server is not running. Start it with:  python -m app.main")
        return False


async def ask(client: httpx.AsyncClient, query: str, session_id: str) -> None:
    payload = {"query": query, "session_id": session_id}
    print("JARVIS > ", end="", flush=True)
    streamed = False
    async with client.stream("POST", f"{BASE_URL}/chat/stream", json=payload, headers=HEADERS, timeout=180.0) as res:
        if res.status_code != 200:
            print(f"\nError {res.status_code}: {(await res.aread()).decode()[:300]}")
            return
        async for line in res.aiter_lines():
            if not line.startswith("data:"):
                continue
            event = json.loads(line[5:])
            if event["type"] == "text":
                streamed = True
                print(event["delta"], end="", flush=True)
            elif event["type"] == "tool":
                print(f"\n  [tool: {event['name']}]", end="", flush=True)
            elif event["type"] == "error":
                print(f"\n⚠️ {event['message']}")
            elif event["type"] == "done":
                result = event["result"]
                if not streamed:
                    print(("\n" if result.get("tools") else "") + result["response"], end="")
                meta = result.get("metadata", {})
                tag = f"{meta.get('provider', '')}/{meta.get('model', '')}".strip("/")
                print(f"\n  ({result['route']} · {tag} · {meta.get('total_latency_ms', 0):.0f} ms)")
                if result.get("status") == "needs_confirmation":
                    print("  ↳ reply 'yes' to confirm, anything else cancels.")
    print("-" * 60)


async def chat_loop() -> None:
    async with httpx.AsyncClient() as client:
        if not await check_server(client):
            return
        print("Type a message (exit to quit).\n")
        session_id = "cli_session_001"
        while True:
            try:
                query = input("You > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                return
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Goodbye!")
                return
            try:
                await ask(client, query, session_id)
            except httpx.HTTPError as exc:
                print(f"\nConnection error: {exc}")


if __name__ == "__main__":
    try:
        asyncio.run(chat_loop())
    except KeyboardInterrupt:
        sys.exit(0)
