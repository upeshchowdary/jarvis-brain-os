"""Interactive Terminal Chat Interface for JARVIS AI Operating System."""

import sys
import asyncio
import httpx
import json

BASE_URL = "http://127.0.0.1:8000"


async def check_server_connection() -> bool:
    """Verify that the FastAPI server is running."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            res = await client.get(f"{BASE_URL}/health")
            if res.status_code == 200:
                data = res.json()
                print(f" Connected to {data.get('app_name')} (Provider: {data.get('active_provider')})")
                return True
    except Exception:
        pass
    print(" Server is not running. Please start the server using: uvicorn app.main:app --reload")
    return False


async def chat_loop():
    print("=" * 60)
    print("  JARVIS AI OPERATING SYSTEM — INTERACTIVE TERMINAL CHAT")
    print("=" * 60)
    
    if not await check_server_connection():
        return

    print("Type your message and press Enter. Type 'exit' or 'quit' to quit.\n")
    session_id = "cli_session_001"

    async with httpx.AsyncClient(timeout=60.0) as client:
        while True:
            try:
                user_input = input("You > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    print("Goodbye!")
                    break

                payload = {
                    "query": user_input,
                    "session_id": session_id,
                }

                print("JARVIS is thinking...", end="\r", flush=True)
                response = await client.post(f"{BASE_URL}/chat", json=payload)
                print(" " * 30, end="\r", flush=True)  # Clear thinking indicator

                if response.status_code == 200:
                    data = response.json()
                    intent_obj = data.get("intent", {})
                    intent_code = intent_obj.get("intent") if isinstance(intent_obj, dict) else str(intent_obj)
                    resp_text = data.get("response", "")
                    meta = data.get("metadata", {})
                    latency = meta.get("total_latency_ms", meta.get("latency_ms", 0))

                    print(f"\nJARVIS [{intent_code} | {latency:.0f}ms] >")
                    print(resp_text)
                    print("-" * 60 + "\n")
                else:
                    print(f"Error {response.status_code}: {response.text}\n")

            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
            except Exception as exc:
                print(f"\nConnection Error: {exc}\n")


if __name__ == "__main__":
    asyncio.run(chat_loop())
