"""Pytest configuration and shared fixtures.

Tests must never perform real GUI/OS actions, on any platform. Windows-only and
GUI modules are therefore replaced with mocks *before* any project import, the
data directory points at a throwaway temp folder, and background services are off.
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock

_TMP = tempfile.mkdtemp(prefix="jarvis-test-")
os.environ["JARVIS_DATA_DIR"] = os.path.join(_TMP, "data")
os.environ["JARVIS_LOG_DIR"] = os.path.join(_TMP, "logs")
os.environ["JARVIS_ENV_FILE"] = os.path.join(_TMP, ".env")
os.environ["JARVIS_API_TOKEN"] = "test-token"
os.environ["BACKGROUND_SERVICES"] = "false"
os.environ["DRY_RUN"] = "false"
os.environ["CONFIRM_MEDIUM_RISK"] = "true"
os.environ["ROUTER_USE_LLM"] = "true"
# No real provider keys in tests: every LLM call is mocked.
for _key in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
             "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY"):
    os.environ[_key] = ""

_GUI_MODULES = [
    "pyautogui", "mouseinfo", "winreg", "win32gui", "win32con", "win32api", "win32process",
    "win32pdh", "win32clipboard", "pywinauto", "pywinauto.keyboard", "pywinauto.mouse",
    "pywinauto.application", "pycaw", "pycaw.pycaw", "comtypes", "wmi", "winocr", "d3dshot",
    "pytesseract", "sounddevice", "pyaudio", "pygetwindow", "pyperclip", "pyttsx3",
    "keyboard", "pystray", "win11toast",
]
for _name in _GUI_MODULES:
    sys.modules[_name] = MagicMock(name=_name)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.database.connection import db_manager  # noqa: E402
from app.main import app  # noqa: E402
from automation.application_controller import dynamic_app_manager  # noqa: E402

# Never scan the real registry / Program Files / Start menu during tests (matters on Windows CI).
dynamic_app_manager._loaded = True
dynamic_app_manager._apps_cache = {}

AUTH = {"Authorization": "Bearer test-token"}


@pytest.fixture(autouse=True)
async def init_test_database():
    """Ensure database tables exist before running test cases."""
    await db_manager.init_db()


@pytest.fixture
def sync_client():
    """Authenticated synchronous test client for FastAPI endpoints."""
    with TestClient(app, headers=AUTH) as client:
        yield client


@pytest.fixture
def anon_client():
    """Unauthenticated client, for checking that routes reject missing tokens."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_client():
    """Authenticated asynchronous HTTP test client for FastAPI app endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", headers=AUTH) as ac:
        yield ac
