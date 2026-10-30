"""Benign questions must never trigger an action.

Covers every query from the original bug report (P0: ordinary questions triggered
reboots / killed JARVIS) plus 100+ more questions full of trigger words.
"""

from unittest.mock import MagicMock

import pytest

from automation.gateway import gateway
from brain import router
from brain.pipeline import brain
from tests.fakes import FakeLLM, call, install, text
from tools.registry import load_builtin_tools

REPORTED = [
    "How do I reboot my router?",
    "What skill python developers need most?",
    "Explain blockchain",
    "Explain how a mutex lock works",
    "What time is it on the clock?",
    "Tips for better sleep",
    "What is open source software?",
    "Give me some context on WW2",
    "How close is the moon?",
    "I want to start learning ML",
    "Should I restart my career?",
    "Summarize this text for me",
    "How long is my commute",
]

MORE = [
    "How do I kill a zombie process in Linux?", "Why does my laptop restart randomly?", "What does shutdown /s /t 0 do?",
    "Is it bad to put my laptop to sleep every night?", "How do I lock a cell in Excel?",
    "What's the best way to text someone after an interview?", "Can you explain how messages are sent over TCP?",
    "Which is better, closing apps or leaving them open?", "What is the volume of a sphere?",
    "How do I mute someone on Instagram?", "Why do people delete their social media?", "What does the kill command do in Unix?",
    "How do I open a PDF on Linux?", "What happens if I empty the recycle bin?", "Explain what a screenshot is",
    "Is it safe to turn off wifi at night?", "When should I restart my router?", "How do I start a podcast?",
    "What should I run first, the tests or the linter?", "Tell me a joke about restarting computers",
    "What's the capital of France?", "How does a lock screen protect my data?", "Why is my python script so slow?",
    "What is the difference between sleep and hibernate?", "How do I send money abroad cheaply?",
    "Do I need to restart after installing updates?", "What's a good message to send for a birthday?",
    "How can I close the gap between theory and practice?", "Who invented the computer mouse?",
    "What are the side effects of too much screen time?", "Can I open a bank account online?",
    "How many processes can a CPU run at once?", "Why did the stock market close lower today?",
    "What is a kill switch in electronics?", "Explain the shutdown of Google Reader",
    "How to start a startup with no money?", "Should I text my ex?", "What does terminate mean in a contract?",
    "How loud is 80 decibels?", "Why does brightness affect battery life?", "Is it better to shutdown or sleep a PC?",
    "What is Python used for?", "How do I become a better developer?", "What's the weather usually like in Goa in June?",
    "How do neural networks learn?", "Explain recursion with an example", "Compare Rust and Go",
    "What are the benefits of meditation?", "What is the meaning of life?", "Who won the 2011 cricket world cup?",
    "How do I fix a merge conflict in git?", "What is a mutex?", "Define entropy",
    "What's the difference between RAM and storage?", "How can I improve my sleep schedule?",
    "Is coffee bad for you?", "Recommend a good sci-fi book", "What should I cook for dinner?",
    "How do vaccines work?", "Why is the sky blue?", "What is quantum computing?",
    "Explain the theory of relativity simply", "How far is Mars from Earth?", "What does HTTP 404 mean?",
    "Error 404 means not found, right?", "How do I deal with stress before exams?",
    "What are good habits for productivity?", "Can you help me understand pointers in C?",
    "Why do cats purr?", "What is the best programming language for beginners?",
    "How should I prepare for a job interview?", "What's a healthy breakfast?", "How does Wi-Fi work?",
    "What happens when a computer boots?", "Tell me about the history of the internet",
    "Summarize the plot of Inception", "Give me ideas for a college project", "What's the GDP of India?",
    "How do I calculate compound interest?", "Is it worth learning Kubernetes?", "What does an API do?",
    "Explain closures in JavaScript", "What's the difference between TCP and UDP?",
    "Why is my phone battery draining fast?", "How do solar panels work?", "What is machine learning?",
    "Could you tell me how DNS works?", "Would it be smart to restart my laptop daily?",
    "Can I run Windows apps on a Mac?", "Do you think I should quit my job?", "What do you think about AI safety?",
    "I am tired today", "I am working on my capstone project", "My day was long", "I feel like sleeping",
    "I like music", "My laptop keeps freezing, any ideas why?", "I need some motivation",
    "Hello JARVIS", "Good morning", "Thanks a lot", "How are you?", "What can you do?",
]

BENIGN = REPORTED + MORE


def test_corpus_is_large_enough():
    assert len(BENIGN) >= 100 and len(set(BENIGN)) == len(BENIGN)


@pytest.mark.parametrize("query", BENIGN)
def test_fast_path_never_picks_a_side_effect(query):
    reg = load_builtin_tools()
    decision = router.fast_route(query)
    if decision is not None:
        assert not reg.get(decision.tool).side_effect, (query, decision)


@pytest.mark.parametrize("query", BENIGN)
def test_questions_and_statements_are_not_imperative(query):
    assert not router.is_imperative(query), query


@pytest.fixture
def os_spies(monkeypatch):
    """Spy on every dangerous primitive; none of them may be touched."""
    from automation.ops import system_impl

    spies = {}
    for name in ("restart_system", "shutdown_system", "sleep_system", "kill_pids", "lock_screen",
                 "toggle_wifi", "empty_recycle_bin", "set_volume", "set_volume_via_keys"):
        spy = MagicMock(return_value=True)
        monkeypatch.setattr(system_impl, name, spy)
        spies[name] = spy
    return spies


@pytest.mark.parametrize("query", REPORTED)
async def test_reported_queries_get_a_normal_answer(monkeypatch, os_spies, query):
    fake = install(monkeypatch, FakeLLM([text("Here is a helpful answer.")]))
    result = await brain.handle(query, f"neg-{abs(hash(query))}")
    assert result.response.endswith("Here is a helpful answer.")
    assert result.route == "agent" and result.status == "ok"
    reg = load_builtin_tools()
    assert all(not reg.get(n).side_effect for n in fake.offered(0)), fake.offered(0)
    assert not any(s.called for s in os_spies.values())


@pytest.mark.parametrize("query", BENIGN)
async def test_adversarial_llm_cannot_act_on_questions(monkeypatch, os_spies, query):
    """Even if the model tries to restart/kill/send, nothing executes and nothing is queued."""
    attempts = [call("power_control", action="restart"), call("kill_process", target="python"),
                call("send_message", app="whatsapp", contact="Mom", text="hi"), text("Answer.")]
    install(monkeypatch, FakeLLM(attempts))
    sid = f"adv-{abs(hash(query))}"
    result = await brain.handle(query, sid)
    assert gateway.pending_for(sid) is None
    assert result.status != "needs_confirmation"
    assert not any(s.called for s in os_spies.values())
