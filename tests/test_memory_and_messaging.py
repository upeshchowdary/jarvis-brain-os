"""Memory extraction correctness and messaging safety logic."""

import subprocess
import sys

import pytest

from automation.gateway import ActionRequest, gateway
from automation.ops.messaging import header_matches, pick_contact
from memory.long_term import extract_explicit_facts, long_term_memory, stable_key


@pytest.mark.parametrize("text", ["I am tired today", "I am working on my capstone project", "I'm looking for a job",
                                  "I am here", "I am asking about python"])
def test_i_am_never_sets_name(text):
    assert not any(c.key == "user_name" for c in extract_explicit_facts(text))


@pytest.mark.parametrize("text,name", [("My name is Rahul", "Rahul"), ("my name is aarav", "Aarav"),
                                       ("call me Ananya please", "Ananya"), ("My name is Sai Kiran", "Sai Kiran")])
def test_names_with_letter_a_are_stored(text, name):
    names = [c.value for c in extract_explicit_facts(text) if c.key == "user_name"]
    assert names == [name]


def test_project_extraction():
    facts = {c.key: c.value for c in extract_explicit_facts("I am working on my capstone project")}
    assert facts.get("current_project") == "capstone project"


def test_like_does_not_capture_whole_sentences():
    assert extract_explicit_facts("I like it when you answer quickly and also tell me jokes") == []


def test_keys_are_stable_across_processes():
    code = "from memory.long_term import stable_key; print(stable_key('note', 'Exam on Friday'))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=".").stdout.strip().splitlines()[-1]
    assert out == stable_key("note", "exam   on friday")


async def test_store_and_recall_and_forget():
    await long_term_memory.auto_extract_facts("My name is Rahul")
    assert await long_term_memory.get_fact("user_name") == "Rahul"
    await long_term_memory.auto_extract_facts("I am tired today")
    assert await long_term_memory.get_fact("user_name") == "Rahul"
    k1 = await long_term_memory.remember("Exam on Friday")
    k2 = await long_term_memory.remember("exam on friday")
    assert k1 == k2
    rel = await long_term_memory.relevant_facts("when is my exam", limit=3)
    assert "user_name" in rel and k1 in rel
    removed = await long_term_memory.forget("exam")
    assert removed and await long_term_memory.get_fact(k1) is None


# ── Messaging safety ─────────────────────────────────────────────────────────

def test_pick_contact_requires_unique_exact_match():
    assert pick_contact(["Rahul", "Rahul Sharma", "Mom"], "rahul") == (0, [])
    assert pick_contact(["Rahul Sharma", "Rahul Verma"], "Rahul")[0] is None
    assert set(pick_contact(["Rahul Sharma", "Rahul Verma"], "Rahul")[1]) == {"Rahul Sharma", "Rahul Verma"}
    assert pick_contact(["Rahul Sharma\nlast seen today", "Mom"], "rahul sharma") == (0, [])
    assert pick_contact(["Mom", "Dad"], "Rahul") == (None, [])


def test_header_verification():
    assert header_matches(["Rahul", "online"], "rahul")
    assert not header_matches(["Rahul Verma"], "Rahul Sharma")
    assert not header_matches([], "Rahul")


async def test_message_send_needs_confirmation_and_shows_preview():
    res = await gateway.execute(ActionRequest("message.send", {"app": "whatsapp", "contact": "Mom", "text": "Reached safely"}), "msg")
    assert res.status == "needs_confirmation" and "Mom" in res.preview and "Reached safely" in res.preview
    gateway.cancel(res.pending_id)


@pytest.mark.parametrize("app", ["discord", "slack"])
async def test_discord_slack_never_fall_back_to_whatsapp(app):
    res = await gateway.execute(ActionRequest("message.send", {"app": app, "contact": "Alex", "text": "hi"}), "msg2")
    assert res.status == "failed" and "isn't supported yet" in res.message and gateway.pending_for("msg2") is None


async def test_email_requires_valid_address_and_confirmation():
    bad = await gateway.execute(ActionRequest("email.compose", {"to": "not-an-email", "body": "x"}), "mail")
    assert bad.status == "failed"
    good = await gateway.execute(ActionRequest("email.compose", {"to": "a@b.com", "subject": "Hi", "body": "Hello"}), "mail")
    assert good.status == "needs_confirmation" and "a@b.com" in good.preview
    gateway.cancel(good.pending_id)
