"""Messaging tools — drafts are always previewed and confirmed by the gateway."""

from __future__ import annotations

from typing import Any

from automation.gateway import ActionRequest, Risk
from tools.base import Tool, ToolContext, ToolResult, obj, via_gateway
from tools.registry import register


@register
class SendMessage(Tool):
    name = "send_message"
    description = (
        "Send a chat message to a contact via WhatsApp or Telegram desktop. Write the final message text yourself "
        "(polite and complete if the user asked you to compose it). The user sees a preview and must confirm. "
        "Discord/Slack are not supported for sending."
    )
    parameters = obj({
        "app": {"type": "string", "enum": ["whatsapp", "telegram", "discord", "slack"]},
        "contact": {"type": "string", "description": "Contact name exactly as the user said it"},
        "text": {"type": "string", "description": "The message to send"},
    }, ["app", "contact", "text"])
    side_effect = True
    risk = Risk.HIGH

    async def run(self, ctx: ToolContext, app: str = "whatsapp", contact: str = "", text: str = "", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("message.send", {"app": app.lower(), "contact": contact, "text": text}))


@register
class ComposeEmail(Tool):
    name = "compose_email"
    description = "Open a pre-filled Gmail draft (the user reviews and presses Send). Write a complete subject and body."
    parameters = obj({
        "to": {"type": "string", "description": "Recipient email address"},
        "subject": {"type": "string"},
        "body": {"type": "string"},
    }, ["to", "body"])
    side_effect = True
    risk = Risk.MEDIUM

    async def run(self, ctx: ToolContext, to: str = "", subject: str = "", body: str = "", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("email.compose", {"to": to, "subject": subject, "body": body}))
