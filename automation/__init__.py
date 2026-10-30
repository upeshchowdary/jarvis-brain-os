"""JARVIS automation: the ActionGateway (single door for side effects), the
computer-use engine, low-level controllers and workflow recording/replay."""

from automation.config import automation_config
from automation.gateway import ActionRequest, GatewayResult, Risk, gateway

__all__ = ["ActionRequest", "GatewayResult", "Risk", "automation_config", "gateway"]
