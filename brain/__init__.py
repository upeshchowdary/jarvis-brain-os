"""JARVIS brain package.

Attributes are resolved lazily so that importing a light submodule
(``brain.utils``, ``brain.brain_config``) never drags in the whole pipeline.
That is what used to cause the ``import memory`` -> ``import brain`` cycle.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "Brain": "brain.pipeline",
    "brain": "brain.pipeline",
    "TurnResult": "brain.pipeline",
    "PersonalityManager": "brain.personality_manager",
    "personality_manager": "brain.personality_manager",
    "PersonalityType": "brain.personality_manager",
    "ConversationManager": "brain.conversation_manager",
    "conversation_manager": "brain.conversation_manager",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'brain' has no attribute {name!r}")
    return getattr(import_module(module), name)
