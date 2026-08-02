"""Prompt Manager for Jinja2 dynamic template compilation across chat, reasoning, planning, coding, and reflection."""

from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from brain.brain_config import brain_config
from brain.logger import logger

PROMPTS_STORE_DIR = Path(__file__).resolve().parent / "prompts_store"


class PromptManager:
    """Loads and compiles reusable prompt templates for Chat, Reasoning, Planning, Coding, Reflection, and Tools."""

    def __init__(self, templates_dir: Optional[Path] = None) -> None:
        self.templates_dir = templates_dir or PROMPTS_STORE_DIR
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_default_templates()

        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _ensure_default_templates(self) -> None:
        """Create default jinja2 template files if they do not exist."""
        defaults = {
            "chat.jinja2": "{{ personality_prompt }}\n\nCurrent Date & Time: {{ current_time }}\nActive Model: {{ active_model }}\nUser Query: {{ user_query }}",
            "reasoning.jinja2": "CRITICAL: You are the REASONING engine of {{ system_name }}.\nBreak down the user query into logical steps.\n\nQuery: {{ user_query }}\n\nOutput JSON strictly:\n{\n  \"reasoning_steps\": [\"step 1\", \"step 2\"]\n}",
            "planning.jinja2": "CRITICAL: You are the PLANNING engine of {{ system_name }}.\nDecompose the goal into hierarchical sub-task execution steps.\n\nGoal: {{ user_query }}\n\nOutput JSON strictly:\n{\n  \"goal\": \"{{ user_query }}\",\n  \"steps\": [{\"step_number\": 1, \"action\": \"action_name\", \"description\": \"details\"}]\n}",
            "coding.jinja2": "CRITICAL: You are the CODE ASSISTANT engine of {{ system_name }}.\nProvide clean, production-grade code adhering to SOLID principles.\n\nTask: {{ user_query }}",
            "reflection.jinja2": "CRITICAL: You are the REFLECTION engine of {{ system_name }}.\nEvaluate the response generated for the user query.\n\nQuery: {{ user_query }}\nGenerated Response: {{ generated_response }}\n\nOutput JSON strictly:\n{\n  \"correctness_score\": 0.95,\n  \"misunderstandings\": [],\n  \"improvement_suggestions\": []\n}",
            "tool_call.jinja2": "Registered Tools:\n{% for tool in registered_tools %}- {{ tool.name }}: {{ tool.description }}\n{% endfor %}\n\nSelect tools if needed for query: {{ user_query }}",
        }

        for filename, content in defaults.items():
            file_path = self.templates_dir / filename
            if not file_path.exists():
                file_path.write_text(content, encoding="utf-8")

    def render(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render specified Jinja2 prompt template with variable context."""
        try:
            if not template_name.endswith(".jinja2"):
                template_name = f"{template_name}.jinja2"

            template = self.env.get_template(template_name)
            default_ctx = {
                "system_name": brain_config.SYSTEM_NAME,
                "system_version": brain_config.SYSTEM_VERSION,
            }
            merged = {**default_ctx, **context}
            return template.render(**merged)
        except TemplateNotFound:
            logger.error(f"Prompt template '{template_name}' not found in {self.templates_dir}")
            raise FileNotFoundError(f"Prompt template '{template_name}' not found.")
        except Exception as exc:
            logger.error(f"Error rendering prompt template '{template_name}': {exc}")
            raise exc


# Global prompt manager singleton
prompt_manager = PromptManager()
