"""Prompt Management Engine using Jinja2 Templating."""

from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from app.domain.interfaces.prompt import BasePromptManager
from app.config.settings import settings
from app.utils.exceptions import PromptNotFoundError
from app.utils.logger import logger

PROMPTS_DIR = Path(__file__).resolve().parent / "templates"


class PromptManager(BasePromptManager):
    """Loads and compiles external prompt templates from template directory."""

    def __init__(self, template_dir: Optional[Path] = None) -> None:
        self.template_dir = template_dir or PROMPTS_DIR
        if not self.template_dir.exists():
            self.template_dir.mkdir(parents=True, exist_ok=True)

        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render_prompt(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render prompt template identified by relative path (e.g., 'system/jarvis_system.jinja2')."""
        try:
            template = self.env.get_template(template_name)
            # Inject default global variables if missing
            default_context = {
                "app_name": settings.APP_NAME,
                "app_env": settings.APP_ENV,
                "provider": settings.LLM_PROVIDER.value,
                "model": settings.MODEL_NAME,
            }
            merged_context = {**default_context, **context}
            rendered = template.render(**merged_context)
            logger.debug(f"Successfully rendered prompt template '{template_name}'")
            return rendered
        except TemplateNotFound as exc:
            logger.error(f"Prompt template not found: {template_name} in {self.template_dir}")
            raise PromptNotFoundError(f"Prompt template '{template_name}' could not be located.") from exc
        except Exception as exc:
            logger.error(f"Error rendering prompt template '{template_name}': {exc}")
            raise PromptNotFoundError(f"Failed to render prompt template '{template_name}': {exc}") from exc


# Global instance singleton
prompt_manager = PromptManager()
