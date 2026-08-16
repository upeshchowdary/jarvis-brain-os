from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.config import router as config_router
from app.api.routes.status import router as status_router
from app.api.routes.automation import router as automation_router

__all__ = ["chat_router", "health_router", "config_router", "status_router", "automation_router"]
