from tools.search_tool import InternetSearchTool
from tools.scraper_tool import WebScraperTool
from tools.filesystem_tool import FileSystemTool
from tools.system_tool import SystemInfoTool
from tools.vision_tool import ScreenVisionTool
from brain.tool_router import tool_router

# Auto-register Phase 2 active tools into central ToolRouter
tool_router.register_tool(InternetSearchTool())
tool_router.register_tool(WebScraperTool())
tool_router.register_tool(FileSystemTool())
tool_router.register_tool(SystemInfoTool())
tool_router.register_tool(ScreenVisionTool())

__all__ = [
    "InternetSearchTool",
    "WebScraperTool",
    "FileSystemTool",
    "SystemInfoTool",
    "ScreenVisionTool",
]
