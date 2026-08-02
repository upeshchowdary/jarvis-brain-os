"""System Telemetry Tool for inspecting CPU, Memory, Disk, and OS metrics."""

import platform
import os
import sys
from typing import Dict, Any
from brain.tool_router import BaseBrainTool
from brain.logger import logger


class SystemInfoTool(BaseBrainTool):
    """Tool for fetching real-time host operating system metrics and hardware status."""

    name: str = "system_info"
    description: str = "Retrieve operating system metrics, platform name, Python version, CPU, and Memory telemetry."
    version: str = "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
        }

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        logger.info("SystemInfoTool gathering telemetry status.")

        try:
            # Try psutil if available, otherwise fallback to platform / os metrics
            cpu_count = os.cpu_count() or 1
            os_name = platform.system()
            os_release = platform.release()
            python_version = platform.python_version()

            memory_info = {}
            try:
                import psutil
                mem = psutil.virtual_memory()
                memory_info = {
                    "total_gb": round(mem.total / (1024 ** 3), 2),
                    "used_gb": round(mem.used / (1024 ** 3), 2),
                    "available_gb": round(mem.available / (1024 ** 3), 2),
                    "usage_percent": mem.percent,
                }
            except ImportError:
                memory_info = {"note": "Install psutil for detailed RAM/Disk metrics."}

            return {
                "success": True,
                "os": f"{os_name} {os_release}",
                "python_version": python_version,
                "cpu_cores": cpu_count,
                "memory": memory_info,
            }

        except Exception as exc:
            logger.error(f"SystemInfoTool execution error: {exc}")
            return {"success": False, "error": str(exc)}
