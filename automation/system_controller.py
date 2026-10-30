"""JARVIS System Controller — Safe file system and command execution.

Wraps the existing FileSystemTool and adds:
- Full file operations (create, read, rename, move, copy, delete, search)
- Safe command execution with stdout/stderr capture
- Confirmation gates for dangerous operations
- All operations go through SafetyManager
"""

import asyncio
import os
import shutil
from pathlib import Path
from typing import Optional

from automation.config import automation_config
from automation.automation_logger import log_action
from automation.safety_manager import safety_manager


import re


def resolve_user_system_path(path_str: str) -> Path:
    """
    Resolve user-friendly path names like 'desktop', 'downloads', 'documents',
    or natural language expressions like:
      - 'can you check what files are there in my desktop'
      - 'can you check the contents of pass file in my desktop'
      - 'is there a file named gem in desktop'
      - 'size of folder named ola in desktop'
      - 'folder maxx res in desktop'
      - 'the folder api_brain on desktop'
    Supports multi-word folder/file names, extensions, and directory queries.
    """
    raw = path_str.strip().strip("'\"")
    if not raw:
        return Path.cwd()

    # Direct existing path check
    direct_p = Path(raw)
    if direct_p.is_absolute() and direct_p.exists():
        return direct_p

    location_hint = ""

    # Detect location first (desktop / downloads / documents …)
    loc_match = re.search(
        r"(?:in|on|at|inside)\s+(?:my\s+)?(desktop|downloads|documents|docs|pictures|music|videos)"
        r"(?:\s+folder|\s+directory)?",
        raw,
        re.IGNORECASE,
    )
    if loc_match:
        location_hint = loc_match.group(1).lower().strip()
    elif any(k in raw.lower() for k in ("desktop", "downloads", "documents", "pictures", "music", "videos")):
        for k in ("desktop", "downloads", "documents", "pictures", "music", "videos"):
            if k in raw.lower():
                location_hint = k
                break

    user_home = Path.home()
    onedrive_desktop = user_home / "OneDrive" / "Desktop"
    onedrive_docs = user_home / "OneDrive" / "Documents"
    onedrive_pics = user_home / "OneDrive" / "Pictures"

    target_dir = Path.cwd()
    if location_hint == "desktop":
        target_dir = onedrive_desktop if onedrive_desktop.exists() else (user_home / "Desktop")
    elif location_hint in ("downloads", "download"):
        target_dir = user_home / "Downloads"
    elif location_hint in ("documents", "docs"):
        target_dir = onedrive_docs if onedrive_docs.exists() else (user_home / "Documents")
    elif location_hint in ("pictures", "pics"):
        target_dir = onedrive_pics if onedrive_pics.exists() else (user_home / "Pictures")
    elif location_hint == "music":
        target_dir = user_home / "Music"
    elif location_hint == "videos":
        target_dir = user_home / "Videos"

    # ── If the input is just a location keyword (bare word like "desktop"), return the directory ──
    # This handles LLM-extracted location hints like "desktop", "downloads folder" etc.
    cleaned_hint = re.sub(r"\s+folder|\s+directory", "", raw.strip(), flags=re.IGNORECASE).strip().lower()
    bare_locations = {"desktop", "downloads", "download", "documents", "docs", "pictures", "pics", "music", "videos", "my desktop", "my downloads", "my documents"}
    if cleaned_hint in bare_locations or location_hint and raw.strip().lower() in (location_hint, f"my {location_hint}", f"{location_hint} folder", f"my {location_hint} folder"):
        return target_dir

    # ── Check if this is a directory-level query (e.g., "what files are in desktop", "list desktop")
    is_dir_query = bool(re.search(
        r"\b(?:what\s+files|which\s+files|show\s+files|list\s+files|list\s+all|check\s+what\s+files|contents\s+of\s+(?:my\s+)?(?:desktop|downloads|documents|docs|pictures|music|videos)|inside\s+(?:my\s+)?(?:desktop|downloads|documents|docs|pictures|music|videos))\b",
        raw,
        re.IGNORECASE,
    ))
    # If the user is just asking to list/view the location itself without naming a specific sub-item
    if is_dir_query and not re.search(r"\b(?:file|folder|directory)\s+named\b|\b(?:contents\s+of\s+file|contents\s+of\s+folder)\b|\b(?:read|view|cat)\s+file\b", raw, re.IGNORECASE):
        # Check if there is a specific file mentioned like "contents of pass file"
        if not re.search(r"\b(?:file|doc)\b", raw, re.IGNORECASE) or "what files" in raw.lower() or "list" in raw.lower():
            return target_dir

    # ── Multi-word folder/file name extraction ───────────────────────────────
    clean_target = ""

    # Pattern 1: "file/folder named <name>" / "file/folder called <name>"
    m_named = re.search(
        r"(?:folder|directory|file)\s+(?:named|called)\s+['\"]?([a-zA-Z0-9_\-\.\s]+?)['\"]?"
        r"(?:\s+(?:in|on|at|inside|present|exists?)\s+|$)",
        raw,
        re.IGNORECASE,
    )
    # Pattern 2: "contents of <name> file" / "contents of <name>" / "content of <name>"
    m_contents = re.search(
        r"(?:contents?|content)\s+of\s+(?:the\s+)?['\"]?([a-zA-Z0-9_\-\.\s]+?)['\"]?"
        r"(?:\s+file|\s+folder)?(?:\s+(?:in|on|at|inside)\s+|$)",
        raw,
        re.IGNORECASE,
    )
    # Pattern 3: "<name> file [present/exists] in desktop" (e.g., "is test_pass file present in my desktop", "test.txt on desktop")
    m_name_file = re.search(
        r"(?:is\s+)?(?:the\s+)?['\"]?([a-zA-Z0-9_\-\.]+?)['\"]?\s+(?:(?:file|folder|document)\s+)?(?:(?:present|exists?|saved|stored)\s+)?(?:in|on|at|inside)\s+(?:my\s+)?(?:desktop|downloads|documents|docs|pictures|music|videos)",
        raw,
        re.IGNORECASE,
    )
    # Pattern 4: "file <name> in desktop" / "folder <name> in desktop"
    m_bare = re.search(
        r"(?:folder|directory|file)\s+['\"]?([a-zA-Z0-9_\-\.\s]+?)['\"]?\s+(?:(?:present|exists?)\s+)?(?:in|on|at|inside)\s+(?:my\s+)?(?:desktop|downloads|documents|docs|pictures|music|videos)",
        raw,
        re.IGNORECASE,
    )
    # Pattern 5: "size of <name> in desktop" / "check if <name> exists in desktop"
    m_simple = re.search(
        r"(?:size\s+of|how\s+big\s+is|check\s+if|check\s+whether|is\s+there\s+a)\s+['\"]?([a-zA-Z0-9_\-\.\s]+?)['\"]?\s+(?:(?:present|exists?)\s+)?(?:in|on|at|inside)\s+(?:my\s+)?(?:desktop|downloads|documents|docs|pictures|music|videos)",
        raw,
        re.IGNORECASE,
    )
    # Pattern 6: "read <name>" / "view <name>" / "cat <name>"
    m_read = re.search(
        r"(?:read|view|cat|open)\s+(?:the\s+)?(?:file\s+)?['\"]?([a-zA-Z0-9_\-\.\s]+?)['\"]?"
        r"(?:\s+(?:in|on|at|inside)\s+|$)",
        raw,
        re.IGNORECASE,
    )

    if m_named:
        clean_target = m_named.group(1).strip()
    elif m_contents and m_contents.group(1).strip().lower() not in ("my", "the", "this", "that", "all", "what"):
        clean_target = m_contents.group(1).strip()
    elif m_name_file and m_name_file.group(1).strip().lower() not in ("what", "which", "there", "any", "a", "the", "this"):
        clean_target = m_name_file.group(1).strip()
    elif m_bare:
        clean_target = m_bare.group(1).strip()
    elif m_simple and m_simple.group(1).strip().lower() not in ("what", "which", "there", "any", "a", "the", "file", "folder"):
        clean_target = m_simple.group(1).strip()
    elif m_read:
        clean_target = m_read.group(1).strip()
    else:
        # Fall back: strip known noise phrases
        noise = re.sub(
            r"\b(?:can\s+you\s+)?(?:check\s+whether|check\s+if|check|is\s+there\s+(?:a\s+)?|is\s+|say|tell\s+me|give\s+me|show\s+me|what\s+is|whats\s+in|what\s+is\s+in|whats\s+inside|get)\s+"
            r"(?:the\s+)?(?:size\s+(?:of\s+)?(?:the\s+)?|how\s+big\s+is\s+(?:the\s+)?|contents?\s+(?:of\s+)?(?:the\s+)?)?",
            "",
            raw,
            flags=re.IGNORECASE,
        )
        noise = re.sub(
            r"\s+(?:(?:present|exists?|saved|stored)\s+)?(?:in|on|at|inside)\s+(?:my\s+)?(?:desktop|downloads|documents|docs|pictures|music|videos)"
            r"(?:\s+folder|\s+directory)?.*$",
            "",
            noise,
            flags=re.IGNORECASE,
        )
        noise = re.sub(r"\b(?:folder|directory|file|present|exists?)\b", "", noise, flags=re.IGNORECASE)
        clean_target = noise.strip().strip("'\"").strip()

    # If clean_target ended up empty and location_hint exists, return target_dir
    if not clean_target:
        return target_dir

    # Handle prefixes like "desktop/foo" or "desktop\foo"
    for prefix, hint in [("desktop", 8), ("downloads", 10), ("documents", 10)]:
        if clean_target.lower().startswith(f"{prefix}/") or clean_target.lower().startswith(f"{prefix}\\"):
            clean_target = clean_target[hint:].strip()
            break

    # Direct match
    cand = target_dir / clean_target
    if cand.exists():
        return cand

    # Case-insensitive match inside target_dir
    if target_dir.exists():
        try:
            for entry in target_dir.iterdir():
                if entry.name.lower() == clean_target.lower():
                    return entry
        except Exception:
            pass

    # Try common extensions (.txt, .json, .py, .md, .csv, .log, .pdf, .docx, .png, .jpg)
    common_exts = [".txt", ".json", ".py", ".md", ".csv", ".log", ".pdf", ".docx", ".png", ".jpg", ".html"]
    if target_dir.exists():
        try:
            for ext in common_exts:
                cand_ext = target_dir / f"{clean_target}{ext}"
                if cand_ext.exists():
                    return cand_ext
            # Case-insensitive with extensions
            for entry in target_dir.iterdir():
                for ext in common_exts:
                    if entry.name.lower() == f"{clean_target.lower()}{ext}":
                        return entry
        except Exception:
            pass

    # Try workspace / current working directory
    cwd_cand = Path.cwd() / clean_target
    if cwd_cand.exists():
        return cwd_cand
    for ext in common_exts:
        cwd_cand_ext = Path.cwd() / f"{clean_target}{ext}"
        if cwd_cand_ext.exists():
            return cwd_cand_ext

    if location_hint:
        return cand  # Return best-guess path even if not found (caller checks .exists())

    p = Path(raw)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


class SystemController:
    """Safe file system and terminal command controller."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run or automation_config.DRY_RUN

    # ── File Operations ──────────────────────────────────────────────

    async def create_file(self, path: str, content: str = "") -> dict:
        allowed, reason = safety_manager.is_allowed("create_file", path)
        if not allowed:
            return {"success": False, "blocked": True, "reason": reason}
        log_action("system", "create_file", path, dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "path": path}
        try:
            p = resolve_user_system_path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            log_action("system", "create_file", str(p), result="CREATED")
            return {"success": True, "action": "create_file", "path": str(p), "size": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_path_size(self, path: str) -> dict:
        """Calculate total size of a file or directory in bytes, KB, MB, GB."""
        log_action("system", "get_path_size", path, dry_run=self.dry_run)
        try:
            p = resolve_user_system_path(path)
            if not p.exists():
                return {"success": False, "error": f"Path not found: '{path}' (checked '{p}')"}

            if p.is_file():
                size_bytes = p.stat().st_size
                num_files = 1
            else:
                size_bytes = 0
                num_files = 0
                for root, _, files in os.walk(p):
                    for f in files:
                        try:
                            fp = os.path.join(root, f)
                            size_bytes += os.path.getsize(fp)
                            num_files += 1
                        except Exception:
                            pass

            def format_bytes(b: int) -> str:
                if b < 1024:
                    return f"{b} B"
                elif b < 1024 * 1024:
                    return f"{b / 1024:.2f} KB"
                elif b < 1024 * 1024 * 1024:
                    return f"{b / (1024 * 1024):.2f} MB"
                else:
                    return f"{b / (1024 * 1024 * 1024):.2f} GB"

            return {
                "success": True,
                "path": str(p),
                "name": p.name,
                "is_dir": p.is_dir(),
                "size_bytes": size_bytes,
                "formatted_size": format_bytes(size_bytes),
                "num_files": num_files,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def list_directory(self, path: str = "") -> dict:
        """List files and folders in the target directory."""
        log_action("system", "list_directory", path, dry_run=self.dry_run)
        try:
            p = resolve_user_system_path(path) if path else Path.cwd()
            if not p.exists() or not p.is_dir():
                return {"success": False, "error": f"Directory not found: '{path}' (checked '{p}')"}

            items = []
            for entry in p.iterdir():
                try:
                    items.append({
                        "name": entry.name,
                        "is_dir": entry.is_dir(),
                        "size_bytes": entry.stat().st_size if entry.is_file() else 0,
                    })
                except Exception:
                    pass
            return {"success": True, "path": str(p), "items": items, "total_items": len(items)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def rename_file(self, src: str, new_name: str) -> dict:
        allowed, reason = safety_manager.is_allowed("rename_file", src)
        if not allowed:
            return {"success": False, "blocked": True, "reason": reason}
        log_action("system", "rename_file", f"{src} -> {new_name}", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}
        try:
            p = Path(src)
            dest = p.parent / new_name
            p.rename(dest)
            return {"success": True, "action": "rename_file", "new_path": str(dest)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def copy_file(self, src: str, dest: str) -> dict:
        allowed, reason = safety_manager.is_allowed("copy_file", src)
        if not allowed:
            return {"success": False, "blocked": True, "reason": reason}
        log_action("system", "copy_file", f"{src} -> {dest}", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}
        try:
            shutil.copy2(src, dest)
            return {"success": True, "action": "copy_file", "src": src, "dest": dest}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def move_file(self, src: str, dest: str, user_confirmed: bool = False) -> dict:
        allowed, reason = safety_manager.is_allowed("move_file", src, user_confirmed)
        if not allowed:
            return {"success": False, "blocked": True,
                    "reason": reason,
                    "confirmation_prompt": safety_manager.format_confirmation_request("move_file", src)}
        log_action("system", "move_file", f"{src} -> {dest}", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True}
        try:
            shutil.move(src, dest)
            return {"success": True, "action": "move_file", "src": src, "dest": dest}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def delete_file(self, path: str, user_confirmed: bool = False) -> dict:
        """Delete a file. ALWAYS requires user confirmation."""
        allowed, reason = safety_manager.is_allowed("delete_file", path, user_confirmed)
        if not allowed:
            return {
                "success": False,
                "blocked": True,
                "reason": reason,
                "confirmation_prompt": safety_manager.format_confirmation_request("delete_file", path),
            }
        log_action("system", "delete_file", path, dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "would_delete": path}
        try:
            p = Path(path)
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(path)
            else:
                return {"success": False, "error": f"Path not found: {path}"}
            return {"success": True, "action": "delete_file", "path": path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def list_files(self, directory: str = ".") -> dict:
        log_action("system", "list_files", directory, dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "items": []}
        try:
            p = Path(directory)
            if not p.exists():
                return {"success": False, "error": f"Directory not found: {directory}"}
            items = [
                {
                    "name": entry.name,
                    "type": "dir" if entry.is_dir() else "file",
                    "size": entry.stat().st_size if entry.is_file() else 0,
                    "path": str(entry),
                }
                for entry in sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
            ]
            return {"success": True, "directory": directory, "items": items, "count": len(items)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def search_files(
        self, pattern: str, directory: str = ".", recursive: bool = True
    ) -> dict:
        log_action("system", "search_files", f"{pattern} in {directory}", dry_run=self.dry_run)
        if self.dry_run:
            return {"success": True, "dry_run": True, "matches": []}
        try:
            p = Path(directory)
            if recursive:
                matches = [str(f) for f in p.rglob(pattern)]
            else:
                matches = [str(f) for f in p.glob(pattern)]
            return {"success": True, "pattern": pattern, "matches": matches, "count": len(matches)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def read_file(self, path: str, max_chars: int = 10000) -> dict:
        """Read text contents of a file safely with encoding fallbacks and truncation safeguards."""
        log_action("system", "read_file", path, dry_run=self.dry_run)
        try:
            p = resolve_user_system_path(path)
            if not p.exists():
                return {"success": False, "error": f"File not found: '{path}' (checked '{p}')"}
            if not p.is_file():
                return {"success": False, "error": f"Target is a directory, not a file: '{p}'"}

            ext = p.suffix.lower()
            size_bytes = p.stat().st_size
            text = ""

            # Check if this is a known binary format without text extraction
            media_binary_exts = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".zip", ".tar", ".gz", ".7z", ".rar", ".exe", ".dll", ".bin", ".iso", ".mp3", ".mp4", ".wav"}

            # 1. PDF Text Extraction (PyMuPDF / PyPDF)
            if ext == ".pdf":
                try:
                    import pymupdf as fitz  # type: ignore[import-untyped]
                    doc = fitz.open(str(p))
                    pages_text = []
                    for i, page in enumerate(doc):
                        t = page.get_text() or ""
                        if t.strip():
                            pages_text.append(f"--- Page {i+1} ---\n{t.strip()}")
                        if len("\n".join(pages_text)) > max_chars:
                            break
                    text = "\n\n".join(pages_text)
                except Exception:
                    try:
                        import pypdf  # type: ignore[import-untyped]
                        reader = pypdf.PdfReader(str(p))
                        pages_text = []
                        for i, page in enumerate(reader.pages[:20]):
                            t = page.extract_text() or ""
                            if t.strip():
                                pages_text.append(f"--- Page {i+1} ---\n{t.strip()}")
                        text = "\n\n".join(pages_text)
                    except Exception:
                        pass

                if not text.strip():
                    text = f"[PDF Document: '{p.name}' ({size_bytes / 1024:.1f} KB)]\nThis is a compiled binary PDF file containing graphic streams and formatted layout."

            # 2. Word (.docx) Extraction
            elif ext in (".docx", ".doc"):
                try:
                    import docx  # type: ignore[import-untyped]
                    doc = docx.Document(str(p))
                    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                except Exception as e:
                    text = f"[Word Document: '{p.name}' ({size_bytes / 1024:.1f} KB)]\nCould not extract Word document text: {e}"

            # 3. Excel (.xlsx, .xls) Extraction
            elif ext in (".xlsx", ".xls"):
                try:
                    import openpyxl  # type: ignore[import-untyped]
                    wb = openpyxl.load_workbook(str(p), data_only=True)
                    lines = []
                    for sheet in wb.worksheets[:3]:
                        lines.append(f"[Sheet: {sheet.title}]")
                        for row in list(sheet.iter_rows(values_only=True))[:50]:
                            row_str = " | ".join(str(c) for c in row if c is not None)
                            if row_str.strip():
                                lines.append(row_str)
                    text = "\n".join(lines)
                except Exception as e:
                    text = f"[Excel Spreadsheet: '{p.name}' ({size_bytes / 1024:.1f} KB)]\nCould not extract Excel data: {e}"

            elif ext in media_binary_exts:
                return {
                    "success": True,
                    "path": str(p),
                    "name": p.name,
                    "size_bytes": size_bytes,
                    "line_count": 0,
                    "is_truncated": False,
                    "content": f"[Binary File: '{p.name}' ({size_bytes / 1024:.1f} KB)]\nThis file is in a binary/compressed format ({ext[1:].upper()}) and cannot be displayed as plain text.",
                }

            else:
                # Check for null bytes in initial chunk (binary check)
                try:
                    sample = p.read_bytes()[:1024]
                    if b"\x00" in sample:
                        return {
                            "success": True,
                            "path": str(p),
                            "name": p.name,
                            "size_bytes": size_bytes,
                            "line_count": 0,
                            "is_truncated": False,
                            "content": f"[Binary File: '{p.name}' ({size_bytes / 1024:.1f} KB)]\nThis file contains binary data and cannot be displayed as plain text.",
                        }
                except Exception:
                    pass

                # Try UTF-8 first, fallback to cp1252 / latin-1
                for encoding in ("utf-8", "cp1252", "latin-1"):
                    try:
                        text = p.read_text(encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    try:
                        raw_bytes = p.read_bytes()
                        text = raw_bytes.decode("utf-8", errors="replace")
                    except Exception as e:
                        return {"success": False, "error": f"Could not decode file content: {e}"}

            total_lines = len(text.splitlines())
            is_truncated = False
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n\n... [Truncated: displaying first {max_chars} of {size_bytes} bytes / {total_lines} lines]"
                is_truncated = True

            return {
                "success": True,
                "path": str(p),
                "name": p.name,
                "size_bytes": size_bytes,
                "line_count": total_lines,
                "is_truncated": is_truncated,
                "content": text,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def inspect_file(self, path: str) -> dict:
        """
        Inspect a file or folder and return comprehensive metadata, file classification,
        and content preview.
        """
        log_action("system", "inspect_file", path, dry_run=self.dry_run)
        try:
            p = resolve_user_system_path(path)
            if not p.exists():
                return {
                    "success": False,
                    "exists": False,
                    "error": f"Item not found: '{path}' (checked '{p}')",
                    "checked_path": str(p),
                }

            is_dir = p.is_dir()
            is_file = p.is_file()
            name = p.name
            ext = p.suffix.lower() if is_file else ""
            stat = p.stat()
            size_bytes = stat.st_size if is_file else 0

            # File classification
            code_exts = {".py": "Python Script", ".js": "JavaScript File", ".ts": "TypeScript File", ".html": "HTML Document", ".css": "CSS Stylesheet", ".json": "JSON Data", ".sql": "SQL Script", ".sh": "Shell Script", ".bat": "Batch Script", ".cpp": "C++ Source", ".c": "C Source"}
            doc_exts = {".txt": "Plain Text Document", ".md": "Markdown Document", ".pdf": "PDF Document", ".docx": "Word Document", ".csv": "CSV Spreadsheet", ".xlsx": "Excel Spreadsheet", ".log": "Log File"}
            img_exts = {".png": "PNG Image", ".jpg": "JPEG Image", ".jpeg": "JPEG Image", ".gif": "GIF Image", ".svg": "SVG Graphic", ".ico": "Icon Image"}
            archive_exts = {".zip": "ZIP Archive", ".tar": "TAR Archive", ".gz": "GZ Archive", ".7z": "7-Zip Archive", ".rar": "RAR Archive"}

            if is_dir:
                file_type = "Directory / Folder"
                num_items = len(list(p.iterdir())) if p.exists() else 0
            elif ext in code_exts:
                file_type = code_exts[ext]
            elif ext in doc_exts:
                file_type = doc_exts[ext]
            elif ext in img_exts:
                file_type = img_exts[ext]
            elif ext in archive_exts:
                file_type = archive_exts[ext]
            else:
                file_type = f"{ext.upper()[1:]} File" if ext else "Binary / Generic File"

            content_preview = ""
            line_count = 0
            if is_file and ext in {".txt", ".md", ".json", ".py", ".js", ".ts", ".html", ".css", ".log", ".csv", ".bat", ".sh", ".sql", ".env", ".yaml", ".yml", ".toml", ".ini", ".cfg", ""}:
                try:
                    for enc in ("utf-8", "cp1252", "latin-1"):
                        try:
                            raw_text = p.read_text(encoding=enc)
                            line_count = len(raw_text.splitlines())
                            content_preview = raw_text[:2000]
                            break
                        except UnicodeDecodeError:
                            continue
                except Exception:
                    pass

            def format_bytes(b: int) -> str:
                if b < 1024:
                    return f"{b} B"
                elif b < 1024 * 1024:
                    return f"{b / 1024:.2f} KB"
                else:
                    return f"{b / (1024 * 1024):.2f} MB"

            return {
                "success": True,
                "exists": True,
                "name": name,
                "path": str(p),
                "is_file": is_file,
                "is_dir": is_dir,
                "file_type": file_type,
                "extension": ext,
                "size_bytes": size_bytes,
                "formatted_size": format_bytes(size_bytes) if is_file else f"{num_items} items",
                "line_count": line_count,
                "content_preview": content_preview,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def check_presence(self, path: str) -> dict:
        """Check whether a file or directory exists and return status with details."""
        return await self.inspect_file(path)

    def file_exists(self, path: str) -> bool:
        p = resolve_user_system_path(path)
        return p.exists()

    # ── Command Execution ────────────────────────────────────────────

    async def run_command(
        self,
        command: str,
        timeout: float | None = None,
        user_confirmed: bool = False,
        cwd: Optional[str] = None,
    ) -> dict:
        """
        Execute a system command safely.
        HIGH-risk commands always require confirmation.
        Returns structured result with stdout, stderr, exit_code.
        """
        allowed, reason = safety_manager.is_allowed("run_command", command, user_confirmed)
        if not allowed:
            return {
                "success": False,
                "blocked": True,
                "reason": reason,
                "confirmation_prompt": safety_manager.format_confirmation_request("run_command", command),
            }

        log_action("system", "run_command", command[:80], dry_run=self.dry_run)
        if self.dry_run:
            return {
                "success": True,
                "dry_run": True,
                "command": command,
                "note": f"Would run: {command}",
            }

        timeout_s = timeout or automation_config.ACTION_TIMEOUT

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_s
                )
            except asyncio.TimeoutError:
                proc.kill()
                return {
                    "success": False,
                    "command": command,
                    "error": f"Command timed out after {timeout_s}s",
                    "exit_code": -1,
                }

            stdout = stdout_b.decode("utf-8", errors="replace").strip()
            stderr = stderr_b.decode("utf-8", errors="replace").strip()
            exit_code = proc.returncode or 0

            return {
                "success": exit_code == 0,
                "command": command,
                "stdout": stdout[:5000],  # Limit output size
                "stderr": stderr[:2000],
                "exit_code": exit_code,
            }
        except Exception as e:
            return {"success": False, "command": command, "error": str(e), "exit_code": -1}


# Global singleton
system_controller = SystemController()

