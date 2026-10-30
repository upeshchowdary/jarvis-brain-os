"""File operations with Recycle-Bin deletes and undo.

* Deleting = ``send2trash`` (Recycle Bin), never permanent.
* Every write/move/rename/organize/bulk-rename records an inverse so "undo that" works.
* System folders (Windows, Program Files, /usr, /etc …) and JARVIS's own data are off limits.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from automation.gateway import OpResult, Preview, Risk, gateway
from automation.paths import resolve
from core.paths import data_dir

CATEGORIES = {
    "Images": {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg", ".heic", ".ico"},
    "Documents": {".pdf", ".doc", ".docx", ".txt", ".md", ".rtf", ".odt", ".pages"},
    "Spreadsheets": {".xls", ".xlsx", ".csv", ".ods"},
    "Presentations": {".ppt", ".pptx", ".key", ".odp"},
    "Archives": {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"},
    "Installers": {".exe", ".msi", ".msix", ".appx", ".dmg", ".deb", ".rpm"},
    "Audio": {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"},
    "Video": {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm"},
    "Code": {".py", ".js", ".ts", ".java", ".c", ".cpp", ".cs", ".go", ".rs", ".html", ".css", ".json", ".ipynb", ".sql"},
}




# Compared both as given and with any drive letter removed, so "\\Windows", "D:\\Windows" and a driveless
# "/etc" on Windows are all caught.
_SYSTEM_ROOTS = ["/windows", "/program files", "/program files (x86)", "/programdata", "/usr", "/etc", "/bin",
                 "/sbin", "/boot", "/system", "/library", "/var", "/opt", "/proc", "/sys", "/dev"]
_DRIVE = re.compile(r"^[a-z]:")


def _protected(p: Path) -> Optional[str]:
    try:
        rp = p.resolve()
    except OSError:
        rp = p
    for candidate in {str(p), str(rp)}:
        s = candidate.replace("\\", "/").lower()
        for form in {s, _DRIVE.sub("", s)}:
            if any(form == r or form.startswith(r + "/") for r in _SYSTEM_ROOTS):
                return f"I won't modify system location {p}."
    data = str(data_dir().resolve()).replace("\\", "/").lower()
    if s == data or s.startswith(data + "/"):
        return "I won't modify JARVIS's own data folder."
    if rp == Path(rp.anchor) or rp == Path.home():
        return f"I won't modify the whole {rp} folder."
    return None


def _unique(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem, suffix, i = dest.stem, dest.suffix, 1
    while (dest.parent / f"{stem} ({i}){suffix}").exists():
        i += 1
    return dest.parent / f"{stem} ({i}){suffix}"


def _check(*paths: Path) -> Optional[str]:
    for p in paths:
        err = _protected(p)
        if err:
            return err
    return None


def _p(args: Dict[str, Any], key: str = "path") -> Path:
    return resolve(str(args.get(key, "")))


# ── Single-file operations ─────────────────────────────────────────────────

@gateway.operation("file.create", Risk.MEDIUM, "Create a text file", preview=lambda a: Preview(f"Create {_p(a)} ({len(str(a.get('content', '')))} characters)"))
async def create_file(path: str, content: str = "", overwrite: bool = False) -> OpResult:
    p = resolve(path)
    if (err := _check(p)):
        return OpResult.fail(err)
    if p.exists() and not overwrite:
        return OpResult.fail(f"{p.name} already exists. Ask me to overwrite it if you want.")
    backup = None
    if p.exists():
        backup = data_dir() / "file_backups" / f"{int(time.time())}_{p.name}"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, backup)
    p.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(p.write_text, content, "utf-8")
    undo = {"op": "file.restore", "args": {"path": str(p), "backup": str(backup)}} if backup else {"op": "file.trash", "args": {"paths": [str(p)]}}
    return OpResult.ok(f"Created {p}.", data={"path": str(p)}, undo=undo)


@gateway.operation("file.append", Risk.MEDIUM, "Append text to a file", preview=lambda a: Preview(f"Append to {_p(a)}: “{str(a.get('content', ''))[:200]}”"))
async def append_file(path: str, content: str) -> OpResult:
    p = resolve(path)
    if (err := _check(p)):
        return OpResult.fail(err)
    backup = None
    if p.exists():
        backup = data_dir() / "file_backups" / f"{int(time.time())}_{p.name}"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, backup)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(("\n" if p.exists() and p.stat().st_size else "") + content)
    undo = {"op": "file.restore", "args": {"path": str(p), "backup": str(backup)}} if backup else {"op": "file.trash", "args": {"paths": [str(p)]}}
    return OpResult.ok(f"Appended to {p.name}.", undo=undo)


@gateway.operation("file.restore", Risk.MEDIUM, "Restore a file from JARVIS's backup")
async def restore_file(path: str, backup: str) -> OpResult:
    src, dst = Path(backup), Path(path)
    if not src.exists():
        return OpResult.fail("The backup no longer exists.")
    shutil.copy2(src, dst)
    return OpResult.ok(f"Restored {dst.name}.")


@gateway.operation("file.mkdir", Risk.MEDIUM, "Create a folder", preview=lambda a: Preview(f"Create folder {_p(a)}"))
async def make_dir(path: str) -> OpResult:
    p = resolve(path)
    if (err := _check(p)):
        return OpResult.fail(err)
    existed = p.exists()
    p.mkdir(parents=True, exist_ok=True)
    return OpResult.ok(f"Folder ready: {p}.", undo=None if existed else {"op": "file.rmdir_empty", "args": {"path": str(p)}})


@gateway.operation("file.rmdir_empty", Risk.MEDIUM, "Remove an empty folder")
async def rmdir_empty(path: str) -> OpResult:
    p = Path(path)
    try:
        p.rmdir()
        return OpResult.ok(f"Removed empty folder {p.name}.")
    except OSError:
        return OpResult.fail(f"{p.name} isn't empty any more, so I left it.")


def _rename_preview(a):
    p = _p(a, "src")
    return Preview(f"Rename {p.name} → {a.get('new_name')}") if p.exists() else Preview(f"{p} doesn't exist.", ok=False)


@gateway.operation("file.rename", Risk.MEDIUM, "Rename a file or folder", preview=_rename_preview)
async def rename(src: str, new_name: str) -> OpResult:
    p = resolve(src)
    dest = p.with_name(Path(new_name).name)
    if (err := _check(p, dest)):
        return OpResult.fail(err)
    if dest.exists():
        return OpResult.fail(f"{dest.name} already exists.")
    p.rename(dest)
    return OpResult.ok(f"Renamed to {dest.name}.", undo={"op": "file.move_back", "args": {"moves": [[str(dest), str(p)]]}})


def _move_preview(a):
    p, d = _p(a, "src"), _p(a, "dest")
    return Preview(f"Move {p} → {d}") if p.exists() else Preview(f"{p} doesn't exist.", ok=False)


@gateway.operation("file.move", Risk.HIGH, "Move a file or folder", preview=_move_preview)
async def move(src: str, dest: str) -> OpResult:
    p, d = resolve(src), resolve(dest)
    if (err := _check(p, d)):
        return OpResult.fail(err)
    target = _unique(d / p.name if d.is_dir() else d)
    target.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(shutil.move, str(p), str(target))
    return OpResult.ok(f"Moved {p.name} to {target.parent}.", undo={"op": "file.move_back", "args": {"moves": [[str(target), str(p)]]}})


@gateway.operation("file.copy", Risk.MEDIUM, "Copy a file or folder", preview=lambda a: Preview(f"Copy {_p(a, 'src')} → {_p(a, 'dest')}"))
async def copy(src: str, dest: str) -> OpResult:
    p, d = resolve(src), resolve(dest)
    if (err := _check(d)):
        return OpResult.fail(err)
    target = _unique(d / p.name if d.is_dir() else d)
    if p.is_dir():
        await asyncio.to_thread(shutil.copytree, p, target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, p, target)
    return OpResult.ok(f"Copied to {target}.", undo={"op": "file.trash", "args": {"paths": [str(target)]}})


@gateway.operation("file.move_back", Risk.MEDIUM, "Undo moves/renames")
async def move_back(moves: List[List[str]]) -> OpResult:
    restored = 0
    for cur, original in reversed(moves):
        c, o = Path(cur), Path(original)
        if c.exists() and not o.exists():
            o.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(c), str(o))
            restored += 1
    return OpResult(restored > 0, f"Restored {restored} item(s) to where they were.")


def _trash_preview(a):
    paths = [resolve(x) for x in a.get("paths", [])]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        return Preview(f"Not found: {', '.join(missing[:5])}", ok=False)
    lines = "\n".join(f"   - {p}" for p in paths[:15])
    return Preview(f"Move {len(paths)} item(s) to the Recycle Bin:\n{lines}")


@gateway.operation("file.trash", Risk.HIGH, "Move files to the Recycle Bin (not permanent)", preview=_trash_preview)
async def trash(paths: List[str]) -> OpResult:
    try:
        from send2trash import send2trash
    except ImportError:
        return OpResult.fail("Deleting needs the 'send2trash' package (pip install send2trash) so nothing is lost permanently.")
    targets = [resolve(x) for x in paths]
    if (err := _check(*targets)):
        return OpResult.fail(err)
    for t in targets:
        await asyncio.to_thread(send2trash, str(t))
    return OpResult.ok(f"Moved {len(targets)} item(s) to the Recycle Bin (restore them from there if needed).")


# ── Bulk operations ────────────────────────────────────────────────────────

def category_for(path: Path) -> str:
    ext = path.suffix.lower()
    for cat, exts in CATEGORIES.items():
        if ext in exts:
            return cat
    return "Other"


def organize_plan(folder: Path, by: str = "type") -> List[tuple[Path, Path]]:
    moves = []
    for f in sorted(folder.iterdir()):
        if not f.is_file() or f.name.startswith(".") or f.suffix.lower() in (".crdownload", ".part", ".tmp"):
            continue
        sub = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m") if by == "date" else category_for(f)
        moves.append((f, folder / sub / f.name))
    return moves


def _organize_preview(a):
    folder = _p(a, "folder")
    if not folder.is_dir():
        return Preview(f"{folder} isn't a folder.", ok=False)
    moves = organize_plan(folder, a.get("by", "type"))
    if not moves:
        return Preview(f"{folder} has no loose files to organize.", ok=False)
    counts: Dict[str, int] = {}
    for _, dest in moves:
        counts[dest.parent.name] = counts.get(dest.parent.name, 0) + 1
    summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    return Preview(f"Organize {len(moves)} files in {folder} by {a.get('by', 'type')} into subfolders ({summary}). You can say 'undo' afterwards.")


@gateway.operation("files.organize", Risk.HIGH, "Sort a folder's files into subfolders by type or month", preview=_organize_preview)
async def organize(folder: str, by: str = "type") -> OpResult:
    root = resolve(folder)
    if (err := _check(root)):
        return OpResult.fail(err)
    done: List[List[str]] = []
    for src, dest in organize_plan(root, by):
        dest = _unique(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.move, str(src), str(dest))
        done.append([str(dest), str(src)])
    return OpResult.ok(f"Organized {len(done)} files in {root.name}.", undo={"op": "file.move_back", "args": {"moves": done}})


def bulk_rename_plan(folder: Path, find: str, replace: str = "", template: str = "", pattern: str = "*") -> List[tuple[Path, Path]]:
    plan = []
    for i, f in enumerate(sorted(p for p in folder.glob(pattern) if p.is_file()), 1):
        if template:
            new = template.replace("{n}", f"{i:03d}").replace("{name}", f.stem).replace("{ext}", f.suffix.lstrip(".")) + ("" if "{ext}" in template else f.suffix)
        elif find:
            new = re.sub(re.escape(find), replace, f.name, flags=re.I)
        else:
            continue
        if new != f.name:
            plan.append((f, f.with_name(new)))
    return plan


def _bulk_preview(a):
    folder = _p(a, "folder")
    plan = bulk_rename_plan(folder, a.get("find", ""), a.get("replace", ""), a.get("template", ""), a.get("pattern", "*"))
    if not plan:
        return Preview("No files would be renamed.", ok=False)
    lines = "\n".join(f"   - {s.name} → {d.name}" for s, d in plan[:12])
    more = f"\n   … and {len(plan) - 12} more" if len(plan) > 12 else ""
    return Preview(f"Rename {len(plan)} files in {folder.name}:\n{lines}{more}")


@gateway.operation("files.bulk_rename", Risk.MEDIUM, "Rename many files (find/replace or a template like 'photo_{n}')", preview=_bulk_preview)
async def bulk_rename(folder: str, find: str = "", replace: str = "", template: str = "", pattern: str = "*") -> OpResult:
    root = resolve(folder)
    if (err := _check(root)):
        return OpResult.fail(err)
    done = []
    for src, dest in bulk_rename_plan(root, find, replace, template, pattern):
        dest = _unique(dest)
        src.rename(dest)
        done.append([str(dest), str(src)])
    return OpResult.ok(f"Renamed {len(done)} files.", undo={"op": "file.move_back", "args": {"moves": done}})


@gateway.operation("files.zip", Risk.MEDIUM, "Compress files/folders into a .zip", preview=lambda a: Preview(f"Zip {len(a.get('paths', []))} item(s) into {_p(a, 'dest')}"))
async def make_zip(paths: List[str], dest: str) -> OpResult:
    out = resolve(dest)
    out = out if out.suffix == ".zip" else out.with_suffix(".zip")
    if (err := _check(out)):
        return OpResult.fail(err)
    out = _unique(out)

    def _zip():
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for raw in paths:
                p = resolve(raw)
                if p.is_dir():
                    for f in p.rglob("*"):
                        if f.is_file():
                            zf.write(f, f.relative_to(p.parent))
                elif p.exists():
                    zf.write(p, p.name)

    await asyncio.to_thread(_zip)
    return OpResult.ok(f"Created {out} ({out.stat().st_size // 1024} KB).", undo={"op": "file.trash", "args": {"paths": [str(out)]}})


@gateway.operation("files.unzip", Risk.MEDIUM, "Extract a .zip archive", preview=lambda a: Preview(f"Extract {_p(a, 'archive')} into {a.get('dest') or 'a folder next to it'}"))
async def unzip(archive: str, dest: str = "") -> OpResult:
    src = resolve(archive)
    target = resolve(dest) if dest else src.with_suffix("")
    if (err := _check(target)):
        return OpResult.fail(err)
    target = _unique(target) if not dest else target

    def _extract():
        with zipfile.ZipFile(src) as zf:
            for member in zf.namelist():  # zip-slip protection
                if not (target / member).resolve().is_relative_to(target.resolve()):
                    raise ValueError(f"unsafe path in archive: {member}")
            zf.extractall(target)

    try:
        await asyncio.to_thread(_extract)
    except (zipfile.BadZipFile, ValueError) as exc:
        return OpResult.fail(f"Couldn't extract: {exc}")
    return OpResult.ok(f"Extracted to {target}.", undo={"op": "file.trash", "args": {"paths": [str(target)]}})
