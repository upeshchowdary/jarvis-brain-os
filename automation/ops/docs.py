"""Document write operations (table extraction to Excel, format conversion)."""

from __future__ import annotations

import asyncio

from automation.gateway import OpResult, Preview, Risk, gateway
from automation.paths import resolve


@gateway.operation("docs.extract_tables", Risk.MEDIUM, "Extract tables from a document into an Excel file",
                   preview=lambda a: Preview(f"Extract tables from {resolve(a.get('path', '')).name} into {a.get('dest') or 'an .xlsx next to it'}"))
async def extract_tables_op(path: str, dest: str = "") -> OpResult:
    from services.documents import extract_tables, write_xlsx

    src = resolve(path)
    out = resolve(dest) if dest else src.with_name(src.stem + "_tables.xlsx")
    try:
        tables = await asyncio.to_thread(extract_tables, src)
        if not tables:
            return OpResult.fail(f"No tables found in {src.name}.")
        await asyncio.to_thread(write_xlsx, tables, out)
    except Exception as exc:
        return OpResult.fail(str(exc))
    return OpResult.ok(f"Saved {len(tables)} table(s) to {out}.", undo={"op": "file.trash", "args": {"paths": [str(out)]}})


@gateway.operation("docs.convert", Risk.MEDIUM, "Convert a document to another format",
                   preview=lambda a: Preview(f"Convert {resolve(a.get('path', '')).name} to .{str(a.get('to', '')).lstrip('.')}"))
async def convert_op(path: str, to: str) -> OpResult:
    from services.documents import convert

    src = resolve(path)
    try:
        out = await asyncio.to_thread(convert, src, to)
    except Exception as exc:
        return OpResult.fail(str(exc))
    return OpResult.ok(f"Converted to {out}.", undo={"op": "file.trash", "args": {"paths": [str(out)]}})
