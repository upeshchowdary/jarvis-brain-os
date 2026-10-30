"""Document tools: summarise a document, ask questions over a folder (local RAG), tables → Excel, convert."""

from __future__ import annotations

import asyncio
from typing import Any

from automation.gateway import ActionRequest, Risk
from automation.paths import resolve
from services.documents import ask_folder, extract_text
from tools.base import Tool, ToolContext, ToolResult, obj, via_gateway
from tools.registry import register


@register
class ReadDocument(Tool):
    name = "document_read"
    description = "Read a PDF/DOCX/PPTX/XLSX/text document so you can summarise it or answer questions about it."
    parameters = obj({"path": {"type": "string"}}, ["path"])
    untrusted_output = True
    sensitive_output = True
    final = False

    async def run(self, ctx: ToolContext, path: str = "", **_: Any) -> ToolResult:
        p = resolve(path)
        if not p.is_file():
            return ToolResult.error(f"I couldn't find {p}.")
        try:
            parts = await asyncio.to_thread(extract_text, p, 40000)
        except ImportError as exc:
            return ToolResult.error(f"Reading {p.suffix} files needs an extra package: {exc.name} (pip install .[docs]).")
        except Exception as exc:
            return ToolResult.error(f"Couldn't read {p.name}: {exc}")
        text = "\n\n".join(f"[{label}]\n{t}" if label else t for label, t in parts)
        return ToolResult.ok(f"Read {p.name} ({len(text)} characters).", content_for_llm=f"Document: {p.name}\n\n{text[:20000]}")


@register
class AskDocuments(Tool):
    name = "documents_ask"
    description = "Answer a question from the documents in a folder (local search over PDFs, Word, slides, notes). Cite file names."
    parameters = obj({"folder": {"type": "string"}, "question": {"type": "string"}}, ["folder", "question"])
    untrusted_output = True
    sensitive_output = True
    final = False

    async def run(self, ctx: ToolContext, folder: str = "documents", question: str = "", **_: Any) -> ToolResult:
        root = resolve(folder)
        if not root.is_dir():
            return ToolResult.error(f"{root} isn't a folder.")
        hits, count = await asyncio.to_thread(ask_folder, root, question)
        if not hits:
            return ToolResult.ok(f"I searched {count} passages in {root.name} but found nothing relevant to that.")
        context = "\n\n".join(f"[{i}] {h['file']}{' ' + h['loc'] if h['loc'] else ''}:\n{h['text']}" for i, h in enumerate(hits, 1))
        return ToolResult.ok(f"Found {len(hits)} relevant passages in {root.name}.", data=hits, content_for_llm=context)


@register
class ExtractTables(Tool):
    name = "extract_tables_to_excel"
    description = "Extract the tables from a DOCX/PDF/Markdown/CSV document into an Excel workbook."
    parameters = obj({"path": {"type": "string"}, "dest": {"type": "string"}}, ["path"])
    side_effect = True
    risk = Risk.MEDIUM

    async def run(self, ctx: ToolContext, path: str = "", dest: str = "", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("docs.extract_tables", {"path": path, "dest": dest}))


@register
class ConvertDocument(Tool):
    name = "convert_document"
    description = "Convert a document: pdf/docx/pptx/xlsx → txt or md, txt/md → docx, csv ↔ xlsx."
    parameters = obj({"path": {"type": "string"}, "to": {"type": "string", "enum": ["txt", "md", "docx", "xlsx", "csv"]}}, ["path", "to"])
    side_effect = True
    risk = Risk.MEDIUM

    async def run(self, ctx: ToolContext, path: str = "", to: str = "txt", **_: Any) -> ToolResult:
        return await via_gateway(ctx, ActionRequest("docs.convert", {"path": path, "to": to}))
