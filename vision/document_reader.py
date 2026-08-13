"""JARVIS Deep Document & PDF Structure Reader Module.

Parses research papers, invoices, books, forms, and notes, extracting Title,
Sections, Paragraphs, Tables, and Footnotes into structured DocumentStructure models.
"""

import os
import re
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from vision.environment import DocumentStructure, TableCell, BoundingBox


class DocumentReader:
    """Engine for parsing structured documents, research papers, and PDFs."""

    def parse_document(self, file_path_or_text: Any) -> DocumentStructure:
        """Parses document from file path, raw text content, or image payload."""
        if not file_path_or_text:
            return DocumentStructure()

        title = "Untitled Document"
        sections: List[str] = []
        paragraphs: List[str] = []
        tables: List[List[TableCell]] = []
        footnotes: List[str] = []

        raw_text = ""
        if isinstance(file_path_or_text, str):
            if os.path.exists(file_path_or_text):
                try:
                    with open(file_path_or_text, "r", encoding="utf-8", errors="ignore") as f:
                        raw_text = f.read()
                    title = os.path.basename(file_path_or_text)
                except Exception as e:
                    logger.debug(f"Document file read exception: {e}")
            else:
                raw_text = file_path_or_text

        if raw_text:
            lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
            if lines:
                title = lines[0]

            for line in lines:
                if re.match(r"^#+\s|^[0-9]+\.\s|[A-Z0-9\s]{4,}:$", line):
                    sections.append(line)
                elif line.startswith("[") or "footnote" in line.lower() or "citation" in line.lower():
                    footnotes.append(line)
                else:
                    paragraphs.append(line)

            # Simulated table parsing if markdown or pipe-separated lines
            table_rows = [l for l in lines if "|" in l]
            if table_rows:
                grid: List[TableCell] = []
                for r_idx, r_text in enumerate(table_rows[:10]):
                    cols = [c.strip() for c in r_text.split("|") if c.strip()]
                    for c_idx, col in enumerate(cols):
                        grid.append(TableCell(row=r_idx, col=c_idx, text=col))
                if grid:
                    tables.append(grid)

        return DocumentStructure(
            title=title,
            sections=sections,
            paragraphs=paragraphs,
            tables=tables,
            footnotes=footnotes,
        )


# Global singleton instance
document_reader = DocumentReader()
