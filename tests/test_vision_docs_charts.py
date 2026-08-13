"""Unit tests for vision.document_reader and vision.chart_reader modules."""

from vision.document_reader import document_reader
from vision.chart_reader import chart_reader
from vision.environment import DocumentStructure, ChartData


def test_document_reader_text_parsing():
    sample_text = """# Quarterly Financial Report

## Executive Summary
Revenue increased by 15% across all regions.

## Financial Table
| Metric | Q1 | Q2 |
| Revenue | $10M | $12M |

[1] Footnote: Audited figures.
"""
    doc = document_reader.parse_document(sample_text)
    assert isinstance(doc, DocumentStructure)
    assert doc.title == "# Quarterly Financial Report"
    assert len(doc.sections) > 0
    assert len(doc.tables) > 0
    assert len(doc.footnotes) > 0


def test_chart_reader_parsing():
    ocr_sample = "2026 Sales Revenue Trend Line Chart\nMonth vs USD\nSeries A"
    chart = chart_reader.parse_chart(None, ocr_text=ocr_sample)

    assert isinstance(chart, ChartData)
    assert chart.chart_type == "line"
    assert "Sales Revenue" in chart.title
