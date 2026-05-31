"""
Routing unit tests — no API key or real PDF required.

Tests:
  1. _is_meaningful_text  thresholds
  2. Text-PDF route  (pdfplumber returns rich text → GPT-4o text path)
  3. Vision route    (pdfplumber returns empty     → GPT-4o vision path)
  4. OCR fallback    (vision raises               → Tesseract path)
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "dummy-key-for-tests")

from app.extractor import (
    _is_meaningful_text,
    _extract_with_retry,
    extract_invoice_from_pdf,
)


RICH_TEXT = (
    "INVOICE #INV-2024-001\n"
    "Vendor: Acme Corporation, 123 Main St, Springfield IL\n"
    "Date: 2024-01-15    Due: 2024-02-15\n"
    "Widget A   x2   $50.00   $100.00\n"
    "Widget B   x1   $75.00    $75.00\n"
    "Subtotal $175.00  Tax $17.50  Total $192.50\n"
) * 3  # repeat so length >> 100


class TestIsMeaningfulText(unittest.TestCase):
    def test_rich_text_is_meaningful(self):
        self.assertTrue(_is_meaningful_text(RICH_TEXT))

    def test_empty_string(self):
        self.assertFalse(_is_meaningful_text(""))

    def test_whitespace_only(self):
        self.assertFalse(_is_meaningful_text("   \n\n  "))

    def test_below_length_threshold(self):
        self.assertFalse(_is_meaningful_text("Invoice #001"))

    def test_garbled_binary(self):
        garbled = "\x00\x01\x02\x03\x04\x05" * 50
        self.assertFalse(_is_meaningful_text(garbled))

    def test_borderline_length(self):
        # exactly 100 printable chars should pass
        text = "A" * 100
        self.assertTrue(_is_meaningful_text(text))


class TestRouting(unittest.TestCase):
    FAKE_RESULT = MagicMock()

    @patch("app.extractor._extract_with_retry", return_value=MagicMock())
    @patch("app.extractor._extract_pdfplumber", return_value=RICH_TEXT)
    def test_text_route_for_rich_pdf(self, mock_plumber, mock_retry):
        with self.assertLogs("app.extractor", level="INFO") as cm:
            extract_invoice_from_pdf(Path("invoice.pdf"))

        log_line = next(l for l in cm.output if "Route" in l)
        self.assertIn("text", log_line)
        self.assertNotIn("vision", log_line)
        mock_retry.assert_called_once()

    @patch("app.ocr.pdf_to_images", return_value=[MagicMock()])
    @patch("app.extractor._extract_with_retry", return_value=MagicMock())
    @patch("app.extractor._extract_pdfplumber", return_value="")
    def test_vision_route_for_empty_pdf(self, mock_plumber, mock_retry, mock_images):
        with self.assertLogs("app.extractor", level="INFO") as cm:
            extract_invoice_from_pdf(Path("scanned.pdf"))

        log_line = next(l for l in cm.output if "Route" in l)
        self.assertIn("vision", log_line)

    @patch("app.ocr.ocr_images", return_value=RICH_TEXT)
    @patch("app.ocr.pdf_to_images", return_value=[MagicMock()])
    @patch("app.extractor._extract_with_retry")
    @patch("app.extractor._extract_pdfplumber", return_value="")
    def test_ocr_fallback_when_vision_fails(
        self, mock_plumber, mock_retry, mock_images, mock_ocr
    ):
        # First call (vision) raises; second call (OCR text) succeeds
        mock_retry.side_effect = [Exception("vision API error"), MagicMock()]

        with self.assertLogs("app.extractor", level="WARNING") as cm:
            extract_invoice_from_pdf(Path("scanned.pdf"))

        self.assertTrue(any("OCR fallback" in l for l in cm.output))
        self.assertEqual(mock_retry.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
