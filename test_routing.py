"""
Unit tests — no API key or real PDF required.

Covers:
  1. _is_meaningful_text  thresholds
  2. Text-PDF route  (pdfplumber returns rich text → GPT-4o text path)
  3. Vision route    (pdfplumber returns empty     → GPT-4o vision path)
  4. OCR fallback    (vision raises               → Tesseract path)
  5. validate_invoice — date, numeric, line-item, and total checks
"""

import os
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "dummy-key-for-tests")

from app.extractor import _is_meaningful_text, extract_invoice_from_pdf
from app.schemas.invoice import InvoiceData, LineItem
from app.validator import ValidationResult, validate_invoice


RICH_TEXT = (
    "INVOICE #INV-2024-001\n"
    "Vendor: Acme Corporation, 123 Main St, Springfield IL\n"
    "Date: 2024-01-15    Due: 2024-02-15\n"
    "Widget A   x2   $50.00   $100.00\n"
    "Widget B   x1   $75.00    $75.00\n"
    "Subtotal $175.00  Tax $17.50  Total $192.50\n"
) * 3


# ---------------------------------------------------------------------------
# _is_meaningful_text
# ---------------------------------------------------------------------------

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
        self.assertFalse(_is_meaningful_text("\x00\x01\x02\x03\x04\x05" * 50))

    def test_exactly_at_threshold(self):
        self.assertTrue(_is_meaningful_text("A" * 100))


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

class TestRouting(unittest.TestCase):
    _INVOICE = InvoiceData()   # bare instance passes validation cleanly

    @patch("app.extractor._extract_with_retry", return_value=InvoiceData())
    @patch("app.extractor._extract_pdfplumber", return_value=RICH_TEXT)
    def test_text_route_for_rich_pdf(self, _plumber, _retry):
        with self.assertLogs("app.extractor", level="INFO") as cm:
            result = extract_invoice_from_pdf(Path("invoice.pdf"))
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(any("Route → text" in l for l in cm.output))

    @patch("app.ocr.pdf_to_images", return_value=[MagicMock()])
    @patch("app.extractor._extract_with_retry", return_value=InvoiceData())
    @patch("app.extractor._extract_pdfplumber", return_value="")
    def test_vision_route_for_empty_pdf(self, _plumber, _retry, _images):
        with self.assertLogs("app.extractor", level="INFO") as cm:
            result = extract_invoice_from_pdf(Path("scanned.pdf"))
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(any("Route → vision" in l for l in cm.output))

    @patch("app.ocr.ocr_images", return_value=RICH_TEXT)
    @patch("app.ocr.pdf_to_images", return_value=[MagicMock()])
    @patch("app.extractor._extract_with_retry")
    @patch("app.extractor._extract_pdfplumber", return_value="")
    def test_ocr_fallback_when_vision_fails(self, _plumber, mock_retry, _images, _ocr):
        mock_retry.side_effect = [Exception("vision API error"), InvoiceData()]
        with self.assertLogs("app.extractor", level="WARNING") as cm:
            result = extract_invoice_from_pdf(Path("scanned.pdf"))
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(any("OCR fallback" in l for l in cm.output))
        self.assertEqual(mock_retry.call_count, 2)


# ---------------------------------------------------------------------------
# validate_invoice
# ---------------------------------------------------------------------------

class TestValidation(unittest.TestCase):

    def _make(self, **kwargs) -> InvoiceData:
        return InvoiceData(**kwargs)

    def test_clean_data_no_warnings(self):
        data = self._make(
            invoice_date=date(2024, 1, 15),
            due_date=date(2024, 2, 15),
            subtotal=175.0,
            tax=17.50,
            total=192.50,
            line_items=[
                LineItem(quantity=2, unit_price=50.0, total=100.0),
                LineItem(quantity=1, unit_price=75.0, total=75.0),
            ],
        )
        result = validate_invoice(data)
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.data.uncertain_fields, [])

    def test_due_date_before_invoice_date(self):
        data = self._make(
            invoice_date=date(2024, 3, 1),
            due_date=date(2024, 1, 1),
        )
        result = validate_invoice(data)
        self.assertTrue(any("earlier than invoice_date" in w for w in result.warnings))
        self.assertIn("due_date", result.data.uncertain_fields)

    def test_year_out_of_range(self):
        data = self._make(invoice_date=date(1850, 1, 1))
        result = validate_invoice(data)
        self.assertTrue(any("year 1850" in w for w in result.warnings))
        self.assertIn("invoice_date", result.data.uncertain_fields)

    def test_negative_total(self):
        data = self._make(subtotal=-10.0, total=-10.0)
        result = validate_invoice(data)
        self.assertTrue(any("negative" in w for w in result.warnings))
        self.assertIn("subtotal", result.data.uncertain_fields)

    def test_line_item_qty_price_mismatch(self):
        data = self._make(
            line_items=[LineItem(quantity=2, unit_price=50.0, total=90.0)]
        )
        result = validate_invoice(data)
        self.assertTrue(any("line_items[0]" in w for w in result.warnings))
        self.assertIn("line_items", result.data.uncertain_fields)

    def test_line_items_sum_mismatch_subtotal(self):
        data = self._make(
            line_items=[LineItem(total=100.0), LineItem(total=75.0)],
            subtotal=200.0,   # should be 175.0
        )
        result = validate_invoice(data)
        self.assertTrue(any("sum" in w and "subtotal" in w for w in result.warnings))
        self.assertIn("subtotal", result.data.uncertain_fields)

    def test_total_mismatch_subtotal_plus_tax(self):
        data = self._make(subtotal=175.0, tax=17.50, total=200.0)  # should be 192.50
        result = validate_invoice(data)
        self.assertTrue(any("total" in w and "subtotal + tax" in w for w in result.warnings))
        self.assertIn("total", result.data.uncertain_fields)

    def test_existing_uncertain_fields_preserved(self):
        data = self._make(uncertain_fields=["vendor_name"], total=-5.0)
        result = validate_invoice(data)
        self.assertIn("vendor_name", result.data.uncertain_fields)
        self.assertIn("total", result.data.uncertain_fields)

    def test_within_tolerance_no_false_positive(self):
        # $0.01 rounding — should not warn
        data = self._make(subtotal=175.0, tax=17.50, total=192.49)
        result = validate_invoice(data)
        total_warnings = [w for w in result.warnings if "total" in w and "subtotal" in w]
        self.assertEqual(total_warnings, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
