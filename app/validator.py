import logging
from dataclasses import dataclass, field

from app.schemas.invoice import InvoiceData

log = logging.getLogger(__name__)

# Floating-point / rounding tolerance for money comparisons
_MONEY_ABS_TOL = 0.02       # $0.02 absolute
_MONEY_REL_TOL = 0.02       # 2% relative (whichever is larger wins)

_MIN_YEAR = 1990
_MAX_YEAR = 2040


@dataclass
class ValidationResult:
    data: InvoiceData
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _near(actual: float, expected: float) -> bool:
    """True when actual is within absolute or relative tolerance of expected."""
    return abs(actual - expected) <= max(_MONEY_ABS_TOL, abs(expected) * _MONEY_REL_TOL)


def _check_dates(data: InvoiceData, warnings: list[str], flagged: set[str]) -> None:
    for fname in ("invoice_date", "due_date"):
        d = getattr(data, fname)
        if d is not None and not (_MIN_YEAR <= d.year <= _MAX_YEAR):
            warnings.append(
                f"{fname}: year {d.year} is outside expected range "
                f"({_MIN_YEAR}–{_MAX_YEAR})"
            )
            flagged.add(fname)

    if data.invoice_date and data.due_date and data.due_date < data.invoice_date:
        warnings.append(
            f"due_date ({data.due_date}) is earlier than invoice_date ({data.invoice_date})"
        )
        flagged.add("due_date")


def _check_numerics(data: InvoiceData, warnings: list[str], flagged: set[str]) -> None:
    for fname in ("subtotal", "tax", "total"):
        v = getattr(data, fname)
        if v is not None and v < 0:
            warnings.append(f"{fname} is negative ({v})")
            flagged.add(fname)


def _check_line_items(data: InvoiceData, warnings: list[str], flagged: set[str]) -> None:
    for i, item in enumerate(data.line_items):
        if item.total is not None and item.total < 0:
            warnings.append(f"line_items[{i}]: total is negative ({item.total})")
            flagged.add("line_items")

        # qty × unit_price should match the per-item total
        if (
            item.quantity is not None
            and item.unit_price is not None
            and item.total is not None
        ):
            expected = item.quantity * item.unit_price
            if not _near(item.total, expected):
                warnings.append(
                    f"line_items[{i}]: total {item.total:.2f} ≠ "
                    f"qty {item.quantity} × unit_price {item.unit_price:.2f} "
                    f"= {expected:.2f}"
                )
                flagged.add("line_items")


def _check_totals(data: InvoiceData, warnings: list[str], flagged: set[str]) -> None:
    # Sum of line item totals should match subtotal
    item_totals = [item.total for item in data.line_items if item.total is not None]
    if item_totals and data.subtotal is not None:
        items_sum = sum(item_totals)
        if not _near(items_sum, data.subtotal):
            warnings.append(
                f"line_items sum ({items_sum:.2f}) does not match "
                f"subtotal ({data.subtotal:.2f})"
            )
            flagged.add("subtotal")

    # subtotal + tax should match total
    if data.subtotal is not None and data.total is not None:
        expected_total = data.subtotal + (data.tax or 0.0)
        if not _near(data.total, expected_total):
            warnings.append(
                f"total ({data.total:.2f}) does not match "
                f"subtotal + tax ({expected_total:.2f})"
            )
            flagged.add("total")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_invoice(data: InvoiceData) -> ValidationResult:
    """
    Run semantic validation on extracted invoice data.

    Never raises — issues are collected as warnings and merged into
    uncertain_fields so downstream consumers can decide how to handle them.
    """
    warnings: list[str] = []
    flagged: set[str] = set(data.uncertain_fields)

    _check_dates(data, warnings, flagged)
    _check_numerics(data, warnings, flagged)
    _check_line_items(data, warnings, flagged)
    _check_totals(data, warnings, flagged)

    for w in warnings:
        log.warning("Validation: %s", w)

    validated = data.model_copy(update={"uncertain_fields": sorted(flagged)})
    return ValidationResult(data=validated, warnings=warnings)
