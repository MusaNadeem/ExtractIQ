from pydantic import BaseModel
from typing import Optional
from datetime import date


class InvoiceSchema(BaseModel):
    invoice_number: Optional[str] = None
    vendor_name: Optional[str] = None
    issue_date: Optional[date] = None
    due_date: Optional[date] = None
    total_amount: Optional[float] = None
    currency: Optional[str] = None
    line_items: list[dict] = []
