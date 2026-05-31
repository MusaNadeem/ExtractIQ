from pydantic import BaseModel
from typing import Optional
from datetime import date


class ReceiptSchema(BaseModel):
    merchant_name: Optional[str] = None
    transaction_date: Optional[date] = None
    total_amount: Optional[float] = None
    tax_amount: Optional[float] = None
    currency: Optional[str] = None
    items: list[dict] = []
    payment_method: Optional[str] = None
