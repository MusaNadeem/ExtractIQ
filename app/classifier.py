from enum import Enum


class DocumentType(str, Enum):
    INVOICE = "invoice"
    RESUME = "resume"
    RECEIPT = "receipt"
    UNKNOWN = "unknown"


def classify_document(text: str) -> DocumentType:
    raise NotImplementedError
