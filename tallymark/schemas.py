from pydantic import BaseModel, Field


class Item(BaseModel):
    name: str = Field(description="item name exactly as printed")
    quantity: float | None = Field(None, description="number of units, if printed")
    price: str | None = Field(None, description="line total as printed, digits and separators only")


class Receipt(BaseModel):
    merchant: str | None = Field(None, description="store or restaurant name")
    date: str | None = Field(None, description="purchase date as printed")
    items: list[Item] = Field(default_factory=list, description="every purchased line item, in order")
    subtotal: str | None = None
    tax: str | None = None
    total: str | None = Field(None, description="final amount paid, as printed")


class Party(BaseModel):
    name: str | None = None
    address: str | None = None


class InvoiceLine(BaseModel):
    description: str
    quantity: float | None = None
    unit_price: str | None = None
    amount: str | None = None


class Invoice(BaseModel):
    invoice_number: str | None = None
    issue_date: str | None = None
    due_date: str | None = None
    seller: Party = Field(default_factory=Party)
    buyer: Party = Field(default_factory=Party)
    lines: list[InvoiceLine] = Field(default_factory=list)
    subtotal: str | None = None
    tax: str | None = None
    total: str | None = None
    currency: str | None = None


SCHEMAS = {"receipt": Receipt, "invoice": Invoice}
