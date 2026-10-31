"""Arithmetic checks on an extraction: do the lines add up to the subtotal, and subtotal plus tax to the total?

A schema guarantees the shape of the output, not that the numbers are right. When the printed numbers disagree with
each other, something was misread, and the result should be looked at by a person.
"""
import re
from decimal import Decimal, InvalidOperation

from .schemas import Invoice, Receipt


def parse_amount(text) -> Decimal | None:
    """Turn a printed amount into a number: '1,234.56', '1.234,56', 'Rp 60.000', '$12', '-4.50'. None if unreadable."""
    if text is None:
        return None
    s = re.sub(r"[^\d,.\-]", "", str(text))
    neg = s.startswith("-")
    s = s.replace("-", "")
    if not re.search(r"\d", s):
        return None
    if "," in s and "." in s:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        s = s.replace("." if dec == "," else ",", "").replace(dec, ".")
    elif "," in s or "." in s:
        sep = "," if "," in s else "."
        head, _, tail = s.rpartition(sep)
        # thousands groups always have three digits, so one separator followed by one or two digits, or by
        # anything after a leading zero ('0.500'), is a decimal point
        if s.count(sep) == 1 and (len(tail) in (1, 2) or head in ("", "0")):
            s = head.replace(sep, "") + "." + tail
        else:
            s = s.replace(sep, "")
    try:
        value = Decimal(s)
    except InvalidOperation:
        return None
    return -value if neg else value


def _close(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) <= max(Decimal("0.02"), abs(b) * Decimal("0.01"))


def check(doc) -> list[str]:
    """Return a list of human-readable problems; an empty list means the numbers agree (or there was nothing to check)."""
    if isinstance(doc, Receipt):
        lines = [parse_amount(i.price) for i in doc.items]
    elif isinstance(doc, Invoice):
        lines = [parse_amount(l.amount) for l in doc.lines]
    else:
        return []
    issues = []
    subtotal, tax, total = parse_amount(doc.subtotal), parse_amount(doc.tax), parse_amount(doc.total)
    known = [x for x in lines if x is not None]
    if known and len(known) == len(lines):
        line_sum = sum(known)
        target, name = (subtotal, "subtotal") if subtotal is not None else (total if tax is None else None, "total")
        if target is not None and not _close(line_sum, target):
            issues.append(f"line items add up to {line_sum}, but the {name} reads {target}")
    if subtotal is not None and total is not None:
        expected = subtotal + (tax or 0)
        if not _close(expected, total):
            what = "subtotal plus tax" if tax is not None else "subtotal"
            issues.append(f"{what} is {expected}, but the total reads {total}")
    if isinstance(doc, Invoice):
        for l in doc.lines:
            q, unit, amount = l.quantity, parse_amount(l.unit_price), parse_amount(l.amount)
            if q is not None and unit is not None and amount is not None and not _close(Decimal(str(q)) * unit, amount):
                issues.append(f"'{l.description}': {q} x {unit} is not {amount}")
    return issues
