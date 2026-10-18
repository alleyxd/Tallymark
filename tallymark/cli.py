import argparse
import csv
import json
from pathlib import Path

from .extract import Extractor

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".pdf"}
CSV_FIELDS = ["file", "kind", "valid", "needs_review", "seconds", "name", "date", "subtotal", "tax", "total", "lines",
              "issues", "error"]


def expand(paths):
    """Files as given, plus every supported document inside any folder, in a stable order."""
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(f for f in p.rglob("*") if f.suffix.lower() in SUFFIXES)
        else:
            yield p


def record(path, kind, r) -> dict:
    out = {"file": str(path), "kind": kind, "valid": r.valid, "needs_review": r.needs_review,
           "seconds": round(r.seconds, 2)}
    if r.valid:
        out.update(r.data.model_dump(), issues=r.issues or [])
    else:
        out.update({"error": r.error, "raw": r.raw[:500]})
    return out


def csv_row(rec: dict) -> dict:
    """One flat row per document: the header fields, and the number of line items."""
    lines = rec.get("items", rec.get("lines")) or []
    return {"file": rec["file"], "kind": rec["kind"], "valid": rec["valid"], "needs_review": rec["needs_review"],
            "seconds": rec["seconds"], "issues": "; ".join(rec.get("issues") or []),
            "name": rec.get("merchant") or (rec.get("seller") or {}).get("name"),
            "date": rec.get("date") or rec.get("issue_date"), "subtotal": rec.get("subtotal"), "tax": rec.get("tax"),
            "total": rec.get("total"), "lines": len(lines), "error": rec.get("error")}


def main():
    ap = argparse.ArgumentParser(prog="tallymark", description="Turn receipts or invoices into validated JSON.")
    ap.add_argument("files", nargs="+", help="images, PDFs, or folders of them")
    ap.add_argument("--kind", default="receipt", choices=["receipt", "invoice"])
    ap.add_argument("--free", action="store_true", help="ask for JSON in the prompt instead of constraining decoding")
    ap.add_argument("--out", help="write all results to a .jsonl (full output) or .csv (one row per document) file")
    args = ap.parse_args()
    if args.out and Path(args.out).suffix.lower() not in (".jsonl", ".csv"):
        ap.error("--out must end in .jsonl or .csv")

    ex = Extractor()
    records = []
    for f in expand(args.files):
        rec = record(f, args.kind, ex.extract(f, args.kind, constrained=not args.free))
        records.append(rec)
        if not args.out:
            print(json.dumps(rec, indent=1, ensure_ascii=False))
        else:
            print(f"{rec['file']}: {'ok' if rec['valid'] else 'invalid'} ({rec['seconds']} s)", flush=True)
    if args.out:
        write(records, Path(args.out))
        print(f"wrote {len(records)} results to {args.out}")


def write(records, out: Path):
    if out.suffix.lower() == ".csv":
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(csv_row(r) for r in records)
    else:
        with out.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
