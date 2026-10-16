import argparse
import json

from .extract import Extractor


def main():
    ap = argparse.ArgumentParser(prog="tallymark", description="Turn a receipt or invoice image into validated JSON.")
    ap.add_argument("files", nargs="+", help="images or PDFs")
    ap.add_argument("--kind", default="receipt", choices=["receipt", "invoice"])
    ap.add_argument("--free", action="store_true", help="ask for JSON in the prompt instead of constraining decoding")
    args = ap.parse_args()
    ex = Extractor()
    for f in args.files:
        r = ex.extract(f, args.kind, constrained=not args.free)
        out = r.data.model_dump() if r.valid else {"error": r.error, "raw": r.raw[:500]}
        print(json.dumps({"file": f, "seconds": round(r.seconds, 2), **out}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
