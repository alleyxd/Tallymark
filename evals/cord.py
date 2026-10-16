"""Benchmark on the CORD-v2 test set (100 photographed receipts with labelled fields).

total     : extracted total equals the labelled total_price (digits compared)
items F1  : item names matched against labelled menu names (normalised, fuzzy at 0.8)
item count: same number of items as the label
valid     : output parsed and validated against the schema

python evals/cord.py --label qwen3-vl-8b [--free] [--limit 100]
"""
import argparse
import difflib
import json
import re
import statistics
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tallymark.extract import Extractor  # noqa: E402


def digits(s) -> str:
    return re.sub(r"\D", "", str(s or ""))


def name(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def item_f1(pred: list[str], gold: list[str]) -> float:
    pred, gold = [name(p) for p in pred if name(p)], [name(g) for g in gold if name(g)]
    if not pred and not gold:
        return 1.0
    left, hit = list(gold), 0
    for p in pred:
        best = max(left, key=lambda g: difflib.SequenceMatcher(None, p, g).ratio(), default=None)
        if best is not None and difflib.SequenceMatcher(None, p, best).ratio() >= 0.8:
            hit += 1
            left.remove(best)
    if not hit:
        return 0.0
    prec, rec = hit / len(pred), hit / len(gold)
    return 2 * prec * rec / (prec + rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--free", action="store_true")
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    ds = load_dataset("naver-clova-ix/cord-v2", split="test").select(range(args.limit))
    ex = Extractor()
    rows = []
    for i, r in enumerate(ds):
        gt = json.loads(r["ground_truth"])["gt_parse"]
        menu = gt.get("menu", [])
        menu = menu if isinstance(menu, list) else [menu]
        gold_items = [m.get("nm", "") for m in menu]
        gold_total = (gt.get("total") or {}).get("total_price")
        res = ex.extract(r["image"], "receipt", constrained=not args.free)
        d = res.data
        rows.append({
            "i": i, "valid": res.valid, "seconds": round(res.seconds, 2),
            "total_ok": bool(d) and gold_total is not None and digits(d.total) == digits(gold_total),
            "items_f1": item_f1([it.name for it in d.items], gold_items) if d else 0.0,
            "count_ok": bool(d) and len(d.items) == len(gold_items),
            "pred_total": d.total if d else None, "gold_total": gold_total,
        })
        print(i, rows[-1]["valid"], rows[-1]["total_ok"], round(rows[-1]["items_f1"], 2), rows[-1]["seconds"], flush=True)

    with_total = [r for r in rows if r["gold_total"] is not None]
    summary = {"label": args.label, "decoding": "free" if args.free else "constrained", "n": len(rows),
               "valid_json": round(statistics.mean(r["valid"] for r in rows), 3),
               "total_accuracy": round(statistics.mean(r["total_ok"] for r in with_total), 3),
               "item_name_f1": round(statistics.mean(r["items_f1"] for r in rows), 3),
               "item_count_accuracy": round(statistics.mean(r["count_ok"] for r in rows), 3),
               "median_seconds": statistics.median(r["seconds"] for r in rows)}
    print(json.dumps(summary, indent=1))
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / f"{args.label}-{summary['decoding']}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1),
                                                                  encoding="utf-8")


if __name__ == "__main__":
    main()
