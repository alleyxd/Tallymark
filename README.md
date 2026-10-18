# Tallymark

[![tests](https://github.com/alleyxd/Tallymark/actions/workflows/tests.yml/badge.svg)](https://github.com/alleyxd/Tallymark/actions/workflows/tests.yml)

Turn a photo or scan of a receipt or invoice into validated JSON with a vision-language model running locally.
The schema is enforced during decoding, so the output always parses, and the result is checked with Pydantic before
it is returned.

```
$ tallymark examples/cord-test-3.jpg
{
 "file": "examples/cord-test-3.jpg",
 "seconds": 1.37,
 "merchant": "Dynamic Bakery & Cafe",
 "date": null,
 "items": [
  {
   "name": "DONAT GULA",
   "quantity": 1.0,
   "price": "11,000"
  }
 ],
 "subtotal": "11,000",
 "tax": null,
 "total": "11,000"
}
```

A real run with Qwen3-VL-4B on a receipt from the CORD-v2 test set (`examples/cord-test-3.jpg`, CC BY 4.0, from
[clovaai/cord](https://github.com/clovaai/cord)). The labelled answer is one "DONAT GULA" at 11,000 with a total of
11,000; the date is not printed on that receipt, so the model returns null instead of inventing one.

## How it works

1. The image (or the first page of a PDF) is resized and sent to an OpenAI-compatible vision model.
2. The request carries the JSON schema of the target type (`Receipt` or `Invoice` in `tallymark/schemas.py`) as a
   `json_schema` response format, so the server only lets the model produce output that fits the schema.
3. The reply is validated with Pydantic and returned as typed data; anything that fails validation is reported,
   never silently passed on.
4. The numbers are checked against each other: line items against the subtotal, subtotal plus tax against the
   total, and quantity times unit price on invoice lines. A schema guarantees the shape of the output, not that the
   digits were read correctly, so any disagreement sets `needs_review` and lists the problem in `issues`.

Adding a document type is a Pydantic model: field descriptions are part of the schema the model sees.

## Benchmark: CORD-v2 receipts

[CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) test set: 100 photographed receipts with labelled
items and totals. `python evals/cord.py --label NAME [--free]`. Served with vLLM on one RTX 5090.

| Model | Schema in the prompt | Schema enforced | Valid JSON | Total | Item F1 | Item count | Median time |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B | no | yes | 99% | 92.6% | 0.904 | 83% | 1.17 s |
| Qwen3-VL-8B | yes | no | 100% | 95.8% | 0.923 | 80% | 1.44 s |
| **Qwen3-VL-8B** | **yes** | **yes** | **100%** | **95.8%** | **0.927** | 81% | 1.50 s |
| Qwen3-VL-4B | yes | no | 100% | 95.8% | 0.876 | 80% | 0.88 s |
| **Qwen3-VL-4B** | **yes** | **yes** | **100%** | **95.8%** | 0.886 | 80% | **0.91 s** |

- **Valid JSON**: the output parsed and passed schema validation.
- **Total**: the extracted total matches the labelled total (digits compared).
- **Item F1**: item names matched to the labelled ones (fuzzy match at 0.8 similarity).
- **Item count**: exactly the labelled number of line items.

What the benchmark showed:

- **Enforcing the schema is not enough on its own.** The first version only passed the schema as a decoding
  constraint. The output had the right shape, but the model never saw the field descriptions: on two receipts it put
  the amount before tax or service charge into `total`, and one reply never finished (it ran for 22 seconds and did
  not produce complete JSON). Putting the schema in the prompt as well fixed both: same guarantee, better content.
- **The 4B model is the better default.** It reads totals as well as the 8B model and is about 40% faster; the 8B
  model is better at item names (F1 0.927 against 0.886), which matters if you need the line items.
- **Item count is the hard part.** Receipts with discount lines, set menus and sub-items are where both models and
  the labels disagree most.

## Quick start

```bash
docker run -d --gpus all -p 8000:8000 --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest --model Qwen/Qwen3-VL-4B-Instruct --served-model-name local \
  --max-model-len 8192 --gpu-memory-utilization 0.75 --limit-mm-per-prompt '{"image": 1}'

uv venv && uv pip install -e ".[dev]"
tallymark path/to/receipt.jpg
tallymark path/to/invoice.pdf --kind invoice
uvicorn tallymark.api:app --port 8070     # POST /extract with a file upload
```

Model and endpoint come from `TALLYMARK_BASE_URL`, `TALLYMARK_MODEL` and `TALLYMARK_API_KEY`; any OpenAI-compatible vision
model works. Hosted APIs that support `json_schema` response formats get the same guarantee.

## Tests

```bash
pytest -q
```

## License

MIT
