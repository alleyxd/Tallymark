import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .schemas import SCHEMAS

INSTRUCTION = ("Extract the {kind} in this image into JSON that matches the schema. Copy names and amounts exactly "
               "as printed. Use null for anything that is not on the document; do not guess.")


@dataclass
class Extraction:
    data: BaseModel | None
    raw: str
    valid: bool
    seconds: float
    error: str | None = None


def image_url(source) -> str:
    """Accepts a path (image or PDF; PDFs use the first page) or a PIL image, returns a base64 data URL."""
    from PIL import Image
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() == ".pdf":
            import pypdfium2
            img = pypdfium2.PdfDocument(path)[0].render(scale=2).to_pil()
        else:
            img = Image.open(path)
    else:
        img = source
    img = img.convert("RGB")
    img.thumbnail((1600, 1600))   # plenty for printed text, keeps the request small
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class Extractor:
    def __init__(self, base_url=None, model=None, api_key=None):
        self.llm = OpenAI(base_url=base_url or os.environ.get("TALLYMARK_BASE_URL", "http://localhost:8000/v1"),
                          api_key=api_key or os.environ.get("TALLYMARK_API_KEY", "local"))
        self.model = model or os.environ.get("TALLYMARK_MODEL", "local")

    def extract(self, source, kind: str = "receipt", constrained: bool = True) -> Extraction:
        schema = SCHEMAS[kind]
        # the schema goes in the prompt either way: constrained decoding enforces its shape, but the model only
        # learns what a field means (e.g. "final amount paid" for total) by reading the descriptions
        text = (INSTRUCTION.format(kind=kind) + "\n\nJSON schema:\n" + json.dumps(schema.model_json_schema())
                + "\n\nReply with the JSON only.")
        kw = {}
        if constrained:
            # the server only lets the model produce tokens that keep the output valid against the schema
            kw["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": kind, "schema": schema.model_json_schema()}}
        t = time.time()
        reply = self.llm.chat.completions.create(
            model=self.model, temperature=0, max_tokens=2048,
            messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url(source)}},
                                                   {"type": "text", "text": text}]}], **kw)
        raw = reply.choices[0].message.content or ""
        secs = time.time() - t
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            return Extraction(schema.model_validate_json(m.group(0) if m else raw), raw, True, secs)
        except (ValidationError, ValueError) as e:
            return Extraction(None, raw, False, secs, str(e)[:300])
