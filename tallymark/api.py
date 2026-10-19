"""POST /extract with a file upload. Start with: uvicorn tallymark.api:app"""
import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from .extract import Extractor

app = FastAPI(title="tallymark")
extractor = Extractor()


@app.post("/extract")
async def extract(file: UploadFile = File(...), kind: str = "receipt"):
    if kind not in ("receipt", "invoice"):
        raise HTTPException(400, "kind must be receipt or invoice")
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "could not read the file as an image")
    r = extractor.extract(img, kind)
    if not r.valid:
        raise HTTPException(422, {"error": r.error})
    return {"kind": kind, "seconds": round(r.seconds, 2), "data": r.data.model_dump(),
            "needs_review": r.needs_review, "issues": r.issues}
