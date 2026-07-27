from pathlib import Path
import io

import fitz
import pytesseract
from fastapi import FastAPI, File, UploadFile
from PIL import Image

app = FastAPI(title="local reimbursement OCR")


def recognize(data: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf") or data[:4] == b"%PDF":
        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            pages.append(pytesseract.image_to_string(Image.open(io.BytesIO(pix.tobytes("png"))), lang="chi_sim+eng"))
        return "\n\n".join(pages).strip()
    return pytesseract.image_to_string(Image.open(io.BytesIO(data)), lang="chi_sim+eng").strip()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    text = recognize(await file.read(), file.filename or "upload")
    return {"success": True, "output": text}
