from __future__ import annotations

import asyncio
import io
import math
import time
import warnings
from pathlib import Path
from typing import Any

import fitz
import pytesseract
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_FILE_BYTES + 1024 * 1024
MAX_PDF_PAGES = 10
MAX_IMAGE_PIXELS = 16_000_000
MAX_TOTAL_RENDERED_PIXELS = 32_000_000
MAX_OUTPUT_CHARS = 100_000
MAX_CONCURRENT_OCR = 2
TESSERACT_TIMEOUT_SECONDS = 30
OCR_REQUEST_TIMEOUT_SECONDS = 60
PDF_RENDER_SCALE = 2
OCR_LANGUAGE = "chi_sim+eng"

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

CONTENT_TYPES = {
    "pdf": {"application/pdf"},
    "png": {"image/png"},
    "jpeg": {"image/jpeg", "image/jpg"},
    "webp": {"image/webp"},
}
FILE_EXTENSIONS = {
    "pdf": {".pdf"},
    "png": {".png"},
    "jpeg": {".jpg", ".jpeg"},
    "webp": {".webp"},
}


class OcrError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class RequestBodyTooLarge(Exception):
    pass


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": code, "message": message},
    )


class RequestBodyLimitMiddleware:
    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("path") != "/ocr":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        try:
            content_length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            content_length = 0
        if content_length > self.max_bytes:
            await _error_response(
                413, "REQUEST_TOO_LARGE", "The OCR request body is too large."
            )(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestBodyTooLarge:
            await _error_response(
                413, "REQUEST_TOO_LARGE", "The OCR request body is too large."
            )(scope, receive, send)


app = FastAPI(title="local reimbursement OCR")
app.add_middleware(RequestBodyLimitMiddleware, max_bytes=MAX_REQUEST_BYTES)
ocr_slots = asyncio.Semaphore(MAX_CONCURRENT_OCR)


@app.exception_handler(OcrError)
async def handle_ocr_error(_request: Request, error: OcrError) -> JSONResponse:
    return _error_response(error.status_code, error.code, error.message)


def _detect_content_kind(data: bytes) -> str:
    if data.startswith(b"%PDF-"):
        return "pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    raise OcrError(415, "UNSUPPORTED_CONTENT", "The uploaded file type is unsupported.")


def _validate_content(data: bytes, filename: str, content_type: str) -> str:
    if not data:
        raise OcrError(422, "EMPTY_FILE", "The uploaded file is empty.")

    kind = _detect_content_kind(data)
    extension = Path(filename).suffix.casefold()
    normalized_content_type = content_type.partition(";")[0].strip().casefold()
    extension_matches = not extension or extension in FILE_EXTENSIONS[kind]
    type_matches = (
        not normalized_content_type
        or normalized_content_type == "application/octet-stream"
        or normalized_content_type in CONTENT_TYPES[kind]
    )
    if not extension_matches or not type_matches:
        raise OcrError(
            415,
            "CONTENT_TYPE_MISMATCH",
            "The filename, media type, and file content do not match.",
        )
    return kind


def _remaining_timeout(deadline: float) -> float:
    remaining = min(TESSERACT_TIMEOUT_SECONDS, deadline - time.monotonic())
    if remaining <= 0:
        raise OcrError(504, "OCR_TIMEOUT", "OCR processing timed out.")
    return remaining


def _ocr_image(image: Image.Image, deadline: float) -> str:
    pixel_count = image.width * image.height
    if pixel_count > MAX_IMAGE_PIXELS:
        raise OcrError(
            422,
            "IMAGE_PIXEL_LIMIT_EXCEEDED",
            f"The image exceeds the {MAX_IMAGE_PIXELS}-pixel limit.",
        )
    try:
        image.load()
        text = pytesseract.image_to_string(
            image,
            lang=OCR_LANGUAGE,
            timeout=_remaining_timeout(deadline),
        ).strip()
        if len(text) > MAX_OUTPUT_CHARS:
            raise OcrError(
                422,
                "OCR_OUTPUT_LIMIT_EXCEEDED",
                f"OCR output exceeds the {MAX_OUTPUT_CHARS}-character limit.",
            )
        return text
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise OcrError(
            422,
            "IMAGE_PIXEL_LIMIT_EXCEEDED",
            f"The image exceeds the {MAX_IMAGE_PIXELS}-pixel limit.",
        ) from exc
    except RuntimeError as exc:
        if "timeout" in str(exc).casefold():
            raise OcrError(504, "OCR_TIMEOUT", "OCR processing timed out.") from exc
        raise OcrError(500, "OCR_FAILED", "OCR processing failed.") from exc


def _recognize_image(data: bytes, deadline: float) -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                return _ocr_image(image, deadline)
    except OcrError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise OcrError(
            422,
            "IMAGE_PIXEL_LIMIT_EXCEEDED",
            f"The image exceeds the {MAX_IMAGE_PIXELS}-pixel limit.",
        ) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise OcrError(
            422, "MALFORMED_IMAGE", "The uploaded image is malformed."
        ) from exc


def _validate_pdf(document: fitz.Document) -> None:
    if document.needs_pass:
        raise OcrError(
            422,
            "ENCRYPTED_PDF_UNSUPPORTED",
            "Encrypted PDF files are not supported.",
        )
    if document.page_count <= 0:
        raise OcrError(422, "MALFORMED_DOCUMENT", "The uploaded PDF is malformed.")
    if document.page_count > MAX_PDF_PAGES:
        raise OcrError(
            422,
            "PDF_PAGE_LIMIT_EXCEEDED",
            f"The PDF exceeds the {MAX_PDF_PAGES}-page limit.",
        )


def _rendered_pixel_count(page: fitz.Page) -> int:
    width = math.ceil(page.rect.width * PDF_RENDER_SCALE)
    height = math.ceil(page.rect.height * PDF_RENDER_SCALE)
    pixel_count = width * height
    if width <= 0 or height <= 0 or pixel_count > MAX_IMAGE_PIXELS:
        raise OcrError(
            422,
            "PDF_PAGE_PIXEL_LIMIT_EXCEEDED",
            f"A rendered PDF page exceeds the {MAX_IMAGE_PIXELS}-pixel limit.",
        )
    return pixel_count


def _append_output(parts: list[str], text: str, output_length: int) -> int:
    separator_length = 2 if parts else 0
    next_length = output_length + separator_length + len(text)
    if next_length > MAX_OUTPUT_CHARS:
        raise OcrError(
            422,
            "OCR_OUTPUT_LIMIT_EXCEEDED",
            f"OCR output exceeds the {MAX_OUTPUT_CHARS}-character limit.",
        )
    parts.append(text)
    return next_length


def _recognize_pdf(data: bytes, deadline: float) -> str:
    try:
        with fitz.open(stream=data, filetype="pdf") as document:
            _validate_pdf(document)
            parts: list[str] = []
            output_length = 0
            rendered_pixels = 0
            for page in document:
                rendered_pixels += _rendered_pixel_count(page)
                if rendered_pixels > MAX_TOTAL_RENDERED_PIXELS:
                    raise OcrError(
                        422,
                        "PDF_PIXEL_LIMIT_EXCEEDED",
                        "The PDF exceeds the total rendered-pixel limit.",
                    )
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE),
                    alpha=False,
                )
                if pixmap.width * pixmap.height > MAX_IMAGE_PIXELS:
                    raise OcrError(
                        422,
                        "PDF_PAGE_PIXEL_LIMIT_EXCEEDED",
                        f"A rendered PDF page exceeds the {MAX_IMAGE_PIXELS}-pixel limit.",
                    )
                with Image.open(io.BytesIO(pixmap.tobytes("png"))) as image:
                    text = _ocr_image(image, deadline)
                output_length = _append_output(parts, text, output_length)
            return "\n\n".join(parts).strip()
    except OcrError:
        raise
    except (fitz.FileDataError, RuntimeError, ValueError) as exc:
        if "timeout" in str(exc).casefold():
            raise OcrError(504, "OCR_TIMEOUT", "OCR processing timed out.") from exc
        raise OcrError(
            422, "MALFORMED_DOCUMENT", "The uploaded PDF is malformed."
        ) from exc


def recognize(data: bytes, kind: str) -> str:
    deadline = time.monotonic() + OCR_REQUEST_TIMEOUT_SECONDS
    if kind == "pdf":
        return _recognize_pdf(data, deadline)
    return _recognize_image(data, deadline)


async def _read_upload(upload: UploadFile) -> bytes:
    data = await upload.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise OcrError(
            413,
            "FILE_TOO_LARGE",
            f"The uploaded file exceeds the {MAX_FILE_BYTES}-byte limit.",
        )
    return data


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(request: Request) -> dict[str, object]:
    try:
        async with request.form(max_files=1, max_fields=0) as form:
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise OcrError(422, "FILE_REQUIRED", "Exactly one file is required.")
            async with ocr_slots:
                data = await _read_upload(upload)
                kind = _validate_content(
                    data,
                    upload.filename or "upload",
                    upload.content_type or "",
                )
                text = await asyncio.wait_for(
                    asyncio.to_thread(recognize, data, kind),
                    timeout=OCR_REQUEST_TIMEOUT_SECONDS + 1,
                )
        return {"success": True, "output": text}
    except RequestBodyTooLarge:
        raise
    except OcrError:
        raise
    except asyncio.TimeoutError as exc:
        raise OcrError(504, "OCR_TIMEOUT", "OCR processing timed out.") from exc
    except HTTPException as exc:
        raise OcrError(
            400, "INVALID_MULTIPART", "The upload request is invalid."
        ) from exc
    except Exception as exc:
        raise OcrError(500, "OCR_FAILED", "OCR processing failed.") from exc
