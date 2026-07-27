FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir fastapi==0.115.6 uvicorn[standard]==0.34.0 python-multipart==0.0.20 pymupdf==1.25.3 pillow==11.1.0 pytesseract==0.3.13
WORKDIR /app
COPY main.py /app/main.py
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
