FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "from transformers import TrOCRProcessor; TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten', cache_dir='/models')"

RUN python -c "from transformers import VisionEncoderDecoderModel; VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten', cache_dir='/models')"

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "965", "app:app"]