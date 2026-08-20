# Hugging Face Spaces (Docker SDK). Spaces routes to port 7860.
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp

EXPOSE 7860

# One worker on purpose: the app loads the 7MB dataset and the models into memory at
# import time, so every extra worker duplicates that. Timeout is generous because
# scoring ~10k invoices in a single request takes a few seconds.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "300", \
     "--access-logfile", "-", "app:app"]
