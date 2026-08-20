# Optional container build. Render deploys this app with its native Python runtime via
# ../render.yaml; this Dockerfile is here so the app can run anywhere that takes a
# container, without changing code.
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp \
    PORT=8080

EXPOSE 8080

# Shell form so $PORT expands at runtime - most hosts inject their own port and a
# hardcoded one fails their health check. One worker on purpose: the dataset and models
# load at import (~190MB resident), so every extra worker duplicates that.
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --access-logfile - app:app
