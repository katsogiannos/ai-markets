FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Render δίνει PORT=10000 (ή ό,τι ορίσει). Το EXPOSE είναι πληροφοριακό.
EXPOSE 10000

# Τρέξε uvicorn στην πόρτα που δίνει το Render, default 10000.
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
