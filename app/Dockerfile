FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Αντιγράφουμε ΟΛΟ το περιεχόμενο του repo (ώστε να βρει templates κλπ)
COPY . /app

# Render δίνει PORT (default 10000). Το EXPOSE είναι πληροφοριακό.
ENV PORT=10000
EXPOSE 10000

# Τρέχουμε uvicorn πάνω στο app/main.py (module: app.main:app)
CMD ["bash","-lc","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
