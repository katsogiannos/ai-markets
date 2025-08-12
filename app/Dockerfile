# app/Dockerfile
FROM python:3.11-slim

# Όλα να τρέχουν μέσα στον /app
WORKDIR /app

# Εγκατάσταση deps (ξεχωρίζουμε requirements για καλύτερο layer caching)
COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Αντιγραφή όλου του κώδικα στο /app
COPY app/ ./

# Render (Docker) απαιτεί να ακούς στο 10000
EXPOSE 10000
ENV PYTHONUNBUFFERED=1

# Εκκίνηση uvicorn στο σταθερό port 10000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
