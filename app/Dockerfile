# Χρησιμοποιούμε Python 3.11
FROM python:3.11

# Ορίζουμε φάκελο εργασίας
WORKDIR /app

# Αντιγράφουμε το requirements.txt
COPY requirements.txt .

# Εγκαθιστούμε dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Αντιγράφουμε όλο τον κώδικα
COPY . /app

# Τρέχουμε την εφαρμογή
CMD ["python", "main.py"]
