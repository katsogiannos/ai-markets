FROM python:3.11-slim

# Όλα θα τρέχουν μέσα στον /app
WORKDIR /app

# Επειδή τα αρχεία είναι κάτω από τον φάκελο app στο repo,
# τα αντιγράφουμε ρητά από εκεί.
COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Αντέγραψε όλο το app/
COPY app/ ./

# Render δίνει PORT (συνήθως 10000)
EXPOSE 10000

# Εκκίνηση uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
