FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and pre-built data artifacts (includes ae_ai4i.pt — avoids retraining on start)
COPY src/ ./src/
COPY data/ ./data/

# PYTHONPATH lets all src.* imports resolve without sys.path hacks
ENV PYTHONPATH=/app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
