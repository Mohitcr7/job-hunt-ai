# Job Hunt AI — container image
# Scrapling's DynamicFetcher drives Chromium, so we start from the Playwright
# base image (Ubuntu + browser system deps preinstalled) instead of python:slim.

FROM mcr.microsoft.com/playwright/python:v1.45.0-jammy

WORKDIR /app

# Install Python deps first — this layer is cached until requirements change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set up the browser Scrapling uses for Naukri / company pages. The base image
# already provides the system libraries; this fetches the browser binary.
RUN scrapling install || true

# Pre-download the embedding model (~90MB) at build time so the first
# pipeline run doesn't stall on the download
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

# data/ (resume, tracker DB) and output/ (application kits) are volumes —
# see docker-compose.yml
RUN mkdir -p data output/applications

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
