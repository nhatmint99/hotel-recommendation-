# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

EXPOSE 8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
CMD bash -lc 'streamlit run app.py --server.address=$STREAMLIT_SERVER_ADDRESS --server.port=${PORT:-8501}'
