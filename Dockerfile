FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download ESM model into image
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('facebook/esm2_t30_150M_UR50D'); \
    AutoModel.from_pretrained('facebook/esm2_t30_150M_UR50D')"

RUN chmod +x start.sh

EXPOSE 8080

CMD ["./start.sh"]
