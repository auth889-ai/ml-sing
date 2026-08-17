# SongForge GPU backend image.
#
# Build from the repository root (the image needs src/ and deploy/backend/):
#   docker build -f deploy/deployment/Dockerfile -t songforge-api .
#
# Run on any CUDA host with >=16 GB VRAM and bf16 (Ampere or newer — T4/Turing
# produces NaN latents on lyrics-to-song and is not supported):
#   docker run --gpus all -p 8000:8000 \
#     -e SONGFORGE_LORA=/models/lora \
#     -v /path/to/models:/models \
#     songforge-api
#
# ACE-Step weights are pulled from the Hugging Face cache on first load; mount
# a persistent volume at /root/.cache/huggingface to avoid re-downloading
# ~15 GB on every container start.

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/backend/requirements.txt /app/deploy/backend/requirements.txt
RUN pip install --no-cache-dir -r deploy/backend/requirements.txt acestep

COPY src /app/src
COPY deploy/backend /app/deploy/backend
COPY deploy/__init__.py /app/deploy/__init__.py

ENV SONGFORGE_OUTPUT=/data/jobs
VOLUME /data

EXPOSE 8000
CMD ["uvicorn", "deploy.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
