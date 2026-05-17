FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# System deps: ffmpeg + COLMAP
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    colmap \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
RUN pip install --no-cache-dir \
    runpod \
    gsplat==1.4.0 \
    nerfview \
    viser \
    opencv-python-headless \
    plyfile \
    tqdm

# Clone gsplat repo for SimpleTrainer
RUN git clone --depth 1 --branch v1.4.0 \
    https://github.com/nerfstudio-project/gsplat.git /app/gsplat

# Worker code
COPY handler.py /app/handler.py
COPY train_gsplat.py /app/train_gsplat.py

ENV PYTHONUNBUFFERED=1
CMD ["python", "/app/handler.py"]
