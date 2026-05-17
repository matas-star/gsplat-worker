FROM dromni/nerfstudio:main

# RunPod serverless SDK
RUN pip install --no-cache-dir runpod

# Worker kodas
COPY worker.py /app/worker.py
COPY train_gsplat.py /app/train_gsplat.py

ENV PYTHONUNBUFFERED=1
CMD ["python", "/app/worker.py"]
