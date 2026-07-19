FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY src ./src
COPY runs/vsl_mvp30_v2_lite_transformer ./runs/vsl_mvp30_v2_lite_transformer
COPY data/practice_videos_web ./data/practice_videos_web

CMD ["python", "-m", "vsl_mvp.deploy_app", "--host", "0.0.0.0", "--port", "7860"]
