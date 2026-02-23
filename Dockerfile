
FROM python:3.9-slim
WORKDIR /app

ons
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir \
    gymnasium \
    stable-baselines3 \
    shimmy \
    pybullet \
    numpy \
    opencv-python \
    face-recognition

ENV PYTHONUNBUFFERED=1

CMD ["python", "train.py"]
