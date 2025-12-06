# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# CHANGED: 'libgl1-mesa-glx' -> 'libgl1' for newer Debian versions
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python packages
RUN pip install --no-cache-dir \
    gymnasium \
    stable-baselines3 \
    shimmy \
    pybullet \
    numpy \
    opencv-python \
    face-recognition

# Define environment variable to ensure output is flushed directly to terminal
ENV PYTHONUNBUFFERED=1

# Run the training script by default
CMD ["python", "train.py"]