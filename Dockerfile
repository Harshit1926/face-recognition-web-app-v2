# Base image: slim Python to keep the image reasonably sized,
# since TensorFlow/DeepFace already add significant weight on their own.
FROM python:3.11-slim

# System-level dependencies needed by OpenCV and DeepFace's image
# processing pipeline. Without these, opencv-python-headless and
# related libraries fail to import at runtime even though pip install
# succeeds (a common, confusing gotcha with this stack).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (separate layer), so Docker can
# cache this step and skip reinstalling everything if only app code
# changes later — meaningfully speeds up rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the actual application code
COPY app.py face_embeddings.py register_routes.py recognition_engine.py .
COPY templates/ ./templates/
COPY static/ ./static/

# Hugging Face Spaces expects the app to listen on port 7860 by default
EXPOSE 7860

# Run with Flask's built-in server. For a small portfolio project this
# is acceptable; for higher-traffic production use, a WSGI server like
# gunicorn would be the more robust choice.
ENV FLASK_RUN_PORT=7860
CMD ["python", "app.py"]