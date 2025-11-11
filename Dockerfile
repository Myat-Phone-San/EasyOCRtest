# Use a Python 3.10 slim image as a base
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files and buffer output
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV, EasyOCR, and pdf2image (poppler-utils)
# libgl1 is often needed by OpenCV in headless environments
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1 \
       poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file (create one below) first to leverage Docker's caching
# NOTE: EasyOCR is large, so this step might take some time.
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the main application code into the container
COPY . /app/

# The application is designed to run on port 5000
EXPOSE 5000

# Set the entrypoint to run the Flask application
# Using a more robust WSGI server like Gunicorn is recommended for production.
# For this example, we'll use the built-in Flask server as in the script.
CMD ["python", "app.py"]