# Use a smaller base image
FROM python:3.12-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set working directory
WORKDIR /usr/src/app

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies directly without virtualenv
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /usr/src/app
USER appuser

# Expose port
EXPOSE 5000

# Run with Gunicorn
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "120", "app:app"]
