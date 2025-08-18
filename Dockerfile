# Use Python 3.11 (works with mediapipe)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Run the app with gunicorn
CMD gunicorn app_final_debug:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
