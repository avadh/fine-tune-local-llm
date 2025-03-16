# Use Python base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ src/
COPY models/ models/

# Expose the API port
EXPOSE 8000

# Start the server
CMD ["python", "src/serve_model.py"]
