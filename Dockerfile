# Use official Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy everything from your project directory into /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your FastAPI app will run on
EXPOSE 8000

# Set environment variables (can override in docker run)
#ENV GOOGLE_API_KEY=your-api-key-here

# Command to run your FastAPI app using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
