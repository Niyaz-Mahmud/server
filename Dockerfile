# Base image with Python
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy the server script
COPY server.py .

# Install dependencies
RUN pip install flwr scikit-learn pandas

# Expose the server port
EXPOSE 8080

# Start the server
CMD ["python", "server.py"]
