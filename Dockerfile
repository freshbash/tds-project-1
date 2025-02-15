# Use a base image with Python and Node.js
FROM node:20-bookworm-slim AS node-base

# Install Python on top of Node.js
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy code and requirements
COPY . .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Prettier globally
RUN npm install -g prettier@3.4.2

# Expose port
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]