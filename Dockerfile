FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Create temp directories with proper permissions
RUN mkdir -p /app/backend/temp_uploads && chmod 777 /app/backend/temp_uploads
RUN mkdir -p /tmp && chmod 777 /tmp
RUN mkdir -p /usr/local/share/nltk_data && chmod -R 755 /usr/local/share/nltk_data

# Install Python dependencies
RUN if [ -f "backend/requirements.txt" ]; then pip install --no-cache-dir -r backend/requirements.txt; fi

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', download_dir='/usr/local/share/nltk_data')"
RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/usr/local/share/nltk_data')" || true

# Pre-download sentence transformer model
RUN mkdir -p /.cache && chmod 777 /.cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Install Node.js dependencies and build frontend
RUN npm install --legacy-peer-deps
RUN npm run build

# Create startup script
RUN echo '#!/bin/bash\n\
export PYTHONPATH=/app/backend:$PYTHONPATH\n\
cd /app/backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &\n\
cd /app && npm start -- --port 7860 --hostname 0.0.0.0\n\
' > start.sh && chmod +x start.sh

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Start both backend and frontend
CMD ["./start.sh"]