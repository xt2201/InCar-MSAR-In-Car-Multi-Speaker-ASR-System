# In-Car Multi-Speaker ASR System
# Base: PyTorch 2.1.0 + CUDA 11.8 + cuDNN 8
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL maintainer="thesis-incar-asr"
LABEL version="1.0.0"
LABEL description="In-Car Multi-Speaker ASR Pipeline using AISHELL-5"

# System dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    libsndfile1 \
    libsox-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Create necessary output directories
RUN mkdir -p \
    data/dev \
    data/eval1 \
    data/eval2 \
    outputs/metrics \
    outputs/figures \
    outputs/tables \
    outputs/separation \
    outputs/ablation

# Expose Streamlit port
EXPOSE 8501

# Default environment variables
ENV PYTHONPATH="/app/src:${PYTHONPATH}"
ENV HF_HOME="/app/.cache/huggingface"
ENV TORCH_HOME="/app/.cache/torch"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; import torchaudio; print('OK')" || exit 1

# Default command: run demo
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
