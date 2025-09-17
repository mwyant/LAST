FROM nvidia/cuda:12.9.0-base-ubuntu24.04
ARG DEBIAN_FRONTEND=noninteractive
ENV PORT=8000
WORKDIR /app

# System deps (include audio libs)
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        bash ca-certificates curl git vim build-essential software-properties-common gnupg wget pkg-config \
	libsndfile1 libsndfile1-dev ffmpeg sox && \
#        apt-get install -y --no-install-recommends \
#        python3 python3-dev python3-venv && \
	add-apt-repository -y ppa:deadsnakes/ppa && \
	apt-get update && \
	apt-get install -y --no-install-recommends python3.11 python3.11-venv python3.11-dev && \
    rm -rf /var/lib/apt/lists/*

# Set cache locations inside the image so downloads land in writable paths
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch
RUN mkdir -p $HF_HOME $TORCH_HOME && chmod -R 777 /app/.cache || true

# Copy requirements
COPY requirements.txt /app/requirements.txt

## Create venv, install wheel-based torch/torchaudio (cu129), then install other deps
#python3.12 run
#RUN python3 -m venv /opt/venv && \
#    /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel && \
#    /opt/venv/bin/pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu129 \
#      torch==2.8.0+cu129 torchvision==0.23.0 torchaudio==2.8.0+cu129 && \
#    /opt/venv/bin/pip install --no-cache-dir nemo_toolkit[tts]==2.4.0 librosa soundfile scipy scikit-learn TTS || true && \
#    /opt/venv/bin/pip install --no-cache-dir -r /app/requirements.txt && \
#    /opt/venv/bin/python -c "import fastapi, uvicorn, torch; print('py ok', torch.__version__, 'cuda_available=', torch.cuda.is_available())"

# create venv with python3.11 and install packages
RUN python3.11 -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu129 \
      torch==2.8.0+cu129 torchvision==0.23.0 torchaudio==2.8.0+cu129 && \
    /opt/venv/bin/pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu129 \
      torchvision torchaudio && \
    /opt/venv/bin/pip install --no-cache-dir demucs && \
    /opt/venv/bin/pip install --no-cache-dir TTS librosa soundfile scipy scikit-learn || true && \
    /opt/venv/bin/pip install --no-cache-dir -r /app/requirements.txt && \
    /opt/venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda=', torch.cuda.is_available())"


ENV PATH="/opt/venv/bin:${PATH}"

# Copy code and templates
COPY app ./app
COPY templates ./templates
COPY gpu_check.py /app/gpu_check.py

EXPOSE 8000
CMD ["sh", "-c", "/opt/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port $PORT --log-level info"]
