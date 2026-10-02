FROM python:3.12.3-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    NO_ALBUMENTATIONS_UPDATE=1 \
    ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gcc \
        libc6-dev \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/hybrid-qtn

COPY requirements.txt .
RUN pip install --requirement requirements.txt

COPY . .

VOLUME ["/workspace/hybrid-qtn/data", "/workspace/hybrid-qtn/checkpoints", "/workspace/hybrid-qtn/outputs"]

ENTRYPOINT ["python"]
CMD ["train.py", "--help"]
