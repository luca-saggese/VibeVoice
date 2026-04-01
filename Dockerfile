FROM dgx-spark-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=3000 \
    MODEL_PATH=microsoft/VibeVoice-Realtime-0.5B \
    MODEL_DEVICE=cuda

WORKDIR /workspace/VibeVoice

COPY . .

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install .

ENV HF_HOME=/huggingface

EXPOSE 3000

CMD ["sh", "-c", "python demo/vibevoice_realtime_demo.py --port ${PORT} --model_path ${MODEL_PATH} --device ${MODEL_DEVICE}"]
