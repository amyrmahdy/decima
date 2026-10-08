# Decima System One server on CPU: POST /v1/systemone (TypeSafe wire format), GET /v1/models.
#
#   docker build -t decima .
#   docker run -p 11436:11436 -v decima-hf:/root/.cache/huggingface decima                       # pulls decima-base on first start
#   docker run -p 11436:11436 -e DECIMA_MODEL=decima-agent decima
#   docker run -p 11436:11436 -v $PWD/export/base2-int8:/model -e DECIMA_MODEL=/model -e DECIMA_NAME=decima-base decima
#
# Runtime only: ONNX Runtime, numpy and the tokenizer. No PyTorch, no GPU.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    DECIMA_MODEL=decima-base DECIMA_THREADS=2 DECIMA_PORT=11436

WORKDIR /app
RUN pip install "onnxruntime>=1.18" "numpy>=1.26" "transformers>=4.40" "huggingface_hub>=0.23"
COPY pyproject.toml README.md LICENSE ./
COPY decima ./decima
COPY bench ./bench
RUN pip install --no-deps .

EXPOSE 11436
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"DECIMA_PORT\"]}/', timeout=3)"
# DECIMA_API_KEY, when set, is required as `Authorization: Bearer <key>`.
CMD ["sh", "-c", "exec decima serve --host 0.0.0.0 --port \"$DECIMA_PORT\" --model \"$DECIMA_MODEL\" --threads \"$DECIMA_THREADS\" ${DECIMA_NAME:+--name \"$DECIMA_NAME\"}"]
