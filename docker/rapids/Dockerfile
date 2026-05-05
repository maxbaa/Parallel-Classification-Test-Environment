ARG RAPIDS_BASE_IMAGE=rapidsai/base:25.04-cuda12.8-py3.11
FROM ${RAPIDS_BASE_IMAGE}

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/Parallel-Classification-Test-Environment

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY datasets ./datasets
COPY scripts ./scripts

RUN python -m pip install --upgrade pip && \
    python -m pip install -e .

CMD ["bash"]
