FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/Parallel-Classification-Test-Environment

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY datasets ./datasets
COPY scripts ./scripts

RUN pip install --upgrade pip && \
    pip install torchgpipe && \
    pip install -e .

CMD ["bash"]
