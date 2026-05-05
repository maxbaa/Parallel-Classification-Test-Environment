ARG THUNDER_BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${THUNDER_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      libomp-dev \
      python3 \
      python3-dev \
      python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /workspace/Parallel-Classification-Test-Environment

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY datasets ./datasets
COPY scripts ./scripts

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install numpy pandas scikit-learn psutil pyyaml matplotlib typing-extensions && \
    python -m pip install git+https://github.com/Xtra-Computing/thundersvm.git && \
    python -m pip install -e .

CMD ["bash"]
