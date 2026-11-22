# EXTRAITS, CPU image.
#
#   docker build -f docker/Dockerfile -t extraits .
#   docker run --rm -v "$PWD/workspace:/workspace" --user "$(id -u):$(id -g)" \
#       extraits process -i /workspace/images -o /workspace/results.csv
#
# The models are NOT in the image: on first use, `process` downloads the official ones
# into the /workspace volume (models/). The EasyOCR weights are in the image.
# Build arg EXTRAS=train adds the training framework (extraits train / finetune).
#
# This image bundles Ultralytics (AGPL-3.0) and is distributed under AGPL-3.0
# (/usr/share/doc/extraits/NOTICE); the EXTRAITS source code is MIT.

FROM python:3.12-slim

LABEL org.opencontainers.image.title="extraits" \
      org.opencontainers.image.description="EXTRAITS: morphological traits from dorsal images of insects" \
      org.opencontainers.image.licenses="AGPL-3.0-only"

# OpenCV runtime libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    EXTRAITS_WORKSPACE=/workspace \
    EASYOCR_MODULE_PATH=/opt/easyocr \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    HOME=/tmp

# CPU-only PyTorch first: the default wheels bundle CUDA (several GB).
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

ARG EXTRAS=""
COPY pyproject.toml README.md /opt/extraits/
COPY src /opt/extraits/src
RUN pip install "/opt/extraits${EXTRAS:+[$EXTRAS]}"

# EasyOCR (scale-bar reading) downloads its weights on first use: fetch them now.
RUN python -c "from extraits.models import prefetch_ocr_weights; prefetch_ocr_weights()" \
    && chmod -R a+rX /opt/easyocr

COPY docker/entrypoint.sh /usr/local/bin/extraits-entrypoint
COPY docker/NOTICE LICENSES/AGPL-3.0.txt /usr/share/doc/extraits/
RUN chmod a+rx /usr/local/bin/extraits-entrypoint

WORKDIR /workspace
VOLUME ["/workspace"]
ENTRYPOINT ["extraits-entrypoint"]
CMD ["--help"]
