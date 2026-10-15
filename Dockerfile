FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04 AS base

# Copy uv tool from build image
COPY --from=ghcr.io/astral-sh/uv:0.12.11 /uv /uvx /bin/

# Set the working directory
WORKDIR /app

ARG TZ=UTC

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=${TZ}
ENV WHISPER_MODEL_FOLDER=/model/whisper/
ENV VIRTUAL_ENV=/opt/venv
ENV UV_LINK_MODE=copy
ENV PATH="/opt/venv/bin:${PATH}"

# Install dependencies. Apt FFmpeg provides shared libav* libraries required by torchcodec.
RUN chmod 1777 /tmp && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y tzdata ca-certificates curl ffmpeg python3 python3-dev build-essential xz-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/* # buildkit

# Set the time zone
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# Create an isolated Python environment for application dependencies.
RUN uv venv "$VIRTUAL_ENV"

# Copy files into the image
# Using external cache to speed up uv pip install. 
# For fully clean build, clear the cache directory with `docker builder prune --filter type=exec.cache` then run `docker build --no-cache`
RUN mkdir -p /app/src
COPY requirements.txt constraints.txt /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt

COPY src /app/src

EXPOSE 7766
EXPOSE 7767

CMD ["sleep", "infinity"]

FROM base AS dev

COPY requirements-dev.txt /app
RUN uv pip install --no-cache -r /app/requirements-dev.txt

WORKDIR /app
ENV PYTHONPATH=/app/src
CMD ["python3", "/app/src/api/app.py"]


# Default target: runs the service from plain Python sources as a non-root user.
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04 AS runtime

WORKDIR /app

ARG TZ=UTC

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=${TZ}
ENV WHISPER_MODEL_FOLDER=/model/whisper/
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# FFmpeg shared libraries are needed by torchcodec.
RUN chmod 1777 /tmp && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y tzdata ffmpeg python3 && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=base /opt/venv /opt/venv
COPY src /app/src

RUN groupadd -g 1655 appuser && \
    useradd -u 1655 -m -s /bin/bash -g appuser appuser && \
    mkdir -p /app/logs /app/temp_files && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 7766

CMD ["python3", "/app/src/api/app.py"]
