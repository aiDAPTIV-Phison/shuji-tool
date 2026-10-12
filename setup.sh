#!/bin/bash

python3 -m nuitka \
    --show-memory \
    --show-progress \
    --static-libpython=no \
    --output-dir=/app/src/out \
    --follow-import-to=api \
    --follow-import-to=transcription \
    --follow-import-to=utils \
    --follow-import-to=vad \
    --include-module=app \
    /app/src/api/app.py
