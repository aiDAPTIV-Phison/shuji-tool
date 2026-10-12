#!/bin/bash

LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "Starting API app ..."
/app/bin/app.bin 2>&1 | tee $LOG_DIR/app.log

echo "All services started successfully!"
