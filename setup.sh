#!/bin/bash

if [[ "$1" == "serve" ]]; then
    echo "[inference-runtime] Starting Runtime"
    exec python3 /main.py
else
    while true; do
        sleep 3600
    done
fi
