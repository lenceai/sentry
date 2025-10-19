#!/bin/bash
# Sentry Inference Launcher
conda run -n sentry python deployment/inference.py "$@"
