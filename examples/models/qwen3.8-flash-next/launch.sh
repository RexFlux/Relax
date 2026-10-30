#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Set these three paths here or provide them through the environment.
set -eo pipefail

export MODEL_DIR="${MODEL_DIR:-}"
export DATA_DIR="${DATA_DIR:-}"
export SAVE_DIR="${SAVE_DIR:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/scripts/run-qwen38-qsa-sft-40xh20.sh" "$@"
