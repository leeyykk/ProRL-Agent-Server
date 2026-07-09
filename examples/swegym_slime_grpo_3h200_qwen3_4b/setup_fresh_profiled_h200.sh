#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# The underlying setup is hardware-neutral despite its historical filename. It
# creates the venv, downloads Qwen3.5-4B, installs Codex, prepares Docker-backed
# SWE-Gym data, and converts the model to a TP2 torch_dist checkpoint.
export SETUP_PROGRAM_NAME="$(basename "${BASH_SOURCE[0]}")"
exec bash "${SCRIPT_DIR}/setup_fresh_b200.sh" "$@"
