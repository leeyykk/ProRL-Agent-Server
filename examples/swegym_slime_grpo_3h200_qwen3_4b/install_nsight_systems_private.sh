#!/usr/bin/env bash
set -euo pipefail

INSTALLER=""
INSTALL_DIR="${HOME}/tools/nsight-systems"

usage() {
    cat <<'EOF'
Usage:
  install_nsight_systems_private.sh --installer PATH [--install-dir PATH]

Installs the NVIDIA Nsight Systems .run package into a user-owned directory.
The system-wide nsys installation and other users are not modified.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --installer) INSTALLER="$2"; shift 2 ;;
        --install-dir) INSTALL_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ -z "${INSTALLER}" ]; then
    echo "ERROR: --installer is required." >&2
    usage >&2
    exit 2
fi
if [ ! -f "${INSTALLER}" ]; then
    echo "ERROR: Nsight Systems installer not found: ${INSTALLER}" >&2
    exit 1
fi

INSTALLER="$(readlink -f "${INSTALLER}")"
mkdir -p "${INSTALL_DIR}"
chmod u+x "${INSTALLER}"
"${INSTALLER}" \
    --accept \
    --nox11 \
    --nochown \
    --target "${INSTALL_DIR}"

NSYS_BIN="$(find "${INSTALL_DIR}" -type f -name nsys -executable | head -1)"
if [ -z "${NSYS_BIN}" ]; then
    echo "ERROR: nsys was not found under ${INSTALL_DIR} after installation." >&2
    exit 1
fi

echo "Private Nsight Systems installation ready:"
echo "  NSYS_BIN=${NSYS_BIN}"
"${NSYS_BIN}" --version
