#!/usr/bin/env bash
set -euo pipefail

# Minimal `apptainer exec` compatibility wrapper for hosts that block
# unprivileged user namespaces. It lazily expands SIF files and executes their
# root filesystems with the proot helper bundled in the user-local Apptainer
# package.

REAL_APPTAINER="${REAL_APPTAINER:-/NHNHOME/home/tools/apptainer-1.5.2/usr/bin/apptainer}"
PROOT_BIN="${PROOT_BIN:-/NHNHOME/home/tools/apptainer-1.5.2/usr/libexec/apptainer/bin/proot}"
UNSQUASHFS_BIN="${UNSQUASHFS_BIN:-/usr/bin/unsquashfs}"
PROOT_ROOTFS_BASE="${PROOT_ROOTFS_BASE:-/NHNHOME/home/prorl_agent_server_env/proot_rootfs}"
PROOT_TMP_DIR="${PROOT_TMP_DIR:-/NHNHOME/home/prorl_agent_server_env/proot_tmp}"

if [ "${1:-}" != "exec" ]; then
    exec "${REAL_APPTAINER}" "$@"
fi
shift

binds=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --nv|--compat|--contain|--containall|--cleanenv)
            shift
            ;;
        --net)
            shift
            ;;
        --network)
            shift 2
            ;;
        --network=*)
            shift
            ;;
        --bind|-B)
            binds+=("$2")
            shift 2
            ;;
        --bind=*)
            binds+=("${1#--bind=}")
            shift
            ;;
        --*)
            echo "Unsupported Apptainer wrapper option: $1" >&2
            exit 2
            ;;
        *)
            break
            ;;
    esac
done

image="${1:?missing SIF image for apptainer exec}"
shift
if [ ! -f "${image}" ]; then
    echo "SIF image not found: ${image}" >&2
    exit 1
fi

mkdir -p "${PROOT_ROOTFS_BASE}" "${PROOT_TMP_DIR}"
image_key="$(basename "${image}" .sif)"
rootfs="${PROOT_ROOTFS_BASE}/${image_key}"
ready="${rootfs}/.polar_proot_ready"
lock="${PROOT_ROOTFS_BASE}/.${image_key}.lock"

(
    flock 9
    if [ ! -f "${ready}" ]; then
        rm -rf "${rootfs}"
        mkdir -p "${rootfs}"
        squashfs="${PROOT_TMP_DIR}/${image_key}.$$.squashfs"
        # SIF filesystem objects produced by Apptainer are 4096-byte aligned;
        # obtain the actual byte offset rather than assuming a fixed header.
        offset="$(
            "${REAL_APPTAINER}" sif list "${image}" |
                awk -F'|' '/FS \(Squashfs/ {
                    value=$4
                    sub(/-.*/, "", value)
                    gsub(/[[:space:]]/, "", value)
                    print value
                    exit
                }'
        )"
        if [ -z "${offset}" ]; then
            echo "Could not locate Squashfs partition in ${image}" >&2
            exit 1
        fi
        dd if="${image}" of="${squashfs}" bs=4096 skip="$((offset / 4096))" status=none
        "${UNSQUASHFS_BIN}" -f -d "${rootfs}" "${squashfs}" >/dev/null
        rm -f "${squashfs}"
        touch "${ready}"
    fi
) 9>"${lock}"

proot_args=(-R "${rootfs}")
for bind in "${binds[@]}"; do
    # Apptainer accepts src:dst:opts while proot accepts src:dst.
    bind="${bind%:ro}"
    bind="${bind%:rw}"
    proot_args+=(-b "${bind}")
done

export PROOT_TMP_DIR
exec "${PROOT_BIN}" "${proot_args[@]}" "$@"
