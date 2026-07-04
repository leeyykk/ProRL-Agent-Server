#!/usr/bin/env bash
set -euo pipefail

GPU_RANGE="${1:-0,1,2,3}"

gpu_selected() {
    local idx="$1"
    local csv=",${GPU_RANGE},"
    [[ "${csv}" == *",${idx},"* ]]
}

declare -A GPU_UUID_TO_INDEX=()
while IFS=, read -r idx uuid; do
    idx="${idx//[[:space:]]/}"
    uuid="${uuid//[[:space:]]/}"
    if gpu_selected "${idx}"; then
        GPU_UUID_TO_INDEX["${uuid}"]="${idx}"
    fi
done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)

echo "Tracking GPUs: ${GPU_RANGE}"
echo

found=0
while IFS=, read -r uuid pid proc mem; do
    uuid="${uuid//[[:space:]]/}"
    pid="${pid//[[:space:]]/}"
    proc="${proc#"${proc%%[![:space:]]*}"}"
    mem="${mem#"${mem%%[![:space:]]*}"}"
    idx="${GPU_UUID_TO_INDEX[${uuid}]:-}"
    if [ -z "${idx}" ]; then
        continue
    fi
    found=1
    cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)"
    owner="other"
    case "${cwd}" in
        *"/ProRL-Agent-Server"*) owner="prorl" ;;
        *"/reverse_distill"*) owner="reverse_distill" ;;
    esac
    printf 'gpu=%s pid=%s mem=%s owner=%s cwd=%s proc=%s\n' \
        "${idx}" "${pid}" "${mem}" "${owner}" "${cwd}" "${proc}"
    ps -o user,pid,ppid,pgid,sid,etime,comm,args -p "${pid}" || true
    echo
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader)

if [ "${found}" = "0" ]; then
    echo "No compute processes found on GPUs ${GPU_RANGE}."
fi
