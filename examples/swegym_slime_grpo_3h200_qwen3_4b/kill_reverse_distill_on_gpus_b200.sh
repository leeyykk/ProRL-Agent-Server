#!/usr/bin/env bash
set -euo pipefail

GPU_RANGE="${1:-0,1,2,3}"
DRY_RUN="${DRY_RUN:-0}"
REVERSE_DISTILL_ROOT="${REVERSE_DISTILL_ROOT:-/home/korea_bupj/reverse_distill}"

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

declare -A PGIDS=()
while IFS=, read -r uuid pid proc mem; do
    uuid="${uuid//[[:space:]]/}"
    pid="${pid//[[:space:]]/}"
    idx="${GPU_UUID_TO_INDEX[${uuid}]:-}"
    if [ -z "${idx}" ] || [ -z "${pid}" ] || [ ! -d "/proc/${pid}" ]; then
        continue
    fi

    cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)"
    if [[ "${cwd}" != "${REVERSE_DISTILL_ROOT}"* ]]; then
        continue
    fi

    pgid="$(ps -o pgid= -p "${pid}" | awk '{gsub(/[[:space:]]/,""); print}')"
    if [ -n "${pgid}" ]; then
        PGIDS["${pgid}"]=1
        printf 'target gpu=%s pid=%s pgid=%s mem=%s cwd=%s proc=%s\n' \
            "${idx}" "${pid}" "${pgid}" "${mem}" "${cwd}" "${proc}"
    fi
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits)

if [ "${#PGIDS[@]}" -eq 0 ]; then
    echo "No reverse_distill compute process groups found on GPUs ${GPU_RANGE}."
    exit 0
fi

echo
for pgid in "${!PGIDS[@]}"; do
    echo "===== reverse_distill PGID ${pgid} ====="
    ps -o user,pid,ppid,pgid,sid,etime,comm,args -g "${pgid}" || true
done

if [ "${DRY_RUN}" = "1" ]; then
    echo
    echo "DRY_RUN=1, not killing anything."
    exit 0
fi

echo
for pgid in "${!PGIDS[@]}"; do
    echo "TERM process group ${pgid}"
    kill -TERM "-${pgid}" 2>/dev/null || true
done

sleep 8

for pgid in "${!PGIDS[@]}"; do
    if ps -g "${pgid}" >/dev/null 2>&1; then
        echo "KILL process group ${pgid}"
        kill -KILL "-${pgid}" 2>/dev/null || true
    fi
done
