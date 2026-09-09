#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --job-name=annotation
#SBATCH --output=annotation_%j.%N.out
#SBATCH --error=annotation_%j.%N.err

# Submit from the repository root after activating the annotation environment.
# Set VLLM_BASE_URL for an existing server, or VLLM_LAUNCH_SCRIPT to start one
# on the first allocated GPU of node 0. The other GPUs run annotation.
set -euo pipefail

usage() {
    echo "Usage: sbatch ... slurm/launch_annotation.sh <dataset> [overrides...]"
    echo "   or: ... --config <resolved.yaml> [overrides...]"
    echo "   or: ... --resume-dir <run_dir> [overrides...]"
}

[ "$#" -gt 0 ] || { usage; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# sbatch copies scripts into its spool, so use the submission directory there.
PIPELINE_ROOT="${PIPELINE_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname -- "$SCRIPT_DIR")}}"
PIPELINE_ROOT="$(cd -- "$PIPELINE_ROOT" && pwd)"
[ -f "$PIPELINE_ROOT/annotate.py" ] || {
    echo "Submit from the repository root or set PIPELINE_ROOT to it." >&2
    exit 2
}
export PIPELINE_ROOT
PIPELINE_PYTHON="${PIPELINE_PYTHON:-python}"

case "$1" in
    --resume-dir|--continue-dir)
        [ "$#" -ge 2 ] || { usage; exit 2; }
        shopt -s nullglob
        CONFIG_MATCHES=("$2"/*_config.yaml)
        shopt -u nullglob
        [ "${#CONFIG_MATCHES[@]}" -eq 1 ] || {
            echo "Expected exactly one *_config.yaml in $2" >&2; exit 2;
        }
        CONFIG_FILE="$(realpath -- "${CONFIG_MATCHES[0]}")"
        ANNOTATOR_ARGS=(--config "$CONFIG_FILE")
        shift 2
        ;;
    --config)
        [ "$#" -ge 2 ] && [ -f "$2" ] || { usage; exit 2; }
        ANNOTATOR_ARGS=(--config "$(realpath -- "$2")")
        shift 2
        ;;
    -*) usage; exit 2 ;;
    *) ANNOTATOR_ARGS=(--dataset "$1"); shift ;;
esac
EXTRA_OVERRIDES=("$@")
# Forward absolute config paths after changing to PIPELINE_ROOT.
if [ "${ANNOTATOR_ARGS[0]}" = --config ]; then
    INVOCATION_ARGS=("${ANNOTATOR_ARGS[@]}" "${EXTRA_OVERRIDES[@]}")
else
    INVOCATION_ARGS=("${ANNOTATOR_ARGS[1]}" "${EXTRA_OVERRIDES[@]}")
fi

if [ -z "${VLLM_BASE_URL:-}" ]; then
    [ -n "${VLLM_LAUNCH_SCRIPT:-}" ] && [ -f "$VLLM_LAUNCH_SCRIPT" ] || {
        echo "Set VLLM_BASE_URL or VLLM_LAUNCH_SCRIPT (see README)." >&2; exit 2;
    }
    VLLM_LAUNCH_SCRIPT="$(realpath -- "$VLLM_LAUNCH_SCRIPT")"
    export VLLM_LAUNCH_SCRIPT
fi
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_HOST_FILE="${VLLM_HOST_FILE:-$PIPELINE_ROOT/runs/slurm/${SLURM_JOB_ID:?}/vlm_host}"
export VLLM_PORT VLLM_HOST_FILE
cd -- "$PIPELINE_ROOT"

if [ "${_WORKER:-}" != 1 ]; then
    mkdir -p "$(dirname -- "$VLLM_HOST_FILE")"
    if [ -z "${VLLM_BASE_URL:-}" ]; then
        rm -f -- "$VLLM_HOST_FILE" "$VLLM_HOST_FILE".done.*
    fi
    SRUN_EXIT=0
    srun --ntasks-per-node=1 --kill-on-bad-exit=1 --export=ALL,_WORKER=1 \
        bash "$PIPELINE_ROOT/slurm/launch_annotation.sh" "${INVOCATION_ARGS[@]}" || SRUN_EXIT=$?
    if [ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ]; then
        "$PIPELINE_PYTHON" annotate.py "${ANNOTATOR_ARGS[@]}" \
            --merge-only "${EXTRA_OVERRIDES[@]}"
    fi
    exit "$SRUN_EXIT"
fi

# Every worker publishes completion, including failures. Node 0 must keep the
# shared VLM alive until the last annotation worker has finished.
LOCAL_VLM=0
[ -n "${VLLM_BASE_URL:-}" ] || LOCAL_VLM=1
finish_worker() {
    local status=$? rank
    trap - EXIT
    if [ "$LOCAL_VLM" = 1 ]; then
        echo "$status" > "$VLLM_HOST_FILE.done.${SLURM_NODEID:-0}"
        if [ -n "${VLLM_PID:-}" ]; then
            if [ "$status" = 0 ]; then
                for ((rank=0; rank<${SLURM_JOB_NUM_NODES:-1}; rank++)); do
                    while [ ! -s "$VLLM_HOST_FILE.done.$rank" ]; do
                        kill -0 "$VLLM_PID" 2>/dev/null || { status=1; break; }
                        sleep 1
                    done
                    [ "$status" = 0 ] || break
                    status="$(cat "$VLLM_HOST_FILE.done.$rank")"
                    [ "$status" = 0 ] || break
                done
            fi
            rm -f -- "$VLLM_HOST_FILE"
            kill "$VLLM_PID" 2>/dev/null || true
            wait "$VLLM_PID" 2>/dev/null || true
        fi
    fi
    exit "$status"
}
trap finish_worker EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a GPU_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t GPU_DEVICES < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
[ "${#GPU_DEVICES[@]}" -gt 0 ] || { echo "No allocated GPUs found" >&2; exit 2; }

if [ -z "${VLLM_BASE_URL:-}" ]; then
    if [ "${SLURM_NODEID:-0}" -eq 0 ]; then
        [ "${#GPU_DEVICES[@]}" -ge 2 ] || {
            echo "Local VLM serving needs at least two GPUs, or use VLLM_BASE_URL." >&2; exit 2;
        }
        CUDA_VISIBLE_DEVICES="${GPU_DEVICES[0]}" PORT="$VLLM_PORT" \
            bash "$VLLM_LAUNCH_SCRIPT" &
        VLLM_PID=$!
        hostname > "$VLLM_HOST_FILE"
        GPU_DEVICES=("${GPU_DEVICES[@]:1}")
    else
        for ((attempt=0; attempt<60; attempt++)); do
            [ -s "$VLLM_HOST_FILE" ] && break
            sleep 10
        done
        [ -s "$VLLM_HOST_FILE" ] || { echo "VLM host file did not appear" >&2; exit 1; }
    fi
    VLLM_BASE_URL="http://$(cat "$VLLM_HOST_FILE"):$VLLM_PORT/v1"
fi

VLLM_HEALTH_URL="${VLLM_BASE_URL%/}"
VLLM_HEALTH_URL="${VLLM_HEALTH_URL%/v1}/health"
for ((attempt=0; attempt<60; attempt++)); do
    curl -sf "$VLLM_HEALTH_URL" >/dev/null && break
    sleep 10
done
[ "$attempt" -lt 60 ] || { echo "VLM unavailable at $VLLM_BASE_URL" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_DEVICES[*]}")"
GPU_IDS=()
for ((i=0; i<${#GPU_DEVICES[@]}; i++)); do GPU_IDS+=("$i"); done
ANNOTATOR_GPU_IDS="[$(IFS=,; echo "${GPU_IDS[*]}")]"
"$PIPELINE_PYTHON" annotate.py "${ANNOTATOR_ARGS[@]}" \
    "annotator.gpu_ids=$ANNOTATOR_GPU_IDS" "vllm.base_url=$VLLM_BASE_URL" \
    "${EXTRA_OVERRIDES[@]}"
