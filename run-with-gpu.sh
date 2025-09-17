#!/usr/bin/env bash
set -euo pipefail

# Defaults
IMAGE="local-last:devlts"
CONTAINER_NAME="last-web"
PORT=8000
HF_TOKEN="${HUGGINGFACE_TOKEN:-}"
HF_HOME="/app/.cache/huggingface"
TORCH_HOME="/app/.cache/torch"
HOST_HF="/docker-mount/LAST/cache/huggingface"
HOST_TORCH="/docker-mount/LAST/cache/torch"
HOST_DATA="/docker-mount/LAST/data"
HOST_TEMPLATES="/docker-mount/LAST/templates"
HOST_BACKUP="/docker-mount/LAST/backup"
DETACH=false
PIDFILE="/tmp/${CONTAINER_NAME}.cid"
RESTART="false"
BUILD=false

usage() {
  cat <<EOF
Usage: $0 [--daemon|-d] [--name <container_name>] [--image <image_tag>]

Options:
  -d, --daemon          Run container detached (background). Container ID written to ${PIDFILE}.
  --name <name>         Container name (default: ${CONTAINER_NAME}).
  --image <image>       Image tag to run (default: ${IMAGE}).
  --help                Show this help.
  --build              Rebuilds image. (default: ${BUILD}).
EOF
  exit 1
}

# parse args
while [ $# -gt 0 ]; do
  case "$1" in
    -d|--daemon) DETACH=true; shift ;;
    --name) CONTAINER_NAME="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    -h|--help) usage ;;
    --restart) RESTART=true; shift ;;
    --build) BUILD=true; shift ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done
# After argument parsing, handle restart without changing DETACH:
if [ "${RESTART}" = true ]; then
  echo "Restart requested for container name='${CONTAINER_NAME}'"

  # check running container with exact name
  RUNNING_CID="$(docker ps -q -f "name=^/${CONTAINER_NAME}$" 2>/dev/null || true)"
  if [ -n "${RUNNING_CID}" ]; then
    echo "Stopping running container ${CONTAINER_NAME} (${RUNNING_CID})..."
    docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  else
    echo "No running container named ${CONTAINER_NAME} found."
  fi

  # check any existing container (stopped or running)
  EXISTING_CID="$(docker ps -a -q -f "name=^/${CONTAINER_NAME}$" 2>/dev/null || true)"
  if [ -n "${EXISTING_CID}" ]; then
    echo "Removing existing container ${CONTAINER_NAME} (${EXISTING_CID})..."
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi

  echo "Restart preflight complete. Will start new container with DETACH=${DETACH}."
fi
# Build with compose (safe)
# only build if requested or image not present
if [ "${BUILD}" = true ] || ! docker image inspect "${IMAGE}" >/dev/null 2>&1 ; then
  echo "Building image ${IMAGE}..."
  docker compose build --pull transcription
else
  echo "Image ${IMAGE} found — skipping build."
fi

# If container exists, stop & remove it so we can start fresh
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Removing existing container named ${CONTAINER_NAME} ..."
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

# Construct run command parts
COMMON_ARGS=(
  --rm
  -p ${PORT}:8000
  -e PORT=${PORT}
  -e HF_HOME=${HF_HOME}
  -e TORCH_HOME=${TORCH_HOME}
  -e HUGGINGFACE_TOKEN=${HF_TOKEN}
  -e HUGGINGFACE_HUB_TOKEN=${HF_TOKEN}
  -v ${HOST_HF}:${HF_HOME}
  -v ${HOST_TORCH}:${TORCH_HOME}
  -v ${HOST_DATA}:/app/data
  -v ${HOST_TEMPLATES}:/app/templates:ro
  -v ${HOST_BACKUP}:/app/backup
  --name "${CONTAINER_NAME}"
)

CMD='/bin/sh -c "/opt/venv/bin/python /app/gpu_check.py; /opt/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port \$PORT --log-level debug"'

if [ "${DETACH}" = true ]; then
  echo "Starting container in detached mode (name=${CONTAINER_NAME}, image=${IMAGE})..."
  CID=$(docker run --gpus all -d "${COMMON_ARGS[@]}" "${IMAGE}" sh -c "${CMD}")
  echo "${CID}" | tee "${PIDFILE}"
  echo "Container started: ${CID}"
  echo "To view logs: docker logs -f ${CID}"
  echo "To stop: docker stop ${CID}"
else
  echo "Starting container in foreground (CTRL+C to stop) (name=${CONTAINER_NAME}, image=${IMAGE})..."
  docker run --gpus all "${COMMON_ARGS[@]}" "${IMAGE}" sh -c "${CMD}"
fi
