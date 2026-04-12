#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash deploy/qwen3vl_4b_llamacpp/infer.sh /path/to/image.jpg" >&2
  exit 1
fi

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${DEPLOY_DIR}/deploy.sh" infer "$1"
