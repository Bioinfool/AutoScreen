#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${RAY_TMPDIR:-/tmp/ray}"

if ! ray status >/dev/null 2>&1; then
  ray start --head --num-cpus="${RAY_NUM_CPUS:-4}" --num-gpus="${RAY_NUM_GPUS:-0}" --disable-usage-stats
fi

exec "$@"
