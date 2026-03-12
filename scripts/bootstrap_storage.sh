#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------
# Bootstrap scratch-backed storage symlinks
# ---------------------------------------

# Prefer Isambard SCRATCHDIR if set (Slurm jobs),
# otherwise fall back to your known /scratch path
: "${SCRATCH_ROOT:=${SCRATCHDIR:-/scratch/u5hv/frixos.u5hv}}"

# Default project name = current repo directory name
: "${PROJECT_NAME:=$(basename "$(pwd)")}"

SCRATCH_DIR="${SCRATCH_ROOT}/${PROJECT_NAME}"

# Create directories in scratch
mkdir -p "${SCRATCH_DIR}/data" \
         "${SCRATCH_DIR}/checkpoints" \
         "${SCRATCH_DIR}/logs"

# Create/refresh symlinks at repo root
for d in data checkpoints logs; do
  if [ -e "$d" ] || [ -L "$d" ]; then
    rm -rf "$d"
  fi
  ln -s "${SCRATCH_DIR}/${d}" "$d"
done

echo "Symlinks created:"
for d in data checkpoints logs; do
  echo "  ${d} -> $(readlink -f "${d}")"
done

