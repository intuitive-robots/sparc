#!/bin/bash
# Download the SAM2.1 Small checkpoint used by the annotation pipeline.
set -euo pipefail
[ "$#" -le 1 ] || { echo "Usage: $0 [checkpoint_directory]" >&2; exit 2; }
CHECKPOINT_DIR="${1:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p -- "$CHECKPOINT_DIR"
CHECKPOINT="$CHECKPOINT_DIR/sam2.1_hiera_small.pt"
URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
if [ -s "$CHECKPOINT" ]; then
    echo "Checkpoint already exists: $CHECKPOINT"
    exit 0
fi
TEMP_FILE="$(mktemp "$CHECKPOINT.part.XXXXXX")"
trap 'rm -f -- "$TEMP_FILE"' EXIT
if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --output "$TEMP_FILE" "$URL"
elif command -v wget >/dev/null 2>&1; then
    wget --output-document="$TEMP_FILE" "$URL"
else
    echo "Install curl or wget to download the checkpoint." >&2
    exit 1
fi
[ -s "$TEMP_FILE" ] || { echo "Downloaded checkpoint is empty" >&2; exit 1; }
mv -- "$TEMP_FILE" "$CHECKPOINT"
echo "Checkpoint ready: $CHECKPOINT"
