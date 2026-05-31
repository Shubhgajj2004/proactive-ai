#!/usr/bin/env bash
# Download the default Piper TTS voice model (en_US-lessac-medium, ~60MB).
# Run once after cloning:  bash scripts/download_piper_voice.sh

set -e
DEST="models/tts"
mkdir -p "$DEST"

BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"

echo "Downloading en_US-lessac-medium.onnx ..."
curl -L -o "$DEST/en_US-lessac-medium.onnx"      "$BASE/en_US-lessac-medium.onnx"
curl -L -o "$DEST/en_US-lessac-medium.onnx.json" "$BASE/en_US-lessac-medium.onnx.json"

echo "Voice model ready at $DEST/"
