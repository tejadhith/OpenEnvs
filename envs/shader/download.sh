#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
ZIP_PATH="/tmp/shaders21k_codes.zip"
GDRIVE_ID="1kIiBdeW9CEIfRlYOYTuxfWvN036k3Iig"

if [ -d "$DATA_DIR/shadertoy" ] && [ "$(ls "$DATA_DIR/shadertoy/" 2>/dev/null | head -1)" ]; then
    echo "Shader data already exists at $DATA_DIR"
    exit 0
fi

echo "Downloading shaders21k codes..."

if command -v gdown &>/dev/null; then
    gdown "$GDRIVE_ID" -O "$ZIP_PATH"
elif command -v curl &>/dev/null; then
    curl -L "https://drive.google.com/uc?export=download&id=$GDRIVE_ID&confirm=t" -o "$ZIP_PATH"
else
    echo "Error: gdown or curl required. Install with: pip install gdown"
    exit 1
fi

echo "Extracting to $DATA_DIR..."
python3 -c "import zipfile; zipfile.ZipFile('$ZIP_PATH').extractall('$SCRIPT_DIR')"
mv "$SCRIPT_DIR/shader_codes" "$DATA_DIR"
rm -f "$ZIP_PATH"

echo "Done. $(find "$DATA_DIR/shadertoy" -type f | wc -l) shaders available."
