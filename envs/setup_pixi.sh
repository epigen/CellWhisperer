#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

echo "==> Step 1: Installing pixi environment..."
pixi install

echo "==> Step 2: Building web frontend (using Node.js from pixi environment)..."
cd modules/cellxgene/client
PUPPETEER_SKIP_DOWNLOAD=true pixi run --manifest-path "$REPO_ROOT/pixi.toml" npm ci
pixi run --manifest-path "$REPO_ROOT/pixi.toml" npm run build configuration/webpack/webpack.config.prod.js
cd ..
mkdir -p server/common/web/{static/assets,templates}
cp client/build/index.html server/common/web/templates/
cp -r client/build/static server/common/web/
cp client/build/csp-hashes.json server/common/web/

cd "$REPO_ROOT"

echo "==> Step 3: Verifying installation..."
pixi run cellxgene --version

echo ""
echo "You are all set! Follow the README to prepare your dataset and to start the CellWhisperer webapp."
