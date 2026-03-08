#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

echo "==> Step 1: Installing pixi environment..."
pixi install

if [ "$(uname -s)" = "Darwin" ]; then
    echo "==> Step 2: Fixing macOS duplicate LC_RPATH in accumulation_tree..."
    # accumulation_tree 0.6.4 ships a .so with a duplicate LC_RPATH which causes
    # dlopen to fail on newer macOS versions. Remove the duplicate rpath entry.
    AT_SO=$(find .pixi/envs/default -name "accumulation_tree*.so" 2>/dev/null | head -1)
    if [ -n "$AT_SO" ]; then
        # Extract rpath entries and find any that appear more than once
        RPATHS=$(otool -l "$AT_SO" 2>/dev/null | awk '/LC_RPATH/{getline;getline;print $2}')
        DUP_RPATHS=$(echo "$RPATHS" | sort | uniq -d)
        if [ -n "$DUP_RPATHS" ]; then
            while IFS= read -r rpath; do
                echo "    Removing duplicate rpath '$rpath' from $AT_SO"
                # delete_rpath removes one occurrence; add it back so exactly one remains
                install_name_tool -delete_rpath "$rpath" "$AT_SO" 2>/dev/null || true
                install_name_tool -add_rpath "$rpath" "$AT_SO" 2>/dev/null || true
            done <<< "$DUP_RPATHS"
        else
            echo "    No duplicate rpath found, skipping."
        fi
    else
        echo "    accumulation_tree .so not found, skipping."
    fi
fi

echo "==> Step 3: Building web frontend (using Node.js from pixi environment)..."
cd modules/cellxgene/client
PUPPETEER_SKIP_DOWNLOAD=true pixi run --manifest-path "$REPO_ROOT/pixi.toml" npm ci
pixi run --manifest-path "$REPO_ROOT/pixi.toml" npm run build configuration/webpack/webpack.config.prod.js
cd ..
mkdir -p server/common/web/{static/assets,templates}
cp client/build/index.html server/common/web/templates/
cp -r client/build/static server/common/web/
cp client/build/csp-hashes.json server/common/web/

cd "$REPO_ROOT"

echo "==> Step 4: Verifying installation..."
pixi run cellxgene --version

echo ""
echo "You are all set! Follow the README to prepare your dataset and to start the CellWhisperer webapp."
