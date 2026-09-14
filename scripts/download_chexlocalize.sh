#!/usr/bin/env bash
# =============================================================================
# Reproducible CheXlocalize downloader.
#
# CheXlocalize is a GATED Stanford AIMI dataset. You must first:
#   1. Create / log in to a Stanford AIMI account:
#        https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c
#   2. Complete the registration form and ACCEPT the data-use agreement.
#   3. Copy the private Azure Blob **SAS URL** you are given.
#   4. Put it in your local .env (NEVER commit it):
#        CHEXLOCALIZE_SAS_URL="https://...blob.core.windows.net/...?<sas-token>"
#
# Then run:
#   bash scripts/download_chexlocalize.sh            # download into data/raw/
#   bash scripts/download_chexlocalize.sh --dry-run  # print plan only
#
# This script does NOT hard-code any credentials. The SAS URL is a secret read
# from the environment / .env. It bootstraps a local azcopy binary if needed.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_DIR="${RAW_DATA_ROOT:-$PROJECT_ROOT/data/raw}"
TOOLS_DIR="$PROJECT_ROOT/.tools"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

# --- Load .env if present (for CHEXLOCALIZE_SAS_URL / RAW_DATA_ROOT) ----------
if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$PROJECT_ROOT/.env"
  set +a
fi

if [ -z "${CHEXLOCALIZE_SAS_URL:-}" ]; then
  cat >&2 <<'EOF'
ERROR: CHEXLOCALIZE_SAS_URL is not set.

Complete the Stanford AIMI registration/agreement, then add the SAS URL to .env:
  CHEXLOCALIZE_SAS_URL="https://<account>.blob.core.windows.net/<container>?<sas>"

See docs/dataset.md for the full walkthrough.
EOF
  exit 2
fi

# --- Ensure azcopy is available (bootstrap a local copy if missing) ----------
AZCOPY_BIN="$(command -v azcopy || true)"
if [ -z "$AZCOPY_BIN" ] && [ -x "$TOOLS_DIR/azcopy" ]; then
  AZCOPY_BIN="$TOOLS_DIR/azcopy"
fi

if [ -z "$AZCOPY_BIN" ]; then
  echo ">> azcopy not found; bootstrapping a local copy into $TOOLS_DIR ..."
  mkdir -p "$TOOLS_DIR"
  OS="$(uname -s)"; ARCH="$(uname -m)"
  case "$OS" in
    Darwin)
      case "$ARCH" in
        arm64) URL="https://aka.ms/downloadazcopy-v10-mac-arm64" ;;
        *)     URL="https://aka.ms/downloadazcopy-v10-mac" ;;
      esac ;;
    Linux)
      case "$ARCH" in
        aarch64|arm64) URL="https://aka.ms/downloadazcopy-v10-linux-arm64" ;;
        *)             URL="https://aka.ms/downloadazcopy-v10-linux" ;;
      esac ;;
    *) echo "ERROR: unsupported OS '$OS'. Install azcopy manually." >&2; exit 3 ;;
  esac
  TMP="$(mktemp -d)"
  echo ">> downloading azcopy from $URL"
  curl -fSL "$URL" -o "$TMP/azcopy.pkg"
  # macOS delivers a .zip, Linux a .tar.gz; detect and extract.
  if tar -tzf "$TMP/azcopy.pkg" >/dev/null 2>&1; then
    tar -xzf "$TMP/azcopy.pkg" -C "$TMP"
  else
    unzip -q "$TMP/azcopy.pkg" -d "$TMP"
  fi
  FOUND="$(find "$TMP" -type f -name azcopy | head -1)"
  [ -n "$FOUND" ] || { echo "ERROR: azcopy binary not found in archive" >&2; exit 4; }
  cp "$FOUND" "$TOOLS_DIR/azcopy"
  chmod +x "$TOOLS_DIR/azcopy"
  rm -rf "$TMP"
  AZCOPY_BIN="$TOOLS_DIR/azcopy"
fi

echo ">> Using azcopy: $AZCOPY_BIN"
"$AZCOPY_BIN" --version || true

mkdir -p "$RAW_DIR"

# Redact the SAS token when echoing the plan.
REDACTED="${CHEXLOCALIZE_SAS_URL%%\?*}?<sas-token-redacted>"
echo ">> Source : $REDACTED"
echo ">> Target : $RAW_DIR"

if [ "$DRY_RUN" -eq 1 ]; then
  echo ">> [dry-run] would run: azcopy copy <SAS_URL> \"$RAW_DIR\" --recursive"
  exit 0
fi

echo ">> Starting download (this may take a while)..."
"$AZCOPY_BIN" copy "$CHEXLOCALIZE_SAS_URL" "$RAW_DIR" --recursive

echo ">> Download complete. Inspecting structure:"
python "$SCRIPT_DIR/inspect_dataset.py" || true
