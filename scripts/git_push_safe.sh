#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/git_push_safe.sh <branch> [proxy_url]
# Example:
#   bash scripts/git_push_safe.sh feat/init-vlm-longmemory-codebook
#   bash scripts/git_push_safe.sh feat/init-vlm-longmemory-codebook http://127.0.0.1:7890

BRANCH="${1:-}"
PROXY_URL="${2:-}"

if [[ -z "${BRANCH}" ]]; then
  echo "Usage: bash scripts/git_push_safe.sh <branch> [proxy_url]"
  exit 1
fi

echo "[1/5] Disable broken IDE askpass channels..."
unset GIT_ASKPASS || true
unset SSH_ASKPASS || true

echo "[2/5] Ensure local credential helper..."
git config credential.helper store

if [[ -n "${PROXY_URL}" ]]; then
  echo "[3/5] Apply repo proxy: ${PROXY_URL}"
  git config http.proxy "${PROXY_URL}"
  git config https.proxy "${PROXY_URL}"
else
  echo "[3/5] No proxy passed, keep existing proxy config."
fi

echo "[4/5] Check github connectivity..."
if ! curl -I https://github.com --max-time 10 >/dev/null 2>&1; then
  echo "ERROR: github.com:443 is unreachable from this machine."
  echo "Try: bash scripts/git_push_safe.sh ${BRANCH} http://<proxy_host>:<proxy_port>"
  exit 2
fi

echo "[5/5] Push branch ${BRANCH} ..."
git push -u origin "${BRANCH}"
echo "Push finished."
