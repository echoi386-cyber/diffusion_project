#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

branch="$(git branch --show-current)"
if [[ "$branch" != "main" ]]; then
  echo "ERROR: You are on branch '$branch'. Switch to main before pushing."
  exit 1
fi

echo "=== Status ==="
git status

echo
echo "=== Tracked cache/conda/vscode files check ==="
bad=$(git ls-files | egrep '^(\.vscode-server/|\.cache/|\.conda/|\.condarc|\.config/|outputs/|runs_|data/)' || true)
if [[ -n "$bad" ]]; then
  echo "ERROR: These tracked paths should not be in git:"
  echo "$bad" | head -n 200
  echo "Fix: git rm -r --cached <path> then commit."
  exit 1
fi
echo "OK: no tracked junk."

echo
echo "=== Diff (what will be committed) ==="
git diff --stat

echo
echo "=== Stage all intended changes ==="
git add -A fourier-equal-snr-diffusion scripts .gitignore

echo
echo "=== Staged summary ==="
git diff --cached --stat

echo
read -p "Commit message (or empty to abort): " msg
if [[ -z "$msg" ]]; then
  echo "Aborted."
  exit 1
fi

git commit -m "$msg" || echo "Nothing to commit."

echo
echo "=== Push ==="
git push
echo "Done."
