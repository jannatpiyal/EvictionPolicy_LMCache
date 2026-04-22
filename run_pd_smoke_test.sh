#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 is required to run the PD smoke test" >&2
  exit 1
fi

python3 - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("torch") is None:
    print("error: torch is required for tests.test_pd_http_smoke", file=sys.stderr)
    sys.exit(1)
PY

echo "Running PD HTTP smoke test..."
python3 -m unittest tests.test_pd_http_smoke
