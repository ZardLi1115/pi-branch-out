#!/bin/bash
set -e
cd /app
if python3 - <<'PY'
from chat.utils import run_cmd
assert run_cmd("true") == 0
assert run_cmd("false") != 0
print("ok")
PY
then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
