#!/bin/bash
set -e
if [ -f /app/NOTES.md ] && grep -q "run_cmd" /app/NOTES.md; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
