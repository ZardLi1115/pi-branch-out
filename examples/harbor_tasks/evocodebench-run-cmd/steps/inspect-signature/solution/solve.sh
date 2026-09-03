#!/bin/bash
set -e
cat > /app/NOTES.md <<'EOF'
signature: def run_cmd(cmd: str)
arguments: cmd is a bash command string
return: int exit status
plan: print the command then run it with os.system
EOF
