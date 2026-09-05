#!/bin/sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export_dir="$repo_dir/runtime/export"
runtime_platform=${PI_RUNTIME_PLATFORM:-linux/amd64}
runtime_arch=${runtime_platform#linux/}
archive="$repo_dir/runtime/pi-runtime-linux-$runtime_arch.tar.gz"

rm -rf "$export_dir"
docker build \
  --platform "$runtime_platform" \
  --target export \
  --output "type=local,dest=$export_dir" \
  "$repo_dir/runtime"
tar -C "$export_dir" -czf "$archive" pi-runtime
test -s "$archive"
rm -rf "$export_dir"
echo "$archive"
