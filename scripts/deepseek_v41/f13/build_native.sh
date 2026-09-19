#!/bin/sh
# Build the F13 native burst reader dylib (CPU-only, pthreads).
# Run under nice on this host. Output sits next to the source so the bench
# loads it by relative path.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
nice -n 19 clang -O2 -Wall -shared -fPIC \
    -o "$HERE/libreader.dylib" \
    "$HERE/native_reader.c" \
    -lpthread
echo "built $HERE/libreader.dylib"
