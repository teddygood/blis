#!/bin/bash

#
# Build BLIS for the wasm32 configuration with Emscripten and run the
# fast testsuite under Node.js.
#
# Usage:
#
#   EMSDK=/path/to/emsdk ci/wasm/do_wasm.sh
#

set -euo pipefail

script_name=${0##*/}

# Run from the top-level source directory, like ci/do_testsuite.sh.
cd "$(dirname "$0")/../.."
DIST_PATH="$(pwd)"

ARTIFACTS_DIR="${ARTIFACTS_DIR:-$DIST_PATH/wasm-artifacts}"
GENERAL_INPUT="testsuite/input.general.fast"
OPS_INPUT="testsuite/input.operations.fast"
TESTSUITE_BIN="test_libblis.x"
TESTSUITE_OUT="output.testsuite"
JOBS="${JOBS:-$(nproc)}"

log() { echo "$script_name: $*"; }
die() { echo "$script_name: error: $*" >&2; exit 1; }

mkdir -p "$ARTIFACTS_DIR"

if [ -n "${EMSDK:-}" ]; then
	[ -f "$EMSDK/emsdk_env.sh" ] || die "EMSDK is set to '$EMSDK' but $EMSDK/emsdk_env.sh does not exist"
	# shellcheck disable=SC1090
	EMSDK_QUIET=1 source "$EMSDK/emsdk_env.sh"
fi

for tool in emcc emar emranlib; do
	command -v "$tool" > /dev/null 2>&1 || die "$tool not found on PATH; set EMSDK=/path/to/emsdk or activate emsdk first"
done

# Prefer an explicit NODE, then the node bundled with emsdk, then PATH.
NODE="${NODE:-${EMSDK_NODE:-node}}"
command -v "$NODE" > /dev/null 2>&1 || die "node executable '$NODE' not found"

# The input fixtures are baked into the binary via --preload-file, so a
# missing file is a hard error.
for f in "$GENERAL_INPUT" "$OPS_INPUT"; do
	[ -f "$f" ] || die "required testsuite input file '$f' is missing"
done

# Record toolchain and commit metadata for the CI artifacts.
{
	echo "date:       $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
	echo "git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
	echo "uname:      $(uname -a)"
	echo "emcc path:  $(command -v emcc)"
	emcc --version | head -1
	echo "node path:  $(command -v "$NODE")"
	"$NODE" --version
} | tee "$ARTIFACTS_DIR/versions.txt"

# A stale in-tree config.mk would contaminate this build.
if [ -f config.mk ]; then
	log "found existing config.mk; running make distclean"
	make distclean
fi

log "configuring (CC=emcc AR=emar RANLIB=emranlib --disable-threading --disable-shared --enable-cblas wasm32)"
if ! ./configure CC=emcc AR=emar RANLIB=emranlib \
	--disable-threading --disable-shared --enable-cblas wasm32 \
	> "$ARTIFACTS_DIR/configure.log" 2>&1; then
	tail -n 40 "$ARTIFACTS_DIR/configure.log" >&2
	die "configure failed; see $ARTIFACTS_DIR/configure.log"
fi

log "building libblis.a (-j$JOBS)"
if ! make -j"$JOBS" > "$ARTIFACTS_DIR/build.log" 2>&1; then
	tail -n 40 "$ARTIFACTS_DIR/build.log" >&2
	die "build failed; see $ARTIFACTS_DIR/build.log"
fi

# Link through the Makefile's LINKER variable so the configure-computed
# LDFLAGS (-lm etc.) are preserved. --preload-file bakes the fast inputs
# into test_libblis.data for the Emscripten virtual FS; the heap must
# grow past the 16MB default for the testsuite's allocations.
emcc_test_linker="emcc"
emcc_test_linker="$emcc_test_linker -sALLOW_MEMORY_GROWTH=1"
emcc_test_linker="$emcc_test_linker --preload-file $GENERAL_INPUT@$GENERAL_INPUT"
emcc_test_linker="$emcc_test_linker --preload-file $OPS_INPUT@$OPS_INPUT"

log "linking $TESTSUITE_BIN (LINKER=\"$emcc_test_linker\")"
if ! make testsuite-bin LINKER="$emcc_test_linker" > "$ARTIFACTS_DIR/link.log" 2>&1; then
	tail -n 40 "$ARTIFACTS_DIR/link.log" >&2
	die "testsuite link failed; see $ARTIFACTS_DIR/link.log"
fi

[ -f "$TESTSUITE_BIN" ] || die "testsuite binary '$TESTSUITE_BIN' was not produced"
[ -f "${TESTSUITE_BIN%.x}.wasm" ] || die "wasm output '${TESTSUITE_BIN%.x}.wasm' was not produced"

log "running testsuite under $NODE"
run_status=0
# Run the binary directly: the Makefile's own run targets pipe through
# tee and would mask the process exit status.
"$NODE" "./$TESTSUITE_BIN" -g "$GENERAL_INPUT" -o "$OPS_INPUT" \
	> "$TESTSUITE_OUT" 2>&1 || run_status=$?

cp "$TESTSUITE_OUT" "$ARTIFACTS_DIR/output.testsuite"

if [ "$run_status" -ne 0 ]; then
	tail -n 40 "$TESTSUITE_OUT" >&2 || true
	die "testsuite process exited with status $run_status; see $ARTIFACTS_DIR/output.testsuite"
fi

ci/wasm/check-testsuite-output.sh "$TESTSUITE_OUT"

# Build and run the double-precision gemm microkernel contract test
# against the archive that was just produced.
TEST_DGEMM_BIN="test-dgemm.js"

log "compiling ci/wasm/test-dgemm.c"
if ! emcc -O2 -msimd128 -Iinclude/wasm32 \
	ci/wasm/test-dgemm.c lib/wasm32/libblis.a -lm -sALLOW_MEMORY_GROWTH=1 \
	-o "$ARTIFACTS_DIR/$TEST_DGEMM_BIN" > "$ARTIFACTS_DIR/test-dgemm.build.log" 2>&1; then
	tail -n 40 "$ARTIFACTS_DIR/test-dgemm.build.log" >&2
	die "test-dgemm compile failed; see $ARTIFACTS_DIR/test-dgemm.build.log"
fi

log "running $TEST_DGEMM_BIN under $NODE"
if ! "$NODE" "$ARTIFACTS_DIR/$TEST_DGEMM_BIN" \
	> "$ARTIFACTS_DIR/test-dgemm.log" 2>&1; then
	tail -n 40 "$ARTIFACTS_DIR/test-dgemm.log" >&2
	die "test-dgemm exited nonzero; see $ARTIFACTS_DIR/test-dgemm.log"
fi
cat "$ARTIFACTS_DIR/test-dgemm.log"


for kernel in sgemm; do
log "compiling and running test-$kernel"
emcc -O2 -msimd128 -Iinclude/wasm32 \
"ci/wasm/test-$kernel.c" lib/wasm32/libblis.a -lm \
-sALLOW_MEMORY_GROWTH=1 -o "$ARTIFACTS_DIR/test-$kernel.js" \
> "$ARTIFACTS_DIR/test-$kernel.build.log" 2>&1
"$NODE" "$ARTIFACTS_DIR/test-$kernel.js" \
> "$ARTIFACTS_DIR/test-$kernel.log" 2>&1
cat "$ARTIFACTS_DIR/test-$kernel.log"
done

log "WASM testsuite passed; artifacts in $ARTIFACTS_DIR"
