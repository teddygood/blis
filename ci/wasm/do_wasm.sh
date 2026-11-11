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

# run_wasm_test <name> <emcc arguments...>: link $ARTIFACTS_DIR/<name>.js,
# run it under $NODE and print its output. A failing step prints the end
# of its log before stopping.
run_wasm_test() {
	local name=$1
	shift
	log "compiling $name"
	if ! emcc "$@" -o "$ARTIFACTS_DIR/$name.js" > "$ARTIFACTS_DIR/$name.build.log" 2>&1; then
		tail -n 40 "$ARTIFACTS_DIR/$name.build.log" >&2
		die "$name failed to build; see $ARTIFACTS_DIR/$name.build.log"
	fi
	log "running $name under $NODE"
	if ! "$NODE" "$ARTIFACTS_DIR/$name.js" > "$ARTIFACTS_DIR/$name.log" 2>&1; then
		tail -n 40 "$ARTIFACTS_DIR/$name.log" >&2
		die "$name exited nonzero; see $ARTIFACTS_DIR/$name.log"
	fi
	cat "$ARTIFACTS_DIR/$name.log"
}

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

# Record toolchain and commit metadata for the CI artifacts. Untracked
# sources count as changes too: the build compiles every file under the
# kernel and config directories.
if git_head="$(git rev-parse HEAD 2>/dev/null)"; then
	git_changes="$(git --no-optional-locks status --porcelain 2>/dev/null || true)"
	git_dirty="$(printf '%s' "$git_changes" | grep -c . || true)"
else
	git_head="unknown"
	git_changes=""
	git_dirty="unknown"
fi
{
	echo "date:       $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
	echo "git commit: $git_head"
	echo "git dirty:  $git_dirty changed or untracked paths"
	if [ -n "$git_changes" ]; then
		printf '%s\n' "$git_changes" | sed -n '1,20s/^/            /p'
	fi
	echo "uname:      $(uname -a)"
	echo "emcc path:  $(command -v emcc)"
	emcc --version | sed -n '1p'
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

# Call the Fortran-77 BLAS symbols with the prototypes an external caller
# such as SciPy uses. wasm-ld only warns about a signature mismatch and
# turns the call into a trap, so the link must treat warnings as errors.
# test-f77-abi.c checks results for a subset of routines;
# test-f77-abi-scipy.c only links a direct call to every BLAS routine
# that SciPy declares.
F77_ABI_FLAGS=( -O2 lib/wasm32/libblis.a -lm -sALLOW_MEMORY_GROWTH=1
	'-Wl,--fatal-warnings' )
run_wasm_test test-f77-abi ci/wasm/test-f77-abi.c "${F77_ABI_FLAGS[@]}"
run_wasm_test test-f77-abi-scipy ci/wasm/test-f77-abi-scipy.c "${F77_ABI_FLAGS[@]}"

# Negative controls: the same links with one deliberately wrong prototype
# must fail, or the probes above could not detect a mismatch.
for probe in test-f77-abi test-f77-abi-scipy; do
	log "checking that a mismatched F77 prototype in $probe.c fails the link"
	neg_log="$ARTIFACTS_DIR/$probe-negative.build.log"
	if emcc -DF77_ABI_NEGATIVE_CONTROL "ci/wasm/$probe.c" "${F77_ABI_FLAGS[@]}" \
		-o "$ARTIFACTS_DIR/$probe-negative.js" > "$neg_log" 2>&1; then
		die "a mismatched F77 prototype in $probe.c linked cleanly; see $neg_log"
	fi
	if ! grep -q 'signature mismatch' "$neg_log"; then
		tail -n 40 "$neg_log" >&2
		die "the $probe negative-control link failed for an unexpected reason; see $neg_log"
	fi
done

# Contract test for the double-precision gemm microkernel and the public
# entry points that reach it, against the archive that was just built.
run_wasm_test test-dgemm -O2 -msimd128 -Iinclude/wasm32 \
	ci/wasm/test-dgemm.c lib/wasm32/libblis.a -lm -sALLOW_MEMORY_GROWTH=1

# The same test under AddressSanitizer. The test and the kernel source are
# instrumented (libblis.a is not), so the direct microkernel cases, which
# pass exact-size buffers, catch out-of-bounds kernel loads and stores.
# The kernel object precedes the archive so that it replaces the archive's
# copy.
run_wasm_test test-dgemm-asan -O1 -g -msimd128 -fsanitize=address \
	-Iinclude/wasm32 ci/wasm/test-dgemm.c \
	kernels/wasm32/3/bli_dgemm_wasm32_simd128_4x4.c \
	lib/wasm32/libblis.a -lm -sALLOW_MEMORY_GROWTH=1

# Single-precision contract test, plain and under AddressSanitizer.
run_wasm_test test-sgemm -O2 -msimd128 -Iinclude/wasm32 \
	ci/wasm/test-sgemm.c lib/wasm32/libblis.a -lm -sALLOW_MEMORY_GROWTH=1
run_wasm_test test-sgemm-asan -O1 -g -msimd128 -fsanitize=address \
	-Iinclude/wasm32 ci/wasm/test-sgemm.c \
	kernels/wasm32/3/bli_sgemm_wasm32_simd128_4x8.c \
	lib/wasm32/libblis.a -lm -sALLOW_MEMORY_GROWTH=1

# Contract test for the small single-precision GEMM handler: plain, under
# AddressSanitizer with the handler source instrumented, and with a 16 KiB
# stack. The handler keeps its 36 KiB of packing buffers off the stack, and
# STACK_OVERFLOW_CHECK=2 aborts the run if any frame outgrows the stack. The
# small-stack build (-DTEST_SMALL_STACK) skips the public calls that BLIS
# computes on its conventional path, which needs more than 16 KiB of stack
# for some of the fallback cases.
run_wasm_test test-small-sup -O2 -msimd128 -Iinclude/wasm32 \
	ci/wasm/test-small-sup.c lib/wasm32/libblis.a -lm -sALLOW_MEMORY_GROWTH=1
run_wasm_test test-small-sup-asan -O1 -g -msimd128 -fsanitize=address \
	-Iinclude/wasm32 ci/wasm/test-small-sup.c \
	kernels/wasm32/3/bli_sgemmsup_wasm32_simd128.c \
	lib/wasm32/libblis.a -lm -sALLOW_MEMORY_GROWTH=1
run_wasm_test test-small-sup-stack16k -O2 -msimd128 -DTEST_SMALL_STACK \
	-Iinclude/wasm32 ci/wasm/test-small-sup.c lib/wasm32/libblis.a -lm \
	-sALLOW_MEMORY_GROWTH=1 -sSTACK_SIZE=16384 -sSTACK_OVERFLOW_CHECK=2

# Contract test for the real dotv and axpyf kernels against the reference
# kernels, plain and under AddressSanitizer with both kernel sources
# instrumented.
run_wasm_test test-l1f -O2 -msimd128 -Iinclude/wasm32 \
	ci/wasm/test-l1f.c lib/wasm32/libblis.a -lm -sALLOW_MEMORY_GROWTH=1
run_wasm_test test-l1f-asan -O1 -g -msimd128 -fsanitize=address \
	-Iinclude/wasm32 ci/wasm/test-l1f.c \
	kernels/wasm32/1/bli_dotv_wasm32_simd128.c \
	kernels/wasm32/1f/bli_axpyf_wasm32_simd128.c \
	lib/wasm32/libblis.a -lm -sALLOW_MEMORY_GROWTH=1

# Contract tests for the trsm microkernels of both real precisions and the
# public cblas_?trsm calls that reach them through gemmtrsm.
run_wasm_test test-dtrsm -O2 -msimd128 -Iinclude/wasm32 \
	ci/wasm/test-dtrsm.c lib/wasm32/libblis.a -lm -sALLOW_MEMORY_GROWTH=1
run_wasm_test test-strsm -O2 -msimd128 -Iinclude/wasm32 \
	ci/wasm/test-strsm.c lib/wasm32/libblis.a -lm -sALLOW_MEMORY_GROWTH=1

log "WASM testsuite passed; artifacts in $ARTIFACTS_DIR"
