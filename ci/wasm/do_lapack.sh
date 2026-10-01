#!/bin/bash

#
# Link the candidate BLIS archive (lib/wasm32/libblis.a, built by
# ci/wasm/do_wasm.sh) into the pinned semicolon-lapack CMocka suite and
# run every registered test as WebAssembly under Node.js.
#
# Usage:
#
#   EMSDK=/path/to/emsdk NODE=/path/to/node-24 ci/wasm/do_lapack.sh
#

set -euo pipefail

script_name=${0##*/}

cd "$(dirname "$0")/../.."
DIST_PATH="$(pwd)"

WORK_DIR="${WORK_DIR:-$DIST_PATH/wasm-lapack-work}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$DIST_PATH/wasm-artifacts/lapack}"
SEMI_LAPACK_REPO="https://github.com/ilayn/semicolon-lapack.git"
SEMI_LAPACK_SHA="${SEMI_LAPACK_SHA:-3adf36b10db0a19dbaf4bed2e28cf08b5f74498e}"
CMOCKA_REPO="https://gitlab.com/cmocka/cmocka.git"
CMOCKA_SHA="${CMOCKA_SHA:-1131a62907287aa9951e22ae93b2e8425fa2de19}"  # tag cmocka-2.0.0
EXPECTED_TEST_COUNT="${EXPECTED_TEST_COUNT:-480}"
REQUIRED_NODE_MAJOR="${REQUIRED_NODE_MAJOR:-24}"
MESON="${MESON:-meson}"
JOBS="${JOBS:-$(nproc)}"
TEST_JOBS="${TEST_JOBS:-2}"
TIMEOUT_MULT="${TIMEOUT_MULT:-4}"
BUILDDIR="builddir"
CHECKER="$DIST_PATH/ci/wasm/check-lapack-results.py"

CANDIDATE_LIB="$DIST_PATH/lib/wasm32/libblis.a"
CANDIDATE_INC="$DIST_PATH/include/wasm32"

log() { echo "$script_name: $*"; }
die() { echo "$script_name: error: $*" >&2; exit 1; }

# Fresh output dirs so stale artifacts cannot masquerade as new evidence.
rm -rf "$ARTIFACTS_DIR" "$WORK_DIR"
mkdir -p "$ARTIFACTS_DIR" "$WORK_DIR"

# copy_log <src> <dst>: copy a log into the artifacts dir, sanitized.
# testlog.json records carry the full process environment and testlog.txt
# opens with an "Inherited environment:" block; both can contain secrets
# and must not reach a published artifact verbatim. Raw files stay in the
# (ignored) work dir for debugging.
copy_log() {
	local src=$1 dst=$2
	case "$dst" in
	*.json)
		python3 - "$src" "$dst" <<'EOF'
import json, sys
out = []
with open(sys.argv[1], encoding="utf-8") as fh:
    for line in fh:
        if line.strip():
            rec = json.loads(line)
            if isinstance(rec, dict) and "env" in rec:
                rec["env"] = {}
            out.append(json.dumps(rec))
with open(sys.argv[2], "w", encoding="utf-8") as fh:
    fh.write("\n".join(out) + "\n" if out else "")
EOF
		;;
	*.txt|*.xml)
		python3 - "$src" "$dst" <<'EOF'
import re, sys
with open(sys.argv[1], encoding="utf-8", errors="replace") as fh:
    s = fh.read()
s = re.sub(r"Inherited environment:.*?(?=\n\n|\Z)",
           "Inherited environment: <stripped by ci/wasm/do_lapack.sh>\n",
           s, flags=re.S)
with open(sys.argv[2], "w", encoding="utf-8") as fh:
    fh.write(s)
EOF
		;;
	*)
		cp "$src" "$dst"
		;;
	esac
}

# copy_testlogs: env-sanitize meson-logs into the artifact dir; used by
# the success path and the failure trap.
copy_testlogs() {
	[ -n "${SEMI_BUILD:-}" ] && [ -d "$SEMI_BUILD/meson-logs" ] || return 0
	for f in meson-log.txt testlog.txt testlog.json testlog.junit.xml; do
		[ -f "$SEMI_BUILD/meson-logs/$f" ] \
			&& copy_log "$SEMI_BUILD/meson-logs/$f" "$ARTIFACTS_DIR/$f" || true
	done
	return 0
}

# Fail-closed artifact hygiene: the workflow uploads $ARTIFACTS_DIR with
# `if: always()`, so anything still in it at exit is published. Every exit
# sweeps for sensitive environment key names and prunes offenders to a
# marker file; names only, never values.
HYGIENE_PATTERN='ORCA_|SSH_AUTH|SSH_ASKPASS|GPG_AGENT|GITHUB_TOKEN|CI_JOB_TOKEN|_PASSWORD=|_SECRET=|_TOKEN='

sweep_artifacts() {
	local offenders f
	offenders="$(grep -rlE "$HYGIENE_PATTERN" "$ARTIFACTS_DIR" 2>/dev/null || true)"
	while IFS= read -r f; do
		[ -n "$f" ] || continue
		printf 'withheld by ci/wasm/do_lapack.sh hygiene sweep: contained sensitive environment entries\n' > "$f.WITHHELD.txt"
		rm -f "$f"
	done <<< "$offenders"
	# Second pass: if anything still matches, withhold the whole directory.
	if grep -rlE "$HYGIENE_PATTERN" "$ARTIFACTS_DIR" > /dev/null 2>&1; then
		rm -rf "$ARTIFACTS_DIR"
		mkdir -p "$ARTIFACTS_DIR"
		printf 'artifacts withheld entirely: sanitization could not be guaranteed\n' > "$ARTIFACTS_DIR/ARTIFACTS-WITHHELD.txt"
	fi
	return 0
}

# On failure preserve meson's logs (env-sanitized), then always sweep so
# the artifact dir is clean before the `if: always()` upload sees it.
preserve_logs() {
	status=$?
	set +e
	[ "$status" -ne 0 ] && copy_testlogs
	sweep_artifacts
	return 0
}
trap preserve_logs EXIT

if [ -n "${EMSDK:-}" ]; then
	[ -f "$EMSDK/emsdk_env.sh" ] || die "EMSDK is set to '$EMSDK' but $EMSDK/emsdk_env.sh does not exist"
	# shellcheck disable=SC1090
	EMSDK_QUIET=1 source "$EMSDK/emsdk_env.sh"
fi

for tool in emcc emar emranlib emcmake cmake ninja pkg-config git python3 "$MESON"; do
	command -v "$tool" > /dev/null 2>&1 || die "$tool not found on PATH; set EMSDK=/path/to/emsdk or activate emsdk first"
done

# llvm-ar/llvm-readobj live in the emsdk bin dir next to emcc; `file`
# only reports "ar archive", not the member format.
EMCC_BIN="$(command -v emcc)"
LLVM_BIN="$(cd "$(dirname "$EMCC_BIN")/.." && pwd)/bin"
for tool in llvm-ar llvm-readobj; do
	[ -x "$LLVM_BIN/$tool" ] || die "$tool not found at $LLVM_BIN; cannot inspect archive member format"
done

# Prefer an explicit NODE, then the node bundled with emsdk, then PATH.
NODE="${NODE:-${EMSDK_NODE:-node}}"
NODE_BIN="$(command -v "$NODE" 2>/dev/null || true)"
[ -n "$NODE_BIN" ] || die "node executable '$NODE' not found"
NODE_BIN="$(cd "$(dirname "$NODE_BIN")" && pwd)/$(basename "$NODE_BIN")"

# emsdk bundles its own node whose version varies by release (22.x in
# emsdk 6.0.5), so activation alone does not guarantee the required major.
node_version="$("$NODE_BIN" --version)"
node_major="${node_version#v}"
node_major="${node_major%%.*}"
if [ -n "$REQUIRED_NODE_MAJOR" ] && [ "$node_major" != "$REQUIRED_NODE_MAJOR" ]; then
	die "node is $node_version but the tests must run under Node $REQUIRED_NODE_MAJOR; pass NODE=/path/to/node-$REQUIRED_NODE_MAJOR (or REQUIRED_NODE_MAJOR= to disable)"
fi

# Defence in depth next to the absolute exe_wrapper in the cross file.
NODE_DIR="$(dirname "$NODE_BIN")"
export PATH="$NODE_DIR:$PATH"
[ "$(command -v node)" = "$NODE_BIN" ] || die "failed to put '$NODE_BIN' first on PATH"

# A host BLAS/cmocka can only leak in through pkg-config, compiler search
# paths or make-style flag variables. Clear all of them; dependency lookup
# is re-allowed only through the explicit settings below.
unset PKG_CONFIG_PATH PKG_CONFIG_SYSROOT_DIR PKG_CONFIG_SYSTEM_INCLUDE_PATH \
	PKG_CONFIG_SYSTEM_LIBRARY_PATH PKG_CONFIG_ALLOW_SYSTEM_CFLAGS \
	PKG_CONFIG_ALLOW_SYSTEM_LIBS PKG_CONFIG_ALLOW_CROSS_COMPILATION \
	CFLAGS CPPFLAGS CXXFLAGS FFLAGS LDFLAGS LIBS CC CXX AR RANLIB LD \
	LIBRARY_PATH LD_LIBRARY_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH \
	CMAKE_PREFIX_PATH CMAKE_INCLUDE_PATH CMAKE_LIBRARY_PATH \
	EMCC_CFLAGS EMCC_CPPFLAGS EMCC_LDFLAGS \
	CMOCKA_TEST_FILTER CMOCKA_SKIP_FILTER || true

# Force cmocka TAP output; the testlog checker relies on this grammar.
export CMOCKA_MESSAGE_OUTPUT=TAP

# Require the in-tree wasm32 build left by ci/wasm/do_wasm.sh; anything
# else is rejected rather than rebuilt here.
for f in config.mk "$CANDIDATE_LIB" "$CANDIDATE_INC/blis.h" "$CANDIDATE_INC/cblas.h"; do
	[ -f "$f" ] || die "candidate build product '$f' is missing; run ci/wasm/do_wasm.sh (or ./configure CC=emcc ... wasm32 && make) in this checkout first"
done
[ -s "$CANDIDATE_LIB" ] || die "candidate archive '$CANDIDATE_LIB' is empty"

cfg() { grep -E "^$1[[:space:]]*:=[[:space:]]*$2\$" config.mk > /dev/null 2>&1; }
cfg CONFIG_NAME wasm32 || die "config.mk is not a wasm32 configuration"
cfg THREADING_MODEL single || die "config.mk is not single-threaded (THREADING_MODEL)"
cfg MK_ENABLE_STATIC yes || die "candidate build does not produce a static archive"
cfg MK_ENABLE_SHARED no || die "candidate build unexpectedly enables shared libraries"
cfg MK_ENABLE_CBLAS yes || die "candidate build does not enable the CBLAS layer"
cfg CC emcc || die "candidate was not built with emcc (config.mk CC)"

# Cross-compilation skips the runtime integer-width probe, so prove the
# configured BLAS integer width at compile time instead.
cat > "$WORK_DIR/intwidth.c" <<'EOF'
#include "blis.h"
_Static_assert(BLIS_BLAS_INT_TYPE_SIZE == 32, "expected 32-bit BLAS integers");
int main(void) { return 0; }
EOF
emcc -fsyntax-only -I"$CANDIDATE_INC" "$WORK_DIR/intwidth.c" \
	|| die "candidate headers do not configure 32-bit BLAS integers"

# Verify a member is a wasm object, not just that the file is an ar
# archive. (No `| head` on llvm-ar output: its early exit would SIGPIPE
# llvm-ar under pipefail.)
members="$("$LLVM_BIN/llvm-ar" t "$CANDIDATE_LIB")"
first_member="${members%%$'\n'*}"
[ -n "$first_member" ] || die "candidate archive has no members"
"$LLVM_BIN/llvm-ar" p "$CANDIDATE_LIB" "$first_member" > "$WORK_DIR/member.o"
member_format="$("$LLVM_BIN/llvm-readobj" -h "$WORK_DIR/member.o" | sed -n 's/.*Format: \(.*\)/\1/p')"
echo "$member_format" | grep -qi 'WASM' \
	|| die "archive member '$first_member' is not a WASM object (llvm-readobj format: $member_format)"
member_count="$(printf '%s\n' "$members" | wc -l)"

candidate_sha_pre="$(sha256sum "$CANDIDATE_LIB" | awk '{print $1}')"
header_sha="$(sha256sum "$CANDIDATE_INC/blis.h" "$CANDIDATE_INC/cblas.h")"
blis_version="$(awk -F':=' '/^VERSION/{gsub(/ /,"",$2); print $2}' config.mk)"
git_head="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
git_dirty="$(git status --porcelain 2>/dev/null | grep -cv '^??' || true)"

{
	echo "date:        $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
	echo "git commit:  $git_head"
	echo "git dirty tracked files: $git_dirty"
	echo "uname:       $(uname -a)"
	echo "emcc path:   $EMCC_BIN"
	emcc --version | sed -n '1p'
	echo "node path:   $NODE_BIN"
	echo "node:        $node_version"
	echo "meson:       $("$MESON" --version)"
	echo "ninja:       $(ninja --version)"
	echo "cmake:       $(cmake --version | sed -n '1p')"
	echo "pkg-config:  $(pkg-config --version)"
	echo "python3:     $(python3 --version 2>&1)"
	echo "blis version (config.mk):  $blis_version"
	echo "candidate libblis.a:       $candidate_sha_pre  $CANDIDATE_LIB"
	echo "candidate archive members: $member_count ($member_format objects; first: $first_member)"
	echo "$header_sha" | sed 's/^/candidate header:      /'
	echo "semicolon-lapack commit: $SEMI_LAPACK_SHA ($SEMI_LAPACK_REPO)"
	echo "cmocka commit:           $CMOCKA_SHA ($CMOCKA_REPO)"
} | tee "$ARTIFACTS_DIR/versions.txt"

# fetch_pinned <name> <repo> <sha> <dest>
fetch_pinned() {
	local name=$1 repo=$2 sha=$3 dest=$4 got
	{
		echo "=== $name ==="
		echo "repo:   $repo"
		echo "pinned: $sha"
	} >> "$ARTIFACTS_DIR/deps.log"

	git init -q "$dest"
	git -C "$dest" remote add origin "$repo"
	if ! git -C "$dest" fetch -q --depth 1 origin "$sha" >> "$ARTIFACTS_DIR/deps.log" 2>&1; then
		log "shallow fetch of $name@$sha refused; fetching full history instead"
		git -C "$dest" fetch -q origin >> "$ARTIFACTS_DIR/deps.log" 2>&1 \
			|| die "fetch of $repo failed; see $ARTIFACTS_DIR/deps.log"
	fi
	git -C "$dest" checkout -q "$sha" >> "$ARTIFACTS_DIR/deps.log" 2>&1 \
		|| die "cannot checkout $name@$sha; see $ARTIFACTS_DIR/deps.log"
	got="$(git -C "$dest" rev-parse HEAD)"
	echo "head:   $got" >> "$ARTIFACTS_DIR/deps.log"
	[ "$got" = "$sha" ] || die "$name checkout resolved to $got, expected pinned $sha"
}

: > "$ARTIFACTS_DIR/deps.log"
SEMI_DIR="$WORK_DIR/semicolon-lapack"
CMOCKA_DIR="$WORK_DIR/cmocka"
CMOCKA_PREFIX="$WORK_DIR/prefix"
PC_DIR="$WORK_DIR/pkgconfig"

log "fetching semicolon-lapack @ $SEMI_LAPACK_SHA"
fetch_pinned semicolon-lapack "$SEMI_LAPACK_REPO" "$SEMI_LAPACK_SHA" "$SEMI_DIR"
log "fetching cmocka @ $CMOCKA_SHA (tag cmocka-2.0.0)"
fetch_pinned cmocka "$CMOCKA_REPO" "$CMOCKA_SHA" "$CMOCKA_DIR"

# Cross-build CMocka 2.0 as a wasm32 static library into a private prefix.

log "building cmocka (emcmake, static, prefix=$CMOCKA_PREFIX)"
if ! emcmake cmake -S "$CMOCKA_DIR" -B "$CMOCKA_DIR/build" \
	-DCMAKE_INSTALL_PREFIX="$CMOCKA_PREFIX" \
	-DCMAKE_INSTALL_LIBDIR=lib \
	-DBUILD_SHARED_LIBS=OFF -DWITH_EXAMPLES=OFF -DUNIT_TESTING=OFF \
	> "$ARTIFACTS_DIR/cmocka-configure.log" 2>&1; then
	tail -n 40 "$ARTIFACTS_DIR/cmocka-configure.log" >&2
	die "cmocka configure failed; see $ARTIFACTS_DIR/cmocka-configure.log"
fi
if ! cmake --build "$CMOCKA_DIR/build" --parallel "$JOBS" > "$ARTIFACTS_DIR/cmocka-build.log" 2>&1; then
	tail -n 40 "$ARTIFACTS_DIR/cmocka-build.log" >&2
	die "cmocka build failed; see $ARTIFACTS_DIR/cmocka-build.log"
fi
if ! cmake --install "$CMOCKA_DIR/build" > "$ARTIFACTS_DIR/cmocka-install.log" 2>&1; then
	tail -n 40 "$ARTIFACTS_DIR/cmocka-install.log" >&2
	die "cmocka install failed; see $ARTIFACTS_DIR/cmocka-install.log"
fi
[ -f "$CMOCKA_PREFIX/lib/libcmocka.a" ] || die "cmocka install did not produce $CMOCKA_PREFIX/lib/libcmocka.a"
[ -f "$CMOCKA_PREFIX/lib/pkgconfig/cmocka.pc" ] || die "expected '$CMOCKA_PREFIX/lib/pkgconfig/cmocka.pc' is missing"
cmocka_sha="$(sha256sum "$CMOCKA_PREFIX/lib/libcmocka.a" | awk '{print $1}')"

# The generated .pc names the candidate archive by absolute path (no
# -lblis indirection); the unique name can only resolve to this file.
mkdir -p "$PC_DIR"
cat > "$PC_DIR/blis-under-test.pc" <<EOF
# Generated by ci/wasm/do_lapack.sh; Libs is the absolute archive path.
prefix=$DIST_PATH
libdir=$DIST_PATH/lib/wasm32
includedir=$CANDIDATE_INC

Name: blis-under-test
Description: wasm32 libblis.a built from the checked-out PR candidate
Version: $blis_version
Libs: $CANDIDATE_LIB -lm
Cflags: -I$CANDIDATE_INC
EOF
cp "$PC_DIR/blis-under-test.pc" "$ARTIFACTS_DIR/blis-under-test.pc"

# Cross-file overlay over upstream's pinned emscripten-wasm.ini: pin
# exe_wrapper to the resolved Node and restrict meson's pkg-config
# search to the generated dirs only.
OVERLAY_INI="$WORK_DIR/lapack-overlay.ini"
cat > "$OVERLAY_INI" <<EOF
# Generated by ci/wasm/do_lapack.sh; overlay for emscripten-wasm.ini
[binaries]
exe_wrapper = '$NODE_BIN'

[properties]
pkg_config_libdir = ['$PC_DIR', '$CMOCKA_PREFIX/lib/pkgconfig']
pkg_config_path = []
EOF
cp "$OVERLAY_INI" "$ARTIFACTS_DIR/lapack-overlay.ini"

# Mirror the restriction for direct pkg-config calls (meson itself only
# honours the ini properties).
export PKG_CONFIG_LIBDIR="$PC_DIR:$CMOCKA_PREFIX/lib/pkgconfig"

pkg-config --exists blis-under-test \
	|| die "pkg-config cannot see blis-under-test (PKG_CONFIG_LIBDIR=$PKG_CONFIG_LIBDIR)"
pc_file="$(pkg-config --variable=pcfiledir blis-under-test)"
[ "$pc_file" = "$PC_DIR" ] \
	|| die "blis-under-test.pc resolved from '$pc_file', expected '$PC_DIR'"
case "$(pkg-config --libs blis-under-test)" in
	*"$CANDIDATE_LIB"*) ;;
	*) die "pkg-config --libs blis-under-test does not contain the absolute candidate archive" ;;
esac
pkg-config --exists cmocka || die "pkg-config cannot see cmocka in the private prefix"
[ "$(pkg-config --variable=pcfiledir cmocka)" = "$CMOCKA_PREFIX/lib/pkgconfig" ] \
	|| die "cmocka.pc resolved outside the private prefix"
# The candidate package must be the ONLY thing the generated dir provides.
[ "$(pkg-config --variable=pcfiledir blis 2>/dev/null || echo none)" = "none" ] \
	|| die "a 'blis' package unexpectedly remains visible under the restricted PKG_CONFIG_LIBDIR"

SEMI_BUILD="$SEMI_DIR/$BUILDDIR"

log "meson setup (-Dblas=blis-under-test, cross-file emscripten-wasm.ini + overlay)"
# shellcheck disable=SC2086
if ! (cd "$SEMI_DIR" && "$MESON" setup "$BUILDDIR" \
	--cross-file emscripten-wasm.ini \
	--cross-file "$OVERLAY_INI" \
	-Dblas=blis-under-test -Dtests=true -Dbenchmarks=false \
	-Dfabi_shim=false -DUSE_INT64=false --default-library=static \
	--wrap-mode=nofallback ${SEMI_LAPACK_MESON_ARGS:-} \
	> "$ARTIFACTS_DIR/meson-setup.log" 2>&1); then
	tail -n 40 "$ARTIFACTS_DIR/meson-setup.log" >&2
	die "meson setup failed; see $ARTIFACTS_DIR/meson-setup.log"
fi

# The setup log must report the private package resolved inside this
# checkout; anything else means a different BLAS was consumed.
if ! grep -F "BLAS: blis-under-test" "$ARTIFACTS_DIR/meson-setup.log" | grep -qF "$DIST_PATH/lib/wasm32"; then
	grep -E 'BLAS|blis' "$ARTIFACTS_DIR/meson-setup.log" >&2 || true
	die "meson did not resolve blis-under-test under $DIST_PATH/lib/wasm32; see $ARTIFACTS_DIR/meson-setup.log"
fi

# Registered-test manifest the checker reconciles against the testlog.
"$MESON" introspect "$SEMI_BUILD" --tests > "$ARTIFACTS_DIR/lapack-tests.json"
ntests="$(python3 -c 'import json, sys; print(len(json.load(sys.stdin)))' < "$ARTIFACTS_DIR/lapack-tests.json")"
log "meson registered $ntests tests"
[ "$ntests" -gt 0 ] || die "meson registered zero tests; refusing to treat an empty suite as a pass"

log "ninja build (-j$JOBS)"
if ! ninja -C "$SEMI_BUILD" -j "$JOBS" > "$ARTIFACTS_DIR/meson-build.log" 2>&1; then
	tail -n 40 "$ARTIFACTS_DIR/meson-build.log" >&2
	die "ninja build failed; see $ARTIFACTS_DIR/meson-build.log"
fi

# Linkage evidence: every test link command must contain the candidate
# archive by absolute path, and no -l<blas> flag may appear.

ninja -C "$SEMI_BUILD" -t commands > "$ARTIFACTS_DIR/link-commands.txt"

test_links="$(grep -cE -- "-o tests/[^[:space:]]+\.js([[:space:]]|$)" "$ARTIFACTS_DIR/link-commands.txt" || true)"
links_with_lib="$(grep -F -- "$CANDIDATE_LIB" "$ARTIFACTS_DIR/link-commands.txt" | grep -cE -- "-o tests/[^[:space:]]+\.js([[:space:]]|$)" || true)"
[ "$test_links" -ge "$ntests" ] || die "found only $test_links test link commands, expected at least $ntests"
[ "$links_with_lib" -eq "$test_links" ] \
	|| die "$((test_links - links_with_lib)) test executable(s) link without the candidate archive"
if grep -E -- "-o tests/[^[:space:]]+\.js([[:space:]]|$)" "$ARTIFACTS_DIR/link-commands.txt" | grep -qE -- '-l(blis|blas|cblas|openblas|lapack-se|refblas)\b'; then
	grep -nE -- '-l(blis|blas|cblas|openblas|refblas)\b' "$ARTIFACTS_DIR/link-commands.txt" >&2
	die "a search-path -l<blas> flag appears in a test link command"
fi

{
	echo "candidate BLIS tree:    $DIST_PATH"
	echo "candidate BLIS commit:  $git_head"
	echo
	echo "candidate libblis.a:    $candidate_sha_pre  $CANDIDATE_LIB"
	echo "cmocka libcmocka.a:     $cmocka_sha  $CMOCKA_PREFIX/lib/libcmocka.a"
	echo
	echo "PKG_CONFIG_LIBDIR:      $PKG_CONFIG_LIBDIR"
	echo "blis-under-test.pc dir: $pc_file"
	echo "pkg-config --modversion blis-under-test: $(pkg-config --modversion blis-under-test)"
	echo "pkg-config --cflags blis-under-test:     $(pkg-config --cflags blis-under-test)"
	echo "pkg-config --libs --static blis-under-test: $(pkg-config --libs --static blis-under-test)"
	echo "pkg-config --modversion cmocka:          $(pkg-config --modversion cmocka)"
	echo
	echo "test link commands containing the candidate archive: $links_with_lib of $test_links"
	echo
	echo "--- meson introspect --dependencies ---"
	"$MESON" introspect "$SEMI_BUILD" --dependencies \
		| python3 -c 'import json, sys; print(json.dumps([d for d in json.load(sys.stdin) if d.get("name") in ("blis-under-test", "cmocka")], indent=2))'
} > "$ARTIFACTS_DIR/linkage.txt" 2>&1

# Trace archive members to verify that the candidate supplies CBLAS
# symbols; a path on a command line is not proof.
trace_relink() {
	local exe=$1 out=$2
	local cmd
	cmd="$(ninja -C "$SEMI_BUILD" -t commands "$exe" | tail -n 1)"
	[ -n "$cmd" ] || { echo "no link command for $exe" >&2; return 1; }
	cmd="${cmd/-o $exe/-o $out} -Wl,--trace"
	(cd "$SEMI_BUILD" && eval "$cmd")
}
for exe in tests/d/test_dchkge.js tests/z/test_zchkge.js; do
	name="${exe##*/}"; name="${name%.js}"
	log "trace relink of $exe"
	trace_out="$WORK_DIR/traced-$name.js"
	if ! trace_relink "$exe" "$trace_out" > "$WORK_DIR/trace-$name.log" 2>&1; then
		tail -n 40 "$WORK_DIR/trace-$name.log" >&2
		die "trace relink of $exe failed; cannot prove candidate supplied symbols"
	fi
	cp "$WORK_DIR/trace-$name.log" "$ARTIFACTS_DIR/trace-$name.log"
	pulled="$(grep -cF "$CANDIDATE_LIB(" "$WORK_DIR/trace-$name.log" || true)"
	cblas_pulled="$(grep -cF "$CANDIDATE_LIB(cblas_" "$WORK_DIR/trace-$name.log" || true)"
	{
		echo
		echo "--- -Wl,--trace of $exe: members pulled from candidate archive ---"
		grep -F "$CANDIDATE_LIB(" "$WORK_DIR/trace-$name.log" | head -20 || true
		echo "... ($pulled members total, $cblas_pulled cblas_*.o)"
	} >> "$ARTIFACTS_DIR/linkage.txt"
	[ "$pulled" -gt 0 ] || die "trace relink of $exe pulled no members from $CANDIDATE_LIB"
	[ "$cblas_pulled" -gt 0 ] || die "trace relink of $exe pulled no cblas_* members from $CANDIDATE_LIB"
done

# Standalone probe: cblas_dgemm executed under the selected node.
PROBE_SRC="$WORK_DIR/linkprobe.c"
cat > "$PROBE_SRC" <<'EOF'
#include <stdio.h>
#include "cblas.h"

/* A = diag(2,3), B = I -> C = A. A wrong or stale BLAS turns this into a
   nonzero exit status or wrong values, not a pass. */
int main(void)
{
	double A[4] = {2.0, 0.0, 0.0, 3.0};
	double B[4] = {1.0, 0.0, 0.0, 1.0};
	double C[4] = {0.0, 0.0, 0.0, 0.0};
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
	            2, 2, 2, 1.0, A, 2, B, 2, 0.0, C, 2);
	if (C[0] != 2.0 || C[3] != 3.0 || C[1] != 0.0 || C[2] != 0.0) {
		printf("linkprobe: cblas_dgemm gave C=[%g %g; %g %g]\n", C[0], C[1], C[2], C[3]);
		return 1;
	}
	printf("linkprobe: cblas_dgemm OK, C=diag(2,3)\n");
	return 0;
}
EOF

log "linking cblas_dgemm probe against the candidate archive"
# shellcheck disable=SC2046  # pkg-config flags are intentionally word-split
if ! emcc -sEXIT_RUNTIME=1 -sALLOW_MEMORY_GROWTH=1 -sERROR_ON_UNDEFINED_SYMBOLS=1 \
	$(pkg-config --cflags blis-under-test) "$PROBE_SRC" \
	$(pkg-config --libs blis-under-test) \
	-Wl,--trace -o "$WORK_DIR/linkprobe.js" > "$ARTIFACTS_DIR/probe.log" 2>&1; then
	tail -n 40 "$ARTIFACTS_DIR/probe.log" >&2
	die "probe link failed; see $ARTIFACTS_DIR/probe.log"
fi
grep -qF "$CANDIDATE_LIB(" "$ARTIFACTS_DIR/probe.log" \
	|| die "probe link trace pulled no members from '$CANDIDATE_LIB'"
{
	echo
	echo "--- cblas_dgemm probe trace members (candidate archive) ---"
	grep -F "$CANDIDATE_LIB(" "$ARTIFACTS_DIR/probe.log" | head -10 || true
	echo
	echo "--- probe run under $NODE_BIN ---"
} >> "$ARTIFACTS_DIR/linkage.txt"
if ! "$NODE_BIN" "$WORK_DIR/linkprobe.js" >> "$ARTIFACTS_DIR/linkage.txt" 2>&1; then
	die "link probe failed under node; see $ARTIFACTS_DIR/linkage.txt"
fi

# The archive must be byte-identical across the whole run.
candidate_sha_post="$(sha256sum "$CANDIDATE_LIB" | awk '{print $1}')"
[ "$candidate_sha_pre" = "$candidate_sha_post" ] \
	|| die "candidate archive changed during the run ($candidate_sha_pre -> $candidate_sha_post)"

log "running $ntests meson tests under node $node_version (num-processes $TEST_JOBS, timeout-multiplier $TIMEOUT_MULT)"
test_status=0
if ! (cd "$SEMI_DIR" && "$MESON" test -C "$BUILDDIR" --no-rebuild \
	--print-errorlogs --num-processes "$TEST_JOBS" \
	--timeout-multiplier "$TIMEOUT_MULT" \
	> "$ARTIFACTS_DIR/meson-test.log" 2>&1); then
	test_status=1
fi

# Preserve meson's logs; on failure also the failing executables.
copy_testlogs
if [ "$test_status" -ne 0 ] && [ -f "$SEMI_BUILD/meson-logs/testlog.json" ]; then
	mkdir -p "$ARTIFACTS_DIR/failed-binaries"
	python3 - "$SEMI_BUILD/meson-logs/testlog.json" "$ARTIFACTS_DIR/failed-binaries" <<'EOF' || true
import json, shutil, sys, os
testlog, destdir = sys.argv[1], sys.argv[2]
for line in open(testlog):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if r.get("result") in ("OK", "EXPECTEDFAIL"):
        continue
    cmd = r.get("command") or []
    if len(cmd) >= 2 and cmd[1].endswith(".js") and os.path.exists(cmd[1]):
        base = os.path.basename(cmd[1])
        shutil.copy2(cmd[1], os.path.join(destdir, base))
        wasm = cmd[1][:-3] + ".wasm"
        if os.path.exists(wasm):
            shutil.copy2(wasm, os.path.join(destdir, os.path.basename(wasm)))
EOF
	tail -n 60 "$ARTIFACTS_DIR/meson-test.log" >&2
	die "meson test failed; see $ARTIFACTS_DIR/meson-test.log (failing binaries preserved in $ARTIFACTS_DIR/failed-binaries)"
fi

# meson's exit status is not sufficient: upstream test mains discard
# cmocka's failure count and return 0, and some smoke executables use the
# exitcode protocol. The checker validates every executable's CMocka TAP
# stream in testlog.json instead.
log "validating TAP output of all executables"
if ! python3 "$CHECKER" \
	--testlog "$SEMI_BUILD/meson-logs/testlog.json" \
	--tests-json "$ARTIFACTS_DIR/lapack-tests.json" \
	--src-dir "$SEMI_DIR" \
	--expected-count "$EXPECTED_TEST_COUNT" \
	--node-path "$NODE_BIN" 2>&1 | tee "$ARTIFACTS_DIR/check-results.txt"; then
	die "testlog TAP validation failed; see $ARTIFACTS_DIR/check-results.txt"
fi

{
	echo
	echo "--- checker summary ---"
	cat "$ARTIFACTS_DIR/check-results.txt"
} >> "$ARTIFACTS_DIR/linkage.txt"

# A green run with a sensitive entry still in an artifact means the
# sanitizer failed: fail here and let the exit trap prune the offender.
if grep -rlE "$HYGIENE_PATTERN" "$ARTIFACTS_DIR" > /dev/null 2>&1; then
	grep -rlE "$HYGIENE_PATTERN" "$ARTIFACTS_DIR" | sed "s|^|leaky artifact: |" >&2
	die "artifacts still contain sensitive environment entries; refusing to finish"
fi

log "semicolon-lapack WASM tests passed ($ntests registered); artifacts in $ARTIFACTS_DIR"
