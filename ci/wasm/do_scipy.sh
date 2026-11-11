#!/bin/bash

#
# Build SciPy for Pyodide against the checked-out BLIS tree and run
# SciPy's own test suite under Node.js.
#
# SciPy's WebAssembly CI (.github/workflows/emscripten.yml in scipy/scipy)
# runs cibuildwheel for cp314-pyodide_wasm32. Its before-build step,
# tools/wheels/cibw_before_build_pyodide.sh, clones pyodide/blis at the
# pyodide-2.1 branch and semicolon-lapack v0.01.5-pre (with the Fortran
# ABI shim), builds both as Emscripten side modules and vendors them into
# the wheel. This script runs that same cibuildwheel build at a pinned
# SciPy commit. The only change is that the BLIS clone comes from this
# checkout's HEAD instead of pyodide/blis.
#
# The Emscripten version is not chosen here or by the emsdk matrix in
# wasm.yml. cibuildwheel picks the Pyodide version for cp314, and that
# Pyodide cross-build environment (xbuildenv) fixes the Emscripten
# version and the Pyodide patches applied to it: cibuildwheel 4.3.4 uses
# Pyodide 314.0.7, which uses Emscripten 5.0.3. CIBW_VERSION must match
# the cibuildwheel pin in SciPy's emscripten.yml at SCIPY_SHA, so the
# toolchain stays the one SciPy's own CI uses.
#
# Usage:
#
#   ci/wasm/do_scipy.sh
#
# Needs git, curl, python3 (3.11 or newer, with venv) and network access.
# Uncommitted changes are not tested: the BLIS build clones HEAD.
#

set -euo pipefail

script_name=${0##*/}

cd "$(dirname "$0")/../.."
DIST_PATH="$(pwd)"

WORK_DIR="${WORK_DIR:-$DIST_PATH/wasm-scipy-work}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$DIST_PATH/wasm-artifacts/scipy}"
SCIPY_REPO="https://github.com/scipy/scipy.git"
# scipy main, 2026-11-09; its own Emscripten/Pyodide CI run passed.
SCIPY_SHA="${SCIPY_SHA:-2bc4099d1e83526edfab71b8af1c88b72af623d1}"
CIBW_VERSION="${CIBW_VERSION:-4.3.4}"
CIBW_BUILD_ID="${CIBW_BUILD_ID:-cp314-pyodide_wasm32}"
# An import problem can turn whole test modules into skips, which pytest
# does not report as a failure. Against pyodide/blis pyodide-2.1 at
# SCIPY_SHA, 87328 tests pass (8464 skipped, 879 xfailed). The floor sits
# about 1.5% lower, so it tolerates small changes in the unpinned test
# requirements but not the loss of a large module such as scipy.linalg.
SCIPY_MIN_PASSED="${SCIPY_MIN_PASSED:-86000}"
JOBS="${JOBS:-$(nproc)}"
PYTHON="${PYTHON:-python3}"

BLIS_BRANCH="blis-under-test"

log() { echo "$script_name: $*"; }
die() { echo "$script_name: error: $*" >&2; exit 1; }

# Fresh output dirs so stale artifacts cannot masquerade as new evidence.
rm -rf "$ARTIFACTS_DIR" "$WORK_DIR"
mkdir -p "$ARTIFACTS_DIR" "$WORK_DIR"

for tool in git curl "$PYTHON"; do
	command -v "$tool" > /dev/null 2>&1 || die "$tool not found on PATH"
done

SCIPY_DIR="$WORK_DIR/scipy"
BLIS_MIRROR="$WORK_DIR/blis.git"
WHEELHOUSE="$WORK_DIR/wheelhouse"
CIBW_LOG="$ARTIFACTS_DIR/cibuildwheel.log"

# On failure keep what is needed to reproduce locally: the wheel and the
# two side modules it vendors. Logs are written to $ARTIFACTS_DIR directly.
keep_failure_files() {
	status=$?
	set +e
	if [ "$status" -ne 0 ]; then
		mkdir -p "$ARTIFACTS_DIR/failure"
		cp "$WORK_DIR"/tested-wheel/*.whl "$ARTIFACTS_DIR/failure/" 2> /dev/null
		cp "$SCIPY_DIR/.blis/lib/libblis.so" "$SCIPY_DIR/.blis/lib/libsemilapack.so" \
			"$ARTIFACTS_DIR/failure/" 2> /dev/null
	fi
	return 0
}
trap keep_failure_files EXIT

git_head="$(git rev-parse HEAD)"
git_dirty="$(git status --porcelain 2>/dev/null | grep -cv '^??' || true)"
if [ "$git_dirty" -ne 0 ]; then
	log "warning: $git_dirty tracked files have uncommitted changes; testing HEAD $git_head without them"
fi

# Expose HEAD as a branch: SciPy's before-build script clones with
# `git clone --depth 1 --branch <ref> <repo>`.
git init -q --bare "$BLIS_MIRROR"
git -C "$BLIS_MIRROR" fetch -q --depth 1 "file://$DIST_PATH" "HEAD:refs/heads/$BLIS_BRANCH" \
	|| die "cannot expose HEAD of $DIST_PATH as a branch"
[ "$(git -C "$BLIS_MIRROR" rev-parse "refs/heads/$BLIS_BRANCH")" = "$git_head" ] \
	|| die "mirror branch does not point at HEAD $git_head"

log "fetching scipy @ $SCIPY_SHA"
: > "$ARTIFACTS_DIR/deps.log"
git init -q "$SCIPY_DIR"
git -C "$SCIPY_DIR" remote add origin "$SCIPY_REPO"
git -C "$SCIPY_DIR" fetch -q --depth 1 origin "$SCIPY_SHA" >> "$ARTIFACTS_DIR/deps.log" 2>&1 \
	|| die "fetch of scipy@$SCIPY_SHA failed; see $ARTIFACTS_DIR/deps.log"
git -C "$SCIPY_DIR" checkout -q FETCH_HEAD >> "$ARTIFACTS_DIR/deps.log" 2>&1 \
	|| die "cannot checkout scipy@$SCIPY_SHA"
[ "$(git -C "$SCIPY_DIR" rev-parse HEAD)" = "$SCIPY_SHA" ] || die "scipy checkout is not $SCIPY_SHA"
git -C "$SCIPY_DIR" submodule update -q --init --recursive --depth 1 >> "$ARTIFACTS_DIR/deps.log" 2>&1 \
	|| die "scipy submodule update failed; see $ARTIFACTS_DIR/deps.log"

# The cibuildwheel version decides the Pyodide and Emscripten versions,
# so it must be the one SciPy's own Pyodide job uses at this commit.
grep -Eq "uses: pypa/cibuildwheel@[0-9a-f]{40} # v${CIBW_VERSION//./\\.}\$" \
	"$SCIPY_DIR/.github/workflows/emscripten.yml" \
	|| die "scipy@$SCIPY_SHA emscripten.yml does not pin cibuildwheel v$CIBW_VERSION; update CIBW_VERSION"

# Point SciPy's before-build script at the mirror. The two assignments
# become overridable; everything else in the script is used unchanged.
before_build="$SCIPY_DIR/tools/wheels/cibw_before_build_pyodide.sh"
[ -f "$before_build" ] || die "scipy@$SCIPY_SHA has no tools/wheels/cibw_before_build_pyodide.sh"
# shellcheck disable=SC2016  # the ${...} defaults are written literally
overridable_repo='BLIS_REPO="${BLIS_REPO:-https://github.com/pyodide/blis}"'
# shellcheck disable=SC2016
overridable_ref='BLIS_REF="${BLIS_REF:-pyodide-2.1}"'
sed -i \
	-e "s|^BLIS_REPO=\"https://github.com/pyodide/blis\"\$|$overridable_repo|" \
	-e "s|^BLIS_REF=\"pyodide-2.1\"\$|$overridable_ref|" \
	"$before_build"
if ! grep -qxF "$overridable_repo" "$before_build" || ! grep -qxF "$overridable_ref" "$before_build"; then
	die "could not redirect the BLIS clone in $before_build; the script changed upstream"
fi
git -C "$SCIPY_DIR" diff > "$ARTIFACTS_DIR/scipy-local.diff"

log "installing cibuildwheel $CIBW_VERSION"
"$PYTHON" -m venv "$WORK_DIR/venv"
"$WORK_DIR/venv/bin/python" -m pip install -q "cibuildwheel==$CIBW_VERSION" \
	> "$ARTIFACTS_DIR/pip.log" 2>&1 || die "cibuildwheel install failed; see $ARTIFACTS_DIR/pip.log"

{
	echo "date:        $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
	echo "git commit:  $git_head"
	echo "git dirty tracked files: $git_dirty"
	echo "uname:       $(uname -a)"
	echo "python3:     $("$PYTHON" --version 2>&1)"
	echo "scipy commit:  $SCIPY_SHA ($SCIPY_REPO)"
	echo "cibuildwheel:  $CIBW_VERSION ($CIBW_BUILD_ID)"
	echo "min passed:    $SCIPY_MIN_PASSED"
} | tee "$ARTIFACTS_DIR/versions.txt"

# cibuildwheel moves the wheel to the output dir only after the tests
# pass. Append a copy step to scipy's own repair command so a failing
# run still leaves the wheel that was tested.
repair_cmd="$("$PYTHON" - "$SCIPY_DIR/pyproject.toml" <<'EOF'
import sys, tomllib
with open(sys.argv[1], "rb") as fh:
    print(tomllib.load(fh)["tool"]["cibuildwheel"]["pyodide"]["repair-wheel-command"])
EOF
)" || die "cannot read scipy's pyodide repair-wheel-command"
mkdir -p "$WORK_DIR/tested-wheel"

# Everything else comes from scipy's pyproject.toml
# ([tool.cibuildwheel.pyodide]: before-build, the test requirements and
# `pytest --pyargs scipy -m 'not slow'`).
log "building and testing scipy with cibuildwheel"
cibw_status=0
(
	cd "$SCIPY_DIR"
	export CIBW_PLATFORM=pyodide CIBW_BUILD="$CIBW_BUILD_ID" CIBW_BUILD_VERBOSITY=1
	export CIBW_REPAIR_WHEEL_COMMAND_PYODIDE="$repair_cmd && cp {dest_dir}/*.whl $WORK_DIR/tested-wheel/"
	export BLIS_REPO="file://$BLIS_MIRROR" BLIS_REF="$BLIS_BRANCH"
	export PYODIDE_JOBS="$JOBS"
	"$WORK_DIR/venv/bin/cibuildwheel" --output-dir "$WHEELHOUSE" .
) 2>&1 | tee "$CIBW_LOG" || cibw_status=$?

# Record the toolchain cibuildwheel resolved.
grep -E 'Installing Pyodide xbuildenv version|Installing Emscripten [0-9]' "$CIBW_LOG" \
	| sed 's/^/toolchain:     /' | tee -a "$ARTIFACTS_DIR/versions.txt" || true

# Summarize pytest's final line and the -ra short summary.
"$PYTHON" - "$CIBW_LOG" "$ARTIFACTS_DIR" <<'EOF' || true
import re, sys
log, out = sys.argv[1], sys.argv[2]
lines = open(log, encoding="utf-8", errors="replace").read().splitlines()
final = [l for l in lines if re.match(r"^=+ .*\b(passed|failed|error|errors)\b.* in [0-9.]+s", l)]
counts = {}
if final:
    for n, what in re.findall(r"(\d+) (passed|failed|skipped|xfailed|xpassed|errors?|warnings?|deselected)", final[-1]):
        counts[what.rstrip("s") if what.startswith(("error", "warning")) else what] = int(n)
with open(f"{out}/summary.txt", "w") as fh:
    fh.write((final[-1] if final else "no pytest summary line found") + "\n")
    for k in ("passed", "failed", "error", "skipped", "xfailed", "xpassed", "deselected", "warning"):
        fh.write(f"{k}: {counts.get(k, 0)}\n")
with open(f"{out}/failures.txt", "w") as fh:
    fh.writelines(l + "\n" for l in lines if l.startswith(("FAILED ", "ERROR ")))
EOF
cat "$ARTIFACTS_DIR/summary.txt" 2> /dev/null || true

if [ "$cibw_status" -ne 0 ]; then
	tail -n 60 "$CIBW_LOG" >&2
	die "cibuildwheel exited with status $cibw_status; see $CIBW_LOG and $ARTIFACTS_DIR/failures.txt"
fi

# The tested wheel must vendor the side modules that the before-build
# step built from this checkout. `pyodide auditwheel repair` copies each
# one to scipy.libs/<name>-<first 8 hex digits of its sha256>.so and
# then edits its dylink section, so match on the name, not the bytes.
wheels=("$WHEELHOUSE"/scipy-*.whl)
[ -f "${wheels[0]}" ] || die "no scipy wheel in $WHEELHOUSE"
"$PYTHON" - "${wheels[0]}" "$SCIPY_DIR/.blis/lib/libblis.so" "$SCIPY_DIR/.blis/lib/libsemilapack.so" <<'EOF' \
	| tee -a "$ARTIFACTS_DIR/versions.txt" || die "the wheel does not vendor the side modules built from this checkout"
import hashlib, sys, zipfile
wheel, *libs = sys.argv[1:]
with zipfile.ZipFile(wheel) as z:
    vendored = [n for n in z.namelist() if n.startswith("scipy.libs/")]
ok = True
for lib in libs:
    with open(lib, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    base = lib.rsplit("/", 1)[-1][:-len(".so")]
    want = f"scipy.libs/{base}-{digest[:8]}.so"
    print(f"{base}.so sha256 {digest}: {'vendored as ' + want if want in vendored else 'NOT in wheel'}")
    ok &= want in vendored
sys.exit(0 if ok else 1)
EOF

grep -q 'in [0-9.]*s' "$ARTIFACTS_DIR/summary.txt" || die "no pytest summary line in $CIBW_LOG"
passed="$(sed -n 's/^passed: //p' "$ARTIFACTS_DIR/summary.txt")"
failed="$(sed -n 's/^failed: //p' "$ARTIFACTS_DIR/summary.txt")"
errors="$(sed -n 's/^error: //p' "$ARTIFACTS_DIR/summary.txt")"
[ "$failed" -eq 0 ] && [ "$errors" -eq 0 ] || die "pytest reported $failed failed and $errors errors"
[ "$passed" -ge "$SCIPY_MIN_PASSED" ] \
	|| die "only $passed scipy tests passed; expected at least $SCIPY_MIN_PASSED (collection or import problem?)"

log "scipy tests passed against BLIS $git_head; artifacts in $ARTIFACTS_DIR"
