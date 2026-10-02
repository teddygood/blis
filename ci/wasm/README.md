# BLIS WASM CI

`do_wasm.sh` builds BLIS for the `wasm32` configuration with Emscripten and
runs the fast BLIS testsuite (`input.general.fast` / `input.operations.fast`,
datatypes `sdcz`) under Node.js. It is driven by
`.github/workflows/wasm.yml` for each Emscripten version in the CI matrix
(5.0.3 and 6.0.5) and can be reproduced locally:

```sh
# Install and activate an emsdk version matching CI (any location works).
git clone --depth 1 --branch 6.0.5 https://github.com/emscripten-core/emsdk.git
cd emsdk && ./emsdk install 6.0.5 && ./emsdk activate 6.0.5 && cd ..

# From the BLIS source tree:
EMSDK=$PWD/emsdk ci/wasm/do_wasm.sh
```

Settings:

- `EMSDK`: path to an emsdk installation; its `emsdk_env.sh` is sourced so
  `emcc`/`emar`/`emranlib` land on `PATH`. Not needed if the toolchain is
  already on `PATH`.
- `NODE`: Node.js executable that runs the testsuite (default:
  `$EMSDK_NODE`, then `node` from `PATH`). CI pins Node 24 via
  `actions/setup-node` and passes it explicitly
  (`NODE="$(command -v node)"`) because emsdk bundles its own Node.
- `JOBS`: `make` parallelism (default: `nproc`).
- `ARTIFACTS_DIR`: output directory (default: `./wasm-artifacts`).

Logs (`configure.log`, `build.log`, `link.log`), `output.testsuite`, and
`versions.txt` land in `wasm-artifacts/` and are uploaded by the workflow
on success and on failure.

After the testsuite, `do_wasm.sh` compiles `ci/wasm/test-dgemm.c` against
the built archive and runs it under the same `NODE`; it is a deterministic
contract test for the double-precision gemm microkernel and its public
CBLAS callers, logging to `test-dgemm.log`.

# semicolon-lapack integration (`do_lapack.sh`)

`do_lapack.sh` builds the pinned CMocka test suite of
[semicolon-lapack](https://github.com/ilayn/semicolon-lapack) against the
`lib/wasm32/libblis.a` left in-tree by `do_wasm.sh` and runs it under
Node.js. It runs as a second stage in the same workflow job, after a
successful `do_wasm.sh`.

These tests cover the static, single-threaded CBLAS build. SciPy and
Pyodide integration require separate tests.

Prerequisites: emsdk plus `git`, `python3`, `pkg-config`, `meson`,
`ninja` and `cmake` (CI pins meson 1.12.1, ninja 1.13.2 and cmake 4.2.3).

Local reproduction (run `do_wasm.sh` first):

```sh
# meson + ninja + cmake are required in addition to emsdk (e.g. a venv):
python3 -m venv .task-tools/meson-venv
.task-tools/meson-venv/bin/pip install 'meson==1.12.1' 'ninja==1.13.2' 'cmake==4.2.3'

EMSDK=$PWD/emsdk NODE="$(command -v node)" \
  PATH="$PWD/.task-tools/meson-venv/bin:$PATH" \
  TEST_JOBS=$(nproc) ci/wasm/do_lapack.sh
```

`NODE` must resolve to Node 24 (`REQUIRED_NODE_MAJOR`, default 24): emsdk
6.0.5 bundles Node 22, so CI captures the `setup-node` path into `NODE`
before `emsdk_env.sh` runs. `TEST_JOBS` defaults to 2 for CI; raise it
locally for a faster suite run. `WORK_DIR` (default `./wasm-lapack-work`)
holds the dependency checkouts and is recreated on every run.
semicolon-lapack and cmocka are pinned by full commit SHA. See the
variable defaults in do_lapack.sh for dependency overrides.

`ci/wasm/check-lapack-results-selftest.sh` exercises the result checker
against fixture streams (passes, `not ok`, bailouts, missing
trailers/plans, duplicate/missing records, wrong node path). At the
pinned revision the suite is 480 executables / 512 CMocka groups with 94
expected upstream subtest skips.

Artifacts land in `wasm-artifacts/lapack/` and are uploaded with the rest
of `wasm-artifacts/` on success and on failure; `linkage.txt` collects
the candidate provenance and link evidence.
