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
