#!/usr/bin/env python3
#
# Compare builds of ci/wasm/bench-gemm.c under Node, for example the same
# benchmark linked against a reference archive and against the SIMD one.
#
# Usage:
#
#   bench-gemm.py [--node NODE] [--emcc EMCC] [--sizes 256,512]
#                 [--processes 3] [--cpu N] [--target-ms 40] [--raw FILE]
#                 LABEL=bench.js [LABEL=bench.js ...]
#
# For each size every build is calibrated, and the repetition count that
# gives at least --target-ms per batch on the fastest build is used for all
# of them. Then --processes rounds each start one fresh process per build,
# alternating the build order between rounds; a process prints 15 batch
# timings and its median is recorded. The report gives, per build, the
# median of the process medians and the range of the process medians as a
# percentage of that median. --cpu pins every process to one CPU with
# taskset; --raw writes every batch timing as CSV.
#
# The report starts with "#" lines that record the provenance: the Node
# version, the first line of `emcc --version` for the emcc on PATH or
# --emcc (run the driver where the benchmarks were built) and the sha256 of
# every .js and .wasm file it runs. With --raw the same lines also go to a
# sidecar next to FILE (out.csv -> out.provenance.txt). The driver stops
# if a file changes during the run.
#

import argparse
import datetime
import hashlib
import math
import os
import shlex
import shutil
import statistics
import subprocess
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def first_line(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as e:
        return f"unavailable ({e.strerror})"
    lines = (p.stdout or p.stderr).splitlines()
    if p.returncode != 0 or not lines:
        return f"unavailable (exit status {p.returncode})"
    return lines[0].strip()


def build_files(js):
    # The .wasm beside the .js; a -sSINGLE_FILE build has none.
    wasm = os.path.splitext(js)[0] + ".wasm"
    return [js, wasm] if os.path.exists(wasm) else [js]


def file_hashes(builds):
    return [(label, f, sha256(f))
            for label, js in builds for f in build_files(js)]


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"bench-gemm.py: {' '.join(cmd)} exited with "
                 f"{p.returncode}:\n{p.stderr}")
    if "check=pass" not in p.stderr:
        sys.exit(f"bench-gemm.py: {' '.join(cmd)} did not pass its check")
    ukr = next((ln for ln in p.stderr.splitlines()
                if ln.startswith("ukr=")), "ukr=?")
    return p.stdout.splitlines(), ukr


def main():
    ap = argparse.ArgumentParser(description="compare bench-gemm.c builds")
    ap.add_argument("--node", default="node")
    ap.add_argument("--emcc", default="emcc",
                    help="emcc whose version is recorded (default: PATH)")
    ap.add_argument("--sizes", default="256")
    ap.add_argument("--processes", type=int, default=3)
    ap.add_argument("--cpu", help="pin processes to this CPU (taskset -c)")
    ap.add_argument("--target-ms", type=float, default=40.0)
    ap.add_argument("--raw", help="write every batch timing to this CSV file")
    ap.add_argument("builds", nargs="+", metavar="LABEL=bench.js")
    args = ap.parse_args()

    builds = [b.split("=", 1) for b in args.builds]
    if any(len(b) != 2 for b in builds):
        ap.error("builds must be given as LABEL=path/to/bench.js")
    prefix = []
    if args.cpu is not None:
        if not shutil.which("taskset"):
            sys.exit("bench-gemm.py: --cpu needs taskset")
        prefix = ["taskset", "-c", args.cpu]
    for _, js in builds:
        if not os.path.isfile(js):
            sys.exit(f"bench-gemm.py: {js} does not exist")

    hashes = file_hashes(builds)
    prov = [
        "date: " + datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command: " + shlex.join(sys.argv),
        f"node: {first_line([args.node, '--version'])} "
        f"({shutil.which(args.node) or args.node})",
        f"emcc: {first_line([args.emcc, '--version'])} "
        f"({shutil.which(args.emcc) or args.emcc})",
    ] + [f"file: {label} sha256 {digest} {path}"
         for label, path, digest in hashes]
    print("\n".join("# " + line for line in prov))

    raw = open(args.raw, "w", encoding="utf-8") if args.raw else None
    if raw:
        raw.write("build,size,process,sample,reps,ms_per_call\n")
        sidecar = os.path.splitext(args.raw)[0] + ".provenance.txt"
        with open(sidecar, "w", encoding="utf-8") as fh:
            fh.write("\n".join(prov) + "\n")
    print("size  build                 microkernel       reps  "
          "median_ms  range_%  process_medians_ms")
    for size in [int(s) for s in args.sizes.split(",")]:
        cal = {}
        for label, js in builds:
            lines, _ = run(prefix + [args.node, js, str(size), "0"])
            cal[label] = float(lines[-1].split(",")[1])
        reps = max(1, math.ceil(args.target_ms / min(cal.values())))

        medians = {label: [] for label, _ in builds}
        ukrs = {}
        for proc in range(args.processes):
            order = builds if proc % 2 == 0 else builds[::-1]
            for label, js in order:
                lines, ukrs[label] = run(
                    prefix + [args.node, js, str(size), str(reps)])
                times = [float(ln.split(",")[3]) for ln in lines]
                if len(times) != 15:
                    sys.exit(f"bench-gemm.py: {js} printed {len(times)} "
                             "samples, expected 15")
                medians[label].append(statistics.median(times))
                if raw:
                    for i, t in enumerate(times):
                        raw.write(f"{label},{size},{proc},{i},{reps},{t:.9f}\n")

        for label, _ in builds:
            m = medians[label]
            mid = statistics.median(m)
            spread = (max(m) - min(m)) / mid * 100
            print(f"{size:<5} {label:<21} {ukrs[label][4:]:<17} {reps:>4}  "
                  f"{mid:9.4f}  {spread:7.1f}  "
                  f"{' '.join(f'{x:.4f}' for x in m)}")
    if raw:
        raw.close()
    if file_hashes(builds) != hashes:
        sys.exit("bench-gemm.py: a benchmark file changed during the run; "
                 "the results do not match the recorded sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
