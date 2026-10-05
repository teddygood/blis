#!/usr/bin/env python3
#
# Validate the meson testlog produced by running the semicolon-lapack
# CMocka suite as WebAssembly under Node.
#
# Upstream test mains discard CMocka's failure count and exit 0, so the
# authoritative signal is each record's TAP stream in testlog.json. Smoke
# executables may emit several CMocka groups per stream, each with its own
# "1..N" plan and "# ok" trailer, so validation is group-aware.
#
# Usage:
#
#   check-lapack-results.py --testlog builddir/meson-logs/testlog.json \
#       --tests-json inventory.json [--src-dir /path/to/semicolon-lapack] \
#       [--expected-count 480] [--node-path /abs/node24]
#
# Exit status is nonzero unless every check passes.
#

import argparse
import json
import re
import sys
from collections import Counter

CHECKER = "check-lapack-results.py"

RE_TAP_VERSION = re.compile(r"^TAP version \d+$")
RE_PLAN = re.compile(r"^1\.\.(\d+)$")
RE_RESULT = re.compile(r"^(ok|not ok) (\d+) - (.+?)( # SKIP)?$")
RE_TRAILER = re.compile(r"^# (ok|not ok) - (.+)$")
RE_BAILOUT = re.compile(r"^Bail out")


def derive_expected_names(src_dir):
    """Derive the registered test names from the semicolon-lapack sources.

    Each precision dir registers dicts of `'<name>': [...]` in
    tests/{d,s,z,c}/meson.build and tests/{d,s,z,c}/smoke/meson.build, and
    tests/meson.build registers 'rng_validation'.
    """
    names = []
    sub_files = [f"tests/{p}/meson.build" for p in ("d", "s", "z", "c")]
    sub_files += [f"tests/{p}/smoke/meson.build" for p in ("d", "s", "z", "c")]
    key_re = re.compile(r"^\s*'([A-Za-z0-9_]+)'\s*:")
    for rel in sub_files:
        path = f"{src_dir}/{rel}"
        try:
            fh = open(path, encoding="utf-8")
        except OSError:
            continue
        in_dict = False
        with fh:
            for line in fh:
                stripped = line.strip()
                if not in_dict:
                    if re.match(r"^\w+\s*=\s*\{\s*$", stripped):
                        in_dict = True
                    continue
                if stripped == "}":
                    in_dict = False
                    continue
                m = key_re.match(line)
                if m:
                    names.append(m.group(1))
    # tests/meson.build: test('rng_validation', ...)
    with open(f"{src_dir}/tests/meson.build", encoding="utf-8") as fh:
        for m in re.finditer(r"test\(\s*'([^']+)'", fh.read()):
            names.append(m.group(1))
    return names


def load_inventory(path):
    """Load test names from `meson introspect --tests` output."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("inventory JSON is not a list")
    names = []
    for item in data:
        if isinstance(item, dict) and "name" in item:
            names.append(item["name"])
        elif isinstance(item, str):
            names.append(item)
        else:
            raise ValueError(f"unrecognized inventory entry: {item!r}")
    return names


def bare_name(record_name):
    """testlog names look like '<suite> - semicolon-lapack:<test>'; the
    registered meson name is the part after the last colon."""
    return record_name.rsplit(":", 1)[-1]


def validate_tap_stream(stdout):
    """Validate one executable's stdout as a CMocka TAP stream.

    Grammar (cmocka-2.0.0 src/cmocka.c):

        TAP version 14                       (once, first line)
        1..N                                 (group plan, N >= 1)
        ok i - name | ok i - name # SKIP     (i = 1..N, in order)
        not ok i - name                      (optionally followed by an
            indented '  ---' ... '  ...' YAML diagnostic block)
        # ok - group | # not ok - group      (group trailer)

    One executable may emit several groups (each a fresh plan with
    result numbering restarting at 1). Anything else is rejected.

    Returns (groups, passed, skipped, errors).
    """
    lines = stdout.split("\n")
    groups = []
    errors = []
    i = 0
    n = len(lines)

    if n == 0 or not RE_TAP_VERSION.match(lines[0] or ""):
        return 0, 0, 0, ["missing 'TAP version' header"]
    i = 1

    while i < n:
        line = lines[i]
        if line == "":
            i += 1
            continue
        plan = RE_PLAN.match(line)
        if plan:
            count = int(plan.group(1))
            i += 1
            results = []  # (number, ok, skipped, name)
            # Read result lines and optional diagnostic blocks.
            while i < n:
                res = RE_RESULT.match(lines[i])
                if res:
                    results.append(
                        (int(res.group(2)), res.group(1) == "ok",
                         bool(res.group(4)), res.group(3)))
                    i += 1
                    # YAML diagnostics are indented lines belonging to a
                    # failing result: '  ---' ... '  ...'
                    while i < n and lines[i].startswith("  "):
                        i += 1
                    continue
                if lines[i].startswith("  "):
                    # stray diagnostics without a preceding result line
                    i += 1
                    continue
                break
            numbers = [r[0] for r in results]
            if numbers != list(range(1, len(results) + 1)):
                errors.append(
                    f"group {len(groups) + 1}: result numbers "
                    f"{numbers[:8]}{'...' if len(numbers) > 8 else ''} "
                    "are not sequential from 1")
            if len(results) != count:
                errors.append(
                    f"group {len(groups) + 1}: plan 1..{count} but "
                    f"{len(results)} result lines found")
            trailer = RE_TRAILER.match(lines[i]) if i < n else None
            if not trailer:
                errors.append(
                    f"group {len(groups) + 1}: missing '# ok/# not ok' "
                    f"trailer after plan 1..{count}")
            else:
                i += 1
            groups.append({
                "plan": count,
                "results": results,
                "trailer_ok": bool(trailer and trailer.group(1) == "ok"),
                "trailer_name": trailer.group(2) if trailer else None,
            })
            continue
        if RE_TRAILER.match(line):
            errors.append(f"trailer without a plan at line {i + 1}: {line!r}")
            i += 1
            continue
        if RE_BAILOUT.match(line):
            errors.append(f"bailout at line {i + 1}: {line!r}")
            i += 1
            continue
        errors.append(f"unrecognized TAP line {i + 1}: {line!r}")
        i += 1

    if not groups:
        errors.append("no TAP groups found")

    passed = sum(1 for g in groups for r in g["results"] if r[1] and not r[2])
    skipped = sum(1 for g in groups for r in g["results"] if r[2])
    failed = sum(1 for g in groups for r in g["results"] if not r[1])

    if failed:
        errors.append(f"{failed} 'not ok' result(s)")
    for gi, g in enumerate(groups, 1):
        if not g["trailer_ok"]:
            errors.append(
                f"group {gi} ({g['trailer_name']}): failing/missing trailer")
    if passed == 0:
        errors.append("no non-skipped passing subtests")

    return len(groups), passed, skipped, errors


def main():
    ap = argparse.ArgumentParser(
        description="Validate semicolon-lapack meson testlog TAP output")
    ap.add_argument("--testlog", required=True,
                    help="path to meson-logs/testlog.json (JSON lines)")
    ap.add_argument("--tests-json", required=True,
                    help="'meson introspect --tests' JSON output")
    ap.add_argument("--src-dir",
                    help="semicolon-lapack source dir for manifest cross-check")
    ap.add_argument("--expected-count", type=int,
                    help="required number of registered tests")
    ap.add_argument("--node-path",
                    help="absolute node path every test command must use")
    args = ap.parse_args()

    errors = []
    warnings = []

    try:
        inventory_names = load_inventory(args.tests_json)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"{CHECKER}: cannot load inventory {args.tests_json}: {e}",
              file=sys.stderr)
        return 1

    if args.expected_count is not None \
            and len(inventory_names) != args.expected_count:
        errors.append(
            f"inventory contains {len(inventory_names)} tests, expected "
            f"{args.expected_count}")

    dup_inv = [n for n, c in Counter(inventory_names).items() if c > 1]
    if dup_inv:
        errors.append(f"duplicate names in test inventory: {dup_inv[:5]}")

    if args.src_dir:
        try:
            source_names = derive_expected_names(args.src_dir)
        except OSError as e:
            print(f"{CHECKER}: cannot derive manifest from {args.src_dir}: {e}",
                  file=sys.stderr)
            return 1
        if sorted(source_names) != sorted(inventory_names):
            only_src = sorted(set(source_names) - set(inventory_names))
            only_inv = sorted(set(inventory_names) - set(source_names))
            errors.append(
                f"source-derived manifest ({len(source_names)} names) does not "
                f"match meson inventory ({len(inventory_names)} names); "
                f"source-only: {only_src[:5]}, inventory-only: {only_inv[:5]}")

    records = []
    try:
        with open(args.testlog, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        errors.append(f"testlog line {lineno} is not JSON: {e}")
    except OSError as e:
        print(f"{CHECKER}: cannot read {args.testlog}: {e}", file=sys.stderr)
        return 1

    seen = Counter(bare_name(r.get("name", "")) for r in records)
    expected = set(inventory_names)
    missing = sorted(expected - set(seen))
    extra = sorted(set(seen) - expected)
    dup = sorted(n for n, c in seen.items() if c > 1)
    if missing:
        errors.append(f"tests missing from testlog ({len(missing)}): "
                      f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    if extra:
        errors.append(f"unexpected tests in testlog ({len(extra)}): "
                      f"{extra[:5]}{'...' if len(extra) > 5 else ''}")
    if dup:
        errors.append(f"duplicate testlog records for: {dup[:5]}")

    total_groups = total_pass = total_skip = 0
    exe_failures = []
    for r in records:
        name = r.get("name", "?")
        rec_errors = []
        if r.get("result") != "OK":
            rec_errors.append(f"meson result={r.get('result')!r}")
        if r.get("returncode") != 0:
            rec_errors.append(f"returncode={r.get('returncode')!r}")
        if r.get("is_fail"):
            rec_errors.append("marked expected-failure")
        if args.node_path:
            cmd = r.get("command") or []
            if not cmd or cmd[0] != args.node_path:
                rec_errors.append(f"command[0]={cmd[0] if cmd else None!r} "
                                  f"!= {args.node_path!r}")
        groups, passed, skipped, tap_errors = \
            validate_tap_stream(r.get("stdout", ""))
        rec_errors.extend(tap_errors)
        total_groups += groups
        total_pass += passed
        total_skip += skipped
        if rec_errors:
            exe_failures.append((name, rec_errors))
            errors.append(f"{name}: {'; '.join(rec_errors)}")

    print(f"{CHECKER}: {len(records)} testlog records, "
          f"{len(inventory_names)} registered tests, "
          f"{total_groups} CMocka groups")
    print(f"{CHECKER}: subtests: {total_pass} passed, {total_skip} skipped, "
          f"{total_groups - len(records)} extra groups beyond the first")
    if total_skip:
        print(f"{CHECKER}: upstream skips are preserved and counted "
              f"({total_skip} subtests marked '# SKIP')")
    for name, errs in exe_failures[:20]:
        print(f"{CHECKER}: {name}: {'; '.join(errs)}")
    if len(exe_failures) > 20:
        print(f"{CHECKER}: ... and {len(exe_failures) - 20} more executables "
              "with errors")
    for w in warnings:
        print(f"{CHECKER}: warning: {w}")

    if errors:
        print(f"{CHECKER}: FAILED with {len(errors)} error(s)", file=sys.stderr)
        return 1
    print(f"{CHECKER}: all {len(records)} executables validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
