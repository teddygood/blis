#!/bin/bash

#
# Self-test fixtures for check-lapack-results.py: each case builds a
# minimal meson testlog.json + test inventory in a temp dir and asserts
# the checker's exit status. "pass-*" fixtures must exit 0, "fail-*"
# fixtures must exit nonzero.
#
# Usage: ci/wasm/check-lapack-results-selftest.sh
#

set -euo pipefail
cd "$(dirname "$0")"
CHECKER="$PWD/check-lapack-results.py"
python3 -m py_compile "$CHECKER"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

NODE_FAKE="/opt/node24/bin/node"

# emit_inventory <file> <test-name>...
emit_inventory() {
	local f=$1; shift
	{
		printf '['
		local first=1
		for t in "$@"; do
			[ "$first" = 1 ] || printf ','
			first=0
			printf '{"name": "%s", "workdir": null, "timeout": 120, "suite": ["s"], "is_parallel": true, "protocol": "exitcode", "cmd": ["node", "x.js"], "env": {}, "depends": []}' "$t"
		done
		printf ']'
	} > "$f"
}

# emit_log <file> [record-spec]...
# record-spec: name|rc|result|stdout  (stdout uses \n escapes)
emit_log() {
	python3 - "$1" "${@:2}" <<'PYEOF'
import json, sys
out, specs = sys.argv[1], sys.argv[2:]
with open(out, "w") as fh:
    for s in specs:
        name, rc, result, stdout = s.split("|", 3)
        rec = {
            "name": "s - semicolon-lapack:" + name,
            "stdout": stdout.replace("\\n", "\n"),
            "returncode": int(rc),
            "result": result,
            "duration": 0.1,
            "command": ["/opt/node24/bin/node", "/b/" + name + ".js"],
            "workdir": None,
            "env": {},
            "is_fail": False,
        }
        fh.write(json.dumps(rec) + "\n")
PYEOF
}

TAP_OK='TAP version 14\n1..2\nok 1 - a\nok 2 - b\n# ok - g\n'
TAP_SKIP='TAP version 14\n1..2\nok 1 - a\nok 2 - b # SKIP\n# ok - g\n'
TAP_ALLSKIP='TAP version 14\n1..2\nok 1 - a # SKIP\nok 2 - b # SKIP\n# ok - g\n'
TAP_NOTOK='TAP version 14\n1..2\nok 1 - a\nnot ok 2 - b\n  ---\n  message: %s\n  severity: fail\n  ...\n# not ok - g\n'
# A second cmocka_run_group_tests group continues the same stream: the
# "TAP version 14" header is emitted once, subsequent groups restart at
# "1..N".
TAP_GROUP2='1..1\nok 1 - c\n# ok - h\n'
TAP_NOTRAILER='TAP version 14\n1..2\nok 1 - a\nok 2 - b\n'
TAP_NOPLAN='TAP version 14\nok 1 - a\n# ok - g\n'
TAP_PLANMISMATCH='TAP version 14\n1..3\nok 1 - a\nok 2 - b\n# ok - g\n'
TAP_NONSEQ='TAP version 14\n1..2\nok 1 - a\nok 3 - b\n# ok - g\n'
TAP_BAILOUT='TAP version 14\n1..2\nok 1 - a\nBail out! boom\n'
TAP_NOHEADER='1..1\nok 1 - a\n# ok - g\n'

passed=0
failures=()

# run_case <expect:pass|fail> <label> [extra checker args...] -- <files...>
run_case() {
	local expect=$1 label=$2; shift 2
	local args=()
	while [ "$1" != "--" ]; do args+=("$1"); shift; done
	shift
	local d="$work/$label"
	mkdir -p "$d"
	local want_code=0
	[ "$expect" = pass ] || want_code=1
	local got=0
	python3 "$CHECKER" --testlog "$d/testlog.json" --tests-json "$d/tests.json" \
		--node-path "$NODE_FAKE" "${args[@]}" > "$d/out.txt" 2>&1 || got=1
	if { [ "$want_code" = 0 ] && [ "$got" = 0 ]; } || \
	   { [ "$want_code" = 1 ] && [ "$got" != 0 ]; }; then
		passed=$((passed + 1))
		echo "ok   $label"
	else
		failures+=("$label")
		echo "FAIL $label (expected $expect, checker exit=$got)" >&2
		sed 's/^/     /' "$d/out.txt" | head -8 >&2
	fi
}

d="$work/pass-single"; mkdir -p "$d"
emit_inventory "$d/tests.json" alpha
emit_log "$d/testlog.json" "alpha|0|OK|$TAP_OK"
run_case pass pass-single -- "$d"

d="$work/pass-multi-group"; mkdir -p "$d"
emit_inventory "$d/tests.json" alpha beta
emit_log "$d/testlog.json" "alpha|0|OK|${TAP_OK}${TAP_GROUP2}" "beta|0|OK|$TAP_OK"
run_case pass pass-multi-group -- "$d"

d="$work/pass-with-skips"; mkdir -p "$d"
emit_inventory "$d/tests.json" alpha
emit_log "$d/testlog.json" "alpha|0|OK|$TAP_SKIP"
run_case pass pass-with-skips -- "$d"

# zero process exit but a 'not ok' record inside the TAP stream
d="$work/fail-not-ok"; mkdir -p "$d"
emit_inventory "$d/tests.json" alpha
emit_log "$d/testlog.json" "alpha|0|OK|$(printf "$TAP_NOTOK" "'assertion failed'")"
run_case fail fail-not-ok -- "$d"

# first group fails, second passes; the group-1 trailer must still fail it
d="$work/fail-first-group"; mkdir -p "$d"
emit_inventory "$d/tests.json" alpha
emit_log "$d/testlog.json" "alpha|0|OK|$(printf "$TAP_NOTOK" "'x'")$TAP_GROUP2"
run_case fail fail-first-group -- "$d"

# empty stdout under a zero exit code
d="$work/fail-empty"; mkdir -p "$d"
emit_inventory "$d/tests.json" alpha
emit_log "$d/testlog.json" "alpha|0|OK|"
run_case fail fail-empty -- "$d"

# every subtest skipped: nothing actually verified
d="$work/fail-all-skipped"; mkdir -p "$d"
emit_inventory "$d/tests.json" alpha
emit_log "$d/testlog.json" "alpha|0|OK|$TAP_ALLSKIP"
run_case fail fail-all-skipped -- "$d"

# plan results emitted but group trailer missing (truncated run)
d="$work/fail-missing-trailer"; mkdir -p "$d"
emit_inventory "$d/tests.json" alpha
emit_log "$d/testlog.json" "alpha|0|OK|$TAP_NOTRAILER"
run_case fail fail-missing-trailer -- "$d"

# results without a preceding plan
d="$work/fail-missing-plan"; mkdir -p "$d"
emit_inventory "$d/tests.json" alpha
emit_log "$d/testlog.json" "alpha|0|OK|$TAP_NOPLAN"
run_case fail fail-missing-plan -- "$d"

# plan promises 3 results, only 2 emitted
d="$work/fail-plan-mismatch"; mkdir -p "$d"
emit_inventory "$d/tests.json" alpha
emit_log "$d/testlog.json" "alpha|0|OK|$TAP_PLANMISMATCH"
run_case fail fail-plan-mismatch -- "$d"

# result numbers not sequential
d="$work/fail-nonseq"; mkdir -p "$d"
emit_inventory "$d/tests.json" alpha
emit_log "$d/testlog.json" "alpha|0|OK|$TAP_NONSEQ"
run_case fail fail-nonseq -- "$d"

# cmocka bailout mid-stream
d="$work/fail-bailout"; mkdir -p "$d"
emit_inventory "$d/tests.json" alpha
emit_log "$d/testlog.json" "alpha|0|OK|$TAP_BAILOUT"
run_case fail fail-bailout -- "$d"

# no TAP header at all
d="$work/fail-no-header"; mkdir -p "$d"
emit_inventory "$d/tests.json" alpha
emit_log "$d/testlog.json" "alpha|0|OK|$TAP_NOHEADER"
run_case fail fail-no-header -- "$d"

# inventory lists a test the log never ran
d="$work/fail-missing-record"; mkdir -p "$d"
emit_inventory "$d/tests.json" alpha beta
emit_log "$d/testlog.json" "alpha|0|OK|$TAP_OK"
run_case fail fail-missing-record -- "$d"

# log contains a test that was never registered
d="$work/fail-unexpected-record"; mkdir -p "$d"
emit_inventory "$d/tests.json" alpha
emit_log "$d/testlog.json" "alpha|0|OK|$TAP_OK" "beta|0|OK|$TAP_OK"
run_case fail fail-unexpected-record -- "$d"

# same executable recorded twice
d="$work/fail-duplicate-record"; mkdir -p "$d"
emit_inventory "$d/tests.json" alpha
emit_log "$d/testlog.json" "alpha|0|OK|$TAP_OK" "alpha|0|OK|$TAP_OK"
run_case fail fail-duplicate-record -- "$d"

# nonzero process exit
d="$work/fail-nonzero-exit"; mkdir -p "$d"
emit_inventory "$d/tests.json" alpha
emit_log "$d/testlog.json" "alpha|1|FAIL|$TAP_OK"
run_case fail fail-nonzero-exit -- "$d"

# meson-level timeout result
d="$work/fail-timeout"; mkdir -p "$d"
emit_inventory "$d/tests.json" alpha
emit_log "$d/testlog.json" "alpha|-9|TIMEOUT|$TAP_OK"
run_case fail fail-timeout -- "$d"

# executable ran under the wrong interpreter path
d="$work/fail-wrong-node"; mkdir -p "$d"
emit_inventory "$d/tests.json" alpha
emit_log "$d/testlog.json" "alpha|0|OK|$TAP_OK"
python3 - "$d/testlog.json" <<'PYEOF'
import json, sys
lines = open(sys.argv[1]).read().replace("/opt/node24/bin/node", "/usr/bin/node22")
open(sys.argv[1], "w").write(lines)
PYEOF
run_case fail fail-wrong-node -- "$d"

# truncated final JSON line
d="$work/fail-truncated-json"; mkdir -p "$d"
emit_inventory "$d/tests.json" alpha
emit_log "$d/testlog.json" "alpha|0|OK|$TAP_OK"
printf '{"name": "s - semicolon-la' >> "$d/testlog.json"
run_case fail fail-truncated-json -- "$d"

# wrong expected count
d="$work/fail-wrong-count"; mkdir -p "$d"
emit_inventory "$d/tests.json" alpha
emit_log "$d/testlog.json" "alpha|0|OK|$TAP_OK"
run_case fail fail-wrong-count --expected-count 7 -- "$d"

total=$((passed + ${#failures[@]}))
echo
echo "$passed/$total fixture cases behaved as expected"
if [ "${#failures[@]}" -gt 0 ]; then
	echo "unexpected results in: ${failures[*]}" >&2
	exit 1
fi
