#!/bin/sh

#
# Validate an output.testsuite file produced by a WASM testsuite run.
#
# Usage: check-testsuite-output.sh <output.testsuite>
#
# Extends the stock testsuite/check-blistest.sh checks (normal exit, no
# FAILURE) by also requiring a real numeric result row, so an empty or
# trivially-aborted run cannot look green.
#

set -eu

script_name=${0##*/}

ansi_red="\033[0;31m"
ansi_normal="\033[0m"

if [ $# -lt 1 ]; then
	printf "${ansi_red}${script_name}: usage: %s <output.testsuite>${ansi_normal}\n" "$script_name" >&2
	exit 1
fi

outfile="$1"

if [ ! -s "$outfile" ]; then
	printf "${ansi_red}${script_name}: testsuite output '%s' is missing or empty.${ansi_normal}\n" "$outfile" >&2
	exit 1
fi

# Require normal completion (testsuite prints "Exiting normally." last).
if ! grep -q 'Exiting normally' "$outfile"; then
	printf "${ansi_red}${script_name}: testsuite did not exit normally (no 'Exiting normally' marker).${ansi_normal}\n" >&2
	exit 1
fi

# Require at least one real numeric result row, e.g.:
#   blis_ztrsm_runu_cc   100  100    11.01  8.29e-17   PASS
# Matching the whole row prevents placeholder text ending in PASS/MARGINAL
# from counting as a real result.
nresults=$(grep -cE '^blis_[[:alnum:]_]+([[:space:]]+[0-9.eE+-]+)+[[:space:]]+(PASS|MARGINAL)$' "$outfile" || true)
if [ "$nresults" -eq 0 ]; then
	printf "${ansi_red}${script_name}: testsuite output contains no numeric test results.${ansi_normal}\n" >&2
	exit 1
fi

# Reject any FAILURE results via the stock BLIS checker.
script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
"$script_dir/../../testsuite/check-blistest.sh" "$outfile"
