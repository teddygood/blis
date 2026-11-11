#!/bin/sh

#
# Validate an output.testsuite file produced by a WASM testsuite run.
#
# Usage: check-testsuite-output.sh <output.testsuite> [<input.general> <input.operations>]
#
# Extends the stock testsuite/check-blistest.sh checks (normal exit, no
# FAILURE) by also requiring a real numeric result row, so an empty or
# trivially-aborted run cannot look green, and by requiring at least one
# result row for every datatype and operation that the input files enable.
# The input files default to the fast testsuite inputs.
#

set -eu

script_name=${0##*/}
script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)

ansi_red="\033[0;31m"
ansi_normal="\033[0m"

if [ $# -lt 1 ]; then
	printf "${ansi_red}${script_name}: usage: %s <output.testsuite> [<input.general> <input.operations>]${ansi_normal}\n" "$script_name" >&2
	exit 1
fi

outfile="$1"
general_in="${2:-$script_dir/../../testsuite/input.general.fast}"
ops_in="${3:-$script_dir/../../testsuite/input.operations.fast}"

if [ ! -s "$outfile" ]; then
	printf "${ansi_red}${script_name}: testsuite output '%s' is missing or empty.${ansi_normal}\n" "$outfile" >&2
	exit 1
fi

# Require normal completion (testsuite prints "Exiting normally." last).
if ! grep -q 'Exiting normally' "$outfile"; then
	printf "${ansi_red}${script_name}: testsuite did not exit normally (no 'Exiting normally' marker).${ansi_normal}\n" >&2
	tail -n 20 "$outfile" >&2
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

# Require a result row for every enabled operation and tested datatype, so
# an operation that silently stops reporting cannot pass. Row names look
# like blis_<dt><operation>[_<params>]. gemm and trsm appear both in the
# "Level-3 micro-kernels" section and in the "Level-3" section, so
# operations from the micro-kernel section are keyed as <operation>_ukr,
# like their rows (blis_dgemm_ukr_c, blis_dtrsm_ukr_l_r), and each
# section's rows count only for that section. The check applies when every
# section switch in the operations file is 1; operation switches of 2
# (test only those operations) are honoured.
for f in "$general_in" "$ops_in"; do
	if [ ! -f "$f" ]; then
		printf "${ansi_red}${script_name}: testsuite input file '%s' is missing.${ansi_normal}\n" "$f" >&2
		exit 1
	fi
done
dts=$(awk '/# Datatype\(s\) to test/ { print $1; exit }' "$general_in")
if ops=$(awk '
	/^# --- / { ukr = ($0 ~ /^# --- Level-3 micro-kernels /); next }
	/^[0-9]+[ \t]+# [A-Z]/ { if ($1 != "1") custom = 1; next }
	/^[0-9]+[ \t]+# [a-z0-9]+[ \t]*$/ {
		o = ukr ? $3 "_ukr" : $3
		v = $1 + 0
		if (!(o in sw) || v > sw[o]) sw[o] = v
		if (v == 2) only = 1
	}
	END {
		if (custom) exit 1
		for (o in sw) if (only ? sw[o] == 2 : sw[o] >= 1) printf "%s ", o
	}' "$ops_in"); then
	if [ -z "$dts" ] || [ -z "$ops" ]; then
		printf "${ansi_red}${script_name}: could not read the tested datatypes or operations from the input files.${ansi_normal}\n" >&2
		exit 1
	fi
	missing=$(awk -v dts="$dts" -v ops="$ops" '
		$1 ~ /^blis_/ && $NF ~ /^(PASS|MARGINAL|FAILURE)$/ {
			np = split(substr($1, 6), part, "_")
			key = part[1]
			if (np > 1 && part[2] == "ukr") key = key "_ukr"
			seen[key] = 1
		}
		END {
			n = split(ops, op, " ")
			for (i = 1; i <= length(dts); ++i)
			for (j = 1; j <= n; ++j)
				if (!((substr(dts, i, 1) op[j]) in seen))
					if (++nmiss <= 20) list = list " " substr(dts, i, 1) op[j]
			if (nmiss) printf "%d (first 20:%s)", nmiss, list
		}' "$outfile")
	if [ -n "$missing" ]; then
		printf "${ansi_red}${script_name}: datatype/operation pairs with no result rows: %s${ansi_normal}\n" "$missing" >&2
		exit 1
	fi
else
	echo "${script_name}: custom section switches in '$ops_in'; skipping the per-operation coverage check"
fi

# Show the failing rows before the stock checker reports the failure.
if grep -q 'FAILURE' "$outfile"; then
	printf "${ansi_red}${script_name}: failing rows (first 40):${ansi_normal}\n" >&2
	grep 'FAILURE' "$outfile" | head -n 40 >&2
fi

# Reject any FAILURE results via the stock BLIS checker.
"$script_dir/../../testsuite/check-blistest.sh" "$outfile"
