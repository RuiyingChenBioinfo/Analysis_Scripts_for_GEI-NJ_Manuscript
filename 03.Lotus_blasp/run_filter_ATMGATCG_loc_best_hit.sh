#!/usr/bin/env bash
set -euo pipefail

# Run from the directory containing the input files, or pass:
# bash run_filter_ATMGATCG_loc_best_hit.sh BLAST.tsv OUTPUT_PREFIX
# Best hit is selected PER LOC across all its protein isoforms and all targets.
# Rank: highest bitscore (column 14), then lowest evalue (column 13).
# Exact ties retain the first input row. No extra BLAST cutoff is applied.

if (( $# > 2 )); then
    echo "Usage: bash $0 [BLAST.tsv] [OUTPUT_PREFIX]" >&2
    exit 1
fi

blast=${1:-Lotus_vs_Arabidopsis.with_LOC.tsv}
prefix=${2:-Lotus_vs_Arabidopsis.loc_best_hit}

[[ -s "$blast" ]] || { echo "Missing or empty input: $blast" >&2; exit 1; }

# 1. Select the best hit BEFORE filtering target IDs. Input need not be sorted.
# BLAST fields may be separated by tabs or spaces; original rows are preserved.
awk '
    NR == 1 {
        if ($1 != "LOC_ID" || $2 != "qseqid" || $3 != "sseqid" ||
            $13 != "evalue" || $14 != "bitscore") {
            print "Unexpected BLAST header/column order" > "/dev/stderr"
            bad = 1; exit 1
        }
        print; next
    }
    NF == 0 { next }
    {
        if (NF < 14 || $1 !~ /^LOC[0-9]+$/ ||
            $13 !~ /^[0-9.]+([eE][-+]?[0-9]+)?$/ ||
            $14 !~ /^[0-9.]+([eE][-+]?[0-9]+)?$/) {
            print "Invalid BLAST row at line " NR > "/dev/stderr"
            bad = 1; exit 1
        }
        loc = $1; bits = $14 + 0; eval = $13 + 0
        if (!(loc in best)) order[++n] = loc
        if (!(loc in best) || bits > best_bits[loc] ||
            (bits == best_bits[loc] && eval < best_eval[loc])) {
            best[loc] = $0; best_bits[loc] = bits; best_eval[loc] = eval
        }
    }
    END { if (!bad) for (i = 1; i <= n; i++) print best[order[i]] }
' "$blast" > "${prefix}.tsv"

# 2. Keep LOCs whose selected best hit starts with ATCG or ATMG.
awk 'NR == 1 || $3 ~ /^(ATCG|ATMG)/' \
    "${prefix}.tsv" > "${prefix}.ATCG_ATMG.tsv"

for file in "${prefix}.tsv" "${prefix}.ATCG_ATMG.tsv"; do
    awk 'END { printf "%s\t%d LOCs\n", FILENAME, NR - 1 }' "$file"
done

