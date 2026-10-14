#!/usr/bin/env bash
set -euo pipefail

# Paths and programs.
BASE_DIR="$PWD"
DATA_DIR="$BASE_DIR/00.Data"
BAM="$DATA_DIR/sec17_with_CB_RG_UB.bam"
REF="$DATA_DIR/genome.fasta"
GATK_OUT="$BASE_DIR/01.GATK_results"
DENDRO_OUT="$BASE_DIR/02.DENDRO_input"
GEINJ_OUT="$BASE_DIR/03.GEI-NJ_input"
GATK="gatk" # version 4.1.8.1
SAMTOOLS="samtools"# version 1.11
PYTHON="python3"
RSCRIPT="Rscript"

# Resources.
THREADS=4
JOBS=4
JAVA_MEMORY="8g"
HC_MEMORY="4g"
FILTER_MEMORY="2g"
HC_THREADS=1

# BAM tags and sample metadata.
CB_TAG="CB"
UB_TAG="UB"
LIBRARY_ID="sec17"
PLATFORM="ILLUMINA"

# Leave empty to skip BQSR.
KNOWN_SITES_VCF=""

# Calling and site filters.
SAMPLE_PLOIDY=2
CALL_CONF=20
DONT_USE_SOFT_CLIPPED_BASES=true
MIN_QUAL=20
MAX_FS=30
MIN_QD=2
MIN_FREQ=0.05
MAX_FREQ=0.95

# Molecule counting.
MIN_MAPQ=20
MIN_BASEQ=10
MAX_PILEUP_DEPTH=1000000
WINDOW_GAP=1000
WINDOW_SPAN=10000

export BAM REF GATK_OUT DENDRO_OUT GEINJ_OUT GATK SAMTOOLS THREADS
export CB_TAG UB_TAG LIBRARY_ID PLATFORM SAMPLE_PLOIDY CALL_CONF
export HC_MEMORY FILTER_MEMORY HC_THREADS DONT_USE_SOFT_CLIPPED_BASES
export MIN_QUAL MAX_FS MIN_QD MIN_FREQ MAX_FREQ MIN_MAPQ MIN_BASEQ
export MAX_PILEUP_DEPTH WINDOW_GAP WINDOW_SPAN

run_python() {
  "$PYTHON" - "$1" <<'PY'
import csv
import gzip
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pysam

gatk_out = Path(os.environ["GATK_OUT"])
dendro_out = Path(os.environ["DENDRO_OUT"])
geinj_out = Path(os.environ["GEINJ_OUT"])
cb_tag = os.environ["CB_TAG"]
ub_tag = os.environ["UB_TAG"]
samtools = os.environ["SAMTOOLS"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def encode_identifier_as_dna(identifier):
    payload = identifier.encode("utf-8")
    alphabet = "ACGT"
    return "".join(alphabet[(byte >> shift) & 3]
                   for byte in len(payload).to_bytes(2, "big") + payload
                   for shift in (6, 4, 2, 0))


def select():
    per_bin = Counter()
    qc = Counter()
    bin_codes = {}
    unsorted = gatk_out / "selected.unsorted.bam"
    with pysam.AlignmentFile(os.environ["BAM"], "rb") as source:
        with pysam.AlignmentFile(str(unsorted), "wb", header=source.header) as target:
            for read in source.fetch(until_eof=True):
                qc["all_alignments"] += 1
                if read.is_unmapped or read.is_secondary or read.is_qcfail:
                    continue
                if not read.has_tag(cb_tag) or not read.has_tag(ub_tag):
                    continue
                barcode = str(read.get_tag(cb_tag))
                umi = str(read.get_tag(ub_tag))
                if not barcode or barcode == "NA" or not umi or umi == "NA":
                    continue
                if barcode not in bin_codes:
                    bin_codes[barcode] = encode_identifier_as_dna(barcode)
                # Encode CB+UB for Picard; keep the original tags.
                read.set_tag("RX", bin_codes[barcode] + encode_identifier_as_dna(umi),
                             value_type="Z")
                target.write(read)
                per_bin[barcode] += 1
                qc["selected_alignments"] += 1
    barcodes = sorted(per_bin)
    if not barcodes:
        raise ValueError("No usable reads with CB and UB tags")
    subprocess.run([samtools, "sort", "-@", os.environ["THREADS"],
                    "-o", str(gatk_out / "selected.bam"), str(unsorted)], check=True)
    subprocess.run([samtools, "index", str(gatk_out / "selected.bam")], check=True)
    unsorted.unlink()
    with (gatk_out / "units.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample_id", "barcode", "selected_alignments"])
        writer.writerows((f"unit_{i:04d}", bc, per_bin[bc])
                         for i, bc in enumerate(barcodes, 1))
    write_json(gatk_out / "selection_qc.json", dict(qc, bins=len(barcodes)))
    print(f"Selected {qc['selected_alignments']} alignments from {len(barcodes)} bins")


def load_units():
    with (gatk_out / "units.tsv").open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def split():
    units = load_units()
    barcode_to_unit = {row["barcode"]: row["sample_id"] for row in units}
    writers = {}
    with pysam.AlignmentFile(os.environ["PROCESSED_BAM"], "rb") as source:
        original_header = source.header.to_dict()
        try:
            for read in source.fetch(until_eof=True):
                unit = barcode_to_unit[str(read.get_tag(cb_tag))]
                if unit not in writers:
                    header = dict(original_header)
                    header["RG"] = [{"ID": unit, "SM": unit,
                                     "LB": os.environ["LIBRARY_ID"],
                                     "PL": os.environ["PLATFORM"], "PU": unit}]
                    writers[unit] = pysam.AlignmentFile(
                        str(gatk_out / "unit_bams" / f"{unit}.bam"), "wb", header=header)
                read.set_tag("RG", unit, value_type="Z")
                writers[unit].write(read)
        finally:
            for writer in writers.values():
                writer.close()
    for unit in writers:
        subprocess.run([samtools, "index",
                        str(gatk_out / "unit_bams" / f"{unit}.bam")], check=True)
    print(f"Wrote {len(writers)} indexed bin BAMs")


def candidate_sites(units):
    called = defaultdict(set)
    min_qual = float(os.environ["MIN_QUAL"])
    min_freq = float(os.environ["MIN_FREQ"])
    max_freq = float(os.environ["MAX_FREQ"])
    for row in units:
        unit = row["sample_id"]
        with pysam.VariantFile(str(gatk_out / "vcfs_filtered" / f"{unit}.vcf.gz")) as vcf:
            for record in vcf.fetch():
                if set(record.filter.keys()) != {"PASS"}:
                    continue
                if record.qual is None or record.qual < min_qual:
                    continue
                if len(record.ref) != 1 or record.ref not in "ACGT":
                    continue
                if not record.alts or len(record.alts) != 1:
                    continue
                alt = record.alts[0]
                if len(alt) != 1 or alt not in "ACGT":
                    continue
                gt = record.samples[unit].get("GT")
                if not gt or 1 not in gt:
                    continue
                called[(record.contig, record.pos, record.ref, alt)].add(unit)
    with pysam.FastaFile(os.environ["REF"]) as fasta:
        order = {chrom: i for i, chrom in enumerate(fasta.references)}
        for chrom, pos, ref, alt in called:
            if fasta.fetch(chrom, pos - 1, pos).upper() != ref:
                raise ValueError(f"Reference mismatch at {chrom}:{pos}")
    positions = Counter(site[:2] for site in called)
    sites = []
    with (gatk_out / "candidate_support.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["chrom", "pos", "ref", "alt", "called_bins", "called_fraction", "retained"])
        for site in sorted(called, key=lambda x: (order[x[0]], x[1], x[2], x[3])):
            support = len(called[site])
            fraction = support / len(units)
            keep = positions[site[:2]] == 1 and min_freq < fraction < max_freq
            writer.writerow([*site, support, f"{fraction:.8f}", int(keep)])
            if keep:
                sites.append(site)
    if not sites:
        raise ValueError("No SNVs passed filtering; see candidate_support.tsv")
    return sites, called


def site_windows(sites):
    max_gap = int(os.environ["WINDOW_GAP"])
    max_span = int(os.environ["WINDOW_SPAN"])
    by_chrom = defaultdict(list)
    for index, (chrom, pos, ref, alt) in enumerate(sites):
        by_chrom[chrom].append((pos - 1, index))
    for chrom, records in by_chrom.items():
        group = [records[0]]
        for item in records[1:]:
            if item[0] - group[-1][0] > max_gap or item[0] - group[0][0] > max_span:
                yield chrom, group
                group = []
            group.append(item)
        yield chrom, group


def count_alleles(sites, units):
    barcode_index = {row["barcode"]: j for j, row in enumerate(units)}
    X = np.zeros((len(sites), len(units)), dtype=np.int32)
    N = np.zeros_like(X)
    qc = Counter()
    min_mapq = int(os.environ["MIN_MAPQ"])
    min_baseq = int(os.environ["MIN_BASEQ"])
    with pysam.AlignmentFile(os.environ["PROCESSED_BAM"], "rb") as bam:
        for chrom, group in site_windows(sites):
            lookup = {pos: index for pos, index in group}
            for column in bam.pileup(chrom, group[0][0], group[-1][0] + 1,
                                     truncate=True, stepper="nofilter", min_base_quality=0,
                                     max_depth=int(os.environ["MAX_PILEUP_DEPTH"])):
                index = lookup.get(column.reference_pos)
                if index is None:
                    continue
                ref, alt = sites[index][2:]
                molecule_bases = {}
                for pile in column.pileups:
                    if pile.is_del or pile.is_refskip:
                        continue
                    read = pile.alignment
                    if (read.is_unmapped or read.is_secondary or read.is_qcfail or
                            read.is_duplicate or read.mapping_quality < min_mapq):
                        continue
                    qpos = pile.query_position
                    if qpos is None or read.query_qualities is None or read.query_qualities[qpos] < min_baseq:
                        continue
                    if not read.has_tag(cb_tag) or not read.has_tag(ub_tag):
                        continue
                    bc = str(read.get_tag(cb_tag))
                    if bc not in barcode_index:
                        continue
                    base = read.query_sequence[qpos].upper()
                    if base not in (ref, alt):
                        continue
                    # Count each CB+UB once per site; drop conflicting alleles.
                    key = (bc, str(read.get_tag(ub_tag)))
                    previous = molecule_bases.get(key)
                    if previous is None:
                        molecule_bases[key] = base
                    elif previous != base:
                        molecule_bases[key] = "?"
                for (bc, _), base in molecule_bases.items():
                    if base == "?":
                        qc["discordant_molecules"] += 1
                        continue
                    j = barcode_index[bc]
                    N[index, j] += 1
                    X[index, j] += int(base == alt)
    qc["covered_entries"] = int(np.count_nonzero(N))
    qc["alt_molecules"] = int(X.sum())
    qc["total_molecules"] = int(N.sum())
    return X, N, qc


def write_matrix(path, sites, barcodes, values, covered=None):
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["locus", *barcodes])
        for i, (site, row) in enumerate(zip(sites, values)):
            locus = f"{site[0]}:{site[1]}:{site[2]}:{site[3]}"
            if covered is None:
                writer.writerow([locus, *row])
            else:
                writer.writerow([locus, *(value if covered[i, j] else "NA"
                                          for j, value in enumerate(row))])


def collect():
    import anndata as ad
    from scipy.sparse import csr_matrix

    units = load_units()
    sites, called = candidate_sites(units)
    X, N, qc = count_alleles(sites, units)
    barcodes = [row["barcode"] for row in units]
    covered = N > 0
    Z = np.full(X.shape, "NA", dtype=object)
    lost_calls = 0
    for i, site in enumerate(sites):
        for j, row in enumerate(units):
            if covered[i, j]:
                Z[i, j] = "1" if row["sample_id"] in called[site] else "0"
            elif row["sample_id"] in called[site]:
                lost_calls += 1
    write_matrix(dendro_out / "X.tsv.gz", sites, barcodes, X, covered)
    write_matrix(dendro_out / "N.tsv.gz", sites, barcodes, N, covered)
    write_matrix(dendro_out / "Z.tsv.gz", sites, barcodes, Z)
    with (dendro_out / "Info.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["locus", "#CHROM", "POS", "REF", "ALT"])
        for chrom, pos, ref, alt in sites:
            writer.writerow([f"{chrom}:{pos}:{ref}:{alt}", chrom, pos, ref, alt])

    # Both tools use the same counts, bins and sites.
    ratio = np.zeros(N.shape, dtype=np.float32)
    np.divide(X, N, out=ratio, where=covered)
    depth = ad.AnnData(X=csr_matrix(N.T.astype(np.uint32)))
    alt_ratio = ad.AnnData(X=csr_matrix(ratio.T))
    loci = [f"{chrom}:{pos}:{ref}:{alt}" for chrom, pos, ref, alt in sites]
    for obj in (depth, alt_ratio):
        obj.obs_names = barcodes
        obj.var_names = loci
    depth.write_h5ad(geinj_out / "sc_depth.h5ad")
    alt_ratio.write_h5ad(geinj_out / "sc_ratio.h5ad")
    metadata = dict(
        source_bam=os.environ["PROCESSED_BAM"], bins=len(units), loci=len(sites),
        count_unit="unique CB+UB molecules after CB+UB-aware duplicate marking",
        X="ALT molecules", N="REF+ALT molecules",
        Z="1 if called, 0 if covered but not called, NA if uncovered",
        site_filters=dict(min_qual=float(os.environ["MIN_QUAL"]),
                          max_FS=float(os.environ["MAX_FS"]),
                          min_QD=float(os.environ["MIN_QD"]),
                          min_call_fraction=float(os.environ["MIN_FREQ"]),
                          max_call_fraction=float(os.environ["MAX_FREQ"]),
                          call_fraction_denominator="all bins",
                          call_fraction_bounds="strict", required_FILTER="PASS"),
        min_mapq=int(os.environ["MIN_MAPQ"]), min_baseq=int(os.environ["MIN_BASEQ"]),
        lost_GATK_calls_after_pileup_filter=lost_calls, counts_qc=dict(qc),
        zero_coverage="TSV: NA; RDA: X=N=0 and Z=NA; H5AD: depth=ratio=0")
    write_json(dendro_out / "conversion_metadata.json", metadata)
    write_json(geinj_out / "conversion_metadata.json",
               dict(metadata, matrix_orientation="bins x loci", ratio_rule="X/N for N>0",
                    Z_used=False))
    print(f"Shared inputs: {len(sites)} SNVs x {len(units)} bins; {lost_calls} calls without count coverage")


if __name__ == "__main__":
    {"select": select, "split": split, "collect": collect}[sys.argv[1]]()
PY
}

mkdir -p "$GATK_OUT/logs" "$GATK_OUT/unit_bams" \
  "$GATK_OUT/vcfs_raw" "$GATK_OUT/vcfs_filtered" "$DENDRO_OUT" "$GEINJ_OUT"

if [[ ! -f "${REF%.*}.dict" ]]; then
  "$GATK" CreateSequenceDictionary -R "$REF" -O "${REF%.*}.dict" \
    > "$GATK_OUT/logs/reference_dict.log" 2>&1
fi

echo "Selecting bins from BAM CB tags"
run_python select > "$GATK_OUT/logs/select.log" 2>&1

echo "Marking duplicates and splitting RNA alignments"
"$GATK" --java-options "-Xmx$JAVA_MEMORY" MarkDuplicates \
  -I "$GATK_OUT/selected.bam" -O "$GATK_OUT/marked.bam" \
  -M "$GATK_OUT/markduplicates_metrics.txt" \
  --BARCODE_TAG RX --REMOVE_DUPLICATES false \
  > "$GATK_OUT/logs/markduplicates.log" 2>&1
"$SAMTOOLS" index "$GATK_OUT/marked.bam"

"$GATK" --java-options "-Xmx$JAVA_MEMORY" SplitNCigarReads \
  -R "$REF" -I "$GATK_OUT/marked.bam" -O "$GATK_OUT/split_ncigar.raw.bam" \
  --create-output-bam-index false \
  > "$GATK_OUT/logs/split_ncigar.log" 2>&1
"$SAMTOOLS" sort -@ "$THREADS" -o "$GATK_OUT/split_ncigar.sorted.bam" \
  "$GATK_OUT/split_ncigar.raw.bam"
"$SAMTOOLS" index "$GATK_OUT/split_ncigar.sorted.bam"
rm "$GATK_OUT/split_ncigar.raw.bam"
PROCESSED_BAM="$GATK_OUT/split_ncigar.sorted.bam"

if [[ -n "$KNOWN_SITES_VCF" ]]; then
  "$GATK" --java-options "-Xmx$JAVA_MEMORY" BaseRecalibrator \
    -R "$REF" -I "$PROCESSED_BAM" --known-sites "$KNOWN_SITES_VCF" \
    -O "$GATK_OUT/recalibration.table" > "$GATK_OUT/logs/base_recalibrator.log" 2>&1
  "$GATK" --java-options "-Xmx$JAVA_MEMORY" ApplyBQSR \
    -R "$REF" -I "$PROCESSED_BAM" --bqsr-recal-file "$GATK_OUT/recalibration.table" \
    -O "$GATK_OUT/bqsr.bam" --create-output-bam-index false \
    > "$GATK_OUT/logs/apply_bqsr.log" 2>&1
  "$SAMTOOLS" index "$GATK_OUT/bqsr.bam"
  PROCESSED_BAM="$GATK_OUT/bqsr.bam"
  echo "BQSR applied: $KNOWN_SITES_VCF" > "$GATK_OUT/bqsr_status.txt"
else
  echo "BQSR skipped: no known-sites VCF supplied" > "$GATK_OUT/bqsr_status.txt"
fi
export PROCESSED_BAM

run_python split > "$GATK_OUT/logs/split_bins.log" 2>&1
echo "Calling SNVs per bin and applying FS/QD filters"
tail -n +2 "$GATK_OUT/units.tsv" | cut -f1 | xargs -r -n1 -P "$JOBS" bash -c '
  set -euo pipefail
  unit=$1
  "$GATK" --java-options "-Xmx$HC_MEMORY" HaplotypeCaller \
    -R "$REF" -I "$GATK_OUT/unit_bams/${unit}.bam" \
    -O "$GATK_OUT/vcfs_raw/${unit}.vcf.gz" --sample-ploidy "$SAMPLE_PLOIDY" \
    --standard-min-confidence-threshold-for-calling "$CALL_CONF" \
    --dont-use-soft-clipped-bases "$DONT_USE_SOFT_CLIPPED_BASES" \
    --native-pair-hmm-threads "$HC_THREADS" \
    > "$GATK_OUT/logs/${unit}.haplotypecaller.log" 2>&1
  "$GATK" --java-options "-Xmx$FILTER_MEMORY" VariantFiltration \
    -R "$REF" -V "$GATK_OUT/vcfs_raw/${unit}.vcf.gz" \
    -O "$GATK_OUT/vcfs_filtered/${unit}.vcf.gz" \
    --filter-name FS --filter-expression "FS > $MAX_FS" \
    --filter-name QD --filter-expression "QD < $MIN_QD" \
    > "$GATK_OUT/logs/${unit}.filter.log" 2>&1
' _

echo "Counting shared sites and writing GEI-NJ input"
run_python collect > "$GATK_OUT/logs/collect.log" 2>&1

echo "Saving DENDRO input"
"$RSCRIPT" --vanilla - > "$GATK_OUT/logs/save_rda.log" 2>&1 <<'RS'
folder <- Sys.getenv("DENDRO_OUT")
read_matrix <- function(name) {
    con <- gzfile(file.path(folder, paste0(name, ".tsv.gz")), "rt")
    on.exit(close(con))
    x <- as.matrix(read.delim(con, row.names = 1, check.names = FALSE,
                              comment.char = "", quote = "", na.strings = "NA"))
    storage.mode(x) <- "double"
    x
}
X <- read_matrix("X")
N <- read_matrix("N")
Z <- read_matrix("Z")
Info <- read.delim(file.path(folder, "Info.tsv"), row.names = 1,
                   check.names = FALSE, comment.char = "", quote = "")
stopifnot(identical(dimnames(X), dimnames(N)),
          identical(dimnames(X), dimnames(Z)),
          identical(rownames(X), rownames(Info)),
          identical(is.na(X), is.na(N)), identical(is.na(N), is.na(Z)),
          all(X >= 0, na.rm = TRUE), all(X <= N, na.rm = TRUE),
          all(Z[!is.na(Z)] %in% c(0, 1)))
# Zero observations are not reference genotypes.
X[is.na(X)] <- 0
N[is.na(N)] <- 0
conversion_metadata <- list(
    count_unit = "unique CB+UB molecules",
    missing_count_rule = "Zero coverage: X=N=0; Z=NA",
    details = readLines(file.path(folder, "conversion_metadata.json"))
)
save(X, N, Z, Info, conversion_metadata,
     file = file.path(folder, "DENDRO_input.rda"))
cat("Saved", nrow(X), "SNVs x", ncol(X), "bins\n")
RS

echo "Finished: $DENDRO_OUT/DENDRO_input.rda"
echo "Finished: $GEINJ_OUT/sc_depth.h5ad"
echo "Finished: $GEINJ_OUT/sc_ratio.h5ad"
