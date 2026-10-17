#!/usr/bin/env python3
"""Remove candidate-gene SNVs from ratio and depth matrices for GEI-NJ input.

SNV names: chromosome:position:REF:ALT, with 1-based positions.
GFF3 gene spans: 1-based, inclusive of both endpoints.
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


INPUT_DIR = Path("03.GEI-NJ_input")
OUTPUT_DIR = Path("03.GEI-NJ_input_rm_organelle")
GENES_FILE = Path("04.Others/candidate_lotus_organelle_list.tsv")
GFF_FILE = Path("/home/chenr/Litianle/data2026/reference/Lotus_japonicus/checkgtf/genes.checked.gff")


def read_gene_intervals(gff_path, candidates):
    rows = []
    with open(gff_path) as handle:
        for line in handle:
            if line.startswith("##FASTA"):
                break
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if fields[2] != "gene":
                continue
            attrs = dict(
                item.split("=", 1)
                for item in fields[8].split(";")
                if "=" in item
            )
            gene_ids = {
                attrs.get(key, "").removeprefix("gene-")
                for key in ("ID", "Name", "gene")
            }
            for gene in sorted(gene_ids & candidates):
                rows.append((gene, fields[0], int(fields[3]), int(fields[4])))
    return pd.DataFrame(rows, columns=["gene", "chrom", "start", "end"])


def find_removed_snvs(snv_names, intervals):
    parts = snv_names.str.split(":", expand=True).to_frame(index=False)
    chrom = parts[0].to_numpy()
    position = parts[1].astype(int).to_numpy()
    hits = []
    for row in intervals.itertuples(index=False):
        overlap = (
            (chrom == row.chrom)
            & (position >= row.start)
            & (position <= row.end)
        )
        for index in np.flatnonzero(overlap):
            hits.append((snv_names[index], row.gene, row.chrom, position[index]))
    return pd.DataFrame(hits, columns=["snv", "gene", "chrom", "position"])


def main():
    candidates = set(GENES_FILE.read_text().split())
    intervals = read_gene_intervals(GFF_FILE, candidates)
    matched = set(intervals["gene"])
    print(f"Candidate genes: {len(candidates)}; found in GFF: {len(matched)}")
    if candidates - matched:
        print("Genes not found: " + ", ".join(sorted(candidates - matched)))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ratio = ad.read_h5ad(INPUT_DIR / "sc_ratio.h5ad")
    depth = ad.read_h5ad(INPUT_DIR / "sc_depth.h5ad")
    removed = find_removed_snvs(ratio.var_names, intervals)

    for kind, data in (("ratio", ratio), ("depth", depth)):
        keep = ~data.var_names.isin(removed["snv"])
        data[:, keep].copy().write_h5ad(
            OUTPUT_DIR / f"sc_{kind}.h5ad", compression="gzip"
        )
        print(
            f"sc_{kind}: {data.n_vars} -> {keep.sum()} SNVs "
            f"(removed {(~keep).sum()}); {data.n_obs} observations"
        )

    removed.assign(dataset="sc").drop_duplicates().to_csv(
        OUTPUT_DIR / "removed_snvs.tsv", sep="\t", index=False
    )


if __name__ == "__main__":
    main()
