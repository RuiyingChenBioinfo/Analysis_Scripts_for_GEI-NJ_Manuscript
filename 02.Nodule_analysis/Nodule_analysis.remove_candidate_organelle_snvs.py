#!/usr/bin/env python3
"""Remove candidate-gene SNVs from LineMut ratio and depth matrices.

SNV names: chromosome:position:REF:ALT, with 1-based positions.
GFF3 gene spans: 1-based, inclusive of both endpoints.
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("LineMut_Call_new"))
    parser.add_argument(
        "--genes", type=Path, default=Path("candidate_lotus_organelle_list.tsv")
    )
    parser.add_argument(
        "--gff", type=Path,
        default=Path(
            "Lotus_japonicus/checkgtf/genes.checked.gff"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("LineMut_Call_new_no_candidate_organelle"),
    )
    args = parser.parse_args()

    candidates = set(args.genes.read_text().split())
    intervals = read_gene_intervals(args.gff, candidates)
    matched = set(intervals["gene"])
    print(f"Candidate genes: {len(candidates)}; found in GFF: {len(matched)}")
    if candidates - matched:
        print("Genes not found: " + ", ".join(sorted(candidates - matched)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    removed_tables = []
    for prefix in ("sc", "unit"):
        ratio = ad.read_h5ad(args.input_dir / f"{prefix}_ratio.h5ad")
        depth = ad.read_h5ad(args.input_dir / f"{prefix}_depth.h5ad")
        removed = find_removed_snvs(ratio.var_names, intervals)
        removed_tables.append(removed.assign(dataset=prefix))

        for kind, data in (("ratio", ratio), ("depth", depth)):
            keep = ~data.var_names.isin(removed["snv"])
            data[:, keep].copy().write_h5ad(
                args.output_dir / f"{prefix}_{kind}.h5ad", compression="gzip"
            )
            print(
                f"{prefix}_{kind}: {data.n_vars} -> {keep.sum()} SNVs "
                f"(removed {(~keep).sum()}); {data.n_obs} observations"
            )

    pd.concat(removed_tables, ignore_index=True).drop_duplicates().to_csv(
        args.output_dir / "removed_snvs.tsv", sep="\t", index=False
    )


if __name__ == "__main__":
    main()
