# Reproduce pooled callability from LineMut unit depth; no ratio input is used.
# Pool cortex and vascular depths before applying the depth threshold.
# Shared/specific percentages use the callable union as their denominator.
# Usage: python lotus_callability_calibration.py --depth unit_depth.h5ad

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def main():
    parser = argparse.ArgumentParser(description="Reproduce Lotus pooled callability calibration.")
    parser.add_argument("--depth", type=Path, required=True, help="LineMut unit_depth H5AD file.")
    parser.add_argument("--min-depth", type=int, default=3, help="Minimum pooled UMI depth (default: 3).")
    parser.add_argument("--output-dir", type=Path, default=Path("lotus_callability_results"))
    args = parser.parse_args()
    if args.min_depth < 1:
        parser.error("--min-depth must be a positive integer.")

    depth = ad.read_h5ad(args.depth)
    infected_name = "Infected_Zone"
    uninfected_names = ["Early_nodule_cortex", "Vascular_tissue"]
    selected = [infected_name, *uninfected_names]
    if not depth.obs_names.is_unique or not depth.var_names.is_unique:
        raise ValueError("Observation names and site identifiers must be unique.")
    missing = sorted(set(selected) - set(depth.obs_names))
    if missing:
        raise ValueError(f"Missing cell-type rows: {missing}")
    if depth.n_vars == 0:
        raise ValueError("The depth matrix contains no sites.")
    if depth.X is None:
        raise ValueError("The depth matrix must be stored in adata.X.")
    values = depth.X.data if sparse.issparse(depth.X) else np.asarray(depth.X)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.rint(values)).any():
        raise ValueError("UMI depths must be finite, nonnegative integer counts.")

    matrix = depth[selected, :].X
    matrix = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    matrix = matrix.astype(np.float64)
    infected_depth = matrix[0]
    uninfected_depth = matrix[1] + matrix[2]
    infected_callable = infected_depth >= args.min_depth
    uninfected_callable = uninfected_depth >= args.min_depth
    codes = infected_callable.astype(np.uint8) + 2 * uninfected_callable.astype(np.uint8)
    categories = np.array(["neither", "infected_only", "uninfected_only", "shared"])

    pass1 = np.bincount(codes, minlength=4)
    pass2 = Counter(
        int(infected_depth[j] >= args.min_depth)
        + 2 * int(uninfected_depth[j] >= args.min_depth)
        for j in range(depth.n_vars)
    )
    infected_sites = set(np.flatnonzero(infected_callable))
    uninfected_sites = set(np.flatnonzero(uninfected_callable))
    pass3 = np.array([
        depth.n_vars - len(infected_sites | uninfected_sites),
        len(infected_sites - uninfected_sites),
        len(uninfected_sites - infected_sites),
        len(infected_sites & uninfected_sites),
    ])
    if not np.array_equal(pass1, [pass2[i] for i in range(4)]) or not np.array_equal(pass1, pass3):
        raise RuntimeError("The three counting methods disagree.")

    counts = dict(zip(categories.tolist(), map(int, pass1)))
    union = int(pass1[1:].sum())
    if union == 0:
        raise ValueError("No sites are callable in either group at this threshold.")
    summary = pd.DataFrame({
        "category": categories,
        "n_sites": pass1,
        "percent_of_all_candidates": pass1 / depth.n_vars * 100,
        "percent_of_callable_union": [np.nan, *(pass1[1:] / union * 100)],
    })
    specific_fraction = (counts["infected_only"] + counts["uninfected_only"]) / union
    symmetric_specific = specific_fraction / 2
    rounded_specific = round(symmetric_specific, 2)
    calibration = pd.DataFrame({
        "pattern": ["A_only", "B_only", "shared"],
        "symmetrized_fraction": [symmetric_specific, symmetric_specific, counts["shared"] / union],
        "rounded_fraction": [rounded_specific, rounded_specific, round(1 - 2 * rounded_specific, 2)],
    })
    per_site = pd.DataFrame({
        "site": depth.var_names,
        "infected_pooled_depth": infected_depth.astype(np.int64),
        "uninfected_pooled_depth": uninfected_depth.astype(np.int64),
        "infected_callable": infected_callable,
        "uninfected_callable": uninfected_callable,
        "category": categories[codes],
    })
    digest = hashlib.sha256()
    with args.depth.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata = {
        "input_file": str(args.depth.resolve()),
        "input_sha256": digest.hexdigest(),
        "input_matrix": "unit_depth.X",
        "ratio_used": False,
        "min_pooled_depth": args.min_depth,
        "infected_units": [infected_name],
        "uninfected_units": uninfected_names,
        "excluded_units": [str(name) for name in depth.obs_names if name not in selected],
        "n_candidate_sites": depth.n_vars,
        "n_callable_union": union,
        "counts": counts,
        "counting_checks": ["numpy_bincount", "per_site_counter", "set_operations"],
        "three_checks_agree": True,
        "rounding_rule": "Round each symmetric specific fraction to the nearest 0.01; assign the remainder to shared.",
        "interpretation": "Coverage patterns within the retained candidate panel provide an empirical proxy for accessibility, not direct biological accessibility estimates.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "callability_summary.csv", index=False)
    calibration.to_csv(args.output_dir / "accessibility_calibration.csv", index=False)
    per_site.to_csv(args.output_dir / "per_site_callability.csv", index=False)
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Candidate sites: {depth.n_vars:,}")
    print(f"Callable in at least one group: {union:,}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(calibration.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
    print("Three counting checks agree.")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
