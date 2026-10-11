# Fit a weighted negative binomial model to Lotus bin50 UMI depths; ratios are unused.
# Define accessibility using pooled unit depth >= 3 and retain bin-level zeros.
# Weight states equally and exclusive/shared entries 1:3 within each state.
# Without --bin-map, recover bin groups from exact sc-to-unit depth aggregation.
# Cross-check the fit using profile likelihood, joint optimization, and a score root.

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import anndata as ad
import numpy as np
import pandas as pd
import scipy
from scipy import optimize, sparse, special, stats


def load_depth(filename):
    data = ad.read_h5ad(filename)
    if not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError(f"Non-unique observation or site names: {filename}")
    if data.X is None or not data.n_obs or not data.n_vars:
        raise ValueError(f"Missing or empty depth matrix: {filename}")
    matrix = data.X.toarray() if sparse.issparse(data.X) else np.asarray(data.X)
    if not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix != np.rint(matrix)).any():
        raise ValueError(f"Depth must contain finite, nonnegative integer counts: {filename}")
    if matrix.max() >= 2**53:
        raise ValueError("Counts exceed the range supported for exact float64 aggregation.")
    return data, matrix.astype(np.float64)


def resolve_bin_groups(sc, unit, sc_depth, unit_depth, map_file):
    details = {}
    if map_file is not None:
        mapping = pd.read_csv(map_file, dtype=str, keep_default_na=False)
        if not {"bin", "unit"}.issubset(mapping.columns):
            raise ValueError("--bin-map must be a CSV with columns named bin and unit.")
        if mapping["bin"].duplicated().any():
            raise ValueError("Each bin must occur once in --bin-map.")
        mapping = mapping.set_index("bin")
        if not sc.obs_names.isin(mapping.index).all():
            raise ValueError("--bin-map does not annotate every sc_depth row.")
        names = mapping.loc[sc.obs_names, "unit"].to_numpy()
        if not np.isin(names, unit.obs_names).all():
            raise ValueError("--bin-map contains a selected bin with an unknown unit label.")
        labels = unit.obs_names.get_indexer(names)
        details["method"] = "supplied_bin_map"
    else:
        coefficients, _, rank, _ = np.linalg.lstsq(sc_depth.T, unit_depth.T, rcond=None)
        labels = coefficients.argmax(axis=1)
        assignment = np.eye(unit.n_obs)[labels]
        deviation = float(np.max(np.abs(coefficients - assignment)))
        if rank != sc.n_obs or deviation > 1e-8:
            raise ValueError("A unique exact bin-to-unit mapping cannot be recovered; provide --bin-map.")
        details.update(method="exact_depth_aggregation_reconstruction", rank=int(rank),
                       maximum_assignment_deviation=deviation)
    reconstructed = np.stack([sc_depth[labels == i].sum(axis=0) for i in range(unit.n_obs)])
    if not np.array_equal(reconstructed, unit_depth):
        raise ValueError("Summed bin depths do not exactly reproduce the matched unit depths.")
    details["all_unit_depths_match_exactly"] = True
    return labels, details


def fit_three_methods(values, probabilities, block_mean):
    mean = float(probabilities @ values)
    variance = float(probabilities @ ((values - mean) ** 2))
    if mean <= 0 or variance <= mean:
        raise ValueError("This calibration requires positive mean and overdispersed depths.")

    def nll(log_parameters):
        mu, size = np.exp(log_parameters)
        return -float(probabilities @ stats.nbinom.logpmf(values, size, size / (size + mu)))

    log_bounds = (-12, 6)
    profile = optimize.minimize_scalar(
        lambda log_size: nll([np.log(mean), log_size]),
        bounds=log_bounds, method="bounded", options={"xatol": 1e-12},
    )
    if not profile.success or not log_bounds[0] + 1e-5 < profile.x < log_bounds[1] - 1e-5:
        raise RuntimeError("Profile likelihood failed or reached the size search boundary.")
    joint = optimize.minimize(
        nll, np.log([0.2, 0.2]), method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 10000},
    )
    if not joint.success:
        raise RuntimeError(f"Joint likelihood optimization failed: {joint.message}")

    def score(size):
        return float(probabilities @ (special.digamma(values + size) - special.digamma(size))
                     + np.log(size / (size + block_mean)))

    if score(1e-4) * score(20.0) >= 0:
        raise RuntimeError("The NB size score root is outside the calibration search interval.")
    root_size = optimize.brentq(score, 1e-4, 20.0, xtol=1e-13)
    joint_mean, joint_size = np.exp(joint.x)
    fits = pd.DataFrame([
        {"method": "profile_likelihood", "mean": mean, "size": float(np.exp(profile.x))},
        {"method": "joint_likelihood", "mean": float(joint_mean), "size": float(joint_size)},
        {"method": "score_root", "mean": block_mean, "size": float(root_size)},
    ])
    if not np.allclose(fits["mean"], mean, rtol=0, atol=1e-6):
        raise RuntimeError("The three mean estimates disagree.")
    if not np.allclose(fits["size"], root_size, rtol=0, atol=1e-6):
        raise RuntimeError("The three size estimates disagree.")
    if not np.isfinite(fits[["mean", "size"]].to_numpy()).all() or abs(score(root_size)) > 1e-7:
        raise RuntimeError("Invalid parameter estimate or score residual.")
    return fits, mean, float(root_size), variance


def sha256(filename):
    digest = hashlib.sha256()
    with Path(filename).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Fit Lotus bin50 depth parameters for simulation.")
    parser.add_argument("--sc-depth", type=Path, required=True)
    parser.add_argument("--unit-depth", type=Path, required=True)
    parser.add_argument("--bin-map", type=Path, help="Optional CSV with bin and unit columns.")
    parser.add_argument("--output-dir", type=Path, default=Path("lotus_nb_depth_fit"))
    args = parser.parse_args()
    sc, sc_depth = load_depth(args.sc_depth)
    unit, unit_depth = load_depth(args.unit_depth)
    if sc.n_vars != unit.n_vars or set(sc.var_names) != set(unit.var_names):
        raise ValueError("sc_depth and unit_depth must contain the same retained candidate sites.")
    sc_depth = sc_depth[:, sc.var_names.get_indexer(unit.var_names)]
    required = ["Infected_Zone", "Early_nodule_cortex", "Vascular_tissue"]
    if not set(required).issubset(unit.obs_names):
        raise ValueError(f"Required unit labels: {required}")
    labels, mapping_details = resolve_bin_groups(sc, unit, sc_depth, unit_depth, args.bin_map)
    infected_index, cortex_index, vascular_index = unit.obs_names.get_indexer(required)
    infected_rows = labels == infected_index
    uninfected_rows = np.isin(labels, [cortex_index, vascular_index])
    infected_callable = unit_depth[infected_index] >= 3
    uninfected_callable = (unit_depth[cortex_index] + unit_depth[vascular_index]) >= 3
    shared = infected_callable & uninfected_callable
    blocks = [
        ("infected_exclusive", infected_rows, infected_callable & ~uninfected_callable, 0.125),
        ("infected_shared", infected_rows, shared, 0.375),
        ("uninfected_exclusive", uninfected_rows, uninfected_callable & ~infected_callable, 0.125),
        ("uninfected_shared", uninfected_rows, shared, 0.375),
    ]
    values = np.arange(int(sc_depth.max()) + 1)
    probabilities = np.zeros(len(values), dtype=float)
    records = []
    for name, rows, columns, weight in blocks:
        entries = sc_depth[rows][:, columns].astype(np.int64).ravel()
        if not entries.size:
            raise ValueError(f"Empty calibration block: {name}; weights cannot be applied.")
        histogram = np.bincount(entries, minlength=len(values))
        probabilities += weight * histogram / entries.size
        records.append({
            "block": name, "n_bins": int(rows.sum()), "n_sites": int(columns.sum()),
            "n_entries": int(entries.size), "weight": weight, "mean": float(entries.mean()),
            "variance": float(entries.var()), "zero_fraction": float(np.mean(entries == 0)),
            "depth_ge3_fraction": float(np.mean(entries >= 3)),
        })
    probabilities /= probabilities.sum()
    block_mean = sum(record["weight"] * record["mean"] for record in records)
    fits, mean, size, variance = fit_three_methods(values, probabilities, block_mean)
    rounded_mean, rounded_size = round(mean, 2), round(size, 2)
    if rounded_mean <= 0 or rounded_size <= 0:
        raise ValueError("Rounding to two decimal places gives nonpositive parameters.")
    empirical = {"distribution": "weighted_empirical", "mean": mean, "variance": variance,
                 "zero_fraction": float(probabilities[0])}
    for threshold in [3, 5, 10, 20, 50]:
        empirical[f"depth_ge{threshold}_fraction"] = float(probabilities[values >= threshold].sum())
    comparisons = [empirical]
    histogram_output = {"depth": values, "weighted_empirical_probability": probabilities}
    for name, mu, k in [("NB_fitted", mean, size), ("NB_rounded", rounded_mean, rounded_size)]:
        p = k / (k + mu)
        record = {"distribution": name, "mean": mu, "variance": mu + mu * mu / k,
                  "zero_fraction": float(stats.nbinom.pmf(0, k, p))}
        for threshold in [3, 5, 10, 20, 50]:
            record[f"depth_ge{threshold}_fraction"] = float(stats.nbinom.sf(threshold - 1, k, p))
        comparisons.append(record)
        histogram_output[f"{name}_probability"] = stats.nbinom.pmf(values, k, p)
    inputs = {"sc_depth": args.sc_depth, "unit_depth": args.unit_depth}
    if args.bin_map is not None:
        inputs["bin_map"] = args.bin_map
    metadata = {
        "inputs": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in inputs.items()},
        "ratio_used": False, "n_candidate_sites": unit.n_vars, "pooled_depth_threshold": 3,
        "infected_bins": int(infected_rows.sum()), "uninfected_bins": int(uninfected_rows.sum()),
        "bin_mapping": mapping_details,
        "calibration": "Bin-level UMI depths, including zeros, within group-accessible retained candidate sites; equal state weights and exclusive/shared weights 1:3.",
        "parameterization": "NB mean=mu, size=k; NumPy/SciPy n=k, p=k/(k+mu); variance=mu+mu^2/k.",
        "fitted_parameters": {"mean": mean, "size": size},
        "rounded_parameters": {"mean": rounded_mean, "size": rounded_size},
        "three_methods_agree": True,
        "method_of_moments_size_for_comparison_only": mean * mean / (variance - mean),
        "scope": "Conditional calibration on the retained LineMut panel; bin50 coverage is a proxy for simulated cell coverage.",
        "versions": {"python": platform.python_version(), "anndata": version("anndata"),
                     "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"bin": sc.obs_names, "unit": unit.obs_names.to_numpy()[labels]}).to_csv(
        args.output_dir / "bin_unit_mapping.csv", index=False)
    pd.DataFrame(records).to_csv(args.output_dir / "calibration_blocks.csv", index=False)
    fits.to_csv(args.output_dir / "nb_fit_three_methods.csv", index=False)
    pd.DataFrame(comparisons).to_csv(args.output_dir / "depth_distribution_comparison.csv", index=False)
    pd.DataFrame(histogram_output).to_csv(args.output_dir / "depth_distribution.csv", index=False)
    (args.output_dir / "fit_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Bins: infected={infected_rows.sum()}, uninfected={uninfected_rows.sum()}")
    print(fits.to_string(index=False, float_format=lambda value: f"{value:.10f}"))
    print(f"Rounded parameters: mean={rounded_mean:.2f}, size={rounded_size:.2f}")
    print("Three fitting methods agree; bin-to-unit depth sums match exactly.")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
