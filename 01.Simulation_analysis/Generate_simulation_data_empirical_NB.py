"""Calibrate empirical coverage and generate terminal-cell SNV observations, and all settings are fixed below.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
from scipy import optimize, sparse, stats


# Settings
outdir = "Simulation_results"
INPUT_DIR = Path("../../NoduleSec_analysis/01.Data_new/03.GEI-NJ_input")
SC_DEPTH = INPUT_DIR / "sc_depth.h5ad"
SC_RATIO = INPUT_DIR / "sc_ratio.h5ad"
META_INFO = Path("../../NoduleSec_analysis/01.Data_new/04.Others/Nodule.Meta_info_282bins.csv")
OUTPUT_DIR = Path(outdir) / "01.Empirical_values"
SIMULATED_DIR = Path(outdir) / "02.Simulated_datasets"
CELLTYPE_COLUMN = "Celltype"
CALLABILITY_GROUP_A = "Vascular_tissue"
CALLABILITY_GROUP_B = "Early_nodule_cortex"
CALLABILITY_DIFF_FACTORS = {"weak": 0.5, "empirical": 1.0, "strong": 1.5}
ROUND_DIGITS = 2

probs = [1e-7, 1e-8, 1e-9]
repeat_seeds = list(range(42, 72))
genome_size = 100_000_000
last_generation = 9
sample_sizes = [100, 200, 500]
min_callable_depth = 1
MIN_CALLABLE_DEPTHS = [min_callable_depth]
beta_concentration = 2.0
observation_error_rate = 0.001  # Symmetric REF/ALT error per observed molecule.
state_start_generation = 2
state_retention_probability = 0.90
permutation_control_prob = 1e-8
permutation_control_n_cells = 200
state_names = ["A", "B"]
# Labels for observed pooled coverage summaries only.
pattern_names = ["A_only", "B_only", "A_B"]
pattern_codes = np.array([1, 2, 3], dtype=np.uint8)
state_bits = np.array([1, 2], dtype=np.uint8)


def load_matrix(filename):
    data = ad.read_h5ad(filename)
    matrix = data.X.toarray() if sparse.issparse(data.X) else np.asarray(data.X)
    return data, matrix.astype(np.float64)


def fit_nb(values, probabilities):
    mean = float(probabilities @ values)

    def nll(log_size):
        size = np.exp(log_size)
        return -float(probabilities @ stats.nbinom.logpmf(
            values, size, size / (size + mean)))

    fit = optimize.minimize_scalar(
        nll, bounds=(-12, 6), method="bounded", options={"xatol": 1e-12},
    )
    if not fit.success:
        raise RuntimeError("NB likelihood optimization failed.")
    return mean, float(np.exp(fit.x))


def fit_site_nb(depth):
    mean = depth.mean(axis=0)

    def nll(log_size):
        size = np.exp(log_size)
        return -float(stats.nbinom.logpmf(
            depth, size, size / (size + mean)).mean())

    fit = optimize.minimize_scalar(
        nll, bounds=(-12, 6), method="bounded", options={"xatol": 1e-12},
    )
    if not fit.success:
        raise RuntimeError("Site-mean NB likelihood optimization failed.")
    return mean, float(np.exp(fit.x))


def calibrate_empirical():
    sc, sc_depth = load_matrix(SC_DEPTH)
    ratio, sc_ratio = load_matrix(SC_RATIO)
    if not (sc.obs_names.equals(ratio.obs_names) and sc.var_names.equals(ratio.var_names)):
        raise ValueError("sc_depth and sc_ratio must have identical bin and site order.")
    if not (np.isfinite(sc_depth).all() and (sc_depth >= 0).all()
            and (sc_depth == np.rint(sc_depth)).all()):
        raise ValueError("Depth must contain nonnegative integer counts.")
    if not (np.isfinite(sc_ratio).all() and (sc_ratio >= 0).all() and (sc_ratio <= 1).all()
            and (sc_ratio[sc_depth == 0] == 0).all()):
        raise ValueError("Ratio must be in [0, 1] and zero at uncovered entries.")
    alt_counts = sc_ratio * sc_depth
    if not np.allclose(alt_counts, np.rint(alt_counts), rtol=0, atol=1e-5):
        raise ValueError("Ratio and depth do not match integer ALT counts.")

    # Match annotations by bin ID; keep all sites and zero depths.
    meta = pd.read_csv(META_INFO, index_col=0)
    celltypes = meta.loc[sc.obs_names, CELLTYPE_COLUMN]
    a_depth = sc_depth[celltypes.eq(CALLABILITY_GROUP_A).to_numpy()]
    b_depth = sc_depth[celltypes.eq(CALLABILITY_GROUP_B).to_numpy()]
    groups = [(CALLABILITY_GROUP_A, a_depth), (CALLABILITY_GROUP_B, b_depth)]

    # Marginal NB fits are reference summaries, not site-level parameters.
    values = np.arange(int(max(a_depth.max(), b_depth.max())) + 1)
    histograms = []
    for _, depth in groups:
        histograms.append(np.bincount(
            depth.astype(np.int64).ravel(), minlength=len(values)) / depth.size)
    distributions = [
        (CALLABILITY_GROUP_A, len(a_depth), histograms[0]),
        (CALLABILITY_GROUP_B, len(b_depth), histograms[1]),
        ("Vascular+Cortex", len(a_depth) + len(b_depth), np.mean(histograms, axis=0)),
    ]
    comparisons = []
    for name, n_bins, probabilities in distributions:
        mean, size = fit_nb(values, probabilities)
        comparisons.append({
            "model": "marginal_NB", "Celltype": name,
            "n_bins": n_bins, "n_sites": sc.n_vars, "mean": mean, "size": size,
            "empirical_variance": float(probabilities @ ((values - mean) ** 2)),
            "NB_variance": mean + mean * mean / size,
            "empirical_zero_fraction": float(probabilities[0]),
            "NB_zero_fraction": float(stats.nbinom.pmf(0, size, size / (size + mean))),
        })

    # One mean per site; refit one size per cell type.
    site_models = []
    for name, depth in groups:
        mean, size = fit_site_nb(depth)
        site_models.append((mean, size))
        comparisons.append({
            "model": "site_mean_NB", "Celltype": name,
            "n_bins": len(depth), "n_sites": sc.n_vars,
            "mean": float(mean.mean()), "size": size,
            "empirical_variance": float(depth.var()),
            "NB_variance": float((mean + mean ** 2 / size).mean() + mean.var()),
            "empirical_zero_fraction": float((depth == 0).mean()),
            "NB_zero_fraction": float(stats.nbinom.pmf(0, size, size / (size + mean)).mean()),
        })
    a_mean, a_size = site_models[0]
    b_mean, b_size = site_models[1]
    a_probability = stats.nbinom.sf(0, a_size, a_size / (a_size + a_mean))
    b_probability = stats.nbinom.sf(0, b_size, b_size / (b_size + b_mean))
    midpoint = (a_probability + b_probability) / 2
    half_difference = (a_probability - b_probability) / 2

    # Scale depth >= 1 probability differences around the paired midpoint.
    profiles = []
    scenarios = {}
    for scenario, factor in CALLABILITY_DIFF_FACTORS.items():
        a_q = midpoint + factor * half_difference
        b_q = midpoint - factor * half_difference
        a_mu = a_size * np.expm1(-np.log1p(-a_q) / a_size)
        b_mu = b_size * np.expm1(-np.log1p(-b_q) / b_size)
        scenarios[scenario] = (a_mu, b_mu)
        profiles.append(pd.DataFrame({
            "scenario": scenario, "difference_factor": factor, "site": sc.var_names,
            "min_callable_depth": 1,
            "group_A": CALLABILITY_GROUP_A, "group_B": CALLABILITY_GROUP_B,
            "mean_A": a_mu, "size_A": a_size, "callability_A": a_q,
            "mean_B": b_mu, "size_B": b_size, "callability_B": b_q,
            "probability_difference_A_minus_B": a_q - b_q,
        }))

    records, per_site, pair_records = [], [], []
    for min_callable_depth in MIN_CALLABLE_DEPTHS:
        a_counts = (a_depth >= min_callable_depth).sum(axis=0)
        b_counts = (b_depth >= min_callable_depth).sum(axis=0)
        a_fraction, b_fraction = a_counts / len(a_depth), b_counts / len(b_depth)
        fractions = {"observed": (a_fraction, b_fraction)}
        for scenario, (a_mu, b_mu) in scenarios.items():
            fractions[scenario] = (
                stats.nbinom.sf(min_callable_depth - 1, a_size, a_size / (a_size + a_mu)),
                stats.nbinom.sf(min_callable_depth - 1, b_size, b_size / (b_size + b_mu)),
            )
        a_fitted, b_fitted = (
            stats.nbinom.sf(min_callable_depth - 1, a_size, a_size / (a_size + a_mean)),
            stats.nbinom.sf(min_callable_depth - 1, b_size, b_size / (b_size + b_mean)),
        )
        per_site.append(pd.DataFrame({
            "min_callable_depth": min_callable_depth, "site": sc.var_names,
            f"{CALLABILITY_GROUP_A}_callable_bins": a_counts,
            f"{CALLABILITY_GROUP_A}_fraction": a_fraction,
            f"{CALLABILITY_GROUP_A}_NB_probability": a_fitted,
            f"{CALLABILITY_GROUP_B}_callable_bins": b_counts,
            f"{CALLABILITY_GROUP_B}_fraction": b_fraction,
            f"{CALLABILITY_GROUP_B}_NB_probability": b_fitted,
            "fraction_difference_A_minus_B": a_fraction - b_fraction,
            "NB_difference_A_minus_B": a_fitted - b_fitted,
        }))
        for scenario, (a_q, b_q) in fractions.items():
            for name, depth, q in [(CALLABILITY_GROUP_A, a_depth, a_q),
                                   (CALLABILITY_GROUP_B, b_depth, b_q)]:
                quantiles = np.quantile(q, [0, 0.25, 0.5, 0.75, 1])
                records.append({
                    "scenario": scenario, "min_callable_depth": min_callable_depth,
                    "Celltype": name, "n_bins": len(depth), "n_sites": sc.n_vars,
                    "mean": float(q.mean()), "min": quantiles[0], "q25": quantiles[1],
                    "median": quantiles[2], "q75": quantiles[3], "max": quantiles[4],
                })
            # Aggregate pair-site events before conditioning on either bin callable.
            shared = float((a_q * b_q).mean())
            a_only = float((a_q * (1 - b_q)).mean())
            b_only = float(((1 - a_q) * b_q).mean())
            neither = float(((1 - a_q) * (1 - b_q)).mean())
            union = shared + a_only + b_only
            a_mu, b_mu = (a_mean, b_mean) if scenario == "observed" else scenarios[scenario]
            pair_records.append({
                "scenario": scenario, "min_callable_depth": min_callable_depth,
                "mean_depth_A": float(a_mu.mean()), "mean_depth_B": float(b_mu.mean()),
                "mean_callability_A": float(a_q.mean()), "mean_callability_B": float(b_q.mean()),
                "mean_absolute_probability_difference": float(np.abs(a_q - b_q).mean()),
                "A_only": a_only, "B_only": b_only, "shared": shared, "neither": neither,
                "A_only_given_either": a_only / union,
                "B_only_given_either": b_only / union, "shared_given_either": shared / union,
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nb_parameters = pd.DataFrame(comparisons)
    pair_summary = pd.DataFrame(pair_records)
    nb_parameters.to_csv(OUTPUT_DIR / "nb_parameters.csv", index=False)
    profiles = pd.concat(profiles, ignore_index=True)
    profiles.to_csv(OUTPUT_DIR / "simulation_nb_parameters.csv", index=False)
    pd.concat(per_site, ignore_index=True).to_csv(OUTPUT_DIR / "per_site_callability.csv", index=False)
    pd.DataFrame(records).to_csv(OUTPUT_DIR / "callability_summary.csv", index=False)
    pair_summary.to_csv(OUTPUT_DIR / "bin_pair_callability.csv", index=False)

    print(f"Bins: A={CALLABILITY_GROUP_A} ({len(a_depth)}), B={CALLABILITY_GROUP_B} ({len(b_depth)})")
    print(f"Candidate sites: {sc.n_vars}; zero depths retained; ratio checked only.")
    print(nb_parameters[["model", "Celltype", "mean", "size"]].round(ROUND_DIGITS).to_string(index=False))
    columns = ["scenario", "mean_depth_A", "mean_depth_B", "mean_callability_A",
               "mean_callability_B", "mean_absolute_probability_difference", "shared_given_either"]
    print(pair_summary.loc[pair_summary.min_callable_depth.eq(1), columns].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print("Callability and gradients use depth >= 1.")
    print("Sample one paired profile per simulated site; keep it for all cells at that site.")
    print("Use site-specific means with their fitted sizes; do not add a fixed zeroing step.")
    print("Fit checks are in-sample and conditional on the retained candidate panel.")
    print(f"Results: {OUTPUT_DIR.resolve()}")

    return profiles


def make_lineage():
    rows = [{"cell_id": "cell_0_0", "lineage": "0", "parent": "", "generation": 0}]
    current = [("0", 0)]
    for generation in range(1, last_generation + 1):
        following = []
        for parent, index in current:
            for side in range(2):
                child_index = 2 * index + side
                child = f"{parent}-{child_index}"
                rows.append({"cell_id": f"cell_{generation}_{child_index}",
                             "lineage": child, "parent": parent, "generation": generation})
                following.append((child, child_index))
        current = following
    return pd.DataFrame(rows).set_index("lineage", drop=False)


def divide_cell_with_mutations(mut0, mut1, n_vars, prob, rng):
    childA0, childA1 = set(mut0), set(mut1)
    childB0, childB1 = set(mut0), set(mut1)
    for inherited, A, B in ((mut0, childA0, childB0), (mut1, childA1, childB1)):
        n_new = rng.poisson((n_vars - len(inherited)) * prob)
        new_sites = set()
        while len(new_sites) < n_new:
            site = int(rng.integers(n_vars))
            if site not in inherited:
                new_sites.add(site)
        for site in sorted(new_sites):
            (A, B)[int(rng.integers(2))].add(site)
    return (childA0, childA1), (childB0, childB1)


def generate_true_genotype(tree, prob, rng_mutation):
    homologs = [(set(), set()) for _ in range(len(tree))]
    for parent in range((len(tree) - 1) // 2):
        homologs[2 * parent + 1], homologs[2 * parent + 2] = divide_cell_with_mutations(
            *homologs[parent], genome_size, prob, rng_mutation)

    keep = np.flatnonzero(tree.generation.to_numpy() == last_generation)
    sites = sorted(set().union(*(h for i in keep for h in homologs[i])))
    site_index = {site: j for j, site in enumerate(sites)}
    genotype = np.zeros((len(keep), len(sites)), dtype=np.float32)
    for i, original_index in enumerate(keep):
        for h in homologs[original_index]:
            genotype[i, [site_index[site] for site in h]] += 0.5
    return tree.iloc[keep].copy(), np.asarray(sites, dtype=np.int64), genotype


def assign_cell_states(tree, rng):
    states = np.full(len(tree), "", dtype="U1")
    founders = np.flatnonzero(tree.generation.to_numpy() == state_start_generation)
    half = len(founders) // 2
    states[founders[:half]] = "A"
    states[founders[half:]] = "B"
    # Each daughter independently retains or switches its parent's state.
    for child in range(founders[-1] + 1, len(tree)):
        parent_state = states[(child - 1) // 2]
        states[child] = (parent_state if rng.random() < state_retention_probability
                         else "B" if parent_state == "A" else "A")
    return states


def generate_depth(states, site_profiles, rng):
    is_a = np.asarray(states)[:, None] == "A"
    mean = np.where(is_a, site_profiles["mean_A"].to_numpy()[None, :],
                    site_profiles["mean_B"].to_numpy()[None, :])
    size = np.where(is_a, site_profiles["size_A"].to_numpy()[None, :],
                    site_profiles["size_B"].to_numpy()[None, :])
    return rng.negative_binomial(size, size / (size + mean)).astype(np.int32)


def generate_allele_counts(genotype, depth, rng, rng_error):
    p = genotype.copy()
    heterozygous = genotype == 0.5
    if beta_concentration is not None:
        p[heterozygous] = rng.beta(beta_concentration / 2, beta_concentration / 2,
                                  size=heterozygous.sum())
    alt = rng.binomial(depth, p).astype(np.int32)
    if observation_error_rate > 0:
        # Draw both directions from the original counts.
        alt_to_ref = rng_error.binomial(alt, observation_error_rate)
        ref_to_alt = rng_error.binomial(depth - alt, observation_error_rate)
        alt = (alt - alt_to_ref + ref_to_alt).astype(np.int32)
    # Zero-depth ratios use zero placeholders; depth determines missingness.
    ratio = np.divide(alt, depth, out=np.zeros(genotype.shape, dtype=np.float32), where=depth > 0)
    return alt, ratio


def fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else np.nan


def summarize(adata, prob, seed, filename):
    depth = adata.layers["depth"]
    alt = adata.layers["alt_count"]
    callable_entries = adata.layers["callable"]
    base = {"prob": prob, "seed": seed,
            "sample_size": adata.n_obs, "file": filename,
            "scenario": adata.uns["simulation"]["scenario"],
            "callability_scenario": adata.uns["simulation"]["callability_scenario"]}
    # Legacy column name; values are cell states.
    states = adata.obs.sim_cell_type.to_numpy()
    pooled = np.stack([depth[states == state].sum(axis=0) for state in state_names])
    # These pattern codes describe observed pooled coverage only.
    observed_codes = ((pooled >= min_callable_depth) * state_bits[:, None]).sum(axis=0)
    n_union = np.count_nonzero(observed_codes)
    minimum = np.where(callable_entries, adata.X, np.inf).min(axis=0)
    maximum = np.where(callable_entries, adata.X, -np.inf).max(axis=0)
    overall = {**base, "n_cells": adata.n_obs, "n_true_snv_sites": adata.n_vars,
               "n_sites_with_alt": int((alt > 0).any(axis=0).sum()),
               "n_sites_with_any_callable_cell": int(callable_entries.any(axis=0).sum()),
               "n_sites_callable_in_pooled_states": n_union,
               "n_variable_observed_sites": int((maximum > minimum).sum()),
               "callable_entry_fraction": fraction(callable_entries.sum(), depth.size)}
    state_records = []
    for i, state in enumerate(state_names):
        selected = states == state
        d = depth[selected]
        c = callable_entries[selected]
        covered = d[c]
        state_records.append({**base, "state": state, "n_cells": int(selected.sum()),
            "n_sites": adata.n_vars, "callable_entry_fraction": fraction(c.sum(), d.size),
            "zero_depth_fraction": fraction((d == 0).sum(), d.size),
            "mean_depth_all_entries": float(d.mean()) if d.size else np.nan,
            "median_depth_callable_entries": float(np.median(covered)) if covered.size else np.nan,
            "n_sites_with_any_callable_cell": int(c.any(axis=0).sum()),
            "pooled_callable_fraction": fraction((pooled[i] >= min_callable_depth).sum(), adata.n_vars)})
    pattern_records = []
    for code, name in [(0, "None")] + list(zip(pattern_codes, pattern_names)):
        n_sites = int((observed_codes == code).sum())
        pattern_records.append({**base, "pattern": name,
            "n_observed_pooled_sites": n_sites,
            "fraction_all_sites": fraction(n_sites, adata.n_vars),
            "fraction_callable_union": fraction(n_sites, n_union) if code else np.nan})
    return overall, state_records, pattern_records


def run_simulations(profiles):
    tree = make_lineage()
    scenario_profiles = {
        name: profiles.loc[profiles.scenario.eq(name)].reset_index(drop=True)
        for name in CALLABILITY_DIFF_FACTORS
    }
    records = {name: ([], [], []) for name in scenario_profiles}
    for name, table in scenario_profiles.items():
        scenario_dir = SIMULATED_DIR / name
        scenario_dir.mkdir(parents=True)
        tree.to_csv(scenario_dir / "true_lineage.csv", index=False)
        pd.DataFrame([{
            "callability_scenario": name,
            "difference_factor": CALLABILITY_DIFF_FACTORS[name],
            "depth_model": "paired_site_mean_NB",
            "mean_depth_A": table.mean_A.mean(), "mean_depth_B": table.mean_B.mean(),
            "depth_size_A": table.size_A.iloc[0], "depth_size_B": table.size_B.iloc[0],
            "mean_callability_A": table.callability_A.mean(),
            "mean_callability_B": table.callability_B.mean(),
            "min_callable_depth": min_callable_depth,
            "beta_concentration": beta_concentration,
            "observation_error_rate": observation_error_rate,
            "state_start_generation": state_start_generation,
            "state_retention_probability": state_retention_probability,
            "state_switch_probability": 1 - state_retention_probability,
            "n_replicates": len(repeat_seeds),
            "permutation_control_prob": permutation_control_prob,
            "permutation_control_n_cells": permutation_control_n_cells,
        }]).to_csv(scenario_dir / "simulation_parameters.csv", index=False)
        for prob in probs:
            for n_cells in sample_sizes:
                (scenario_dir / f"prob_{prob:.0e}" / f"cells_{n_cells}").mkdir(parents=True)

    for prob in probs:
        prob_seed = int(round(-np.log10(prob)))
        for seed in repeat_seeds:
            streams = np.random.SeedSequence([seed, prob_seed]).spawn(11)
            rng_mutation, rng_subset, rng_type, rng_pattern = [
                np.random.default_rng(s) for s in streams[:4]]
            rng_permutation = np.random.default_rng(streams[7])
            observation_streams = [
                streams[i].spawn(len(scenario_profiles)) for i in [4, 5, 6, 8, 9, 10]]
            obs, sites, genotype = generate_true_genotype(tree, prob, rng_mutation)
            sample_order = rng_subset.permutation(len(obs))
            lineage_states = assign_cell_states(tree, rng_type)
            states = lineage_states[tree.generation.to_numpy() == last_generation]
            # Share truth, states, sampled cells and profile indices across gradients.
            profile_index = rng_pattern.integers(len(next(iter(scenario_profiles.values()))), size=len(sites))
            if prob == permutation_control_prob:
                permutation_order = rng_permutation.permutation(permutation_control_n_cells)
            # Legacy column name; values are cell states.
            obs["sim_cell_type"] = states
            obs["root_branch"] = [lineage.split("-")[1] for lineage in obs.lineage]
            obs["sampling_rank"] = np.argsort(sample_order)

            for scenario_index, (name, table) in enumerate(scenario_profiles.items()):
                scenario_dir = SIMULATED_DIR / name
                rng_depth, rng_alt, rng_error, rng_control_depth, rng_control_alt, rng_control_error = [
                    np.random.default_rng(s[scenario_index]) for s in observation_streams]
                site_profiles = table.iloc[profile_index].reset_index(drop=True)
                depth = generate_depth(states, site_profiles, rng_depth)
                alt, ratio = generate_allele_counts(genotype, depth, rng_alt, rng_error)
                var = pd.DataFrame({
                    "position": sites + 1,
                    "empirical_profile_index": profile_index,
                    "empirical_site": site_profiles.site.to_numpy(),
                    **{column: site_profiles[column].to_numpy() for column in
                       ["mean_A", "mean_B", "size_A", "size_B", "callability_A", "callability_B"]},
                }, index=pd.Index([f"SNV_{site + 1}" for site in sites], name="SNV"))
                adata = ad.AnnData(X=ratio, obs=obs.copy(), var=var)
                adata.layers["true_genotype"] = genotype
                adata.layers["depth"] = depth
                adata.layers["alt_count"] = alt
                adata.layers["callable"] = depth >= min_callable_depth
                adata.uns["simulation"] = {
                    "prob": prob, "seed": seed, "scenario": "main",
                    "callability_scenario": name,
                    "difference_factor": CALLABILITY_DIFF_FACTORS[name],
                    "genome_size_per_homolog": genome_size,
                    "last_generation": last_generation,
                    "state_start_generation": state_start_generation,
                    "initial_state_assignment": "First founder branch A; second founder branch B",
                    "state_retention_probability": state_retention_probability,
                    "state_switch_probability": 1 - state_retention_probability,
                    "min_callable_depth": min_callable_depth,
                    "beta_concentration": "None" if beta_concentration is None else beta_concentration,
                    "observation_error_rate": observation_error_rate,
                    "depth_model": "paired_site_mean_NB",
                    "empirical_group_A": CALLABILITY_GROUP_A,
                    "empirical_group_B": CALLABILITY_GROUP_B,
                    "true_lineage_file": "../../true_lineage.csv",
                }
                overall_records, state_records, pattern_records = records[name]
                for n_cells in sample_sizes:
                    keep = np.sort(sample_order[:n_cells])
                    # Filter on truth, retaining missed ALT observations.
                    retained_sites = (genotype[keep] > 0).any(axis=0)
                    sampled = adata[keep, retained_sites].copy()
                    sampled.uns["simulation"].update({
                        "n_sampled_cells": sampled.n_obs,
                        "n_snv_sites": sampled.n_vars,
                    })
                    filename = os.path.join(f"prob_{prob:.0e}", f"cells_{n_cells}",
                        f"diploid_gen{last_generation}_n{n_cells}_seed{seed}.h5ad")
                    outputs = [(sampled, filename)]
                    if prob == permutation_control_prob and n_cells == permutation_control_n_cells:
                        control = sampled.copy()
                        control.obs["sim_cell_type_original"] = sampled.obs.sim_cell_type.to_numpy()
                        control.obs["sim_cell_type"] = sampled.obs.sim_cell_type.to_numpy()[permutation_order]
                        # Re-simulate observations after permuting terminal states.
                        control_depth = generate_depth(control.obs.sim_cell_type.to_numpy(),
                                                       control.var, rng_control_depth)
                        control_alt, control_ratio = generate_allele_counts(
                            control.layers["true_genotype"], control_depth, rng_control_alt, rng_control_error)
                        control.X = control_ratio
                        control.layers["depth"] = control_depth
                        control.layers["alt_count"] = control_alt
                        control.layers["callable"] = control_depth >= min_callable_depth
                        control.uns["simulation"].update({
                            "scenario": "state_permutation",
                            "paired_main_file": os.path.join("..", "..", "..", filename),
                            "true_lineage_file": "../../../true_lineage.csv",
                        })
                        control_filename = os.path.join("state_permutation", filename)
                        (scenario_dir / control_filename).parent.mkdir(parents=True, exist_ok=True)
                        outputs.append((control, control_filename))
                    for result, result_filename in outputs:
                        overall, state_summary, patterns = summarize(result, prob, seed, result_filename)
                        overall_records.append(overall)
                        state_records.extend(state_summary)
                        pattern_records.extend(patterns)
                        result.write_h5ad(scenario_dir / result_filename, compression="gzip")
                        print(f"{name}/{overall['scenario']}: prob={prob:.0e}, seed={seed}: "
                              f"{result.n_obs} cells, {result.n_vars} SNVs, "
                              f"callable={overall['callable_entry_fraction']:.4%}", flush=True)
    for name, (overall_records, state_records, pattern_records) in records.items():
        scenario_dir = SIMULATED_DIR / name
        pd.DataFrame(overall_records).to_csv(scenario_dir / "summary_simulations.csv", index=False)
        pd.DataFrame(state_records).to_csv(scenario_dir / "summary_states.csv", index=False)
        pd.DataFrame(pattern_records).to_csv(scenario_dir / "summary_patterns.csv", index=False)
        print(f"Saved {len(overall_records)} h5ad files to {scenario_dir}.")


def main():
    if os.path.exists(outdir):
        raise SystemExit(f"The results path already exists: {outdir}. Stop running.")
    os.makedirs(outdir)
    profiles = calibrate_empirical()
    run_simulations(profiles)
    archive = shutil.make_archive(outdir, "gztar", root_dir=Path(outdir).parent, base_dir=Path(outdir).name)
    print(f"Archive: {archive}")


if __name__ == "__main__":
    main()
