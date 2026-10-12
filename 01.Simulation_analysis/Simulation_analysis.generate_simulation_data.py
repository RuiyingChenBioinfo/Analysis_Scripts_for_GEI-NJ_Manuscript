"""Generate terminal-cell samples with fixed-parameter coverage and allelic sampling.

Run with: python generate_simulation_data.py
No empirical input files are required.
Requires numpy, pandas, scipy, and anndata.
Each rate has 50 independent replicates, each with four nested sample sizes.
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import nbinom


# Settings
outdir = "Simulation_data_generation"

p_ups = [1e-7, 1e-8, 1e-9]
repeat_seeds = list(range(42, 92))
genome_size = 100_000_000
last_generation = 9
sample_sizes = [50, 100, 200, 500]
min_callable_depth = 3
beta_concentration = 2.0  # DAESC phi=0.5
depth_mean = 0.18
depth_size = 0.18  # NB variance = mean + mean**2 / size.

state_names = ["A", "B"]
pattern_names = ["A_only", "B_only", "A_B"]
pattern_codes = np.array([1, 2, 3], dtype=np.uint8)
pattern_fractions = np.array([0.20, 0.20, 0.60])
state_bits = np.array([1, 2], dtype=np.uint8)


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


def divide_cell_with_mutations(mut0, mut1, n_vars, p_up, rng):
    childA0, childA1 = set(mut0), set(mut1)
    childB0, childB1 = set(mut0), set(mut1)
    for inherited, A, B in ((mut0, childA0, childB0), (mut1, childA1, childB1)):
        n_new = rng.poisson((n_vars - len(inherited)) * p_up)
        new_sites = set()
        while len(new_sites) < n_new:
            site = int(rng.integers(n_vars))
            if site not in inherited:
                new_sites.add(site)
        for site in sorted(new_sites):
            (A, B)[int(rng.integers(2))].add(site)
    return (childA0, childA1), (childB0, childB1)


def generate_true_genotype(tree, mu, rng_mutation):
    homologs = [(set(), set()) for _ in range(len(tree))]
    for parent in range((len(tree) - 1) // 2):
        homologs[2 * parent + 1], homologs[2 * parent + 2] = divide_cell_with_mutations(
            *homologs[parent], genome_size, mu, rng_mutation)

    keep = np.flatnonzero(tree.generation.to_numpy() == last_generation)
    sites = sorted(set().union(*(h for i in keep for h in homologs[i])))
    site_index = {site: j for j, site in enumerate(sites)}
    genotype = np.zeros((len(keep), len(sites)), dtype=np.float32)
    for i, original_index in enumerate(keep):
        for h in homologs[original_index]:
            genotype[i, [site_index[site] for site in h]] += 0.5
    return tree.iloc[keep].copy(), np.asarray(sites, dtype=np.int64), genotype


def assign_cell_types(obs, rng):
    types = []
    for lineage in obs.lineage:
        p_a = 0.7 if lineage.split("-")[1] == "0" else 0.3
        types.append("A" if rng.random() < p_a else "B")
    return np.asarray(types)


def assign_patterns(n_sites, rng):
    expected = n_sites * pattern_fractions
    counts = np.floor(expected).astype(int)
    # Break rounding ties without favoring A or B.
    remainder_order = rng.permutation(len(counts))
    remainder_order = remainder_order[np.argsort(-(expected - counts)[remainder_order], kind="stable")]
    counts[remainder_order[:n_sites - counts.sum()]] += 1
    codes = np.repeat(pattern_codes, counts)
    rng.shuffle(codes)
    return codes


def generate_depth(types, codes, rng):
    bits = np.where(types == "A", state_bits[0], state_bits[1])
    accessible = (bits[:, None] & codes[None, :]) > 0
    depth = np.zeros(accessible.shape, dtype=np.int32)
    probability = depth_size / (depth_size + depth_mean)
    depth[accessible] = rng.negative_binomial(depth_size, probability, size=accessible.sum())
    return depth, accessible


def generate_allele_counts(genotype, depth, rng):
    p = genotype.copy()
    heterozygous = genotype == 0.5
    if beta_concentration is not None:
        p[heterozygous] = rng.beta(beta_concentration / 2, beta_concentration / 2,
                                  size=heterozygous.sum())
    alt = rng.binomial(depth, p).astype(np.int32)
    # Zero-depth ratios use zero placeholders; depth determines missingness.
    ratio = np.divide(alt, depth, out=np.zeros(genotype.shape, dtype=np.float32), where=depth > 0)
    return alt, ratio


def fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else np.nan


def summarize(adata, mu, seed, filename):
    depth = adata.layers["depth"]
    alt = adata.layers["alt_count"]
    callable_entries = adata.layers["callable"]
    base = {"mutation_probability": mu, "seed": seed,
            "sample_size": adata.n_obs, "file": filename}
    pooled = np.stack([depth[adata.obs.sim_cell_type == state].sum(axis=0) for state in state_names])
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
        selected = adata.obs.sim_cell_type.to_numpy() == state
        d = depth[selected]
        c = callable_entries[selected]
        covered = d[c]
        state_records.append({**base, "state": state, "n_cells": int(selected.sum()),
            "n_sites": adata.n_vars, "callable_entry_fraction": fraction(c.sum(), d.size),
            "zero_depth_fraction": fraction((d == 0).sum(), d.size),
            "accessible_entry_fraction": fraction(adata.layers["expression_accessible"][selected].sum(), d.size),
            "mean_depth_all_entries": float(d.mean()) if d.size else np.nan,
            "median_depth_callable_entries": float(np.median(covered)) if covered.size else np.nan,
            "n_sites_with_any_callable_cell": int(c.any(axis=0).sum()),
            "pooled_callable_fraction": fraction((pooled[i] >= min_callable_depth).sum(), adata.n_vars)})
    pattern_records = []
    target_by_code = dict(zip(pattern_codes, pattern_fractions))
    for code, name in [(0, "None")] + list(zip(pattern_codes, pattern_names)):
        n_sites = int((observed_codes == code).sum())
        n_preset = int((adata.var.pattern_code == code).sum())
        pattern_records.append({**base, "pattern": name,
            "target_fraction": target_by_code.get(code, 0.0),
            "n_preset_sites": n_preset,
            "preset_fraction_in_sample": fraction(n_preset, adata.n_vars),
            "n_observed_pooled_sites": n_sites,
            "fraction_all_sites": fraction(n_sites, adata.n_vars),
            "fraction_callable_union": fraction(n_sites, n_union) if code else np.nan})
    return overall, state_records, pattern_records


def main():
    if os.path.exists(outdir):
        raise SystemExit(f"The results path already exists: {outdir}. Stop running.")
    os.makedirs(outdir)
    tree = make_lineage()
    tree.to_csv(os.path.join(outdir, "true_lineage.csv"), index=False)
    probability = depth_size / (depth_size + depth_mean)
    pd.DataFrame([{"depth_mean": depth_mean, "depth_size": depth_size,
                  "depth_variance": depth_mean + depth_mean**2 / depth_size,
                  "zero_fraction_accessible": nbinom.pmf(0, depth_size, probability),
                  "callable_fraction_accessible": nbinom.sf(min_callable_depth - 1, depth_size, probability),
                  "min_callable_depth": min_callable_depth,
                  "beta_concentration": beta_concentration}]).to_csv(
                      os.path.join(outdir, "simulation_parameters.csv"), index=False)
    overall_records, state_records, pattern_records = [], [], []
    for mu in p_ups:
        rate_seed = int(round(-np.log10(mu)))
        rate_dir = os.path.join(outdir, f"mu_{mu:.0e}")
        for n_cells in sample_sizes:
            os.makedirs(os.path.join(rate_dir, f"cells_{n_cells}"))
        for seed in repeat_seeds:
            streams = np.random.SeedSequence([seed, rate_seed]).spawn(6)
            rng_mutation, rng_subset, rng_type, rng_pattern, rng_depth, rng_alt = [
                np.random.default_rng(s) for s in streams]
            obs, sites, genotype = generate_true_genotype(tree, mu, rng_mutation)
            sample_order = rng_subset.permutation(len(obs))
            types = assign_cell_types(obs, rng_type)
            codes = assign_patterns(len(sites), rng_pattern)
            depth, accessible = generate_depth(types, codes, rng_depth)
            alt, ratio = generate_allele_counts(genotype, depth, rng_alt)
            obs["sim_cell_type"] = types
            obs["root_branch"] = [lineage.split("-")[1] for lineage in obs.lineage]
            obs["sampling_rank"] = np.argsort(sample_order)
            name_by_code = dict(zip(pattern_codes, pattern_names))
            var = pd.DataFrame({"position": sites + 1, "pattern_code": codes,
                "callability_class": [name_by_code[c] for c in codes]},
                index=pd.Index([f"SNV_{site + 1}" for site in sites], name="SNV"))
            adata = ad.AnnData(X=ratio, obs=obs, var=var)
            adata.layers["true_genotype"] = genotype
            adata.layers["depth"] = depth
            adata.layers["alt_count"] = alt
            adata.layers["expression_accessible"] = accessible
            adata.layers["callable"] = depth >= min_callable_depth
            adata.uns["simulation"] = {
                "mutation_probability": mu, "seed": seed, "seed_entropy": [seed, rate_seed],
                "rng_streams": ["mutation", "subsampling", "cell_type", "patterns", "depth", "alt"],
                "genome_size_per_homolog": genome_size, "last_generation": last_generation,
                "n_full_terminal_cells": len(obs), "n_full_snv_sites": len(sites),
                "sample_sizes": sample_sizes,
                "sampling_model": "Nested prefixes of one uniform random permutation of all terminal cells; shared genotype, labels, coverage and ALT counts across sizes.",
                "state_model": "A probability 0.7 in root branch 0 and 0.3 in root branch 1; remaining cells are B. Labels are assigned once before sampling.",
                "mutation_model": "Poisson((L - inherited sites) * mu) per mother division and homolog; each new mutation goes to one random daughter.",
                "min_callable_depth": min_callable_depth,
                "allele_model": "Binomial" if beta_concentration is None else "Beta-binomial for heterozygotes; error-free homozygotes",
                "beta_concentration": "None" if beta_concentration is None else beta_concentration,
                "beta_parameter_source": "Default kappa=2 corresponds to DAESC phi=0.5; concentration is a preset, not estimated from Lotus.",
                "depth_mean": depth_mean, "depth_size": depth_size,
                "depth_model": "Independent NB draws at accessible cell-site entries; the same fixed parameters for A and B; inaccessible entries set to zero.",
                "depth_unit": "UMI-level locus coverage; fixed parameters calibrated at bin50 scale as a cell-level proxy",
                "depth_calibration": "One-time nodule bin50 fit; 112 infected and 104 uninfected bins. Pooled state depth >=3 defines accessibility. Keep bin-level zeros; balance states equally and weight exclusive/shared accessible entries 1:3. Fitted mean=0.1799617 and size=0.1798755, rounded to 0.18. No empirical data are read at runtime.",
                "depth_limitation": "A marginal approximation: matches mean and zero fraction but underestimates the rare high-depth tail; cell/site correlations are not modeled.",
                "ratio_at_zero_depth": "Zero placeholder; use depth or callable to distinguish missing from REF",
                "pattern_names": pattern_names, "pattern_fractions": pattern_fractions,
                "n_full_pattern_sites": [int((codes == c).sum()) for c in pattern_codes],
                "pattern_assignment": "Assigned on the full terminal-cell SNV panel; proportions can vary after subset-specific truth filtering.",
                "true_lineage_file": "../../true_lineage.csv",
                "site_selection": "Sites mutated in at least one sampled true genotype; observed all-zero sites retained.",
                "references": ["10.1038/s41467-023-42016-9", "10.1186/s13059-021-02367-2", "10.1002/evl3.121"],
            }
            for n_cells in sample_sizes:
                keep = np.sort(sample_order[:n_cells])
                # Filter on truth, retaining missed ALT observations.
                retained_sites = (genotype[keep] > 0).any(axis=0)
                sampled = adata[keep, retained_sites].copy()
                sampled.uns["simulation"]["n_sampled_cells"] = n_cells
                filename = os.path.join(f"mu_{mu:.0e}", f"cells_{n_cells}",
                    f"diploid_gen{last_generation}_n{n_cells}_seed{seed}.h5ad")
                overall, states, patterns = summarize(sampled, mu, seed, filename)
                overall_records.append(overall)
                state_records.extend(states)
                pattern_records.extend(patterns)
                sampled.write_h5ad(os.path.join(outdir, filename), compression="gzip")
                print(f"mu={mu:.0e}, seed={seed}: {sampled.n_obs} cells, {sampled.n_vars} SNVs, "
                      f"callable={overall['callable_entry_fraction']:.4%}", flush=True)
    pd.DataFrame(overall_records).to_csv(os.path.join(outdir, "summary_simulations.csv"), index=False)
    pd.DataFrame(state_records).to_csv(os.path.join(outdir, "summary_states.csv"), index=False)
    pd.DataFrame(pattern_records).to_csv(os.path.join(outdir, "summary_patterns.csv"), index=False)
    print(f"Saved {len(overall_records)} h5ad files to {outdir}.")


if __name__ == "__main__":
    main()
