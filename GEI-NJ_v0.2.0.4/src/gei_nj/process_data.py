def _dense(values):
    import numpy as np
    from scipy import sparse

    return np.asarray(values.toarray() if sparse.issparse(values) else values, dtype=float)


def _depth_ratio(adata, depth_layer="total_depth", warn_nan=False):
    import warnings
    import numpy as np

    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Cell and SNV names must be unique.")
    if depth_layer not in adata.layers:
        raise KeyError(f"Missing '{depth_layer}' in adata.layers.")
    depth = _dense(adata.layers[depth_layer])
    ratio = _dense(adata.X)
    if not np.isfinite(depth).all() or (depth < 0).any() or (depth != np.floor(depth)).any():
        raise ValueError("Depth must contain finite, non-negative integer counts; NaN is not allowed.")
    if np.isinf(ratio).any() or ((ratio < 0) | (ratio > 1)).any():
        raise ValueError("SNV ratios must be in [0, 1]; infinity is not allowed.")
    covered = depth > 0
    if np.isnan(ratio[covered]).any():
        raise ValueError("NaN ratios at positive depth are not allowed; provide the observed ALT/depth ratio.")
    if ((~covered) & np.isfinite(ratio) & (ratio != 0)).any():
        raise ValueError("At zero depth, SNV ratios must be 0 or NaN to represent missing observations.")
    n_missing = int((np.isnan(ratio) & ~covered).sum())
    if warn_nan and n_missing:
        warnings.warn(f"{n_missing} NaN ratio entries at zero depth are treated as missing.",
                      UserWarning, stacklevel=3)
    return depth, ratio


def _clear_processed_layers(adata, clear_state=False):
    settings = adata.uns.pop("geinj_zero_filling", {})
    layers = {"callable", "zero_filled_state"}
    layers.update(settings[key] for key in ["callable_layer", "output_layer"] if key in settings)
    if clear_state:
        state = adata.uns.pop("geinj_state", {})
        layers.update(["state", state.get("layer_name", "state")])
    for layer in layers:
        if layer in adata.layers:
            del adata.layers[layer]


def comb_depth_ratio(ad_depth, ad_ratio):
    """Combine matched ratio and depth AnnData objects, aligning their names.

    Ratios are raw ALT/depth values. Zero-depth ratios may be 0 or NaN;
    NaN ratios at positive depth and non-finite depths are rejected.
    The returned copy keeps raw ratios in X and stores a total_depth layer.
    """
    for adata in [ad_depth, ad_ratio]:
        if not adata.obs_names.is_unique or not adata.var_names.is_unique:
            raise ValueError("Cell and SNV names must be unique.")
    if (set(ad_depth.obs_names) != set(ad_ratio.obs_names)
            or set(ad_depth.var_names) != set(ad_ratio.var_names)):
        raise ValueError("Depth and ratio inputs must contain the same cell and SNV names.")
    adata = ad_ratio.copy()
    adata.layers["total_depth"] = ad_depth[adata.obs_names, adata.var_names].X.copy()
    _depth_ratio(adata, warn_nan=not adata.uns.get("geinj_input_checked", False))
    adata.uns["geinj_input_checked"] = True
    return adata


def _identify_germline(depth, ratio, min_callable_cells_germline, min_pooled_ratio_germline, min_E_germline):
    import numpy as np
    from scipy.special import logsumexp

    callable_sites = depth > 0
    all_alt = callable_sites & (ratio == 1.)
    n_callable = callable_sites.sum(axis=0)
    failures = n_callable - all_alt.sum(axis=0)
    total_depth = depth.sum(axis=0)
    alt = np.where(callable_sites, ratio, 0.) * depth
    pooled_ratio = np.divide(alt.sum(axis=0), total_depth,
        out=np.full(depth.shape[1], np.nan), where=total_depth > 0)
    upper = (np.log1p(depth) * all_alt).sum(axis=0)
    log_e = np.full(depth.shape[1], np.nan)
    threshold = np.log(min_E_germline) - 1e-12
    selected = np.flatnonzero((n_callable >= min_callable_cells_germline) & (upper >= threshold))
    nodes, weights = np.polynomial.legendre.leggauss(64)
    lam = (nodes + 1.) / 2.
    logweights = np.log(weights / 2.) + 3. * np.log(lam)
    logweights -= logsumexp(logweights)
    for start in range(0, len(selected), 512):
        idx = selected[start:start + 512]
        site_depth, success = depth[:, idx], all_alt[:, idx]
        components = np.empty((len(lam), len(idx)))
        for j, value in enumerate(lam):
            components[j] = ((np.log1p(value * site_depth) * success).sum(axis=0)
                + failures[idx] * np.log1p(-value) + logweights[j])
        perfect = np.where(failures[idx] == 0, upper[idx], -np.inf)
        log_e[idx] = np.logaddexp(perfect, logsumexp(components, axis=0)) - np.log(2.)
    return (n_callable >= min_callable_cells_germline) & (
        (pooled_ratio >= min_pooled_ratio_germline) | (log_e >= threshold))


def filter_snvs(
    adata,
    min_total_depth=5,
    remove_germline=True,
    min_callable_cells_germline=3,
    min_pooled_ratio_germline=0.90,
    min_E_germline=10.,
    exclude_snvs=None,
    depth_layer="total_depth",
    copy=True,
    verbose=True,
):
    """Filter SNVs by total depth, germline evidence and an optional TSV list.

    Germline candidates require min_callable_cells_germline entries with depth > 0,
    and either pooled ALT ratio >= min_pooled_ratio_germline or E >= min_E_germline.
    exclude_snvs is a one-column, headerless TSV of exact SNV identifiers.
    Filtering uses raw ratios in X and clears previously processed layers.
    Germline thresholds apply only when remove_germline=True.
    """
    import numpy as np
    import pandas as pd

    if not np.isfinite(min_total_depth) or min_total_depth < 1 or min_total_depth != int(min_total_depth):
        raise ValueError("min_total_depth must be a positive integer.")
    if remove_germline:
        if (not np.isfinite(min_callable_cells_germline) or min_callable_cells_germline < 1
                or min_callable_cells_germline != int(min_callable_cells_germline)):
            raise ValueError("min_callable_cells_germline must be a positive integer.")
        if not np.isfinite(min_pooled_ratio_germline) or not 0 <= min_pooled_ratio_germline <= 1:
            raise ValueError("min_pooled_ratio_germline must be in [0, 1].")
        if not np.isfinite(min_E_germline) or min_E_germline <= 0:
            raise ValueError("min_E_germline must be finite and positive.")
    depth, ratio = _depth_ratio(adata, depth_layer,
                              warn_nan=not adata.uns.get("geinj_input_checked", False))
    enough_depth = depth.sum(axis=0) >= min_total_depth
    germline = (_identify_germline(depth, ratio, min_callable_cells_germline,
                                   min_pooled_ratio_germline, min_E_germline)
                if remove_germline else np.zeros(adata.n_vars, dtype=bool))
    user_sites = set()
    if exclude_snvs is not None:
        table = pd.read_csv(exclude_snvs, sep="\t", header=None, dtype=str, keep_default_na=False)
        if table.shape[1] != 1:
            raise ValueError("exclude_snvs must be a one-column TSV without a header.")
        user_sites = set(table.iloc[:, 0])
    user_remove = adata.var_names.isin(user_sites)
    keep = enough_depth & ~germline & ~user_remove
    diagnostics = dict(min_total_depth=int(min_total_depth), remove_germline=bool(remove_germline),
        min_callable_cells_germline=int(min_callable_cells_germline),
        min_pooled_ratio_germline=float(min_pooled_ratio_germline), min_E_germline=float(min_E_germline),
        n_sites_input=adata.n_vars, n_sites_retained=int(keep.sum()),
        n_low_total_depth_sites_removed=int((~enough_depth).sum()),
        n_germline_candidates_removed=int((enough_depth & germline).sum()),
        n_user_snvs_matched=int(user_remove.sum()),
        n_user_snvs_removed=int((enough_depth & ~germline & user_remove).sum()),
        n_user_snvs_unmatched=len(user_sites - set(adata.var_names)))
    if copy:
        ad = adata[:, keep].copy()
    else:
        adata._inplace_subset_var(keep)
        ad = adata
    _clear_processed_layers(ad, clear_state=True)
    ad.uns["geinj_input_checked"] = True
    ad.uns["geinj_filter"] = diagnostics
    if verbose:
        print(f"SNV number before filter: {diagnostics['n_sites_input']}")
        print(f"SNV number after filter: {ad.n_vars}")
        if exclude_snvs is not None:
            print(f"User-defined SNVs matched: {diagnostics['n_user_snvs_matched']}; "
                  f"not present: {diagnostics['n_user_snvs_unmatched']}")
    return ad


def assign_state(
    adata,
    method="two-state",
    state_low=0.10,
    layer_name="state",
    depth_layer="total_depth",
    copy=True,
):
    """Assign binary ALT states or continuous ratios without changing X.

    two-state assigns 1 to ratios >= state_low and 0 to lower ratios.
    continuous uses the original ratios and ignores state_low.
    layer_name stores temporary states; zero-depth entries remain missing.
    zero_filling consumes this layer and stores the filled states separately.
    """
    import numpy as np

    if method not in ["two-state", "continuous"]:
        raise ValueError("method must be 'two-state' or 'continuous'.")
    if method == "two-state" and (not np.isfinite(state_low) or not 0 < state_low <= 1):
        raise ValueError("state_low must be in (0, 1].")
    depth, ratio = _depth_ratio(adata, depth_layer,
                              warn_nan=not adata.uns.get("geinj_input_checked", False))
    ad = adata.copy() if copy else adata
    _clear_processed_layers(ad, clear_state=True)
    values = ratio.copy()
    observed = depth > 0
    values[~observed] = np.nan
    if method == "two-state":
        values[observed] = (values[observed] >= state_low).astype(float)
    ad.layers[layer_name] = values
    ad.uns["geinj_input_checked"] = True
    ad.uns["geinj_state"] = dict(method=method, layer_name=layer_name)
    if method == "two-state":
        ad.uns["geinj_state"]["state_low"] = float(state_low)
    return ad


def zero_filling(
    adata,
    min_callable_depth=1,
    depth_layer="total_depth",
    callable_layer="callable",
    output_layer="zero_filled_state",
    copy=True,
    verbose=True,
):
    """Fill non-callable states with zero, retaining callable values and raw X.

    Read the temporary layer created by assign_state and remove it after
    writing output_layer. Re-run assign_state before changing the threshold.
    """
    import numpy as np

    if (not np.isfinite(min_callable_depth) or min_callable_depth < 1
            or min_callable_depth != int(min_callable_depth)):
        raise ValueError("min_callable_depth must be a positive integer.")
    depth, _ = _depth_ratio(adata, depth_layer,
                          warn_nan=not adata.uns.get("geinj_input_checked", False))
    state_layer = adata.uns.get("geinj_state", {}).get("layer_name", "state")
    if state_layer not in adata.layers:
        raise KeyError(f"Missing '{state_layer}' in adata.layers; run assign_state before zero_filling.")
    values = _dense(adata.layers[state_layer]).copy()
    if not np.isfinite(values[depth > 0]).all():
        raise ValueError("State values at positive depth must be finite; NaN cannot be silently filled.")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("State values must be in [0, 1].")
    ad = adata.copy() if copy else adata
    _clear_processed_layers(ad)
    callable_sites = depth >= min_callable_depth
    values[~callable_sites] = 0.
    ad.layers[callable_layer] = callable_sites.astype("int8")
    ad.layers[output_layer] = values
    if state_layer != output_layer:
        del ad.layers[state_layer]
    ad.uns["geinj_input_checked"] = True
    ad.uns["geinj_zero_filling"] = dict(min_callable_depth=int(min_callable_depth),
        callable_layer=callable_layer, output_layer=output_layer)
    if verbose:
        print(f"Callable SNV entries: {int(callable_sites.sum())}")
        print(f"Non-callable entries filled with zero: {int((~callable_sites).sum())}")
    return ad
