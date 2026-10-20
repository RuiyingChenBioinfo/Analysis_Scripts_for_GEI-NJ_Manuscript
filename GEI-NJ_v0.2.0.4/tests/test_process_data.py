import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse
from scipy.integrate import quad

from gei_nj import comb_depth_ratio, filter_snvs, assign_state, zero_filling


def make_input(ratio, depth, names=None, as_sparse=False):
    ratio, depth = np.asarray(ratio, dtype=float), np.asarray(depth, dtype=float)
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(len(ratio))])
    var = pd.DataFrame(index=names or [f"snv{i}" for i in range(ratio.shape[1])])
    if as_sparse:
        ratio, depth = sparse.csr_matrix(ratio), sparse.csr_matrix(depth)
    adata = AnnData(ratio, obs=obs, var=var)
    adata.layers["total_depth"] = depth
    return adata


def dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


@pytest.mark.parametrize("as_sparse", [False, True])
def test_missing_ratio_is_warned_and_does_not_remove_whole_site(as_sparse):
    adata = make_input([[np.nan, 0], [0.2, 0.09], [0.5, 0.1]],
                       [[0, 0], [3, 2], [4, 3]], as_sparse=as_sparse)
    depth = AnnData(adata.layers["total_depth"].copy(), obs=adata.obs, var=adata.var)
    with pytest.warns(Warning, match="NaN"):
        combined = comb_depth_ratio(depth, adata)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        filtered = filter_snvs(combined, remove_germline=False, verbose=False)
        assigned = assign_state(filtered)
        filled = zero_filling(assigned, verbose=False)
    assert list(filtered.var_names) == ["snv0", "snv1"]
    np.testing.assert_allclose(dense(assigned.layers["state"]), [[np.nan, np.nan], [1, 0], [1, 1]],
                               equal_nan=True)
    np.testing.assert_array_equal(dense(filled.layers["zero_filled_state"]), [[0, 0], [1, 0], [1, 1]])
    np.testing.assert_allclose(dense(filled.X), dense(adata.X), equal_nan=True)
    assert "original_ratio" not in filled.layers
    assert "state" not in filled.layers


@pytest.mark.parametrize("ratio,depth", [
    ([[np.nan]], [[1]]), ([[np.inf]], [[1]]), ([[1.01]], [[1]]),
    ([[-0.1]], [[1]]), ([[0.1]], [[0]]), ([[0]], [[-1]]),
    ([[0]], [[1.5]]), ([[0]], [[np.nan]]), ([[0]], [[np.inf]]),
])
def test_invalid_observed_values_are_rejected(ratio, depth):
    adata = make_input(ratio, depth)
    with pytest.raises(ValueError):
        comb_depth_ratio(AnnData(adata.layers["total_depth"], obs=adata.obs, var=adata.var), adata)


def test_comb_aligns_axes_and_rejects_ambiguous_ids():
    adata = make_input([[0.1, 0.2], [0.3, 0.4]], [[10, 20], [30, 40]])
    depth = AnnData(adata.layers["total_depth"], obs=adata.obs, var=adata.var)
    combined = comb_depth_ratio(depth[::-1, ::-1].copy(), adata)
    np.testing.assert_array_equal(dense(combined.layers["total_depth"]), [[10, 20], [30, 40]])
    depth.obs_names = ["cell0", "different_cell"]
    with pytest.raises(ValueError):
        comb_depth_ratio(depth, adata)
    depth.obs_names = ["cell0", "cell0"]
    with pytest.raises(ValueError):
        comb_depth_ratio(depth, adata)


@pytest.mark.parametrize("as_sparse", [False, True])
def test_state_boundary_callability_and_continuous_restore(as_sparse):
    raw = [[0, 0.099, 0.1, 0.9, 0], [0.4, 0, 0.1, 0.5, 1]]
    depth = [[4, 3, 2, 1, 0], [1, 2, 3, 4, 5]]
    adata = make_input(raw, depth, as_sparse=as_sparse)
    assigned = assign_state(adata)
    np.testing.assert_allclose(dense(assigned.layers["state"]), [[0, 0, 1, 1, np.nan], [1, 0, 1, 1, 1]],
                               equal_nan=True)
    filled = zero_filling(assigned, min_callable_depth=3, verbose=False)
    np.testing.assert_array_equal(dense(filled.layers["zero_filled_state"]), [[0, 0, 0, 0, 0], [0, 0, 1, 1, 1]])
    np.testing.assert_array_equal(dense(filled.layers["callable"]), np.asarray(depth) >= 3)
    np.testing.assert_array_equal(dense(filled.X), raw)
    assert "state" not in filled.layers
    restored = zero_filling(assign_state(filled, method="continuous"), verbose=False)
    np.testing.assert_allclose(dense(restored.layers["zero_filled_state"]), raw)
    np.testing.assert_array_equal(dense(restored.X), raw)
    np.testing.assert_array_equal(dense(adata.X), raw)


def test_filter_uses_raw_ratios_after_state_assignment():
    adata = make_input([[0.2], [0.2], [0.2]], [[10], [10], [10]])
    filtered = filter_snvs(zero_filling(assign_state(adata), verbose=False), verbose=False)
    assert filtered.n_vars == 1
    np.testing.assert_array_equal(dense(filtered.X), [[0.2], [0.2], [0.2]])
    assert "zero_filled_state" not in filtered.layers
    germline = make_input([[1], [1], [1]], [[2], [2], [2]])
    filled = zero_filling(assign_state(germline), min_callable_depth=3, verbose=False)
    assert not np.any(filled.layers["zero_filled_state"])
    assert filter_snvs(filled, verbose=False).n_vars == 0


def reference_e(depth, ratio):
    callable_ = depth > 0
    depth, ratio = depth[callable_], ratio[callable_]
    all_alt = ratio == 1
    integral = quad(lambda value: 4 * value**3 * np.prod(1 + value * depth[all_alt])
                    * (1 - value)**sum(~all_alt), 0, 1, epsabs=1e-11)[0]
    endpoint = np.prod(1 + depth) if all_alt.all() else 0
    return 0.5 * (endpoint + integral)


@pytest.mark.parametrize("alt_depth", [4, 6, 7])
def test_e_branch_matches_independent_integration(alt_depth):
    depth = np.array([alt_depth, alt_depth, alt_depth, 10.])
    ratio = np.array([1., 1., 1., 0.])
    evidence = reference_e(depth, ratio)
    adata = make_input(ratio[:, None], depth[:, None])
    below = filter_snvs(adata, min_pooled_ratio_germline=1, min_E_germline=evidence - 1e-6, verbose=False)
    above = filter_snvs(adata, min_pooled_ratio_germline=1, min_E_germline=evidence + 1e-6, verbose=False)
    assert below.n_vars == 0
    assert above.n_vars == 1


def test_germline_or_callable_count_depth_and_no_alt_rounding():
    adata = make_input(
        [[0.9, 1, 1, 0.96, 1, 0.2], [0.9, 1, 1, 0.96, 1, 0.2],
         [0.9, 1, 0, 0.96, 1, 0.2], [0, 0, 0, 0, 0, 0.2]],
        [[20, 6, 10, 2, 1, 5], [20, 6, 10, 2, 1, 5],
         [20, 6, 0, 2, 1, 5], [0, 10, 0, 0, 0, 5]],
        ["pooled_only", "e_only", "two_cells", "rounded_alt", "low_depth", "somatic"],
    )
    default = filter_snvs(adata, verbose=False)
    assert list(default.var_names) == ["two_cells", "somatic"]
    no_filter = filter_snvs(adata, remove_germline=False, verbose=False)
    assert list(no_filter.var_names) == ["pooled_only", "e_only", "two_cells", "rounded_alt", "somatic"]
    strict_pool = filter_snvs(adata, min_pooled_ratio_germline=1, verbose=False)
    assert "rounded_alt" in strict_pool.var_names
    two_cells = filter_snvs(adata, min_callable_cells_germline=2, verbose=False)
    assert "two_cells" not in two_cells.var_names
    depth_one = filter_snvs(adata, min_total_depth=1, remove_germline=False, verbose=False)
    assert "low_depth" in depth_one.var_names


def test_tsv_exclusion_is_explicit_and_exact(tmp_path):
    name = "NC_080041.1:122054955:A:G"
    adata = make_input([[0.2, 0.2, 0.2]] * 3, [[5, 5, 5]] * 3,
                       [name, name + "_other", "somatic"])
    path = tmp_path / "exclude.tsv"
    path.write_bytes((name + "\r\nunknown_locus\r\n").encode())
    assert filter_snvs(adata, verbose=False).n_vars == 3
    filtered = filter_snvs(adata, exclude_snvs=path, verbose=False)
    assert list(filtered.var_names) == [name + "_other", "somatic"]


def test_custom_layers_and_inplace_state_lifecycle():
    raw = [[0.05, 0.5, 0], [0.2, 0, 0.4], [0.3, 0.1, 0]]
    adata = make_input(raw, [[20, 2, 0], [5, 0, 5], [10, 10, 0]])
    assigned = assign_state(adata, layer_name="assigned", copy=False)
    assert assigned is adata
    filled = zero_filling(assigned, min_callable_depth=3, callable_layer="covered",
                          output_layer="filled", copy=False, verbose=False)
    assert filled is adata
    assert "assigned" not in adata.layers
    assert "original_ratio" not in adata.layers
    np.testing.assert_array_equal(dense(adata.X), raw)
    np.testing.assert_array_equal(dense(adata.layers["filled"]), [[0, 0, 0], [1, 0, 1], [1, 1, 0]])
    assigned = assign_state(adata, state_low=0.4, copy=False)
    assert "filled" not in assigned.layers
    assert "covered" not in assigned.layers
    filled = zero_filling(assigned, copy=False, verbose=False)
    np.testing.assert_array_equal(dense(filled.layers["zero_filled_state"]),
                                  [[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    filtered = filter_snvs(filled, min_total_depth=10, copy=False, verbose=False)
    assert filtered is adata
    assert list(filtered.var_names) == ["snv0", "snv1"]
    assert "state" not in filtered.layers
    assert "zero_filled_state" not in filtered.layers
    assert "callable" not in filtered.layers
    np.testing.assert_array_equal(dense(filtered.X), np.asarray(raw)[:, :2])
