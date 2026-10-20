# GEI-NJ

GEI-NJ provides an integrated Python workflow for cell relationship analysis from matched SNV ratio and sequencing-depth `AnnData` matrices.

Version **0.2.0.4** uses SNV filtering, two-state ALT representation, zero-filling, Manhattan L1 distances, RapidNJ tree construction, and weighted least squares (WLS) rooting. Continuous ALT ratios are also supported.

**Note:** GEI-NJ does not perform SNV calling. Use [GATK Haplotype Caller](https://gatk.broadinstitute.org/hc/en-us/articles/360037225632-HaplotypeCaller), [LineMut Call](https://github.com/RuiyingChenBioinfo/LineMut_v0.2.0) or other suitable workflows to generate the SNV ratio and depth inputs.

## Requirements and installation

Python 3.9+, with `anndata`, `biopython`, `matplotlib`, `numpy`, `pandas`, and `scipy`. Install the Python package from this source directory:

```bash
python -m pip install .
```

RapidNJ is a separate executable and must be on `PATH`, or supplied through `rapidnj_binary`. The benchmark and CI use [RapidNJ](https://github.com/somme89/rapidNJ) at the following pinned commit. On Linux or macOS with `git`, `make`, and a C++ compiler:

```bash
git clone https://github.com/somme89/rapidNJ.git
git -C rapidNJ checkout ed2d36e219d9db16778b941b5054c0fd021b528a
make -C rapidNJ -j2
export PATH="$PWD/rapidNJ/bin:$PATH"
```

## Input encoding

Rows represent cells or cell groups; columns represent SNVs. Depth and ratio objects must have matching, unique observation and SNV names. `comb_depth_ratio()` aligns their order using the ratio object.

- Depths must be finite, nonnegative integer counts.
- At positive depth, ratios must be finite values in `[0, 1]`, calculated as ALT support divided by total support.
- At zero depth, encode the ratio as `0` or `NaN`. Both represent an unobserved entry. Input NaNs produce a warning; NaNs at positive depth raise an error.
- Use original observed ratios, not genotype calls or coarsely rounded ratios. Germline evidence uses exact all-ALT observations (`ratio == 1`).

> [!CAUTION]
> **Invalid depth values, incompatible names, and invalid ratios raise errors. A missing entry does not silently delete an entire SNV. During state assignment, zero-depth entries are marked as NaN in the temporary `state` layer until zero-filling. `X` retains the original ratios.**

## Basic workflow

```python
import anndata as ad
from gei_nj import (
    comb_depth_ratio,
    filter_snvs,
    assign_state,
    zero_filling,
    build_nj,
)

ad_depth = ad.read_h5ad("unit_depth.h5ad")
ad_ratio = ad.read_h5ad("unit_ratio.h5ad")
adata = comb_depth_ratio(ad_depth, ad_ratio)

adata = filter_snvs(
    adata,
    min_total_depth=5,
    remove_germline=True,
    min_callable_cells_germline=3,
    min_pooled_ratio_germline=0.90,
    min_E_germline=10,
)
adata = assign_state(adata, method="two-state", state_low=0.10)
adata = zero_filling(adata, min_callable_depth=1)
tree = build_nj(
    adata,
    wls_weight_power=2,
    do_plot=True,
    use_branch_lengths=False,
    savepath="geinj_tree.pdf",
    newick_path="geinj_tree.nwk",
)
```

A complete example is provided in the [tutorial](./assets/Tutorial_GEI_NJ.html). Function signatures and parameter descriptions are provided in the [API reference](./assets/API.html).

### SNV filtering

`filter_snvs()` first excludes SNVs whose depth summed across cells is below `min_total_depth`. With `remove_germline=True`, it also excludes candidate homozygous ALT germline variants satisfying:

```python
remove = (n_callable >= min_callable_cells_germline) & (
    (pooled_ratio >= min_pooled_ratio_germline) | (E >= min_E_germline)
)
```

The three parameters ending in `_germline` apply only when `remove_germline=True`. Here, `n_callable` counts cells with positive depth. `pooled_ratio` is the depth-weighted raw ALT ratio. The e-value combines depth and all-ALT observations across cells; it is an evidence statistic, not a germline probability. This rule identifies candidates with strong all-ALT signals and does not establish their biological origin. It is not intended to detect all heterozygous germline variants. The later `min_callable_depth` setting does not change this filtering rule.

To exclude user-defined SNVs, pass a single-column, headerless TSV with identifiers matching `adata.var_names`:

```python
adata = filter_snvs(
    adata,
    exclude_snvs="examples/organelle_SNV_candidates.tsv",
)
```

The supplied [TSV example](./examples/organelle_SNV_candidates.tsv) contains six user-specified loci. It is applied only when explicitly provided and should be replaced with a list appropriate for the dataset. GEI-NJ does not identify or correct RNA editing; known editing sites or other unwanted loci can be excluded this way. Set `remove_germline=False` to disable candidate germline filtering while retaining depth and user-list filtering.

### States and zero-filling

`method="two-state"` assigns observed ratios below `state_low=0.10` to 0 and ratios at or above it to 1. `method="continuous"` assigns the raw ALT ratios as continuous states. Both methods use zero-filling at entries with depth below `min_callable_depth`; at the default of 1, only zero-depth entries are filled. Other entries retain their selected state. Zero-filling does not use neighboring cells.

For the continuous alternative, use the same filtered input and rerun state assignment and zero-filling:

```python
adata_cont = assign_state(adata, method="continuous")
adata_cont = zero_filling(adata_cont, min_callable_depth=1)
tree_cont = build_nj(adata_cont, do_plot=False, newick_path="continuous_tree.nwk")
```

`assign_state()` reads the original ratios in `adata.X` and writes a temporary `state` layer. `zero_filling()` writes `zero_filled_state` and `callable`, then removes the temporary state layer. Neither step overwrites `X`. To change the state representation or `min_callable_depth`, rerun `assign_state()` and then `zero_filling()` from the preserved original ratios. Reassigning states clears the previous filled matrix and callability mask; filtering again also clears processed states. Preprocessing functions return a copy by default; `copy=False` updates the object in place.

### Tree construction and plotting

`build_nj()` reads `adata.layers["zero_filled_state"]`, calculates ordinary summed L1 distances between real cells, and calls RapidNJ. It then places a synthetic all-reference profile on this fixed tree by constrained WLS, using weights `(1 + distance_to_reference) ** -wls_weight_power`. The default power is 2; power 0 gives equal weights. The fitted attachment point defines the root, and the returned tree contains only real cells. Negative NJ branch lengths are set to zero before WLS fitting, without collapsing their splits.

The default `use_branch_lengths=False` draws a topology view with aligned tips. Set it to `True` to draw according to the fitted branch lengths. This option changes only the display; the topology, root, and saved Newick branch lengths are unchanged. Branch lengths are derived from the selected SNV representation and distance model; they are not calibrated mutation counts or elapsed time.

Use `group_key` to color cells by an `adata.obs` column. `do_plot=False` skips plotting, while still constructing the tree and writing `newick_path` if provided. `return_tree_info=True` returns `(tree, info)` instead of the Biopython `Tree` alone.

### Processed AnnData structure

| Location | Content |
| --- | --- |
| `adata.X` | Original observed ratios at retained sites, preserving zero-depth 0 or NaN encoding |
| `adata.layers["total_depth"]` | Total sequencing depth for each cell–SNV entry |
| `adata.layers["state"]` | Assigned states; temporary layer removed after zero-filling |
| `adata.layers["callable"]` | Binary mask defined by `min_callable_depth` |
| `adata.layers["zero_filled_state"]` | Zero-filled state matrix used for tree construction |

When custom `layer_name`, `callable_layer`, or `output_layer` names are supplied, the workflow records them in `adata.uns` for the following steps. `build_nj()` uses the recorded filled output layer and does not fall back to `X`.

## Tests and version changes

```bash
python -m pip install -e ".[test]"
python -m pytest
```

RapidNJ must be available for the tree-construction integration tests. GitHub Actions installs the pinned RapidNJ version and runs tests on Python 3.9 and 3.12.

This version replaces the former PCA workflow. Use `filter_snvs()`, `assign_state()`, `zero_filling()`, and `build_nj()` in place of the previous filtering, state assignment, mean-imputation, and PCA tree-building functions. See the [changelog](./CHANGELOG.md) and [release instructions](./RELEASE.md).

## Citations

Software citation metadata is provided in [CITATION.cff](./CITATION.cff). A version-specific DOI should be added after the exact release is archived; no DOI has been assigned in this source package.

For RapidNJ, cite: Simonsen M, Mailund T, Pedersen CNS. Rapid neighbour-joining. Algorithms in Bioinformatics. 2008. p. 113–122. [doi:10.1007/978-3-540-87361-7_10](https://doi.org/10.1007/978-3-540-87361-7_10).

## Contact

* Ruiying Chen (陈睿颖), <chenruiying@genomics.cn>
* Chao Qin (秦超), <qinchao@genomics.cn>
* Hai-Xi Sun (孙海汐), <sunhaixi@genomics.cn>
