import os
import re
import shutil
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from Bio import Phylo
from scipy.spatial.distance import pdist, squareform

import gei_nj
from gei_nj import build_nj, assign_state, zero_filling
from gei_nj.construct_nj import _build_nj_tree, _tree_coordinates


@pytest.fixture
def rapidnj():
    binary = os.environ.get("RAPIDNJ_BINARY") or shutil.which("rapidnj")
    if binary is None:
        pytest.skip("Set RAPIDNJ_BINARY or install RapidNJ for integration tests.")
    return binary


@pytest.fixture
def lineage():
    names = [f"{i:03b}" for i in range(8)]
    prefixes = sorted({name[:length] for name in names for length in (1, 2, 3)})
    prefixes += [name for i, name in enumerate(names) for _ in range(i)]
    states = np.array([[float(name.startswith(prefix)) for prefix in prefixes] for name in names])
    adata = AnnData(states * 0.5, obs=pd.DataFrame(index=names))
    adata.layers["total_depth"] = np.full(states.shape, 10.)
    return zero_filling(assign_state(adata), verbose=False)


def splits(tree):
    full = frozenset(tip.name for tip in tree.get_terminals())
    result = set()
    for clade in tree.find_clades():
        side = frozenset(tip.name for tip in clade.get_terminals())
        other = full - side
        if min(len(side), len(other)) >= 2:
            result.add(frozenset([side, other]))
    return result


def rooted_lengths(tree):
    return {frozenset(tip.name for tip in clade.get_terminals()): clade.branch_length
            for clade in tree.find_clades()}


@pytest.mark.parametrize("power", [0, 1, 2, 3])
def test_wls_recovers_additive_root_without_changing_backbone(lineage, rapidnj, power):
    names = lineage.obs_names.tolist()
    distances = squareform(pdist(lineage.layers["zero_filled_state"], metric="cityblock"))
    backbone = _build_nj_tree(distances, names, rapidnj)
    tree = build_nj(lineage, do_plot=False, wls_weight_power=power, rapidnj_binary=rapidnj)
    assert tree.rooted
    assert sorted(tip.name for tip in tree.get_terminals()) == names
    assert splits(tree) == splits(backbone)
    assert {frozenset(tip.name for tip in clade.get_terminals()) for clade in tree.root.clades} == {
        frozenset(names[:4]), frozenset(names[4:])}
    actual = [[tree.distance(first, second) for second in names] for first in names]
    np.testing.assert_allclose(actual, distances, atol=1e-6)


def test_plotting_and_newick_preserve_root_and_lengths(lineage, rapidnj, tmp_path):
    baseline = build_nj(lineage, do_plot=False, rapidnj_binary=rapidnj)
    x_topology, _ = _tree_coordinates(baseline)
    assert len({x_topology[tip] for tip in baseline.get_terminals()}) == 1
    x_lengths, _ = _tree_coordinates(baseline, use_branch_lengths=True)
    assert len({x_lengths[tip] for tip in baseline.get_terminals()}) > 1
    lineage.obs["group"] = ["A"] * 4 + ["B"] * 4
    for use_lengths in [False, True]:
        image_path = tmp_path / f"tree_{use_lengths}.png"
        newick_path = tmp_path / f"tree_{use_lengths}.nwk"
        tree, info = build_nj(
            lineage, do_plot=True, rapidnj_binary=rapidnj, use_branch_lengths=use_lengths,
            group_key="group", show_leaf_labels=True, group_colors={"A": "red", "B": "blue"},
            savepath=image_path, newick_path=newick_path, return_tree_info=True,
        )
        assert rooted_lengths(tree) == rooted_lengths(baseline)
        assert image_path.stat().st_size > 0
        assert isinstance(info, dict)
        restored = Phylo.read(newick_path, "newick")
        assert splits(restored) == splits(baseline)
        for first in lineage.obs_names:
            for second in lineage.obs_names:
                assert restored.distance(first, second) == pytest.approx(baseline.distance(first, second), abs=1e-5)


def test_all_zero_profiles_fail_before_tree_construction(rapidnj):
    adata = AnnData(np.zeros((4, 2)))
    adata.layers["zero_filled_state"] = adata.X.copy()
    with pytest.raises(ValueError, match="[Zz]ero|informative|identical"):
        build_nj(adata, do_plot=False, rapidnj_binary=rapidnj)


def test_release_version_alignment():
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    declared = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE).group(1)
    assert declared == gei_nj.__version__ == version("gei-nj") == "0.2.0.4"


def test_build_uses_filled_layer_instead_of_x(lineage, rapidnj):
    baseline = build_nj(lineage, do_plot=False, rapidnj_binary=rapidnj)
    lineage.X = np.zeros(lineage.shape)
    tree = build_nj(lineage, do_plot=False, rapidnj_binary=rapidnj)
    assert rooted_lengths(tree) == rooted_lengths(baseline)
    del lineage.layers["zero_filled_state"]
    with pytest.raises((KeyError, ValueError), match="zero_filling|zero_filled_state"):
        build_nj(lineage, do_plot=False, rapidnj_binary=rapidnj)


def test_build_uses_custom_zero_filled_layer(lineage, rapidnj):
    baseline = build_nj(lineage, do_plot=False, rapidnj_binary=rapidnj)
    assigned = assign_state(lineage, layer_name="assigned")
    filled = zero_filling(assigned, output_layer="filled", verbose=False)
    tree = build_nj(filled, do_plot=False, rapidnj_binary=rapidnj)
    assert "assigned" not in filled.layers
    assert "zero_filled_state" not in filled.layers
    assert rooted_lengths(tree) == rooted_lengths(baseline)
