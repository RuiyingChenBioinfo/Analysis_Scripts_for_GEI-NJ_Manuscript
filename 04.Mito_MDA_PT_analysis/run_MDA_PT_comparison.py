"""Compare four MDA_PT trees using the author's filtered AFM."""
from pathlib import Path
import copy
import json
import subprocess
import sys

# Fixed paths and analysis settings.
HERE = Path(__file__).resolve().parent
AFM_FILE = HERE / "source_data/data/lineage_inference/UPMGA/MDA_PT/4efe3d4d1f/afm_filtered.h5ad"
OUTPUT_DIR = HERE / "output"
INPUT_DIR = OUTPUT_DIR / "inputs"
R_LIBRARY = Path("/home/chenr/Litianle/DENDRO_test/R_library")
RAPIDNJ = "/home/chenr/00.Tools/rapidNJ-master/bin/rapidnj"
RSCRIPT = "Rscript"
COLLAPSE_EPS = 1e-9
DISTANCE_DECIMALS = 9
TREE_FIGSIZE = (11, 16)
COMPARISON_FIGSIZE = (17, 4.6)
GROUPS = ["GEI_NJ_low_0p01", "GEI_NJ_continuous", "DENDRO_all_default",
          "DENDRO_no_SNV_prefilter"]
GROUP_LABELS = ["GEI-NJ\nbinary (0.01)", "GEI-NJ\ncontinuous", "DENDRO\ndefault filtering",
                "DENDRO\nno SNV prefilter"]
GROUP_COLORS = ["#d54000", "#E69F00", "#578c8c", "#5976e2"]
CLONE_COLORS = [
    "#456c99", "#ff9d26", "#00cc6a", "#f22693", "#a673ff",
    "#66662e", "#c10717", "#1fccca", "#ff9999", "#919900",
    "#994567", "#990091", "#076600", "#1f665d", "#26c2ff",
    "#99a2ff", "#ff73ea", "#684599", "#996600", "#84cc3d",
    "#cca27a", "#99322e", "#5c9965", "#ff764d", "#ff99cf",
    "#bb7acc", "#cc0048", "#ccb65c", "#3d53cc", "#3dcc9e",
    "#229917", "#ff9f73", "#819945", "#cc3d76", "#2e8799",
    "#0f6632", "#cc5c69", "#cc4e00", "#995c2e", "#cc3da3",
    "#659900", "#5c73cc", "#ab5ccc",
]
METRICS = [("dendrogram_purity", "Pair-weighted dendrogram purity"),
           ("macro_dendrogram_purity", "Clone-weighted dendrogram purity"),
           ("pairwise_AP", "Same-clone pair average precision")]

sys.setrecursionlimit(20000)
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Bio import Phylo
import gei_nj


def run_dendro():
    # Numeric arguments are omitted to preserve native DENDRO defaults.
    code = r'''
input_dir <- INPUT_VALUE
output_dir <- OUTPUT_VALUE
.libPaths(c(R_LIBRARY_VALUE, .libPaths()))
suppressPackageStartupMessages(library(DENDRO))
read_matrix <- function(name) {
    as.matrix(read.csv(file.path(input_dir, paste0(name, ".csv")),
                       row.names=1, check.names=FALSE, na.strings="NA"))
}
X <- read_matrix("ALT_counts")
N <- read_matrix("total_depth")
Z <- read_matrix("mutation_calls")
for (group in c("DENDRO_all_default", "DENDRO_no_SNV_prefilter")) {
    out <- file.path(output_dir, "results", group)
    if (group == "DENDRO_all_default") {
        qc <- DENDRO::FilterCellMutation(X, N, Z, plot=FALSE)
    } else {
        # Skip the SNV prefilter; retain native cell QC.
        total <- colSums(N)
        bounds <- c(pmax(0, round(mean(total)-5*sd(total))),
                    round(mean(total)+5*sd(total)))
        keep_cell <- total>bounds[1] & total<bounds[2]
        qc <- list(X=X[,keep_cell,drop=FALSE], N=N[,keep_cell,drop=FALSE],
                   Z=Z[,keep_cell,drop=FALSE])
    }
    d <- DENDRO::DENDRO.dist(qc$X, qc$N, qc$Z, show.progress=FALSE)
    hc <- DENDRO::DENDRO.cluster(d, plot=FALSE)
    tree <- ape::as.phylo(hc)
    ape::write.tree(tree, file=file.path(out, "tree.nwk"), digits=17)
}
'''
    code = code.replace("INPUT_VALUE", json.dumps(str(INPUT_DIR)))
    code = code.replace("OUTPUT_VALUE", json.dumps(str(OUTPUT_DIR)))
    code = code.replace("R_LIBRARY_VALUE", json.dumps(str(R_LIBRARY)))
    script = OUTPUT_DIR / "run_DENDRO.R"
    script.write_text(code, encoding="utf-8")
    subprocess.run([RSCRIPT, str(script)], check=True)


def collapsed(tree):
    # Remove zero-length internal edges for purity only.
    result = copy.deepcopy(tree)
    for node in result.find_clades(order="postorder"):
        children = []
        for child in node.clades:
            if child.clades and float(child.branch_length or 0) <= COLLAPSE_EPS:
                for grand in child.clades:
                    grand.branch_length = float(grand.branch_length or 0) + float(child.branch_length or 0)
                children.extend(child.clades)
            else:
                children.append(child)
        node.clades = children
    return result


def evaluate(tree, labels):
    names = [tip.name for tip in tree.get_terminals()]
    truth, clones = pd.factorize(labels.loc[names, "GBC"], sort=True)
    n, ng = len(names), len(clones)
    lookup = {name:i for i, name in enumerate(names)}
    sizes = np.bincount(truth, minlength=ng)
    pairs = sizes*(sizes-1)/2
    contribution = np.zeros(ng)
    counts = {}
    for node in collapsed(tree).find_clades(order="postorder"):
        if not node.clades:
            counts[node] = np.bincount([truth[lookup[node.name]]], minlength=ng)
        else:
            cc = np.asarray([counts[ch] for ch in node.clades])
            counts[node] = cc.sum(axis=0)
            at_lca = (counts[node]*(counts[node]-1)-(cc*(cc-1)).sum(axis=0))/2
            contribution += at_lca*counts[node]/counts[node].sum()
    perclone = np.divide(contribution, pairs, out=np.full(ng, np.nan), where=pairs>0)
    # AP uses original patristic distances and grouped numerical ties.
    levels = tree.depths()
    leaf_depths = np.zeros(n)
    for tip in tree.get_terminals():
        leaf_depths[lookup[tip.name]] = levels[tip]
    distances = np.zeros((n,n))
    descendants = {}
    for node in tree.find_clades(order="postorder"):
        if not node.clades:
            descendants[node] = np.array([lookup[node.name]])
        else:
            previous = descendants[node.clades[0]]
            for child in node.clades[1:]:
                current = descendants[child]
                block = leaf_depths[previous,None]+leaf_depths[current][None,:]-2*levels[node]
                distances[np.ix_(previous,current)] = block
                distances[np.ix_(current,previous)] = block.T
                previous = np.concatenate([previous,current])
            descendants[node] = previous
    distances = np.maximum(distances, 0)
    ii,jj = np.triu_indices(n,1)
    values = np.round(distances[ii,jj], DISTANCE_DECIMALS)
    order = np.argsort(values, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(values[order])!=0), len(order)-1]
    positive = truth[ii]==truth[jj]
    tp = np.cumsum(positive[order])[ends]
    precision = tp/(ends+1)
    recall = tp/positive.sum()
    ap = float(np.dot(np.diff(np.r_[0,recall]), precision))
    return dict(dendrogram_purity=float(contribution.sum()/pairs.sum()),
                macro_dendrogram_purity=float(np.nanmean(perclone)), pairwise_AP=ap)


def plot_metrics(summary):
    fig, axes = plt.subplots(1, 3, figsize=COMPARISON_FIGSIZE)
    for ax, (column, title) in zip(axes, METRICS):
        bars = ax.bar(GROUP_LABELS, summary.loc[GROUPS, column], color=GROUP_COLORS)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set_ylim(0, 1)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pdf/06.three_comparison_metrics.pdf")
    fig.savefig(OUTPUT_DIR / "png/06.three_comparison_metrics.png", dpi=180)
    plt.close(fig)


def main():
    for path in [INPUT_DIR, OUTPUT_DIR / "pdf", OUTPUT_DIR / "png",
                 *[OUTPUT_DIR / "results" / group for group in GROUPS]]:
        path.mkdir(parents=True, exist_ok=True)
    # Keep author-filtered counts and coverage unchanged.
    afm = ad.read_h5ad(AFM_FILE)
    x = pd.DataFrame(afm.layers["AD"].toarray().T, index=afm.var_names, columns=afm.obs_names)
    n = pd.DataFrame(afm.layers["site_coverage"].toarray().T, index=afm.var_names, columns=afm.obs_names)
    z = pd.DataFrame(afm.layers["bin"].toarray().T, index=afm.var_names, columns=afm.obs_names)
    x.to_csv(INPUT_DIR / "ALT_counts.csv")
    n.to_csv(INPUT_DIR / "total_depth.csv")
    z.to_csv(INPUT_DIR / "mutation_calls.csv")
    labels = afm.obs[["GBC"]].astype(str).copy()
    clone_names = {gbc:f"Clone{i+1:02d}" for i, gbc in enumerate(sorted(labels.GBC.unique()))}
    labels["clone_name"] = labels.GBC.map(clone_names)
    labels.to_csv(INPUT_DIR / "cell_GBC_labels.csv", index_label="cell_id")
    key = pd.DataFrame({"GBC":clone_names.keys(), "clone_name":clone_names.values(),
                        "color":CLONE_COLORS})
    key.to_csv(OUTPUT_DIR / "GBC_clone_key.csv", index=False)

    obs, var = pd.DataFrame(index=x.columns), pd.DataFrame(index=x.index)
    ratio = ad.AnnData((x/n).to_numpy().T, obs=obs.copy(), var=var.copy())
    depth = ad.AnnData(n.to_numpy().T, obs=obs.copy(), var=var.copy())
    adata = gei_nj.comb_depth_ratio(depth, ratio)
    adata = gei_nj.filter_snvs(adata)
    trees = {}
    for group, mode in zip(GROUPS[:2], ["binary", "continuous"]):
        states = gei_nj.assign_state(adata, method=mode)
        states = gei_nj.zero_filling(states)
        states.obs["clone_name"] = labels.loc[states.obs_names, "clone_name"]
        trees[group] = gei_nj.build_nj(
            states, do_plot=(mode == "binary"), rapidnj_binary=str(RAPIDNJ),
            group_key="clone_name", group_colors=dict(zip(key.clone_name, key.color)),
            leafcolor_onside=True, layout="rectangular", figsize=TREE_FIGSIZE,
            savepath=OUTPUT_DIR / "pdf/GEI_NJ_binary_rectangular.pdf",
        )
        Phylo.write(trees[group], OUTPUT_DIR / "results" / group / "tree.nwk",
                    "newick", format_branch_length="%1.17g")

    run_dendro()
    for group in GROUPS[2:]:
        trees[group] = Phylo.read(OUTPUT_DIR / "results" / group / "tree.nwk", "newick")
    summary = pd.DataFrame([evaluate(trees[group], labels) for group in GROUPS],
                           index=pd.Index(GROUPS, name="group"))
    summary.to_csv(OUTPUT_DIR / "comparison_metrics.csv")
    plot_metrics(summary)

    # Centered raw-AF PCA supplies display coordinates only.
    matrix = (x/n).to_numpy().T
    u, s, vt = np.linalg.svd(matrix-matrix.mean(axis=0), full_matrices=False)
    signs = np.sign(vt[np.arange(2), np.argmax(np.abs(vt[:2]), axis=1)])
    coords_path = OUTPUT_DIR / "SNV_PCA_coordinates.tsv"
    pd.DataFrame(u[:,:2]*s[:2]*signs, index=x.columns).to_csv(coords_path, sep="\t", header=False)
    gei_nj.view_tree_coords(
        trees["GEI_NJ_low_0p01"], coords_path=coords_path,
        savepath=OUTPUT_DIR / "GEI_NJ_binary_rectangular.html",
        x_label="SNV PC1", y_label="SNV PC2",
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
