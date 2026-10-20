def _build_nj_tree(distances, names, binary):
    """Build the real-cell backbone with RapidNJ."""
    from pathlib import Path
    from tempfile import TemporaryDirectory
    import subprocess
    from Bio import Phylo
    from Bio.Phylo.NewickIO import NewickError

    aliases = [f"T{i:06d}" for i in range(len(names))]
    original_names = dict(zip(aliases, names))
    with TemporaryDirectory(prefix="rapidnj_") as directory:
        input_path = Path(directory) / "distances.phy"
        output_path = Path(directory) / "tree.nwk"
        with input_path.open("w") as handle:
            handle.write(f"{len(aliases)}\n")
            for alias, row in zip(aliases, distances):
                handle.write(f"{alias:<10}" + " ".join(format(value, ".17g") for value in row) + "\n")
        command = [str(binary), str(input_path), "-i", "pd", "-o", "t", "-x", str(output_path)]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"RapidNJ exit code {completed.returncode}: {detail[:2000]}")
        try:
            tree = Phylo.read(output_path, "newick")
        except (OSError, ValueError, NewickError) as error:
            raise RuntimeError(f"Cannot read the RapidNJ tree: {error}") from error
    tips = tree.get_terminals()
    returned = [tip.name for tip in tips]
    if len(returned) != len(aliases) or set(returned) != set(aliases):
        raise RuntimeError("RapidNJ returned missing, duplicate or unexpected cell names.")
    for tip in tips:
        tip.name = original_names[tip.name]
    return tree


def _root_zero_reference(tree, names, reference_distances, weight_power=2.):
    """Place a zero reference by constrained WLS on every backbone edge."""
    import numpy as np
    from Bio.Phylo.BaseTree import Clade, Tree

    nodes = list(tree.find_clades(order='preorder'))
    node_index = {node: i for i, node in enumerate(nodes)}
    tip_index = {name: i for i, name in enumerate(names)}
    children = [[] for node in nodes]
    adjacency = [[] for node in nodes]
    edges = []
    levels = np.zeros(len(nodes))
    n_negative = 0
    for parent, node in enumerate(nodes):
        for child in node.clades:
            index = node_index[child]
            length = float(child.branch_length or 0.)
            n_negative += int(length < 0.)
            length = max(length, 0.)
            children[parent].append(index)
            adjacency[parent].append((index, length))
            adjacency[index].append((parent, length))
            edges.append((parent, index, length))
            levels[index] = levels[parent] + length
    descendants = np.zeros((len(nodes), len(names)), dtype=bool)
    for index in range(len(nodes) - 1, -1, -1):
        if children[index]:
            descendants[index] = descendants[children[index]].any(axis=0)
        else:
            descendants[index, tip_index[nodes[index].name]] = True
    distances = np.empty((len(nodes), len(names)))
    for index, node in enumerate(nodes):
        if not children[index]:
            distances[0, tip_index[node.name]] = levels[index]
    for parent, child, length in edges:
        distances[child] = distances[parent] + length
        distances[child, descendants[child]] -= 2. * length
    reference_distances = np.asarray(reference_distances, dtype=float)
    weights = (1. + reference_distances) ** -weight_power
    total_weight = weights.sum()
    best = None
    for parent, child, length in edges:
        inside = descendants[child]
        sign = np.where(inside, -1., 1.)
        residual = reference_distances - distances[parent]
        outside_mean = np.dot(weights[~inside], residual[~inside]) / weights[~inside].sum()
        inside_mean = np.dot(weights[inside], residual[inside]) / weights[inside].sum()
        position = (outside_mean - inside_mean) / 2.
        pendant = (outside_mean + inside_mean) / 2.
        candidates = []
        if 0. <= position <= length and pendant >= 0.:
            candidates.append((position, pendant))
        candidates.extend([
            (0., max(np.dot(weights, residual) / total_weight, 0.)),
            (length, max(np.dot(weights, residual - sign * length) / total_weight, 0.)),
            (float(np.clip(np.dot(weights * sign, residual) / total_weight, 0., length)), 0.),
        ])
        for position, pendant in candidates:
            difference = residual - pendant - sign * position
            loss = float(np.dot(weights, difference * difference))
            if best is None or loss < best[0]:
                best = loss, parent, child, length, position, pendant
    loss, parent, child, length, position, pendant = best

    def orient(index, previous, branch_length):
        node = nodes[index]
        return Clade(branch_length=branch_length, name=node.name,
            confidence=node.confidence,
            clades=[orient(neighbor, index, distance)
                for neighbor, distance in adjacency[index] if neighbor != previous])

    root = Clade(branch_length=0., clades=[
        orient(parent, child, position), orient(child, parent, length - position)])
    rooted = Tree(root=root, rooted=True)
    diagnostics = dict(wls_loss=loss, wls_pendant_length=pendant,
        wls_edge_length=length, wls_edge_position=position,
        wls_root_at_endpoint=position == 0. or position == length,
        wls_negative_branches_projected=n_negative)
    return rooted, diagnostics


def _tree_coordinates(tree, use_branch_lengths=False):
    """Return display coordinates without changing the tree."""
    tips = tree.get_terminals()
    y = {tip: float(i) for i, tip in enumerate(tips)}
    heights = {}
    for node in tree.find_clades(order="postorder"):
        if node.clades:
            y[node] = sum(y[child] for child in node.clades) / len(node.clades)
            heights[node] = 1 + max(heights[child] for child in node.clades)
        else:
            heights[node] = 0
    if use_branch_lengths:
        x = tree.depths()
    else:
        x = {node: heights[tree.root] - height for node, height in heights.items()}
    return x, y


def _plot_tree(tree, adata, use_branch_lengths, group_key, show_leaf_labels,
               show_legend, group_colors, figsize, title_prefix, savepath,
               leafcolor_onside):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from pathlib import Path

    x, y = _tree_coordinates(tree, use_branch_lengths)
    tips = tree.get_terminals()
    color_map, groups = {}, {}
    if group_key is not None:
        values = adata.obs[group_key]
        categories = list(values.drop_duplicates())
        palette = plt.get_cmap("tab10" if len(categories) <= 10 else "tab20")
        color_map = {group: palette(i % palette.N) for i, group in enumerate(categories)}
        if isinstance(group_colors, dict):
            color_map.update(group_colors)
        elif group_colors is not None:
            if len(group_colors) != len(categories):
                raise ValueError("group_colors must match the number of groups.")
            color_map = dict(zip(categories, group_colors))
        groups = values.to_dict()
    fig, ax = plt.subplots(figsize=figsize)
    for node in tree.find_clades():
        if node.clades:
            ax.vlines(x[node], min(y[c] for c in node.clades),
                      max(y[c] for c in node.clades), color="black", linewidth=0.8)
            for child in node.clades:
                color = color_map.get(groups.get(child.name), "black") if child.is_terminal() else "black"
                ax.hlines(y[child], x[node], x[child], color=color, linewidth=0.8)
    span = max(x.values()) or 1.
    for tip in tips:
        color = color_map.get(groups.get(tip.name), "black")
        if leafcolor_onside and group_key is not None:
            ax.plot(span * 1.02, y[tip], marker="s", color=color, markersize=4)
        if show_leaf_labels:
            label_x = span * 1.04 if leafcolor_onside else x[tip] + span * 0.02
            ax.text(label_x, y[tip], tip.name, va="center", fontsize=8, color=color)
    if group_key is not None and show_legend:
        handles = [Line2D([0], [0], color=color, label=str(group), linewidth=2)
                   for group, color in color_map.items()]
        ax.legend(handles=handles, title=group_key, loc="upper left",
                  bbox_to_anchor=(1.02, 1.), frameon=False)
    ax.set_ylim(len(tips) - 0.5, -0.5)
    ax.set_xlim(-span * 0.02, span * 1.06)
    ax.set_yticks([])
    ax.set_title(title_prefix)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    if use_branch_lengths:
        ax.set_xlabel("SNV profile distance (L1)")
    else:
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    fig.tight_layout()
    if savepath is not None:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def build_nj(
    adata,
    do_plot=True,
    wls_weight_power=2.,
    use_branch_lengths=False,
    rapidnj_binary="rapidnj",
    group_key=None,
    show_leaf_labels=False,
    show_legend=True,
    group_colors=None,
    figsize=(10, 7),
    title_prefix="GEI-NJ",
    savepath=None,
    newick_path=None,
    print_tree_info=False,
    return_tree_info=False,
    leafcolor_onside=False,
):
    """Build a summed-L1 RapidNJ tree and root it with a zero reference.

    Parameters
    ----------
    adata : AnnData
        Cells by SNVs, with raw ratios in X and finite processed values in
        layers['zero_filled_state'], or the output_layer set by zero_filling.
    do_plot : bool
        Display the rooted tree and save it if savepath is supplied.
    wls_weight_power : float
        Use weights (1 + distance_to_zero) ** (-wls_weight_power).
    use_branch_lengths : bool
        False aligns tips in a topology view. True uses inferred lengths.
        Both options retain the same returned tree and Newick lengths.
    rapidnj_binary : str
        Installed RapidNJ command or executable path.
    group_key : str or None
        Optional obs column for categorical cell colors.
    show_leaf_labels, show_legend, leafcolor_onside : bool
        Show cell names, group legend, or aligned group markers.
    group_colors : dict, sequence or None
        Group colors; sequences follow first appearance in obs.
    figsize : tuple
        Figure dimensions in inches.
    title_prefix : str
        Figure title.
    savepath : str or None
        Figure output, used only when do_plot=True.
    newick_path : str or None
        Save the rooted tree with its inferred branch lengths.
    print_tree_info, return_tree_info : bool
        Print diagnostics or return (tree, diagnostics).

    Returns
    -------
    Bio.Phylo.BaseTree.Tree or tuple
        Rooted real-cell tree; no synthetic tip is retained.

    Notes
    -----
    WLS searches every fixed backbone edge with nonnegative attachment
    position and reference pendant length. Negative NJ branch lengths are
    projected to zero before WLS, matching the benchmark. Branch lengths
    describe SNV-profile distances, not developmental time or mutation counts.
    """
    import io
    import shutil
    from pathlib import Path
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    from Bio import Phylo
    from .process_data import _dense

    state_layer = adata.uns.get("geinj_zero_filling", {}).get("output_layer", "zero_filled_state")
    if state_layer not in adata.layers:
        raise KeyError(f"Missing '{state_layer}' in adata.layers; run zero_filling before build_nj.")
    features = _dense(adata.layers[state_layer])
    names = adata.obs_names.tolist()
    if adata.n_obs < 3 or adata.n_vars == 0:
        raise ValueError("build_nj requires at least three cells and one retained SNV.")
    if not adata.obs_names.is_unique or any(not name for name in names):
        raise ValueError("Cell names must be nonempty and unique.")
    if not np.isfinite(features).all() or ((features < 0.) | (features > 1.)).any():
        raise ValueError(f"'{state_layer}' must contain finite values in [0, 1]; run zero_filling before build_nj.")
    if not np.isfinite(wls_weight_power) or wls_weight_power < 0.:
        raise ValueError("wls_weight_power must be finite and nonnegative.")
    if do_plot and group_key is not None and group_key not in adata.obs:
        raise KeyError(f"group_key='{group_key}' not found in adata.obs.")
    condensed = pdist(features, metric="cityblock")
    if not condensed.any():
        raise ValueError("All cells have identical profiles; no informative NJ tree can be constructed.")
    binary = shutil.which(str(rapidnj_binary))
    if binary is None:
        raise FileNotFoundError("RapidNJ was not found; install RapidNJ or set rapidnj_binary to its executable path.")
    tree = _build_nj_tree(squareform(condensed), names, binary)
    tree, diagnostics = _root_zero_reference(tree, names, features.sum(axis=1), wls_weight_power)
    info = dict(n_cells=adata.n_obs, n_snvs=adata.n_vars, distance="L1",
                wls_weight_power=wls_weight_power, **diagnostics)
    if newick_path is not None:
        Path(newick_path).parent.mkdir(parents=True, exist_ok=True)
        Phylo.write(tree, str(newick_path), "newick", format_branch_length="%1.12g")
    if print_tree_info or return_tree_info:
        buffer = io.StringIO()
        Phylo.write(tree, buffer, "newick", format_branch_length="%1.12g")
        info["newick"] = buffer.getvalue().strip()
    if print_tree_info:
        print(info)
    if do_plot:
        _plot_tree(tree, adata, use_branch_lengths, group_key, show_leaf_labels,
                   show_legend, group_colors, figsize, title_prefix, savepath,
                   leafcolor_onside)
    return (tree, info) if return_tree_info else tree
