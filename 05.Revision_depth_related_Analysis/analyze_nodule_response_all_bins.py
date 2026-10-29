from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import rdata
from Bio import Phylo
from scipy import sparse
from scipy.linalg import eigh, helmert
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, false_discovery_control
from threadpoolctl import threadpool_limits
from gei_nj import comb_depth_ratio, filter_snvs


# %% 1. Set paths and parameters.
DATA_DIR = Path('data')
OUTPUT_DIR = Path('Nodule_response_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_RANDOMIZATIONS = 999
SEED = 42
N_NEIGHBORS = 6


# %% 2. Match bins and retain SNVs with default settings.
depth = ad.read_h5ad(DATA_DIR / 'sc_depth.h5ad')
ratio = ad.read_h5ad(DATA_DIR / 'sc_ratio.h5ad')
adata = filter_snvs(comb_depth_ratio(depth, ratio))
barcodes = adata.obs_names.tolist()
N = adata.layers['total_depth'].toarray().astype(float)
n, n_snvs = N.shape
callable_entries = N >= 1

annotation = pd.read_csv(
    DATA_DIR / 'cell_type.tsv', sep='\t', header=None,
    names=['barcode', 'cell_type']).set_index('barcode').loc[barcodes]
annotation['cell_type'] = annotation.cell_type.replace({'mixed': 'Mixed'})
coordinates = pd.read_csv(
    DATA_DIR / 'cell_coordinates.tsv', sep='\t', header=None,
    names=['barcode', 'x', 'y']).set_index('barcode').loc[barcodes]


# %% 3. Read the saved binary tree and summarize coverage.
tree = Phylo.read(DATA_DIR / 'GEI-NJ_binary.nwk', 'newick')
tips = {tip.name: tip for tip in tree.get_terminals()}
tree_depths = tree.depths()
annotation['root_to_tip'] = [tree_depths[tips[name]] for name in barcodes]
annotation['mean_snv_depth'] = N.mean(axis=1)  # Include zero-depth entries.
annotation['callable_fraction'] = callable_entries.mean(axis=1)
annotation[['x', 'y']] = coordinates[['x', 'y']]
annotation.to_csv(OUTPUT_DIR / 'bin_metrics.csv')
pd.Series(adata.var_names).to_csv(
    OUTPUT_DIR / 'retained_snvs.tsv', sep='\t', index=False, header=False)

cell_types = ['Infected_Zone', 'Early_nodule_cortex', 'Vascular_tissue',
              'Mixed', 'Meristems']
summary = []
for cell_type in cell_types:
    selected = annotation.cell_type.to_numpy() == cell_type
    values = N[selected]
    summary.append({'cell_type': cell_type, 'n_bins': int(selected.sum()),
                    'median_umi_depth_callable': np.median(values[values >= 1]),
                    'callable_fraction': (values >= 1).mean()})
summary = pd.DataFrame(summary)
summary.to_csv(OUTPUT_DIR / 'callability_by_cell_type.csv', index=False)

correlations = pd.DataFrame([
    {'n_bins': n, 'variable': key,
     'spearman_rho': spearmanr(annotation[key], annotation.root_to_tip).statistic}
    for key in ['mean_snv_depth', 'callable_fraction']])
correlations.to_csv(OUTPUT_DIR / 'root_to_tip_correlations.csv', index=False)


# %% 4. Read the existing log-normalized HVG expression.
def r_attributes(obj):
    result = {}
    node = obj.attributes
    while node is not None and node.value is not None:
        result[rdata.conversion.convert(node.tag)] = node.value[0]
        node = node.value[1]
    return result


seurat = r_attributes(rdata.parser.parse_file(DATA_DIR / 'Nodule_SeuratObject.rds').object)
assay = r_attributes(seurat['assays'].value[0])  # RNA assay.
slot = r_attributes(assay['data'])
expression_sparse = sparse.csc_matrix(
    (rdata.conversion.convert(slot['x']), rdata.conversion.convert(slot['i']),
     rdata.conversion.convert(slot['p'])),
    shape=tuple(rdata.conversion.convert(slot['Dim'])))
genes, expression_barcodes = rdata.conversion.convert(slot['Dimnames'])
hvgs = rdata.conversion.convert(assay['var.features'])
gene_index = pd.Index(genes).get_indexer(hvgs)
bin_index = pd.Index(expression_barcodes).get_indexer(barcodes)
expression = expression_sparse[gene_index][:, bin_index].T.toarray()


# %% 5. Calculate pairwise distances using actual branch lengths.
upper = np.triu_indices(n, 1)
tree_distance = np.array([
    tree.distance(tips[barcodes[i]], tips[barcodes[j]]) for i, j in zip(*upper)])
depth_values = np.log1p(annotation.mean_snv_depth.to_numpy())[:, None]
distances = {'SNV depth': pdist(depth_values),
             'Gene expression': pdist(expression),
             'Spatial position': pdist(coordinates[['x', 'y']].to_numpy())}
tree_centered = tree_distance - tree_distance.mean()
tree_unit = tree_centered / np.linalg.norm(tree_centered)


def distance_correlation(values):
    centered = values - values.mean()
    return float(centered @ tree_unit / np.linalg.norm(centered))


# %% 6. Construct the spatial weights and Moran eigenvectors.
spatial_distance = squareform(distances['Spatial position'])
neighbor_radius = np.sort(spatial_distance, axis=1)[:, N_NEIGHBORS]
W = spatial_distance <= neighbor_radius[:, None]  # Include tied neighbors.
np.fill_diagonal(W, False)
W = (W | W.T).astype(float)
H = helmert(n, full=False).T
eigenvalues, vectors = eigh(H.T @ W @ H)
MEM = H @ vectors[:, ::-1]


# %% 7. Randomize spatial labels and Moran coefficients.
rng = np.random.default_rng(SEED)
permutations = np.array([rng.permutation(n) for _ in range(N_RANDOMIZATIONS)])
rng = np.random.default_rng(SEED)  # Keep separate reproducible streams.
signs = rng.choice([-1., 1.], size=(N_RANDOMIZATIONS, n - 1))
observed = {name: distance_correlation(values) for name, values in distances.items()}
nulls = {}

with threadpool_limits(limits=1):
    nulls['Spatial position_ordinary'] = np.array([
        distance_correlation(spatial_distance[perm[upper[0]], perm[upper[1]]])
        for perm in permutations])

    for name, values in [('SNV depth', depth_values), ('Gene expression', expression)]:
        centered = values - values.mean(axis=0)
        coefficients = MEM.T @ centered
        nulls[name + '_MSR'] = np.array([
            distance_correlation(pdist(MEM @ (sign[:, None] * coefficients)))
            for sign in signs])  # Share signs across all genes.


# %% 8. Calculate equal-tail two-sided P values and BH q values.
def two_sided_p(observed, simulated):
    lower = (1 + np.count_nonzero(simulated <= observed)) / (len(simulated) + 1)
    upper = (1 + np.count_nonzero(simulated >= observed)) / (len(simulated) + 1)
    return min(1., 2 * min(lower, upper))


rows = []
for name in distances:
    primary = 'ordinary' if name == 'Spatial position' else 'MSR'
    simulated = nulls[name + '_' + primary]
    rows.append({'comparison': name, 'mantel_r': observed[name], 'primary_null': primary,
                 'p_two_sided': two_sided_p(observed[name], simulated),
                 'null_mean_r': simulated.mean(),
                 'null_q025_r': np.quantile(simulated, .025),
                 'null_q975_r': np.quantile(simulated, .975),
                 'n_randomizations': N_RANDOMIZATIONS, 'seed': SEED})
results = pd.DataFrame(rows)
results['bh_q'] = false_discovery_control(results.p_two_sided.to_numpy(), method='bh')
results.to_csv(OUTPUT_DIR / 'distance_associations.csv', index=False)
pd.DataFrame(nulls).to_csv(OUTPUT_DIR / 'randomization_statistics.csv', index=False)

print(f'{n} bins | {n_snvs} retained SNVs | {len(hvgs)} HVGs')
print(summary.to_string(index=False))
print(correlations.to_string(index=False))
print(results.to_string(index=False))
