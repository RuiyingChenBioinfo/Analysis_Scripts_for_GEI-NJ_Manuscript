"""Benchmark binary/WLS NJ, module ablations and parameter sensitivity."""
"""--time_mem   1 WORKERS   Runtime + Peak memory for 2 main methods"""
"""--benchmark_only   30 WORKERS   Benchmark + Ablation"""
"""--sensitivity_only   30 WORKERS   Sensitivity"""
"""--organized   30 WORKERS   Benchmark + grouped ablation/sensitivity; no time/memory"""
"""No args   30 WORKERS    Benchmark + Ablation + Sensitivity"""

import os

# Apply before importing numerical libraries or spawning children.
THREAD_ENV = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS']
for name in THREAD_ENV:
    os.environ[name] = '1'
os.environ['OMP_DYNAMIC'] = 'FALSE'
os.environ['MKL_DYNAMIC'] = 'FALSE'

from multiprocessing import get_context
from threading import Event, Thread
from pathlib import Path
from time import perf_counter
from datetime import datetime
import hashlib
import importlib.metadata
import json
from math import comb
import platform
import shutil
import subprocess
import sys
from textwrap import fill
from tempfile import TemporaryDirectory

import anndata as ad
import h5py
import gei_nj
from gei_nj.construct_nj import _build_nj_tree, _root_zero_reference
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree
from Bio.Phylo.NewickIO import NewickError
import numpy as np
import pandas as pd
import psutil
from scipy.spatial.distance import pdist, squareform
from threadpoolctl import threadpool_limits

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import false_discovery_control, t, wilcoxon

# Edit settings here.
SENSITIVITY_ONLY = '--sensitivity_only' in sys.argv[1:]
BENCHMARK_ONLY = '--benchmark_only' in sys.argv[1:]
TIME_MEM = '--time_mem' in sys.argv[1:]
ORGANIZED = '--organized' in sys.argv[1:]
if TIME_MEM:
    try:
        from anndata.io import read_elem
    except ImportError:
        from anndata.experimental import read_elem
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR.parent / '01.Simulation_data_generation' / 'Simulation_results' / '02.Simulated_datasets'
OUTPUT_DIR = BASE_DIR / ('Organized_results' if ORGANIZED else 'Runtime_memory_results' if TIME_MEM else
    'Sensitivity_results' if SENSITIVITY_ONLY else 'Benchmark_results')
RESULTS = OUTPUT_DIR / 'benchmark_results.csv'
FIGURE_DIR = OUTPUT_DIR / 'figures'
RESOURCE_RESULTS = OUTPUT_DIR / 'runtime_memory_summary.csv'
MEMORY_INTERVAL = 0.01  # Seconds between process-tree RSS samples.
RAPIDNJ_BINARY = 'rapidnj'  # Executable name or full path.
RSCRIPT_BINARY = '/usr/bin/Rscript'
DENDRO_R_LIBRARY = Path('~/Litianle/DENDRO_test/R_library').expanduser()
WORKERS = 1 if TIME_MEM else 30
SEEDS = None
SAMPLE_SIZES = None
PROBS = None
SCENARIOS = ['main', 'state_permutation']
SAVE_TREES = True
TIP_ORDER_SEED = 20261014
OUTGROUP = 'Synthetic_Non_SNV_Outgroup'
N_NEIGHBORS = 10
MIN_CALLABLE_DEPTH = 1
MIN_CALLABLE_CELLS = 3
MIN_POOLED_RATIO = 0.9
MIN_E = 10.
MIN_TOTAL_DEPTH = 5
WLS_WEIGHT_POWER = 2.
STATE_LOW = 0.01
PLOT_SEED = 20261014
DPI = 300
QGS_N = 2
QGS_METRIC = f'Qgs{QGS_N}'
TARGET_METHOD = 'geinj_L1'
RESOURCE_METHODS = [TARGET_METHOD, 'geinj_no_revision_L1']
DENDRO_METHOD = 'dendro_default'
DENDRO_FILTERED_METHOD = 'dendro_filtered'
DENDRO_RESULTS = OUTPUT_DIR / 'dendro_matched_benchmark_results.csv'
SENSITIVITY_DIR = OUTPUT_DIR if SENSITIVITY_ONLY else OUTPUT_DIR / 'Sensitivity_results'
SENSITIVITY_RESULTS = SENSITIVITY_DIR / 'sensitivity_results.csv'
SENSITIVITY_VALUES = {
    'MIN_TOTAL_DEPTH': [1, 3, 5, 10],
    'MIN_CALLABLE_DEPTH': [1, 2, 3, 5],
    'MIN_POOLED_RATIO': [0.85, 0.90, 0.95],
    'MIN_E': [5., 10., 20.],
    'STATE_LOW': [0.005, 0.01, 0.1, 0.2],
    'WLS_WEIGHT_POWER': [0., 1., 2., 3.],
}


class PeakMemory:
    """Sample this worker and its children; report summed RSS in MiB."""
    def __init__(self):
        self.process = psutil.Process()
        self.peak = 0
        self.stop = Event()
        self.thread = Thread(target=self.sample, daemon=True)

    def sample_once(self):
        rss = 0
        for process in [self.process] + self.process.children(recursive=True):
            try:
                rss += process.memory_info().rss
            except psutil.NoSuchProcess:
                pass  # A child may exit between samples.
        self.peak = max(self.peak, rss)

    def sample(self):
        while True:
            self.sample_once()
            if self.stop.wait(MEMORY_INTERVAL):
                break

    def __enter__(self):
        self.sample_once()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample_once()


class RapidNJTiming:
    """Time only RapidNJ execution, including calls inside GEI-NJ."""
    def __init__(self, binary):
        self.binary = str(binary)
        self.seconds = 0.
        self.calls = 0

    def run(self, command, *args, **kwargs):
        if str(command[0]) != self.binary:
            return self.original_run(command, *args, **kwargs)
        started = perf_counter()
        self.calls += 1
        try:
            return self.original_run(command, *args, **kwargs)
        finally:
            self.seconds += perf_counter() - started

    def __enter__(self):
        self.original_run = subprocess.run
        subprocess.run = self.run
        return self

    def __exit__(self, *args):
        subprocess.run = self.original_run


class RapidNJError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


DENDRO_R_SCRIPT = r'''
options(mc.cores = 1L)
args <- commandArgs(trailingOnly = TRUE)
r_lib <- args[1]
work_dir <- args[2]
result <- data.frame(
    dendro_n_sites_retained = NA_integer_,
    dendro_n_cells_retained = NA_integer_,
    dendro_cell_retained_fraction = NA_real_,
    dendro_version = NA_character_,
    status = "dendro_failed", backend_error = ""
)
tryCatch({
    .libPaths(c(r_lib, .libPaths()))
    suppressPackageStartupMessages(library(DENDRO, lib.loc = r_lib))
    result$dendro_version <- as.character(packageVersion("DENDRO", lib.loc = r_lib))
    X <- as.matrix(read.delim(file.path(work_dir, "X.tsv"),
        row.names = 1, check.names = FALSE))
    N <- as.matrix(read.delim(file.path(work_dir, "N.tsv"),
        row.names = 1, check.names = FALSE))
    Z <- 1L * (X > 0)
    Z[N == 0] <- NA
    qc <- FilterCellMutation(X, N, Z, plot = FALSE)
    result$dendro_n_sites_retained <- nrow(qc$X)
    result$dendro_n_cells_retained <- ncol(qc$X)
    result$dendro_cell_retained_fraction <- ncol(qc$X) / ncol(X)
    if (ncol(qc$X) < 3L) {
        result$status <- "dendro_too_few_cells"
        result$backend_error <- "Fewer than 3 cells retained after default filtering."
    } else {
        d <- DENDRO.dist(qc$X, qc$N, qc$Z, show.progress = FALSE)
        hc <- DENDRO.cluster(d, plot = FALSE)
        tree <- ape::as.phylo(hc)
        ape::write.tree(tree, file = file.path(work_dir, "tree.nwk"))
        result$status <- "ok"
    }
}, error = function(error) {
    result$status <<- "dendro_failed"
    result$backend_error <<- conditionMessage(error)
    message(conditionMessage(error))
})
write.csv(result, file.path(work_dir, "metadata.csv"), row.names = FALSE, na = "NA")
'''


def run_dendro(depth, alt, names, output, dataset_id):
    """Run default DENDRO on observed counts."""
    names = list(names)
    aliases = [f'T{i:06d}' for i in range(len(names))]
    original_names = dict(zip(aliases, names))
    info = dict(dendro_n_sites_input=alt.shape[1],
        dendro_n_sites_retained=np.nan, dendro_n_cells_retained=np.nan,
        dendro_cell_retained_fraction=np.nan,
        dendro_version=np.nan, status='dendro_failed', backend_error='')
    tree, log = None, ''
    with TemporaryDirectory(prefix='dendro_') as directory:
        directory = Path(directory)
        try:
            pd.DataFrame(alt.T, columns=aliases).to_csv(directory / 'X.tsv', sep='\t')
            pd.DataFrame(depth.T, columns=aliases).to_csv(directory / 'N.tsv', sep='\t')
            command = [str(RSCRIPT_BINARY), '--vanilla', '-',
                str(DENDRO_R_LIBRARY), str(directory)]
            completed = subprocess.run(command, input=DENDRO_R_SCRIPT,
                capture_output=True, text=True, check=False)
            log = completed.stdout + '\n' + completed.stderr
            metadata = directory / 'metadata.csv'
            if metadata.exists():
                info.update(pd.read_csv(metadata, keep_default_na=False,
                    na_values=['NA']).iloc[0].to_dict())
            if completed.returncode:
                detail = completed.stderr.strip() or completed.stdout.strip()
                error = f'Rscript exit code {completed.returncode}: {detail[-2000:]}'
                info['status'] = 'dendro_failed'
                info['backend_error'] = '\n'.join(filter(None, [info['backend_error'], error]))
            elif info['status'] == 'ok':
                tree = Phylo.read(directory / 'tree.nwk', 'newick')
                for tip in tree.get_terminals():
                    tip.name = original_names[tip.name]
                tree.rooted = True
            elif not metadata.exists():
                info['backend_error'] = 'Rscript did not write DENDRO metadata.'
        except (OSError, ValueError, KeyError, NewickError) as error:
            tree = None
            info['status'] = 'dendro_failed'
            info['backend_error'] = '\n'.join(filter(None,
                [info['backend_error'], str(error)]))
            log += '\n' + str(error)
    log_dir = Path(output) / 'dendro_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f'{dataset_id}.log').write_text(log, encoding='utf-8')
    return tree, info


def build_nj_tree(distance_matrix, names, binary, outgroup_name='Synthetic_Non_SNV_Outgroup'):
    """Build NJ and optionally root on the supplied outgroup."""
    distances = np.asarray(distance_matrix, dtype=float)
    names = list(names)
    if distances.shape != (len(names), len(names)) or not np.isfinite(distances).all():
        raise ValueError('RapidNJ requires a finite square distance matrix.')
    if len(set(names)) != len(names) or (outgroup_name is not None and outgroup_name not in names):
        raise ValueError('Tip names must be unique; a requested outgroup must be present.')
    try:
        tree = _build_nj_tree(distances, names, binary)
    except OSError as error:
        raise RapidNJError('rapidnj_execution_failed', str(error)) from error
    if outgroup_name is None:
        return tree
    outgroup = next(tip for tip in tree.get_terminals() if tip.name == outgroup_name)
    try:
        tree.root_with_outgroup(outgroup)
    except ValueError as error:
        raise RapidNJError('rapidnj_rooting_failed', str(error)) from error
    return tree



def configurations():
    methods = [
        ('complete_genotype', 'Complete-genotype', 'reference'),
        ('zero_filled_no_revision', 'Raw ratio / Zero-filled', 'primary'),
        ('zero_filled', 'Zero-filled', 'primary'),
        ('knn_normalized_L1', 'kNN (normalized L1)', 'primary'),
        ('mean', 'Mean imputation', 'primary'),
    ]
    configs = []
    for method, label, group in methods:
        state_model = ('true' if group == 'reference' else
                       'raw' if method == 'zero_filled_no_revision' else 'threshold')
        for metric in ['L1', 'L2']:
            separator = '_final_' if method.startswith('knn_') else '_'
            method_id = ('zero_filled_no_wls' if method == 'zero_filled' and metric == 'L1'
                         else method + separator + metric)
            configs.append(dict(method_id=method_id,
                method=method, method_label=label + ' | ' + metric,
                method_group=group, final_metric=metric,
                state_model=state_model,
                root_method='outgroup', remove_germline=group != 'reference',
                n_neighbors=N_NEIGHBORS if method.startswith('knn_') else 0))
    for metric, label in [('L1', 'mean L1'), ('L2', 'RMS L2')]:
        configs.append(dict(method_id='pairwise_' + metric, method='pairwise',
            method_label='Pairwise deletion / ' + label, method_group='primary',
            final_metric=metric, state_model='threshold', root_method='outgroup',
            remove_germline=True, n_neighbors=0))
    target = dict(method_id=TARGET_METHOD, method='zero_filled',
        method_label='GEI-NJ / Binary / Zero-filled / L1 / WLS',
        method_group='primary', final_metric='L1', state_model='threshold',
        root_method='wls_ingroup', remove_germline=True, n_neighbors=0)
    configs.insert(2, target)
    configs.insert(3, dict(target, method_id='geinj_no_revision_L1',
        method_label='Continuous ALT ratio / Zero-filled / L1 / WLS', state_model='raw'))
    ablations = [
        ('geinj_L2', 'L2 distance', dict(final_metric='L2')),
        ('geinj_mean_L1', 'Mean imputation', dict(method='mean')),
        ('geinj_knn_L1', 'kNN imputation', dict(method='knn_normalized_L1', n_neighbors=N_NEIGHBORS)),
    ]
    for method_id, label, changes in ablations:
        configs.append(dict(target, method_id=method_id, method_label=label,
            method_group='ablation', **changes))
    configs.append(dict(method_id=DENDRO_METHOD, method=DENDRO_METHOD,
        method_label='DENDRO defaults | ward.D', method_group='external',
        final_metric='ward.D', state_model='native', root_method='native',
        remove_germline=False, n_neighbors=0))
    configs.append(dict(configs[-1], method_id=DENDRO_FILTERED_METHOD,
        method_label='DENDRO / Filtered SNVs / ward.D', remove_germline=True))
    return configs


def dense(values):
    return np.asarray(values.toarray() if hasattr(values, 'toarray') else values, dtype=float)


def sensitivity_configurations():
    target = next(config for config in configurations() if config['method_id'] == TARGET_METHOD)
    defaults = dict(min_total_depth=MIN_TOTAL_DEPTH, min_callable_depth=MIN_CALLABLE_DEPTH,
        min_pooled_ratio=MIN_POOLED_RATIO, min_E=MIN_E, state_low=STATE_LOW,
        wls_weight_power=WLS_WEIGHT_POWER)
    target = dict(target, **defaults, use_e=True, method_group='sensitivity')
    configs, groups = [target], {}
    for parameter, values in SENSITIVITY_VALUES.items():
        key = 'min_E' if parameter == 'MIN_E' else parameter.lower()
        groups[parameter] = []
        for value in values:
            if value == defaults[key]:
                method_id, label = TARGET_METHOD, f'{value:g} (default)'
            else:
                method_id, label = f'sensitivity_{parameter.lower()}_{value:g}', f'{value:g}'
                config = dict(target, method_id=method_id, method_label=f'{parameter} = {label}')
                config[key] = value
                if parameter == 'WLS_WEIGHT_POWER':
                    config['score_unrooted'] = False
                configs.append(config)
            groups[parameter].append((method_id, label))
    configs.extend([
        dict(target, method_id='geinj_no_revision_L1', state_model='raw',
            method_label='Continuous ALT ratio / Zero-filled / L1 / WLS'),
        dict(target, method_id='geinj_no_germline_L1', remove_germline=False,
            method_label='No germline filtering'),
        dict(target, method_id='sensitivity_pooled_only', use_e=False,
            method_label='Pooled ratio only'),
    ])
    groups['STATE_LOW'].append(('geinj_no_revision_L1', 'Continuous'))
    groups['GERMLINE_FILTERING'] = [('geinj_no_germline_L1', 'No filtering'),
        ('sensitivity_pooled_only', 'Pooled ratio only'), (TARGET_METHOD, 'Pooled ratio + E')]
    return configs, groups


def prepare_panel(depth, alt, site_names, remove_germline=True,
                  min_total_depth=MIN_TOTAL_DEPTH, min_callable_depth=MIN_CALLABLE_DEPTH,
                  min_pooled_ratio=MIN_POOLED_RATIO, min_E=MIN_E, use_e=True,
                  germline_truth=None):
    """Apply package filtering and retain benchmark diagnostics."""
    covered = (depth > 0).any(axis=0)
    enough_depth = depth.sum(axis=0) >= min_total_depth
    ratio = np.divide(alt, depth, out=np.zeros(depth.shape), where=depth > 0)
    panel = ad.AnnData(X=ratio, var=pd.DataFrame(index=site_names), layers={'total_depth': depth})
    panel = gei_nj.filter_snvs(panel, min_total_depth=min_total_depth,
        remove_germline=remove_germline and use_e, min_callable_cells_germline=MIN_CALLABLE_CELLS,
        min_pooled_ratio_germline=min_pooled_ratio, min_E_germline=min_E, copy=False, verbose=False)
    if remove_germline and not use_e:
        # Pooled-only ablation is not exposed by the package.
        total_depth = depth.sum(axis=0)
        pooled_ratio = np.divide(alt.sum(axis=0), total_depth,
            out=np.full(depth.shape[1], np.nan), where=total_depth > 0)
        remove = ((depth > 0).sum(axis=0) >= MIN_CALLABLE_CELLS) & (pooled_ratio >= min_pooled_ratio)
        panel = panel[:, ~remove[enough_depth]].copy()
    keep = site_names.isin(panel.var_names)
    remove = enough_depth & ~keep
    observed = dense(panel.layers['total_depth']) >= min_callable_depth
    diagnostics = dict(n_sites_input=int(len(keep)), n_sites_retained=int(keep.sum()),
        n_all_zero_depth_sites_removed=int((~covered).sum()),
        n_low_total_depth_sites_removed=int((covered & ~enough_depth).sum()),
        n_germline_candidates_removed=int(remove.sum()),
        n_cells_zero_observed=int((~observed.any(axis=1)).sum()),
        n_observed_entries=int(observed.sum()),
        observed_fraction=float(observed.mean()) if observed.size else 0.,
        panel_sha256=hashlib.sha256(json.dumps(list(map(str, site_names[keep]))).encode()).hexdigest())
    if germline_truth is not None:
        positive = enough_depth & germline_truth
        negative = enough_depth & ~germline_truth
        tp, fn = int((positive & remove).sum()), int((positive & ~remove).sum())
        fp, tn = int((negative & remove).sum()), int((negative & ~remove).sum())
        diagnostics.update(germline_tp=tp, germline_fn=fn, germline_fp=fp, germline_tn=tn,
            n_germline_eligible=tp+fn, n_somatic_eligible=fp+tn,
            germline_fnr=fn/(tp+fn) if tp+fn else np.nan,
            germline_fpr=fp/(fp+tn) if fp+tn else np.nan)
    return panel, keep, diagnostics



def normalized_l1(values):
    observed = np.isfinite(values)
    filled = np.nan_to_num(values, nan=0.)
    distances = np.full((len(values), len(values)), np.nan)
    for i in range(len(values)):
        common = observed[i] & observed[i:]
        shared = common.sum(axis=1)
        sums = (np.abs(filled[i] - filled[i:]) * common).sum(axis=1)
        distances[i, i:] = np.divide(sums, shared,
            out=np.full(len(shared), np.nan), where=shared > 0)
        distances[i:, i] = distances[i, i:]
    return distances


def impute_neighbors(values, distances):
    """Use site-specific observed donors and preserve the original kNN ties."""
    observed = np.isfinite(values)
    features = values.copy()
    n_fallback = 0
    for col in range(values.shape[1]):
        receivers = np.flatnonzero(~observed[:, col])
        if not len(receivers):
            continue
        donors = np.flatnonzero(observed[:, col])
        subset = distances[np.ix_(receivers, donors)]
        fallback = np.isnan(subset).all(axis=1)
        if fallback.any():
            features[receivers[fallback], col] = values[donors, col].mean()
            n_fallback += int(fallback.sum())
        receivers, subset = receivers[~fallback], subset[~fallback]
        if not len(receivers):
            continue
        k = min(N_NEIGHBORS, len(donors))
        nearest = np.argpartition(subset, k - 1, axis=1)[:, :k]
        chosen_distances = np.take_along_axis(subset, nearest, axis=1)
        weight = np.ones_like(chosen_distances)
        weight[np.isnan(chosen_distances)] = 0.
        chosen_values = values[donors, col].take(nearest)
        features[receivers, col] = np.ma.average(
            np.ma.array(chosen_values, mask=np.zeros_like(chosen_values, dtype=bool)),
            axis=1, weights=weight).data
    n_missing = int((~observed).sum())
    return features, dict(knn_missing_n_entries=n_missing,
        knn_mean_fallback_n_entries=n_fallback,
        knn_mean_fallback_fraction=n_fallback / n_missing if n_missing else 0.)


def prepare_representation(values, method):
    if method == 'knn_normalized_L1':
        return impute_neighbors(values, normalized_l1(values))
    if method == 'mean':
        return np.where(np.isfinite(values), values, np.nanmean(values, axis=0)), {}
    if method in {'zero_filled', 'zero_filled_no_revision'}:
        return np.nan_to_num(values, nan=0.), {}
    return values, {}


def root_zero_reference(tree, names, reference_distances, weight_power=2.):
    return _root_zero_reference(tree, names, reference_distances, weight_power)



def pairwise_distance(values, metric):
    """Normalize differences by the number of shared callable sites."""
    features = np.vstack([values, np.zeros(values.shape[1])])
    if metric == 'L1':
        distances = normalized_l1(features)
    else:
        observed = np.isfinite(features)
        filled = np.nan_to_num(features, nan=0.)
        distances = np.full((len(features), len(features)), np.nan)
        for i in range(len(features)):
            common = observed[i] & observed[i:]
            shared = common.sum(axis=1)
            sums = ((filled[i] - filled[i:]) ** 2 * common).sum(axis=1)
            means = np.divide(sums, shared,
                out=np.full(len(shared), np.nan), where=shared > 0)
            distances[i, i:] = np.sqrt(means)
            distances[i:, i] = distances[i, i:]
    np.fill_diagonal(distances, 0.)
    return distances


def prepare_truth(names, lineages):
    """Prepare the binary simulation genealogy on the sampled tips only."""
    n = len(names)
    if len(set(names)) != n or len(lineages) != n or OUTGROUP in names:
        raise ValueError('Truth requires unique real tip names and one lineage per tip.')
    paths = [tuple(lineage.split('-')) for lineage in lineages]
    prefixes, groups = {}, {}
    for i, path in enumerate(paths):
        for length in range(1, len(path) + 1):
            key = path[:length]
            prefixes[key] = prefixes.get(key, 0) | (1 << i)
        if not ORGANIZED:
            # Include generation 0.
            key = path[:QGS_N + 1]
            groups[key] = groups.get(key, 0) | (1 << i)
    # Cache one MRCA descendant mask per unordered pair; the empty prefix is
    # the root when sampled lineage strings do not share their first token.
    prefixes[()] = (1 << n) - 1
    pair_indices = np.triu_indices(n, 1)
    pair_mrca_masks = []
    for i, j in zip(*pair_indices):
        common_length = 0
        for first, second in zip(paths[i], paths[j]):
            if first != second:
                break
            common_length += 1
        pair_mrca_masks.append(prefixes[paths[i][:common_length]])
    triplet_denominator = comb(n, 3)
    resolved_truth = sum(n - mask.bit_count() for mask in pair_mrca_masks)
    if resolved_truth != triplet_denominator:
        raise ValueError('Rooted triplet scoring requires a fully resolved simulation truth; '
                         'check lineage strings and sampled-tip identities.')
    full = (1 << n) - 1
    rooted_clades = {mask for mask in prefixes.values() if 1 < mask.bit_count() < n}
    unrooted_splits = {min(mask, full ^ mask) for mask in rooted_clades
                       if mask.bit_count() < n - 1}
    return dict(index={name: i for i, name in enumerate(names)}, groups=list(groups.values()),
                n_tips=n, pair_indices=pair_indices,
                pair_mrca_masks=pair_mrca_masks, triplet_denominator=triplet_denominator,
                unrooted_splits=unrooted_splits)


def mask_indices(mask):
    """Return sampled-tip indices represented by an integer bit mask."""
    indices = []
    while mask:
        bit = mask & -mask
        indices.append(bit.bit_length() - 1)
        mask ^= bit
    return np.asarray(indices, dtype=np.intp)


def score_tree(tree, truth, subset=False, score_unrooted=True):
    """Score QGS, exact rooted triplet distance and unrooted nRF.

    Truth must be binary (checked once in prepare_truth). For each pair a,b,
    c supports ab|c in both trees iff c is outside the union of their MRCA
    descendant sets. Summing over pairs counts each agreeing triplet once.
    subset=True scores the induced tree on truth tips.
    Inferred polytomies disagree with resolved truth; zero-length branches
    are preserved. Branch lengths and the synthetic outgroup are excluded.
    """
    index, n = truth['index'], truth['n_tips']
    members, clades, seen = {}, set(), set()
    pair_mrca_masks = np.zeros((n, n), dtype=object)
    for node in tree.find_clades(order='postorder'):
        if node.is_terminal():
            name = str(node.name)
            if name == OUTGROUP or (subset and name not in index):
                mask = 0
            else:
                if name not in index or name in seen:
                    raise ValueError('Unexpected or duplicate sampled tip.')
                seen.add(name)
                mask = 1 << index[name]
        else:
            child_masks = [members.pop(id(child)) for child in node.clades]
            mask = 0
            for child_mask in child_masks:
                mask |= child_mask
            # Each real-tip pair crosses children at exactly one node: its
            # MRCA. Empty outgroup branches and unary nodes add no pairs.
            previous = np.empty(0, dtype=np.intp)
            for child_mask in child_masks:
                if not child_mask:
                    continue
                current = mask_indices(child_mask)
                if previous.size:
                    pair_mrca_masks[np.ix_(previous, current)] = mask
                    pair_mrca_masks[np.ix_(current, previous)] = mask
                previous = np.concatenate((previous, current))
        members[id(node)] = mask
        if mask:
            clades.add(mask)
    if seen != set(index):
        raise ValueError('Missing sampled tips.')
    inferred_pairs = pair_mrca_masks[truth['pair_indices']]
    agreeing_triplets = sum(n - (actual | predicted).bit_count()
        for actual, predicted in zip(truth['pair_mrca_masks'], inferred_pairs))
    triplet_denominator = truth['triplet_denominator']
    raw_triplet = triplet_denominator - agreeing_triplets
    if not 0 <= raw_triplet <= triplet_denominator:
        raise ValueError('Invalid rooted triplet counts; check the tree topology.')
    scores = {'triplet_distance': raw_triplet / triplet_denominator if triplet_denominator else np.nan}
    if not ORGANIZED:
        recovery = []
        for group in truth['groups']:
            size = group.bit_count()
            largest = max(mask.bit_count() for mask in clades if mask & group == mask)
            recovery.append((largest - 1) / (size - 1) if size > 1 else 0.)
        scores[QGS_METRIC] = float(100 * np.mean(recovery))
    if score_unrooted:
        full = (1 << n) - 1
        unrooted_splits = {min(mask, full ^ mask) for mask in clades
                           if 1 < mask.bit_count() < n - 1}
        actual = truth['unrooted_splits']
        denominator = len(actual) + len(unrooted_splits)
        scores['unrooted_nrf'] = len(actual ^ unrooted_splits) / denominator if denominator else 0.
    return scores


def baseline_tree(features, names, config, binary, row, founder=None, panel=None):
    if founder is None:
        founder = np.zeros(features.shape[1])
    evidence = np.nan_to_num(features, nan=0.) - founder
    row['n_cells_zero_evidence'] = int((~evidence.any(axis=1)).sum())
    row['all_evidence_zero'] = not bool(evidence.any())
    if config['root_method'] == 'wls_ingroup' and config['final_metric'] == 'L1' and np.isfinite(features).all():
        # Count identical profiles without repeating the package distance calculation.
        _, counts = np.unique(features, axis=0, return_counts=True)
        row.update(n_undefined_pairs=0, n_undefined_ingroup_pairs=0,
            zero_distance_ingroup_fraction=float((counts * (counts-1)).sum() / (len(names) * (len(names)-1))))
        if len(counts) == 1:
            row['status'] = 'degenerate_distance'
            return None
        panel.layers['zero_filled_state'] = features
        try:
            tree, diagnostics = gei_nj.build_nj(panel, do_plot=False,
                wls_weight_power=config.get('wls_weight_power', WLS_WEIGHT_POWER),
                rapidnj_binary=str(binary), return_tree_info=True)
        except OSError as error:
            raise RapidNJError('rapidnj_execution_failed', str(error)) from error
        row.update({key: value for key, value in diagnostics.items() if key.startswith('wls_')})
        return tree
    if config['method'] == 'pairwise':
        distance = pairwise_distance(features, config['final_metric'])
    else:
        coordinates = np.vstack([features, founder])
        distance = squareform(pdist(coordinates,
            metric='cityblock' if config['final_metric'] == 'L1' else 'euclidean'))
    pairs = distance[np.triu_indices(len(names) + 1, 1)]
    ingroup = distance[:len(names), :len(names)][np.triu_indices(len(names), 1)]
    row.update(n_undefined_pairs=int((~np.isfinite(pairs)).sum()),
        n_undefined_ingroup_pairs=int((~np.isfinite(ingroup)).sum()),
        zero_distance_ingroup_fraction=float((ingroup == 0.).mean()))
    if not np.isfinite(distance).all():
        row['status'] = 'undefined_pairwise_distance'
        return None
    if not np.any(ingroup > 0.):
        row['status'] = 'degenerate_distance'
        return None
    if config['root_method'] == 'wls_ingroup':
        tree = build_nj_tree(distance[:-1, :-1], names, binary, outgroup_name=None)
    else:
        tree = build_nj_tree(distance, names + [OUTGROUP], binary)
    if config['root_method'] == 'wls_ingroup':
        tree, diagnostics = root_zero_reference(tree, names, distance[:-1, -1],
            config.get('wls_weight_power', WLS_WEIGHT_POWER))
        row.update(diagnostics)
    return tree



def evaluate_dataset(task):
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))
    path, config, output, rapidnj_binary = task
    with threadpool_limits(limits=1):
        if TIME_MEM:
            with PeakMemory() as memory, RapidNJTiming(rapidnj_binary) as timing:
                started = perf_counter()
                row, tree, names, lineages = _evaluate_dataset(path, config, output, rapidnj_binary)
                row['total_runtime_seconds'] = perf_counter() - started
            row.update(peak_memory_mib=memory.peak / 1024 ** 2,
                rapidnj_runtime_seconds=timing.seconds, rapidnj_calls=timing.calls)
        else:
            row, tree, names, lineages = _evaluate_dataset(path, config, output, rapidnj_binary)
        if not TIME_MEM and tree is not None:
            if config['method'] == DENDRO_METHOD:
                retained = {tip.name for tip in tree.get_terminals()}
                selected = [i for i, name in enumerate(names) if name in retained]
                names, lineages = [names[i] for i in selected], [lineages[i] for i in selected]
            row.update(score_tree(tree, prepare_truth(names, lineages),
                score_unrooted=config.get('score_unrooted', True)))
        target = output / 'parts' / f"{row['dataset_id']}__{config['method_id']}.csv"
        pd.DataFrame([row]).to_csv(target, index=False)
    return str(target)


def _evaluate_dataset(path, config, output, rapidnj_binary):
    if TIME_MEM:
        with h5py.File(path, 'r') as handle:
            obs = pd.DataFrame(index=read_elem(handle['obs'][handle['obs'].attrs['_index']]))
            var = pd.DataFrame(index=read_elem(handle['var'][handle['var'].attrs['_index']]))
            simulation = {key: read_elem(handle['uns']['simulation'][key])
                for key in ['seed', 'prob', 'scenario', 'callability_scenario',
                            'germline_background', 'difference_factor']}
            data = ad.AnnData(obs=obs, var=var, uns={'simulation': simulation},
                layers={key: read_elem(handle['layers'][key]) for key in ['depth', 'alt_count']})
    else:
        data = ad.read_h5ad(path)
    simulation = data.uns['simulation']
    seed, prob, n = int(simulation['seed']), float(simulation['prob']), data.n_obs
    scenario = str(simulation['scenario'])
    callability_scenario = str(simulation['callability_scenario'])
    germline_background = str(simulation['germline_background'])
    dataset_id = f'{germline_background}_{callability_scenario}_{scenario}_prob{prob:g}_n{n}_seed{seed}'
    depth, alt = dense(data.layers['depth']), dense(data.layers['alt_count'])
    names = list(map(str, data.obs_names))
    lineages = [] if TIME_MEM else list(map(str, data.obs.lineage))
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(round(-np.log10(prob))), n, TIP_ORDER_SEED]))
    order = rng.permutation(n)
    depth, alt = depth[order], alt[order]
    names = [names[i] for i in order]
    if not TIME_MEM:
        lineages = [lineages[i] for i in order]
    method = config['method']
    row = dict(dataset_id=dataset_id, input_file=str(path.resolve()),
        scenario=scenario, seed=seed, prob=prob, sample_size=n, n_cells=n,
        callability_scenario=callability_scenario, germline_background=germline_background,
        difference_factor=float(simulation['difference_factor']),
        n_snv_sites=data.n_vars, n_sites_input=data.n_vars,
        n_sites_retained=np.nan, n_cells_retained=n,
        nj_backend='DENDRO' if method == DENDRO_METHOD else 'RapidNJ',
        **config, technical_success=False, supported_success=False,
        **{endpoint: np.nan for endpoint in ENDPOINTS},
        n_undefined_pairs=np.nan, n_undefined_ingroup_pairs=np.nan,
        n_cells_zero_evidence=np.nan, all_evidence_zero=False,
        status='no_observed_sites', backend_error='', tree_path='')
    tree = None
    try:
        if method == DENDRO_METHOD:
            if config['remove_germline']:
                _, keep, diagnostics = prepare_panel(depth, alt, data.var_names)
                row.update(diagnostics)
                log_id = dataset_id + '__' + config['method_id']
            else:
                keep = (depth > 0).any(axis=0)
                log_id = dataset_id
            tree, info = run_dendro(depth[:, keep], alt[:, keep], names, output, log_id)
            row.update(info)
            row.update(n_sites_retained=info['dendro_n_sites_retained'],
                n_cells_retained=info['dendro_n_cells_retained'])
        else:
            panel, keep, diagnostics = prepare_panel(depth, alt, data.var_names,
                remove_germline=config['remove_germline'],
                min_total_depth=config.get('min_total_depth', MIN_TOTAL_DEPTH),
                min_callable_depth=config.get('min_callable_depth', MIN_CALLABLE_DEPTH),
                min_pooled_ratio=config.get('min_pooled_ratio', MIN_POOLED_RATIO),
                min_E=config.get('min_E', MIN_E), use_e=config.get('use_e', True),
                germline_truth=(data.var.is_germline.to_numpy(dtype=bool)
                    if config['method_group'] == 'sensitivity' else None))
            row.update(diagnostics)
            if panel.n_vars:
                panel.obs_names = names
                founder = None
                if method == 'complete_genotype':
                    # Truth enters only the two reference configurations.
                    features = dense(data.layers['true_genotype'])[order][:, keep]
                    founder = data.var.is_germline.to_numpy(dtype=float)[keep]
                else:
                    panel = gei_nj.assign_state(panel,
                        method='binary' if config['state_model'] == 'threshold' else 'continuous',
                        state_low=config.get('state_low', STATE_LOW), copy=False)
                    min_callable_depth = config.get('min_callable_depth', MIN_CALLABLE_DEPTH)
                    if method in {'zero_filled', 'zero_filled_no_revision'}:
                        panel = gei_nj.zero_filling(panel, min_callable_depth=min_callable_depth,
                                                   copy=False, verbose=False)
                        features = dense(panel.layers['zero_filled_state'])
                    else:
                        values = dense(panel.layers['state']).copy()
                        values[dense(panel.layers['total_depth']) < min_callable_depth] = np.nan
                        features, method_diagnostics = prepare_representation(values, method)
                        row.update(method_diagnostics)
                tree = baseline_tree(features, names, config, rapidnj_binary, row, founder, panel)
    except RapidNJError as error:
        row.update(status=error.code, backend_error=str(error))
    except RuntimeError as error:
        detail = str(error)
        status = ('rapidnj_parse_failed' if detail.startswith('Cannot read the RapidNJ tree:') else
            'rapidnj_tip_set_failed' if detail.startswith('RapidNJ returned') else 'rapidnj_execution_failed')
        row.update(status=status, backend_error=detail)
    except subprocess.CalledProcessError as error:
        row.update(status='rapidnj_execution_failed', backend_error=str(error.stderr))
    if tree is not None:
        tree_dir = output / 'trees' / dataset_id
        tree_dir.mkdir(parents=True, exist_ok=True)
        tree_path = tree_dir / (config['method_id'] + '.nwk')
        Phylo.write(tree, tree_path, 'newick', format_branch_length='%1.12g')
        row.update(technical_success=True, status='ok', tree_path=str(tree_path),
            supported_success=(method in {'complete_genotype', DENDRO_METHOD}
                or row['n_cells_zero_observed'] == 0) and not row['all_evidence_zero'])
    return row, tree, names, lineages


def compare_dendro(rows):
    """Score saved trees on shared cells retained by successful DENDRO runs."""
    dendro = rows[rows.method.eq(DENDRO_METHOD) & rows.technical_success]
    context = dict(evaluation_scope='dendro_common_retained_cells',
        n_evaluation_cells=0, evaluation_tip_sha256='')
    truth = None
    if len(dendro):
        tip_sets = [{tip.name for tip in Phylo.read(path, 'newick').get_terminals()}
                    for path in dendro.tree_path]
        retained = set.intersection(*tip_sets)
        context['n_evaluation_cells'] = len(retained)
        context['evaluation_tip_sha256'] = hashlib.sha256(json.dumps(sorted(retained)).encode()).hexdigest()
        if len(retained) >= 3:
            data = ad.read_h5ad(dendro.iloc[0].input_file)
            selected = data.obs.loc[data.obs_names.isin(retained)]
            truth = prepare_truth(list(map(str, selected.index)), list(map(str, selected.lineage)))
    matched = []
    for row in rows.to_dict('records'):
        item = dict(row, **context)
        item.update(full_tree_success=row['technical_success'], technical_success=False,
            supported_success=False, status='dendro_subset_unavailable',
            **{endpoint: np.nan for endpoint in ENDPOINTS})
        if truth is not None:
            item['status'] = row['status']
            if row['technical_success']:
                tree = Phylo.read(row['tree_path'], 'newick')
                item.update(score_tree(tree, truth, subset=True))
                item.update(technical_success=True, supported_success=not row['all_evidence_zero'])
        elif row['method'] == DENDRO_METHOD and not row['technical_success']:
            item['status'] = row['status']
        matched.append(item)
    return matched


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simulation_paths(scenarios=SCENARIOS):
    paths = sorted(path for directory in INPUT_DIR.iterdir() if directory.is_dir()
        for background in [directory / 'with_germline']
        for scenario in scenarios
        for path in (background if scenario == 'main' else background / scenario).glob('prob_*/cells_*/*.h5ad'))
    paths = [path for path in paths
        if (SEEDS is None or int(path.stem.rsplit('seed', 1)[1]) in SEEDS)
        and (SAMPLE_SIZES is None or int(path.parent.name.split('_')[-1]) in SAMPLE_SIZES)
        and (PROBS is None or float(path.parent.parent.name.split('_')[-1]) in PROBS)]
    if not paths:
        raise ValueError('No simulation files found for the selected scenarios.')
    return paths


def rapidnj_executable():
    executable = shutil.which(str(RAPIDNJ_BINARY))
    if executable is None:
        raise FileNotFoundError('Set RAPIDNJ_BINARY to an installed RapidNJ executable.')
    return Path(executable).resolve()


def run_tasks(tasks, workers):
    np.random.default_rng(TIP_ORDER_SEED).shuffle(tasks)
    completed = []
    # A fresh worker prevents memory carryover between methods.
    with get_context('spawn').Pool(processes=workers, maxtasksperchild=1) as pool:
        for result in pool.imap_unordered(evaluate_dataset, tasks, chunksize=1):
            completed.append(result)
            if len(completed) % 10 == 0 or len(completed) == len(tasks):
                print(f'Completed {len(completed)}/{len(tasks)} dataset-method runs', flush=True)
    results = pd.concat([pd.read_csv(path) for path in sorted(completed)], ignore_index=True)
    assert len(results) == len(tasks)
    assert not results.duplicated(['dataset_id', 'method_id']).any()
    return results


def run_benchmark():
    paths = simulation_paths()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'parts').mkdir(exist_ok=True)
    (OUTPUT_DIR / 'parts_dendro').mkdir(exist_ok=True)
    rapidnj_binary = rapidnj_executable()
    configs = configurations()
    manifest = dict(input_dir=str(INPUT_DIR.resolve()),
        input_files=[dict(path=str(path.resolve()), sha256=sha256(path)) for path in paths],
        sources={Path(__file__).name: sha256(Path(__file__).resolve())},
        geinj_version=gei_nj.__version__, geinj_source=str(Path(gei_nj.__file__).resolve()),
        nj_backend='RapidNJ', rapidnj_binary=str(rapidnj_binary),
        rapidnj_binary_sha256=sha256(rapidnj_binary),
        phylip_precision='.17g', phylip_layout='full square matrix',
        configs=configs, n_neighbors=N_NEIGHBORS,
        target_method=TARGET_METHOD,
        parameter_protocol='ALT threshold 0.01 and WLS power 2 are fixed. Selected seeds are summarized together; this is not a separate held-out validation report.',
        target_rule=dict(min_callable_depth=MIN_CALLABLE_DEPTH,
            representation='ALT/depth >= 0.01 gives 1, otherwise 0; fill missing entries with zero.',
            distance='L1: sum of absolute SNV differences.',
            backbone='RapidNJ on real cells only.',
            rooting='Fit zero-reference distances on every nonnegative-projected backbone edge; do not add a tip.',
            zero_evidence='Retain zero-evidence cells; identical profiles are unresolved.'),
        wls_rule=dict(weight_power=WLS_WEIGHT_POWER,
            weights='(1 + distance to zero reference) ** -weight_power',
            objective='Sum of weighted squared residuals between reference distances and pendant + backbone distances.',
            constraints='Edge coordinate in [0, edge length]; pendant >= 0.',
            solver='Evaluate the unconstrained solution and all three boundary optima on every edge.',
            branches='Project negative lengths to zero for fitting; preserve all backbone splits.'),
        state_revision_rule=dict(threshold=STATE_LOW, states=[0., 1.],
            rule='At or above threshold: 1; below threshold: 0. Missing stays NaN.',
            scope='All binary configurations use the same encoding before imputation and rooting.'),
        germline_rule=dict(min_callable_cells=MIN_CALLABLE_CELLS, min_pooled_ratio=MIN_POOLED_RATIO,
            min_E=MIN_E, rule='(n_callable >= 3) & ((pooled_ratio >= 0.9) | (E >= 10))',
            input='Raw ALT and depth before revision or imputation; callable if depth > 0.',
            evidence='q=1/(depth+1); 0.5*M(1) + 0.5*Beta(4,1) mixture; 64 Gauss-Legendre nodes.',
            scope='Target and simple baselines; exclude complete-genotype and native DENDRO.'),
        tree_site_qc=dict(min_total_depth=MIN_TOTAL_DEPTH,
            scope='All NJ configurations, including references; native DENDRO keeps its own QC.'),
        ablations=dict(geinj_L2='Use L2 for both cell and zero-reference distances.',
            geinj_mean_L1='Replace zero filling with mean imputation.',
            geinj_knn_L1='Replace zero filling with kNN imputation.',
            zero_filled_no_wls='Reuse the original zero-outgroup benchmark; changes both NJ backbone input and rooting.'),
        knn_rule=dict(n_neighbors=N_NEIGHBORS,
            neighbor_distance='Mean L1 on shared callable sites; identical for final L1 and L2.',
            imputation='Up to k callable donors per missing site; equal mean, continuous output; site mean if all donor distances are undefined.',
            ties='Original np.argpartition rule in the fixed randomized tip order.'),
        dendro_rule=dict(rscript=RSCRIPT_BINARY, r_library=str(DENDRO_R_LIBRARY),
            input='Observed ALT and depth; Z=1 if ALT>0, NA if depth=0; no true genotypes.',
            parameters='Installed FilterCellMutation, DENDRO.dist and DENDRO.cluster defaults.',
            tree='Native rooted cell clustering tree; no synthetic outgroup.',
            evaluation='At least 3 common cells retained by successful DENDRO variants; score all methods on identical tips.',
            retained_sites='Count after FilterCellMutation; internal distance filtering is in logs.',
            failures='Retained as attempted repeats; unavailable scores are NA.'),
        dendro_filtered_rule=dict(input='Raw ALT and depth at the same SNVs retained by the two main methods.',
            filtering='Reuse prepare_panel total-depth QC and germline filtering; no encoding or imputation.',
            downstream='Unchanged default DENDRO filtering, distances and clustering.'),
        tip_order_seed=TIP_ORDER_SEED,
        input_filters=dict(seeds=SEEDS, sample_sizes=SAMPLE_SIZES, probs=PROBS,
            germline_background='with_germline'),
        baseline_panel_rule='Shared raw-count germline filtering and total-depth QC; continuous-ratio configurations have no state revision, other observed-data NJ configurations use threshold states.',
        pairwise_rule=dict(L1='Mean absolute difference on shared callable sites.',
            L2='Square root of mean squared difference on shared callable sites.'),
        scenarios=SCENARIOS, save_trees=SAVE_TREES,
        reference_rule='True genotypes on all sampled tips and sites passing shared total-depth QC. Keep germline; no predicted filtering, revision or imputation. Founder: germline=1, somatic=0. Ideal input reference, not a strict performance bound.',
        control_rule='Read existing state-permuted observations; do not permute evaluation labels.',
        qgs_rule=dict(generation=QGS_N,
            rule='100 times the mean of (largest pure clade size - 1) / (group size - 1) across represented generation groups; singleton groups score zero.',
            scope='Use evaluated real tips only; exclude outgroup. Higher is better.'),
        triplet_rule='Exact normalized discordant rooted triplets from lineage truth; exclude outgroup.',
        nrf_rule='Symmetric difference / sum of nontrivial unrooted split counts; denominator zero gives zero. Exclude outgroup and use matched tips.',
        evaluation_priority='Triplet distance and unrooted nRF: lower is better; Qgs2: higher is better.',
        paired_tests=('Organized comparison families: see organized/plot_manifest.json.' if ORGANIZED else
            'Separate BH families for benchmark and ablation within each background and callability setting. The four target contrasts share the ablation family across endpoints and scenarios.'),
        callability_scenarios=sorted({path.relative_to(INPUT_DIR).parts[0] for path in paths}))
    if ORGANIZED:
        manifest.pop('qgs_rule')
        manifest['evaluation_priority'] = 'Triplet distance and unrooted nRF: lower is better.'
    (OUTPUT_DIR / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    environment = dict(python=sys.version, platform=platform.platform(), workers=WORKERS,
        numerical_threads_per_worker=1,
        thread_environment={name: os.environ[name] for name in THREAD_ENV},
        r_mc_cores=1,
        packages={name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'pandas',
            'anndata', 'biopython', 'threadpoolctl', 'matplotlib', 'psutil']})
    (OUTPUT_DIR / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    tasks = [(path, config, OUTPUT_DIR, rapidnj_binary) for path in paths for config in configs]
    all_results = run_tasks(tasks, workers=WORKERS)
    results = all_results[all_results.method.ne(DENDRO_METHOD)].copy()
    results.to_csv(RESULTS, index=False)
    matched_rows = []
    for dataset_id, rows in all_results.groupby('dataset_id', sort=True):
        matched = compare_dendro(rows)
        matched_rows.extend(matched)
        pd.DataFrame(matched).to_csv(OUTPUT_DIR / 'parts_dendro' / f'{dataset_id}.csv', index=False)
    matched = pd.DataFrame(matched_rows)
    matched.to_csv(DENDRO_RESULTS, index=False)
    all_results[all_results.method_id.eq(DENDRO_METHOD)].to_csv(OUTPUT_DIR / 'dendro_results.csv', index=False)
    all_results[all_results.method_id.eq(DENDRO_FILTERED_METHOD)].to_csv(
        OUTPUT_DIR / 'dendro_filtered_results.csv', index=False)
    if not SAVE_TREES:
        for tree_path in all_results.loc[all_results.technical_success, 'tree_path']:
            Path(tree_path).unlink()
    return results


def run_time_mem():
    paths = simulation_paths()
    configs = [config for config in configurations() if config['method_id'] in RESOURCE_METHODS]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'parts').mkdir(exist_ok=True)
    rapidnj_binary = rapidnj_executable()
    manifest = dict(input_dir=str(INPUT_DIR.resolve()),
        input_files=[dict(path=str(path.resolve()), sha256=sha256(path)) for path in paths],
        sources={Path(__file__).name: sha256(Path(__file__).resolve())},
        geinj_version=gei_nj.__version__, geinj_source=str(Path(gei_nj.__file__).resolve()),
        rapidnj_binary=str(rapidnj_binary), rapidnj_binary_sha256=sha256(rapidnj_binary),
        configs=configs, tip_order_seed=TIP_ORDER_SEED, save_trees=SAVE_TREES,
        input_filters=dict(seeds=SEEDS, sample_sizes=SAMPLE_SIZES, probs=PROBS,
            scenarios=SCENARIOS, germline_background='with_germline'),
        resource_rule=dict(scope='Two main configurations only; includes failed attempts.',
            total_runtime='Wall time from selective H5AD loading through preprocessing, inference and final Newick output. Loads only depth, ALT counts, cell/site names and required run metadata. Excludes worker/import startup, scoring, CSV summaries and plotting.',
            rapidnj_runtime='Only the RapidNJ subprocess wall time; excludes distances, PHYLIP writing and parsing.',
            peak_memory='Maximum sampled sum of fresh worker and descendant RSS in MiB; excludes parent. Includes Python baseline; shared pages may be counted more than once.',
            memory_interval_seconds=MEMORY_INTERVAL, workers=WORKERS,
            fresh_worker_per_dataset_method=True,
            task_order='All dataset-method tasks shuffled with TIP_ORDER_SEED.',
            resource_plots='Successful runs only; same linear scale per metric across methods.'),
        python=sys.version, platform=platform.platform(), numerical_threads_per_worker=1,
        thread_environment={name: os.environ[name] for name in THREAD_ENV},
        packages={name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'pandas',
            'anndata', 'biopython', 'threadpoolctl', 'matplotlib', 'psutil']})
    (OUTPUT_DIR / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    tasks = [(path, config, OUTPUT_DIR, rapidnj_binary) for path in paths for config in configs]
    results = run_tasks(tasks, workers=WORKERS)
    resource_columns = ['dataset_id', 'input_file', 'germline_background', 'callability_scenario', 'difference_factor',
        'scenario', 'prob', 'seed', 'method_id', 'method_label', 'method_group', 'final_metric',
        'n_cells', 'n_snv_sites', 'n_cells_retained', 'n_sites_retained',
        'total_runtime_seconds', 'rapidnj_runtime_seconds', 'rapidnj_calls', 'peak_memory_mib',
        'technical_success', 'status']
    results[resource_columns].to_csv(RESOURCE_RESULTS, index=False)
    if not SAVE_TREES:
        for tree_path in results.loc[results.technical_success, 'tree_path']:
            Path(tree_path).unlink()
    return results


def run_sensitivity():
    scenarios = SCENARIOS if ORGANIZED else ['main']
    paths = simulation_paths(scenarios=scenarios)
    configs, groups = sensitivity_configurations()
    SENSITIVITY_DIR.mkdir(parents=True, exist_ok=True)
    (SENSITIVITY_DIR / 'parts').mkdir(exist_ok=True)
    rapidnj_binary = rapidnj_executable()
    tasks = [(path, config, SENSITIVITY_DIR, rapidnj_binary) for path in paths for config in configs]
    manifest = dict(input_dir=str(INPUT_DIR.resolve()),
        input_files=[dict(path=str(path.resolve()), sha256=sha256(path)) for path in paths],
        sources={Path(__file__).name: sha256(Path(__file__).resolve())},
        geinj_version=gei_nj.__version__, geinj_source=str(Path(gei_nj.__file__).resolve()),
        rapidnj_binary=str(rapidnj_binary), rapidnj_binary_sha256=sha256(rapidnj_binary),
        configs=configs, parameter_groups=groups, target_method=TARGET_METHOD,
        input_filters=dict(seeds=SEEDS, sample_sizes=SAMPLE_SIZES, probs=PROBS,
            scenarios=scenarios, germline_background='with_germline'),
        protocol='One parameter at a time around the binary zero-filled L1/WLS default; reuse the default within sensitivity panels.',
        background_rule=('With-germline main and existing state-permuted inputs for all axes and filtering controls.'
            if ORGANIZED else 'With-germline main inputs for all axes and filtering controls.'),
        germline_rule=dict(min_callable_cells=MIN_CALLABLE_CELLS,
            callable='Raw depth > 0; independent of MIN_CALLABLE_DEPTH.',
            rule='(n_callable >= min_callable_cells) & ((pooled_ratio >= min_pooled_ratio) | (E >= min_E))',
            pooled_only='The same callable-cell requirement and pooled threshold; disable only the E branch.',
            evidence='q=1/(depth+1); 0.5*M(1) + 0.5*Beta(4,1) mixture; 64 Gauss-Legendre nodes.',
            evaluation='FP/FN after total-depth QC, fixed across germline comparisons. Truth never enters filtering.',
            denominators='FPR: eligible somatic sites; FNR: eligible germline sites. Zero denominator gives NA.'),
        endpoints=ENDPOINTS, qgs_generation=QGS_N, wls_endpoints=['triplet_distance', QGS_METRIC],
        wls_weights='(1 + distance to zero reference) ** -weight_power',
        paired_rule='Pair seeds within background, callability, scenario, probability and sample size; default minus setting. Negative favors the default for distances; positive favors the default for Qgs2.',
        paired_tests=('Organized comparison families: see ../organized/plot_manifest.json.' if ORGANIZED else
            'Separate BH family for each sensitivity axis, background and callability setting, across endpoints and conditions.'),
        tip_order_seed=TIP_ORDER_SEED, save_trees=SAVE_TREES, workers=WORKERS,
        python=sys.version, packages={name: importlib.metadata.version(name)
            for name in ['numpy', 'scipy', 'pandas', 'anndata', 'biopython', 'matplotlib']})
    if ORGANIZED:
        manifest.pop('qgs_generation')
        manifest['wls_endpoints'] = ['triplet_distance']
        manifest['paired_rule'] = ('Pair seeds within background, callability, scenario, probability and sample size; '
            'default minus setting. Negative favors the default for distances.')
    (SENSITIVITY_DIR / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    results = run_tasks(tasks, workers=WORKERS)
    results.to_csv(SENSITIVITY_RESULTS, index=False)
    if not SAVE_TREES:
        for tree_path in results.loc[results.technical_success, 'tree_path']:
            Path(tree_path).unlink()
    return results


METHODS = {
    'complete_genotype_L1': ('Complete-genotype / L1', '#000000'),
    'complete_genotype_L2': ('Complete-genotype / L2', '#000000'),
    TARGET_METHOD: ('GEI-NJ / Binary / Zero-filled / L1 / WLS', '#d54000'),
    'geinj_no_revision_L1': ('Continuous ALT ratio / Zero-filled / L1 / WLS', '#E69F00'),
    'zero_filled_no_revision_L1': ('Raw ratio / Zero-filled / L1', '#1B9E77'),
    'zero_filled_no_revision_L2': ('Raw ratio / Zero-filled / L2', '#1B9E77'),
    'zero_filled_no_wls': ('Binary / Zero-filled / L1 / No WLS', '#56B4E9'),
    'zero_filled_L2': ('Binary / Zero-filled / L2', '#56B4E9'),
    'knn_normalized_L1_final_L1': ('kNN (normalized L1) / L1', '#0072b2'),
    'knn_normalized_L1_final_L2': ('kNN (normalized L1) / L2', '#0072b2'),
    'mean_L1': ('Mean imputation / L1', '#CC79A7'),
    'mean_L2': ('Mean imputation / L2', '#CC79A7'),
    'pairwise_L1': ('Pairwise deletion / mean L1', '#666666'),
    'pairwise_L2': ('Pairwise deletion / RMS L2', '#666666'),
    DENDRO_FILTERED_METHOD: ('DENDRO / Filtered SNVs / ward.D', '#a24db8'),
    DENDRO_METHOD: ('DENDRO defaults / ward.D', '#6A3D9A'),
    'geinj_L2': ('Binary / Zero-filled / L2 / WLS', '#A6761D'),
    'geinj_mean_L1': ('Binary / Mean imputation / WLS', '#CC79A7'),
    'geinj_knn_L1': ('Binary / kNN imputation / WLS', '#0072b2'),
}
REFERENCES = ['complete_genotype_L1', 'complete_genotype_L2']
SR_ABLATIONS = ['geinj_L2', 'geinj_mean_L1', 'geinj_knn_L1', 'zero_filled_no_wls']
MAIN_METHODS = REFERENCES + [
    TARGET_METHOD, 'geinj_no_revision_L1', 'zero_filled_no_revision_L1', 'zero_filled_no_revision_L2',
    'zero_filled_no_wls', 'zero_filled_L2',
    'knn_normalized_L1_final_L1', 'knn_normalized_L1_final_L2',
    'mean_L1', 'mean_L2', 'pairwise_L1', 'pairwise_L2',
]
COMPARATORS = [method for method in MAIN_METHODS
               if method not in REFERENCES and method != TARGET_METHOD]
CONTROL_METHODS = list(MAIN_METHODS)
MATCHED_METHODS = MAIN_METHODS[:-2] + [DENDRO_FILTERED_METHOD, DENDRO_METHOD] + MAIN_METHODS[-2:]

ENDPOINTS = {
    'triplet_distance': 'Normalized rooted triplet distance',
    'unrooted_nrf': 'Normalized unrooted RF distance',
}
if not ORGANIZED:
    ENDPOINTS[QGS_METRIC] = f'{QGS_METRIC} (%)'
ENDPOINT_DIRECTIONS = {
    'triplet_distance': -1,
    'unrooted_nrf': -1,
    QGS_METRIC: 1,
}
PAIRED_ENDPOINT_LABELS = {
    'triplet_distance': 'Δ normalized rooted triplet distance',
    'unrooted_nrf': 'Δ normalized unrooted RF distance',
    QGS_METRIC: f'Δ {QGS_METRIC} (percentage points)',
}
PAIR_KEYS = ['scenario', 'prob', 'sample_size']
EXPERIMENT_KEYS = ['germline_background', 'callability_scenario']
CALLABILITY_COLORS = {'weak': '#56B4E9', 'empirical': '#009E73'}
# Benchmark statuses that confirm no NJ tree could be built or parsed.
# Tip validation, rooting, scoring, and unknown failures may occur after a
# tree has been generated, so they retain the generic N/A label.
NJ_CONSTRUCTION_FAILURE_STATUSES = {
    'no_observed_sites', 'undefined_pairwise_distance', 'degenerate_distance',
    'rapidnj_execution_failed', 'rapidnj_parse_failed',
}


def read_results(path, scenario=None):
    data = pd.read_csv(path)
    missing_endpoints = [endpoint for endpoint in ENDPOINTS if endpoint not in data.columns]
    if missing_endpoints:
        raise ValueError(
            'Missing required benchmark metric columns: ' + ', '.join(missing_endpoints)
            + '. Run the benchmark to generate all metrics before plotting.')
    if data.duplicated(EXPERIMENT_KEYS + PAIR_KEYS + ['seed', 'method_id']).any():
        raise ValueError('Repeated seed-method rows within a condition.')
    if data.technical_success.dtype != bool:
        data['technical_success'] = data.technical_success.map(
            {'True': True, 'False': False, 'true': True, 'false': False})
    if scenario is not None:
        data = data[data.scenario.eq(scenario)].copy()
    return data


def summarize(data, endpoint):
    rows = []
    keys = PAIR_KEYS + ['method_id']
    for key, group in data.groupby(keys, sort=False):
        success = group[group.technical_success]
        values = success[endpoint].dropna().to_numpy(dtype=float)
        n = len(values)
        mean = float(values.mean()) if n else np.nan
        half = float(t.ppf(.975, n - 1) * values.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        no_nj_tree = (len(group) > 0 and group.technical_success.eq(False).all()
                      and 'status' in group.columns
                      and group['status'].isin(NJ_CONSTRUCTION_FAILURE_STATUSES).all())
        rows.append(dict(zip(keys, key), n_attempted=len(group), n_success=len(success),
            no_nj_tree=bool(no_nj_tree), mean=mean, ci_low=mean-half, ci_high=mean+half))
    return pd.DataFrame(rows).set_index(keys)


def state_comparison_data(data):
    keys = EXPERIMENT_KEYS + ['prob', 'sample_size', 'seed']
    control = data[data.scenario.eq('state_permutation')]
    matched = control[keys].drop_duplicates()
    main = data[data.scenario.eq('main')].merge(matched, on=keys, how='inner')
    return pd.concat([main, control], ignore_index=True)


def paired_effects(data, target_method=TARGET_METHOD, comparators=COMPARATORS, endpoints=None):
    rows, per_seed = [], []
    keys = EXPERIMENT_KEYS + PAIR_KEYS
    for key, group in data.groupby(keys, sort=True):
        for endpoint in (ENDPOINTS if endpoints is None else endpoints):
            columns = ['seed', 'technical_success', endpoint]
            target = group[group.method_id.eq(target_method)][columns]
            for comparator in comparators:
                other = group[group.method_id.eq(comparator)][columns]
                joined = target.merge(other, on='seed', how='outer',
                                      suffixes=('_target', '_comparator'), indicator=True)
                both = joined['_merge'].eq('both')
                target_ok = joined.technical_success_target.eq(True)
                other_ok = joined.technical_success_comparator.eq(True)
                target_values = joined[f'{endpoint}_target'].to_numpy(dtype=float)
                other_values = joined[f'{endpoint}_comparator'].to_numpy(dtype=float)
                paired = both & target_ok & other_ok & np.isfinite(target_values) & np.isfinite(other_values)
                delta = target_values[paired] - other_values[paired]
                test_delta = np.round(delta, 12)
                n = len(delta)
                direction = ENDPOINT_DIRECTIONS[endpoint]
                improved = int(np.count_nonzero(direction * test_delta > 0))
                tied = int(np.count_nonzero(test_delta == 0))
                worse = int(np.count_nonzero(direction * test_delta < 0))
                if not n:
                    statistic, p_value = np.nan, np.nan
                elif not np.any(test_delta):
                    statistic, p_value = 0., 1.
                else:
                    test = wilcoxon(test_delta, alternative='two-sided',
                                    zero_method='wilcox', method='auto')
                    statistic, p_value = float(test.statistic), float(test.pvalue)
                mean = float(delta.mean()) if n else np.nan
                half = float(t.ppf(.975, n-1) * delta.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
                context = dict(zip(keys, key), endpoint=endpoint,
                               target_method=target_method, comparator_method=comparator)
                rows.append(dict(context,
                    n_target_attempted=len(target), n_comparator_attempted=len(other),
                    n_attempted_both=int(both.sum()), n_target_success=int(target_ok.sum()),
                    n_comparator_success=int(other_ok.sum()), n_paired=n,
                    n_excluded=int(both.sum())-n,
                    median_difference=float(np.median(delta)) if n else np.nan,
                    mean_difference=mean, mean_ci_low=mean-half, mean_ci_high=mean+half,
                    n_improved=improved, n_tied=tied, n_worse=worse,
                    fraction_improved=improved/n if n else np.nan,
                    fraction_tied=tied/n if n else np.nan,
                    fraction_worse=worse/n if n else np.nan,
                    wilcoxon_statistic=statistic, p_raw=p_value))
                for i, item in joined.iterrows():
                    is_paired = bool(paired.loc[i])
                    difference = target_values[i]-other_values[i] if is_paired else np.nan
                    rounded = float(np.round(difference, 12)) if is_paired else np.nan
                    outcome = ('improved' if direction*rounded > 0 else
                               'worse' if direction*rounded < 0 else 'tied') if is_paired else 'excluded'
                    per_seed.append(dict(context, seed=int(item.seed),
                        attempted_both=bool(both.loc[i]), target_success=bool(target_ok.loc[i]),
                        comparator_success=bool(other_ok.loc[i]), paired=is_paired,
                        target_value=target_values[i], comparator_value=other_values[i],
                        difference=difference, difference_for_test=rounded, outcome=outcome))
    summary = pd.DataFrame(rows)
    summary['p_bh'] = np.nan
    summary['bh_family_size'] = 0
    for _, family in summary.groupby(EXPERIMENT_KEYS, sort=False):
        valid = family.index[family.p_raw.notna()]
        if len(valid):
            summary.loc[valid, 'p_bh'] = false_discovery_control(summary.loc[valid, 'p_raw'], method='bh')
        summary.loc[family.index, 'bh_family_size'] = len(valid)
    return summary, pd.DataFrame(per_seed)


def _facets(data, control_comparison=False):
    if control_comparison:
        control = data[data.scenario.eq('state_permutation')]
        conditions = control[['prob', 'sample_size']].drop_duplicates()
        conditions = conditions.sort_values(['prob', 'sample_size'], ascending=[False, True])
        return [[(scenario, prob, size) for scenario in ['main', 'state_permutation']]
                for prob, size in conditions.itertuples(index=False, name=None)]
    rates = sorted(data.prob.unique(), reverse=True)
    sizes = sorted(data.sample_size.unique())
    scenario = data.scenario.iloc[0]
    return [[(scenario, prob, size) for size in sizes] for prob in rates]


def _method_positions(methods, same_group_gap=1.0):
    positions = [0.]
    for previous, current in zip(methods, methods[1:]):
        same_group = previous.rsplit('_L', 1)[0] == current.rsplit('_L', 1)[0]
        positions.append(positions[-1] + (same_group_gap if same_group else 1.25))
    return np.asarray(positions)


def _grid(facets, methods, control_comparison=False, paired=False, matched=False):
    positions = _method_positions(methods)
    nrows, ncols = len(facets), len(facets[0])
    # Scale each plotting area relative to the original layout, retaining
    # the physical margins and spacing between panels at the same font sizes.
    original_positions = _method_positions(methods, same_group_gap=.75)
    original_width = (7.5 + 6.1*ncols) if paired else (6.4 + 5.3*ncols)
    original_height = 1.6 + (.29*(original_positions[-1]+1)+.85)*nrows
    right_margin = 1.75 if paired else .9
    original_wspace = .48 if paired else .23
    original_panel_width = (original_width-4.6-right_margin) / (ncols+original_wspace*(ncols-1))
    original_panel_height = (original_height-1.6) / (nrows+.27*(nrows-1))
    width = 4.95 + right_margin + original_panel_width*(.5*ncols+original_wspace*(ncols-1))
    height = 1.6 + original_panel_height*(.75*.75*nrows+.27*(nrows-1))
    bottom_margin = 1.2 if matched else .95
    top_margin = 1.15 if matched or paired else .65
    height += bottom_margin + top_margin - 1.6
    height = max(height, bottom_margin + top_margin + .8*nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height),
                             squeeze=False, sharex=paired, sharey=paired)
    fig.subplots_adjust(left=4.95/width, right=1.-(1.75 if paired else .9)/width,
                        bottom=bottom_margin/height, top=1.-top_margin/height,
                        wspace=original_wspace/.5, hspace=.27/(.75*.75))
    for r, row in enumerate(facets):
        for c, (scenario, prob, size) in enumerate(row):
            ax = axes[r, c]
            exponent = int(round(np.log10(prob)))
            title = rf'Probability setting: $1\times10^{{{exponent}}}$'
            subtitle = f'{int(size)} cells'
            if matched:
                subtitle = f'{int(size)} input cells'
            if control_comparison:
                subtitle = ('Main' if scenario == 'main' else 'State-permuted') + '   |   ' + subtitle
            title += '\n' + subtitle
            ax.set_title(title, fontsize=14, pad=12, fontweight='medium')
            ax.set_ylim(positions[-1]+.55, -.65)
            ax.set_yticks(positions)
            ax.tick_params(axis='y', length=0, pad=11, labelsize=12)
            ax.tick_params(axis='x', labelsize=11, length=3, color='#999999')
            ax.grid(axis='x', color='#DFE3E8', linewidth=.7, zorder=0)
            ax.set_axisbelow(True)
            for side in ['top', 'right', 'left']:
                ax.spines[side].set_visible(False)
            ax.spines['bottom'].set_color('#999999')
            if c:
                ax.tick_params(labelleft=False)
    return fig, axes


def _save(fig, output, stem, dpi):
    fig.savefig(output / f'{stem}.pdf', dpi=dpi, facecolor='white')
    plt.close(fig)


def plot_accuracy(data, endpoint, methods, output, stem, dpi,
                  control_comparison=False, matched=False):
    summary = summarize(data, endpoint)
    metric = ENDPOINTS[endpoint]
    positions = _method_positions(methods)
    facets = _facets(data, control_comparison)
    fig, axes = _grid(facets, methods, control_comparison, matched=matched)
    if matched:
        fig.suptitle('Scores on shared DENDRO-retained cells', fontsize=14, y=1.-.08/fig.get_figheight())
    labels = [METHODS[method][0] for method in methods]
    for r, row_facets in enumerate(facets):
        for c, (scenario, prob, size) in enumerate(row_facets):
            ax = axes[r, c]
            index = summary.index
            values = summary[(index.get_level_values('scenario') == scenario)
                & (index.get_level_values('prob') == prob)
                & (index.get_level_values('sample_size') == size)
                & index.get_level_values('method_id').isin(methods)
                & summary.n_success.ge(5)][
                    ['mean', 'ci_low', 'ci_high']].to_numpy(dtype=float)
            finite_values = values[np.isfinite(values)]
            if finite_values.size:
                lower, upper = float(finite_values.min()), float(finite_values.max())
            else:
                lower, upper = (0., 100.) if endpoint == QGS_METRIC else (0., 1.)
            ax.set_xscale('linear')
            padding = .025 * max(upper-lower, .01)
            limits = (lower-padding, upper+padding)
            if ENDPOINT_DIRECTIONS[endpoint] < 0:
                limits = limits[::-1]
            ax.set_xlim(*limits)
            if not c:
                ax.set_yticklabels(labels)
            if r == len(facets)-1:
                ax.set_xlabel(metric, fontsize=13, labelpad=9)
            groups = []
            for i, method in enumerate(methods):
                if i and method.rsplit('_L', 1)[0] == methods[i-1].rsplit('_L', 1)[0]:
                    groups[-1].append(i)
                else:
                    groups.append([i])
            for group, indices in enumerate(groups):
                if group % 2 == 0:
                    ax.axhspan(positions[indices[0]]-.45, positions[indices[-1]]+.45,
                               color='#F4F6F8', zorder=-1)
            for y, method in zip(positions, methods):
                key = (scenario, prob, size, method)
                if key not in summary.index:
                    ax.text(.5, y, 'Not run', transform=ax.get_yaxis_transform(),
                            ha='center', va='center', fontsize=11, color='#777777')
                    continue
                result = summary.loc[key]
                color = METHODS[method][1]
                marker = ('D' if method in {DENDRO_METHOD, DENDRO_FILTERED_METHOD}
                          else 's' if method.endswith('L2') else 'o')
                if result.n_success >= 5 and np.isfinite(result['mean']):
                    if np.isfinite(result.ci_low):
                        ax.plot([result.ci_low, result.ci_high], [y, y], color=color,
                                linewidth=2.2, solid_capstyle='round', zorder=2)
                    ax.plot(result['mean'], y, marker=marker, color=color,
                            markersize=6.7, markeredgecolor='white',
                            markeredgewidth=.65, linestyle='none', zorder=3, clip_on=not matched)
                else:
                    missing_label = ('Less than 5 NJ tree constructed' if result.n_success < 5 else
                        'No evaluable DENDRO tree' if method in {DENDRO_METHOD, DENDRO_FILTERED_METHOD} else
                        'No NJ tree constructed' if result.no_nj_tree else 'N/A')
                    ax.text(.5, y, missing_label, transform=ax.get_yaxis_transform(),
                            ha='center', va='center', color='#777777', fontsize=11)
                ax.text(1.035, y, f'{int(result.n_success)}/{int(result.n_attempted)}',
                        transform=ax.get_yaxis_transform(), va='center', ha='left',
                        fontsize=10.5, color='#444444', clip_on=False)
            ax.text(1.035, 1.015, 'n', transform=ax.transAxes,
                    va='bottom', ha='left', fontsize=11, color='#444444')
    handles = []
    if any(method.endswith('L1') for method in methods):
        handles.append(Line2D([], [], marker='o', color='#555555', linestyle='none',
            label='NJ distance: L1', markersize=6))
    if any(method.endswith('L2') for method in methods):
        handles.append(Line2D([], [], marker='s', color='#555555', linestyle='none',
            label='NJ distance: L2', markersize=6))
    if DENDRO_METHOD in methods:
        handles.append(Line2D([], [], marker='D', color=METHODS[DENDRO_METHOD][1],
            linestyle='none', label='DENDRO: ward.D', markersize=6))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.01),
               ncol=len(handles), fontsize=11, frameon=False, handletextpad=.45, borderaxespad=0.)
    _save(fig, output, stem, dpi)


def plot_paired(summary, per_seed, scenario, endpoint, output, stem, dpi,
                target_method=TARGET_METHOD, comparators=COMPARATORS, matched=False):
    selected = summary[summary.scenario.eq(scenario) & summary.endpoint.eq(endpoint)
                       & summary.target_method.eq(target_method)]
    callability = selected.callability_scenario.iloc[0]
    background = selected.germline_background.iloc[0].replace('_', ' ').capitalize()
    point_color = CALLABILITY_COLORS[callability]
    points = per_seed[per_seed.scenario.eq(scenario) & per_seed.endpoint.eq(endpoint)
                      & per_seed.target_method.eq(target_method) & per_seed.paired]
    facets = _facets(selected)
    positions = _method_positions(comparators)
    fig, axes = _grid(facets, comparators, paired=True, matched=matched)
    title = f'{callability.capitalize()} callability | {background}'
    title += ' | Main' if scenario == 'main' else ' | State-permuted'
    if matched:
        title += ' | DENDRO-retained cells'
    fig.suptitle(title, fontsize=14, y=1.-.08/fig.get_figheight())
    indexed = selected.set_index(PAIR_KEYS + ['comparator_method'])
    values = points.difference.to_numpy(dtype=float)
    bound = float(np.max(np.abs(values))) if len(values) else 0.
    if bound == 0:
        bound = .01
    limits = (-1.12*bound, 1.12*bound)
    if ENDPOINT_DIRECTIONS[endpoint] < 0:
        limits = limits[::-1]
    label = PAIRED_ENDPOINT_LABELS[endpoint]
    rng = np.random.default_rng(PLOT_SEED)
    for r, row_facets in enumerate(facets):
        for c, (_, prob, size) in enumerate(row_facets):
            ax = axes[r, c]
            ax.set_xlim(*limits)
            ax.axvline(0., color='#555555', linewidth=1., linestyle='--', zorder=1)
            if not c:
                ax.set_yticklabels(['vs ' + METHODS[m][0] for m in comparators])
            if r == len(facets)-1:
                ax.set_xlabel(label, fontsize=13, labelpad=9)
            for y, comparator in zip(positions, comparators):
                result = indexed.loc[(scenario, prob, size, comparator)]
                subset = points[points.prob.eq(prob) & points.sample_size.eq(size)
                                & points.comparator_method.eq(comparator)].sort_values('seed')
                if result.n_paired >= 5:
                    jitter = rng.uniform(-.16, .16, len(subset))
                    ax.scatter(subset.difference, y+jitter, s=8, color=point_color, alpha=.55,
                               linewidths=0, zorder=2)
                    ax.plot(result.median_difference, y, marker='D', color=METHODS[target_method][1],
                            markersize=7, markeredgecolor='white', markeredgewidth=.65,
                            linestyle='none', zorder=3)
                else:
                    ax.text(.5, y, 'Less than 5 NJ tree constructed', transform=ax.get_yaxis_transform(),
                            ha='center', va='center', color='#777777', fontsize=11)
                p_label = f'{result.p_bh:.2g}' if result.n_paired >= 5 and np.isfinite(result.p_bh) else 'NA'
                ax.text(1.035, y, f'{int(result.n_paired)}   |   {p_label}',
                        transform=ax.get_yaxis_transform(), va='center', ha='left',
                        fontsize=10.5, color='#444444', clip_on=False)
            ax.text(1.035, 1.015, 'n   |   pBH', transform=ax.transAxes,
                    va='bottom', ha='left', fontsize=11, color='#444444')
    handles = [Line2D([], [], marker='o', color=point_color, linestyle='none',
                      label=f'Paired replicate ({callability.capitalize()})', markersize=4),
               Line2D([], [], marker='D', color=METHODS[target_method][1], linestyle='none',
                      label=f'Median difference ({METHODS[target_method][0]} − comparator)', markersize=6)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .01),
               ncol=2, fontsize=11, frameon=False, handletextpad=.45, borderaxespad=0.)
    _save(fig, output, stem, dpi)


def setup_plot_style():
    font = 'Arial' if any(f.name == 'Arial' for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
    font_path = font_manager.findfont(font_manager.FontProperties(family=font), fallback_to_default=False)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    print(f'Font: {font_name}; Arial: {font_name == "Arial"}; file: {font_path}')
    plt.rcParams.update({'font.family':font, 'mathtext.fontset':'custom',
        'mathtext.rm':font, 'mathtext.it':f'{font}:italic', 'mathtext.bf':f'{font}:bold',
        'mathtext.sf':font, 'mathtext.cal':font, 'mathtext.default':'regular',
        'pdf.fonttype':42, 'ps.fonttype':42,
        'axes.titlelocation':'center', 'savefig.facecolor':'white', 'axes.unicode_minus':False})


def plot_results():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()
    data = read_results(RESULTS)
    data = data[data.scenario.isin(['main', 'state_permutation'])
                & data.method_id.isin(METHODS)].copy()
    summaries, differences = [], []
    for (background, callability), selected in data.groupby(EXPERIMENT_KEYS, sort=False):
        figure_dir = FIGURE_DIR / background / callability
        figure_dir.mkdir(parents=True, exist_ok=True)
        main_data = selected[selected.scenario.eq('main')]
        control_data = state_comparison_data(selected)
        summary, per_seed = paired_effects(selected)
        summaries.append(summary)
        differences.append(per_seed)
        for endpoint in ENDPOINTS:
            numbers = {'triplet_distance': (1, 3, 5, 7), 'unrooted_nrf': (2, 4, 6, 8),
                       QGS_METRIC: (27, 28, 29, 30)}[endpoint]
            plot_accuracy(main_data, endpoint, MAIN_METHODS, figure_dir,
                          f'{numbers[0]:02d}_main_{endpoint}', DPI)
            plot_accuracy(control_data, endpoint, CONTROL_METHODS, figure_dir,
                          f'{numbers[1]:02d}_state_control_{endpoint}', DPI, control_comparison=True)
            plot_paired(summary, per_seed, 'main', endpoint, figure_dir,
                        f'{numbers[2]:02d}_paired_{endpoint}', DPI)
            plot_paired(summary, per_seed, 'state_permutation', endpoint, figure_dir,
                        f'{numbers[3]:02d}_control_paired_{endpoint}', DPI)
    pd.concat(summaries, ignore_index=True).to_csv(OUTPUT_DIR / 'paired_effect_summary.csv', index=False)
    pd.concat(differences, ignore_index=True).to_csv(OUTPUT_DIR / 'paired_differences.csv', index=False)
    plot_dendro_results()
    plot_shared_ablations()
    print(f'Saved benchmark figures to {FIGURE_DIR}; statistics to {OUTPUT_DIR}')


def summarize_dendro(data):
    rows = []
    for key, group in data.groupby(PAIR_KEYS, sort=True):
        qc = group[group.dendro_cell_retained_fraction.notna()]
        rows.append(dict(zip(PAIR_KEYS, key), n_attempted=len(group),
            n_evaluable=int(group.technical_success.sum()), n_qc_completed=len(qc),
            evaluable_tree_percent=100. * group.technical_success.mean(),
            mean_cell_retained_percent=100. * qc.dendro_cell_retained_fraction.mean(),
            mean_cells_retained=qc.dendro_n_cells_retained.mean(),
            median_sites_retained=qc.dendro_n_sites_retained.median()))
    return pd.DataFrame(rows)


def plot_dendro_retention(summary, output):
    scenarios = [name for name in SCENARIOS if name in set(summary.scenario)]
    nrows = max(int(summary.scenario.eq(name).sum()) for name in scenarios)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(7.5*len(scenarios), 1.8+.65*nrows),
                             squeeze=False)
    fig.subplots_adjust(left=.12, right=.92, bottom=.17, top=.90, wspace=.65)
    for c, scenario in enumerate(scenarios):
        ax = axes[0, c]
        selected = summary[summary.scenario.eq(scenario)].sort_values(
            ['prob', 'sample_size'], ascending=[False, True])
        positions = np.arange(len(selected))
        for y, row in zip(positions, selected.itertuples()):
            ax.barh(y-.17, row.evaluable_tree_percent, height=.30,
                    color=METHODS[DENDRO_METHOD][1])
            if np.isfinite(row.mean_cell_retained_percent):
                ax.barh(y+.17, row.mean_cell_retained_percent, height=.30, color='#0072b2')
            else:
                ax.text(2., y+.17, 'N/A', va='center', fontsize=10)
            ax.text(1.025, y-.17, f'{row.n_evaluable}/{row.n_attempted}',
                    transform=ax.get_yaxis_transform(), va='center', fontsize=10)
            ax.text(1.025, y+.17, f'{row.n_qc_completed}/{row.n_attempted}',
                    transform=ax.get_yaxis_transform(), va='center', fontsize=10)
        labels = [rf'$1\times10^{{{int(round(np.log10(row.prob)))}}}$ | {int(row.sample_size)} cells'
                  for row in selected.itertuples()]
        ax.set_yticks(positions, labels)
        ax.set_ylim(len(selected)-.5, -.6)
        ax.set_xlim(0., 100.)
        ax.set_xlabel('Percent (%)', fontsize=12)
        ax.set_title('Main' if scenario == 'main' else 'State-permuted', fontsize=14)
        ax.text(1.025, 1.015, 'n', transform=ax.transAxes, fontsize=11)
        ax.grid(axis='x', color='#DFE3E8', linewidth=.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y', length=0, labelsize=11)
        for side in ['top', 'right', 'left']:
            ax.spines[side].set_visible(False)
    handles = [Line2D([], [], color=METHODS[DENDRO_METHOD][1], linewidth=7,
                   label='Evaluable trees (at least 3 cells)'),
               Line2D([], [], color='#0072b2', linewidth=7,
                   label='Mean cells retained (completed QC)')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .04),
               ncol=2, frameon=False, fontsize=11)
    fig.text(.5, .015, 'n: evaluable / attempted repeats; QC completed / attempted repeats',
             ha='center', fontsize=10)
    _save(fig, output, '17_dendro_retention', DPI)


def plot_dendro_results():
    data = read_results(DENDRO_RESULTS)
    native = read_results(OUTPUT_DIR / 'dendro_results.csv')
    retentions = []
    for (background, callability), selected in data.groupby(EXPERIMENT_KEYS, sort=False):
        figure_dir = FIGURE_DIR / background / callability
        figure_dir.mkdir(parents=True, exist_ok=True)
        main_data = selected[selected.scenario.eq('main')]
        control_data = state_comparison_data(selected)
        for endpoint in ENDPOINTS:
            numbers = {'triplet_distance': (9, 11), 'unrooted_nrf': (10, 12),
                       QGS_METRIC: (31, 32)}[endpoint]
            plot_accuracy(main_data, endpoint, MATCHED_METHODS, figure_dir,
                          f'{numbers[0]:02d}_dendro_matched_main_{endpoint}', DPI, matched=True)
            plot_accuracy(control_data, endpoint, MATCHED_METHODS, figure_dir,
                          f'{numbers[1]:02d}_dendro_matched_state_control_{endpoint}', DPI,
                          control_comparison=True, matched=True)
        retention = summarize_dendro(native[native.germline_background.eq(background)
            & native.callability_scenario.eq(callability)])
        retentions.append(retention.assign(germline_background=background, callability_scenario=callability))
        plot_dendro_retention(retention, figure_dir)
    pd.concat(retentions, ignore_index=True).to_csv(OUTPUT_DIR / 'dendro_retention_summary.csv', index=False)


def plot_shared_ablations():
    data = read_results(RESULTS)
    methods = [TARGET_METHOD] + SR_ABLATIONS
    data = data[data.scenario.isin(['main', 'state_permutation'])
                & data.method_id.isin(methods)].copy()
    summaries, differences = [], []
    for (background, callability), selected in data.groupby(EXPERIMENT_KEYS, sort=False):
        figure_dir = FIGURE_DIR / background / callability
        figure_dir.mkdir(parents=True, exist_ok=True)
        main_data = selected[selected.scenario.eq('main')]
        control_data = state_comparison_data(selected)
        summary, per_seed = paired_effects(selected, comparators=SR_ABLATIONS)
        summaries.append(summary)
        differences.append(per_seed)
        for endpoint in ENDPOINTS:
            numbers = {'triplet_distance': (19, 21, 23, 25), 'unrooted_nrf': (20, 22, 24, 26),
                       QGS_METRIC: (33, 34, 35, 36)}[endpoint]
            plot_accuracy(main_data, endpoint, methods, figure_dir,
                          f'{numbers[0]:02d}_shared_ablation_main_{endpoint}', DPI)
            plot_accuracy(control_data, endpoint, methods, figure_dir,
                          f'{numbers[1]:02d}_shared_ablation_state_control_{endpoint}', DPI,
                          control_comparison=True)
            for number, scenario in zip(numbers[2:], ['main', 'state_permutation']):
                plot_paired(summary, per_seed, scenario, endpoint, figure_dir,
                            f'{number:02d}_shared_ablation_{scenario}_paired_{endpoint}', DPI,
                            comparators=SR_ABLATIONS)
    pd.concat(summaries, ignore_index=True).to_csv(
        OUTPUT_DIR / 'shared_ablation_paired_effect_summary.csv', index=False)
    pd.concat(differences, ignore_index=True).to_csv(
        OUTPUT_DIR / 'shared_ablation_paired_differences.csv', index=False)


def plot_resources():
    setup_plot_style()
    resources = pd.read_csv(RESOURCE_RESULTS)
    metrics = [
        ('total_runtime_seconds', 'Total runtime (s)'),
        ('peak_memory_mib', 'Peak memory (MiB)'),
    ]
    # Rocket, dark-to-red range (0.0-0.6).
    colors = ['#03051a', '#1a102a', '#33183c', '#4c1d4b', '#681f55',
              '#841e5a', '#a11a5b', '#bd1655', '#d62449', '#e83f3f']
    cmap = LinearSegmentedColormap.from_list('rocket_red', colors)
    for background, data in resources.groupby('germline_background', sort=False):
        data = data[data.technical_success.eq(True)].copy()
        figure_dir = FIGURE_DIR / background
        figure_dir.mkdir(parents=True, exist_ok=True)
        if len(data):
            norms = {}
            for metric, _ in metrics:
                values = data[metric]
                norms[metric] = Normalize(vmin=values.min(), vmax=values.max())
            x_min, x_max = data.n_cells.min(), data.n_cells.max()
            y_min, y_max = data.n_snv_sites.min(), data.n_snv_sites.max()
            x_padding = .12 * max(x_max-x_min, 1.)
            y_padding = .08 * max(y_max-y_min, 1.)
        with PdfPages(figure_dir / '18_runtime_memory.pdf') as pdf:
            fig, axes = plt.subplots(2, 2, figsize=(11., 9.0))
            fig.subplots_adjust(left=.09, right=.92, bottom=.08, top=.88, wspace=.42, hspace=.50)
            fig.suptitle(f'GEI-NJ / Zero-filled / L1 / WLS | {background.replace("_", " ").capitalize()}',
                         fontsize=15, y=.97)
            for method, method_axes in zip(RESOURCE_METHODS, axes):
                method_label = 'Binary' if method == TARGET_METHOD else 'Continuous'
                selected = data[data.method_id.eq(method)]
                for ax, (metric, label) in zip(method_axes, metrics):
                    ax.set_title(f'{method_label} | {label}', fontsize=12, pad=10)
                    if selected.empty:
                        ax.text(.5, .5, 'No successful runs', transform=ax.transAxes,
                                ha='center', va='center', fontsize=11)
                        ax.set_axis_off()
                        continue
                    points_data = selected.sort_values(metric)
                    values = points_data[metric].to_numpy(dtype=float)
                    norm = norms[metric]
                    sizes = 18. + 142. * norm(values)
                    points = ax.scatter(points_data.n_cells, points_data.n_snv_sites,
                        c=values, s=sizes, cmap=cmap, norm=norm,
                        edgecolors='white', linewidths=.25, zorder=3)
                    ax.set_xlabel('Number of cells', fontsize=11)
                    ax.set_ylabel('Number of SNV sites', fontsize=11)
                    ax.set_xscale('linear')
                    ax.set_yscale('linear')
                    ax.set_xticks(sorted(data.n_cells.unique()))
                    ax.ticklabel_format(axis='both', style='plain', useOffset=False)
                    ax.set_xlim(x_min-x_padding, x_max+x_padding)
                    ax.set_ylim(y_min-y_padding, y_max+y_padding)
                    ax.grid(color='#DFE3E8', linewidth=.7, zorder=0)
                    for side in ['top', 'right']:
                        ax.spines[side].set_visible(False)
                    colorbar = fig.colorbar(points, ax=ax, fraction=.046, pad=.035)
                    colorbar.set_label('MiB' if metric == 'peak_memory_mib' else 'Seconds', fontsize=10)
                    colorbar.ax.tick_params(labelsize=9)
            pdf.savefig(fig, dpi=DPI, facecolor='white')
            plt.close(fig)


def summarize_sensitivity(data, endpoints, filtering=False):
    rows = []
    keys = EXPERIMENT_KEYS + PAIR_KEYS + ['method_id']
    for key, group in data.groupby(keys, sort=True):
        success = group[group.technical_success]
        for endpoint in endpoints:
            source = group if filtering else success
            values = source[endpoint].dropna().to_numpy(dtype=float)
            n = len(values)
            mean = float(values.mean()) if n else np.nan
            half = float(t.ppf(.975, n-1) * values.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            rows.append(dict(zip(keys, key), endpoint=endpoint,
                n_attempted=len(group), n_success=len(success), n_evaluated=n,
                mean=mean, ci_low=mean-half, ci_high=mean+half))
    return pd.DataFrame(rows)


def plot_sensitivity_accuracy(summary, settings, parameter, labels, output, stem):
    methods = [method for method, _ in settings]
    positions = _method_positions(methods)
    facets = _facets(summary)
    fig, axes = _grid(facets, methods)
    panel_height = .7*len(methods)+.35
    fig.set_figheight(2.45+panel_height*len(facets)+len(facets)-1)
    fig.subplots_adjust(top=1.-1.15/fig.get_figheight(), bottom=1.3/fig.get_figheight(),
                        hspace=1./panel_height)
    first = summary.iloc[0]
    title = (f'{parameter} | '
             f'{first.germline_background.replace("_", " ")} | {first.scenario.replace("_", " ")}')
    fig.suptitle(title, fontsize=13, y=1.-.08/fig.get_figheight())
    endpoint = first.endpoint
    indexed = summary.set_index(PAIR_KEYS + ['method_id', 'callability_scenario'])
    for r, row_facets in enumerate(facets):
        for c, (scenario, prob, size) in enumerate(row_facets):
            ax = axes[r, c]
            subset = summary[summary.prob.eq(prob) & summary.sample_size.eq(size)]
            if endpoint in ENDPOINTS:
                subset = subset[subset.n_success.ge(5)]
            values = subset[['mean', 'ci_low', 'ci_high']].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            lower, upper = ((float(finite.min()), float(finite.max())) if finite.size
                            else (0., 100.) if endpoint == QGS_METRIC else (0., 1.))
            padding = .06 * max(upper-lower, .01)
            limits = (lower-padding, upper+padding)
            if ENDPOINT_DIRECTIONS.get(endpoint, -1) < 0:
                limits = limits[::-1]
            ax.set_xlim(*limits)
            if not c:
                ax.set_yticklabels([label for _, label in settings])
            if r == len(facets)-1:
                ax.set_xlabel(labels[endpoint], fontsize=12, labelpad=9)
            for i, (y, method) in enumerate(zip(positions, methods)):
                if i % 2 == 0:
                    ax.axhspan(y-.45, y+.45, color='#F4F6F8', zorder=-1)
                for offset, (callability, color) in zip([-.125, .125], CALLABILITY_COLORS.items()):
                    key = (scenario, prob, size, method, callability)
                    if key not in indexed.index:
                        continue
                    result = indexed.loc[key]
                    row_y = y + offset
                    if endpoint in ENDPOINTS and result.n_success < 5:
                        ax.text(.5, row_y, 'Less than 5 NJ tree constructed', transform=ax.get_yaxis_transform(),
                                ha='center', va='center', color='#777777', fontsize=10)
                    elif np.isfinite(result['mean']):
                        if np.isfinite(result.ci_low):
                            ax.plot([result.ci_low, result.ci_high], [row_y, row_y], color=color,
                                    linewidth=2.2, solid_capstyle='round', zorder=2)
                        ax.plot(result['mean'], row_y, marker='o', color=color,
                                markersize=6.7, markeredgecolor='white', markeredgewidth=.65,
                                linestyle='none', zorder=3)
                    else:
                        ax.text(.5, row_y, 'N/A', transform=ax.get_yaxis_transform(),
                                ha='center', va='center', color=color, fontsize=10)
                    ax.text(1.035, row_y, f'{int(result.n_evaluated)}/{int(result.n_attempted)}',
                            transform=ax.get_yaxis_transform(), va='center', fontsize=10,
                            color=color, clip_on=False)
            ax.text(1.035, 1.015, 'n', transform=ax.transAxes, fontsize=11, color='#444444')
    fig.text(.5, .58/fig.get_figheight(), 'Mean and 95% CI across repeats; n: evaluated / attempted',
             ha='center', fontsize=10)
    handles = [Line2D([], [], marker='o', color=color, linestyle='none',
                      label=callability.capitalize(), markersize=5)
               for callability, color in CALLABILITY_COLORS.items()]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .04/fig.get_figheight()),
               ncol=2, fontsize=10, frameon=False)
    _save(fig, output, stem, DPI)


def plot_sensitivity_paired(summary, per_seed, settings, parameter, output, stem):
    comparators = [method for method, _ in settings if method != TARGET_METHOD]
    labels = dict(settings)
    positions = _method_positions(comparators)
    facets = _facets(summary)
    fig, axes = _grid(facets, comparators, paired=True)
    panel_height = .7*len(comparators)+.35
    fig.set_figheight(2.75+panel_height*len(facets)+len(facets)-1)
    fig.subplots_adjust(top=1.-1.15/fig.get_figheight(), bottom=1.6/fig.get_figheight(),
                        hspace=1./panel_height)
    first = summary.iloc[0]
    title = (f'{parameter} | '
             f'{first.germline_background.replace("_", " ")} | {first.scenario.replace("_", " ")}')
    fig.suptitle(title, fontsize=13, y=1.-.08/fig.get_figheight())
    indexed = summary.set_index(PAIR_KEYS + ['comparator_method', 'callability_scenario'])
    points = per_seed[per_seed.paired]
    bound = float(points.difference.abs().max()) if len(points) else 0.
    bound = 1.12 * max(bound, .01)
    rng = np.random.default_rng(PLOT_SEED)
    for r, row_facets in enumerate(facets):
        for c, (scenario, prob, size) in enumerate(row_facets):
            ax = axes[r, c]
            limits = (-bound, bound)
            if ENDPOINT_DIRECTIONS[first.endpoint] < 0:
                limits = limits[::-1]
            ax.set_xlim(*limits)
            ax.axvline(0., color='#555555', linewidth=1., linestyle='--', zorder=1)
            if not c:
                ax.set_yticklabels(['vs ' + labels[method] for method in comparators])
            if r == len(facets)-1:
                ax.set_xlabel(PAIRED_ENDPOINT_LABELS[first.endpoint] + '\n(default − setting)',
                              fontsize=12, labelpad=9)
            for y, method in zip(positions, comparators):
                for offset, (callability, color) in zip([-.125, .125], CALLABILITY_COLORS.items()):
                    key = (scenario, prob, size, method, callability)
                    if key not in indexed.index:
                        continue
                    result = indexed.loc[key]
                    row_y = y + offset
                    subset = points[points.prob.eq(prob) & points.sample_size.eq(size)
                                    & points.comparator_method.eq(method)
                                    & points.callability_scenario.eq(callability)].sort_values('seed')
                    if result.n_paired >= 5:
                        ax.scatter(subset.difference, row_y+rng.uniform(-.07, .07, len(subset)),
                                   s=8, color=color, alpha=.55, linewidths=0, zorder=2)
                        ax.plot(result.median_difference, row_y, marker='D', color=color,
                                markersize=7, markeredgecolor='white', markeredgewidth=.65,
                                linestyle='none', zorder=3)
                    else:
                        ax.text(.5, row_y, 'Less than 5 NJ tree constructed', transform=ax.get_yaxis_transform(),
                                ha='center', va='center', color='#777777', fontsize=10)
                    p_label = f'{result.p_bh:.2g}' if result.n_paired >= 5 and np.isfinite(result.p_bh) else 'NA'
                    ax.text(1.035, row_y, f'{int(result.n_paired)}   |   {p_label}',
                            transform=ax.get_yaxis_transform(), va='center', fontsize=10,
                            color=color, clip_on=False)
            ax.text(1.035, 1.015, 'n   |   pBH', transform=ax.transAxes,
                    fontsize=11, color='#444444')
    fig.text(.5, .58/fig.get_figheight(), 'Points: paired repeats; diamonds: median difference (default − setting)',
             ha='center', fontsize=10)
    handles = [Line2D([], [], marker='o', color=color, linestyle='none',
                      label=callability.capitalize(), markersize=5)
               for callability, color in CALLABILITY_COLORS.items()]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .04/fig.get_figheight()),
               ncol=2, fontsize=10, frameon=False)
    _save(fig, output, stem, DPI)


def plot_sensitivity_results():
    setup_plot_style()
    data = read_results(SENSITIVITY_RESULTS, scenario='main')
    data = data[data.germline_background.eq('with_germline')]
    _, groups = sensitivity_configurations()
    germline_parameters = {'MIN_POOLED_RATIO', 'MIN_E', 'GERMLINE_FILTERING'}
    germline_labels = dict(germline_fp='Somatic sites removed (FP)',
        germline_fn='Germline sites retained (FN)',
        germline_fpr='False positive rate', germline_fnr='False negative rate')
    summaries, pair_summaries, differences, germline_summaries = [], [], [], []
    for parameter, settings in groups.items():
        methods = [method for method, _ in settings]
        selected = data[data.method_id.isin(methods)].copy()
        endpoints = ['triplet_distance', QGS_METRIC] if parameter == 'WLS_WEIGHT_POWER' else list(ENDPOINTS)
        summary = summarize_sensitivity(selected, endpoints)
        summary['sensitivity_parameter'] = parameter
        summary['setting'] = summary.method_id.map(dict(settings))
        summaries.append(summary)
        paired, per_seed = paired_effects(selected,
            comparators=[method for method in methods if method != TARGET_METHOD], endpoints=endpoints)
        paired['sensitivity_parameter'] = parameter
        per_seed['sensitivity_parameter'] = parameter
        paired['setting'] = paired.comparator_method.map(dict(settings))
        per_seed['setting'] = per_seed.comparator_method.map(dict(settings))
        pair_summaries.append(paired)
        differences.append(per_seed)
        keys = ['germline_background', 'scenario', 'endpoint']
        for key, subset in summary.groupby(keys, sort=True):
            background, scenario, endpoint = key
            output = SENSITIVITY_DIR / 'figures' / background / scenario
            output.mkdir(parents=True, exist_ok=True)
            stem = parameter.lower() + '_' + endpoint
            plot_sensitivity_accuracy(subset, settings, parameter, ENDPOINTS, output, stem)
            matches = np.ones(len(paired), dtype=bool)
            point_matches = np.ones(len(per_seed), dtype=bool)
            for column, value in zip(keys, key):
                matches &= paired[column].eq(value).to_numpy()
                point_matches &= per_seed[column].eq(value).to_numpy()
            plot_sensitivity_paired(paired.loc[matches], per_seed.loc[point_matches],
                                    settings, parameter, output, stem + '_paired')
        if parameter in germline_parameters:
            germline = summarize_sensitivity(selected, germline_labels, filtering=True)
            germline['sensitivity_parameter'] = parameter
            germline['setting'] = germline.method_id.map(dict(settings))
            germline_summaries.append(germline)
            for (background, scenario, endpoint), subset in germline.groupby(keys, sort=True):
                output = SENSITIVITY_DIR / 'figures' / background / scenario
                plot_sensitivity_accuracy(subset, settings, parameter, germline_labels,
                    output, parameter.lower() + '_' + endpoint)
    for tables, name in [(summaries, 'sensitivity_summary.csv'),
                         (pair_summaries, 'sensitivity_paired_effect_summary.csv'),
                         (differences, 'sensitivity_paired_differences.csv'),
                         (germline_summaries, 'germline_filtering_summary.csv')]:
        pd.concat(tables, ignore_index=True).to_csv(SENSITIVITY_DIR / name, index=False)
    print(f'Saved sensitivity figures and statistics to {SENSITIVITY_DIR}')


def organized_groups():
    _, groups = sensitivity_configurations()
    default = (TARGET_METHOD, 'Pooled ratio + E (default)')
    return {
        'DISTANCE': [(TARGET_METHOD, 'L1 (default)'), ('geinj_L2', 'L2')],
        'IMPUTATION': [(TARGET_METHOD, 'Zero-filled (default)'),
            ('geinj_mean_L1', 'Mean imputation'), ('geinj_knn_L1', 'kNN imputation')],
        'STATE_LOW': [(TARGET_METHOD, f'Binary {STATE_LOW:g} (default)')]
            + [(method, 'Continuous' if method == 'geinj_no_revision_L1' else 'Binary ' + label)
               for method, label in groups['STATE_LOW'] if method != TARGET_METHOD],
        'WLS': [(TARGET_METHOD, f'WLS power {WLS_WEIGHT_POWER:g} (default)')]
            + [(method, 'WLS power ' + label) for method, label in groups['WLS_WEIGHT_POWER']
               if method != TARGET_METHOD]
            + [('zero_filled_no_wls', 'No WLS / Zero-outgroup NJ')],
        'GERMLINE_FILTERING': [default]
            + [(method, label) for method, label in groups['GERMLINE_FILTERING']
               if method != TARGET_METHOD]
            + [(method, 'Pooled ratio = ' + label) for method, label in groups['MIN_POOLED_RATIO']
               if method != TARGET_METHOD]
            + [(method, 'E = ' + label) for method, label in groups['MIN_E']
               if method != TARGET_METHOD],
        'MIN_TOTAL_DEPTH': groups['MIN_TOTAL_DEPTH'],
        'MIN_CALLABLE_DEPTH': groups['MIN_CALLABLE_DEPTH'],
    }


def plot_organized_comparison(blocks, endpoint, output, stem, title, control=False):
    rows = []
    for block in blocks:
        data = state_comparison_data(block['data']) if control else block['data'][block['data'].scenario.eq('main')]
        conditions = data[['scenario', 'prob']].drop_duplicates().sort_values(
            ['scenario', 'prob'], ascending=[True, False])
        for scenario, prob in conditions.itertuples(index=False, name=None):
            sizes = sorted(data.loc[data.scenario.eq(scenario) & data.prob.eq(prob), 'sample_size'].unique())
            rows.append((block, scenario, prob, sizes))
    labels = [fill(label, 32, break_long_words=False, break_on_hyphens=False)
              for block in blocks for _, label in block['settings']]
    left = max(1.75, .085*max(len(line) for label in labels for line in label.splitlines())+.18)
    accuracy_width, paired_width = 2.285, .84
    block_width = 2.97 + paired_width + .95
    callability_gap = (5.70-block_width)*.8
    column_lefts = [left, left+block_width+callability_gap]
    width = left+2*block_width+callability_gap+.35
    heights = [.6*((len(block['settings'])-1)*.75+1.10) for block, _, _, _ in rows]
    height = 1.4+sum(heights)+.69*(len(rows)-1)+1.15
    fig = plt.figure(figsize=(width, height))
    fig.suptitle(title, fontsize=13, y=1.-.12/height)
    for x, callability in zip(column_lefts, ['weak', 'empirical']):
        fig.text((x+block_width/2)/width, 1.-.56/height,
                 callability.capitalize()+' callability', ha='center', va='top', fontsize=12)
    shades = {100: .52, 200: .26, 500: 0.}
    offsets = {100: -.24, 200: 0., 500: .24}
    keys = PAIR_KEYS + ['callability_scenario']
    rng = np.random.default_rng(PLOT_SEED)
    top = 1.4
    for r, ((block, scenario, prob, sizes), panel_height) in enumerate(zip(rows, heights)):
        settings = block['settings']
        positions = np.arange(len(settings))*.75
        summary = block['summary'][block['summary'].endpoint.eq(endpoint)]
        paired = block['paired'][block['paired'].endpoint.eq(endpoint)]
        points = block['per_seed'][block['per_seed'].endpoint.eq(endpoint) & block['per_seed'].paired]
        valid = paired[paired.n_target_success.ge(5) & paired.n_comparator_success.ge(5) & paired.n_paired.ge(5)]
        points = points.merge(valid[keys+['comparator_method']], on=keys+['comparator_method'])
        selected_scenarios = ['main', 'state_permutation'] if control else ['main']
        points = points[points.scenario.isin(selected_scenarios)]
        if control:
            points = points[points.prob.eq(prob) & points.sample_size.isin(sizes)]
        bound = 1.12*max(float(points.difference.abs().max()) if len(points) else 0., .01)
        indexed = summary.set_index(keys+['method_id'])
        pair_indexed = paired.set_index(keys+['comparator_method'])
        point_groups = dict(tuple(points.groupby(keys+['comparator_method'], sort=False)))
        row_title = rf'$1\times10^{{{int(round(np.log10(prob)))}}}$'
        if control:
            row_title = ('Main' if scenario == 'main' else 'State-permuted')+' | '+row_title
        if len(blocks) > 1:
            section = {'MIN_TOTAL_DEPTH': 'Total depth', 'MIN_CALLABLE_DEPTH': 'Callable depth'}
            row_title = section[block['title']]+'\n'+row_title
        title_top = top-(.50 if len(blocks) > 1 else .38)
        fig.text(.32/width, 1.-title_top/height, row_title, fontsize=12, va='top')
        for c, (x, callability) in enumerate(zip(column_lefts, ['weak', 'empirical'])):
            bottom = 1.-(top+panel_height)/height
            ax = fig.add_axes([x/width, bottom, accuracy_width/width, panel_height/height])
            pair_ax = fig.add_axes([(x+2.97)/width, bottom, paired_width/width, panel_height/height])
            for current in [ax, pair_ax]:
                current.set_ylim(positions[-1]+.55, -.55)
                current.set_yticks(positions)
                current.tick_params(axis='y', length=0, pad=11, labelsize=10.5)
                current.tick_params(axis='x', labelsize=10, length=3, color='#999999')
                current.grid(axis='x', color='#DFE3E8', linewidth=.7)
                current.set_axisbelow(True)
                for side in ['top', 'right', 'left']:
                    current.spines[side].set_visible(False)
                current.spines['bottom'].set_color('#999999')
                for i in range(0, len(settings), 2):
                    current.axhspan(positions[i]-.34, positions[i]+.34, color='#F4F6F8', zorder=-1)
            ax.set_yticklabels([fill(label, 32, break_long_words=False, break_on_hyphens=False)
                for _, label in settings] if c == 0 else ['']*len(settings))
            pair_ax.set_yticklabels(['']*len(settings))
            if r == 0:
                ax.set_title('Accuracy', fontsize=11, pad=22)
                pair_ax.set_title('Paired', fontsize=11, pad=22)
            selected = summary[summary.scenario.eq(scenario) & summary.prob.eq(prob)
                & summary.sample_size.isin(sizes) & summary.callability_scenario.eq(callability)
                & summary.n_success.ge(5)]
            values = selected[['mean', 'ci_low', 'ci_high']].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            lower, upper = (float(finite.min()), float(finite.max())) if finite.size else (0., 1.)
            padding = .04*max(upper-lower, .01)
            ax.set_xlim(upper+padding, lower-padding)
            ax.xaxis.set_major_locator(MaxNLocator(3))
            pair_ax.set_xlim(bound, -bound)
            pair_ax.xaxis.set_major_locator(MaxNLocator(2))
            pair_ax.axvline(0., color='#555555', linewidth=1., linestyle='--', zorder=1)
            for y, (method, _) in zip(positions, settings):
                for size in sizes:
                    yy = y+offsets[size]
                    base_color = np.asarray(to_rgb(METHODS.get(method, ('', '#777777'))[1]))
                    color = tuple(base_color*(1-shades[size])+shades[size])
                    key = (scenario, prob, size, callability, method)
                    result = indexed.loc[key]
                    if result.n_success < 5:
                        ax.text(.5, yy, 'Less than 5 NJ tree constructed', transform=ax.get_yaxis_transform(),
                            ha='center', va='center', fontsize=8, color='#777777')
                    elif np.isfinite(result['mean']):
                        if np.isfinite(result.ci_low):
                            ax.plot([result.ci_low, result.ci_high], [yy, yy], color=color,
                                    linewidth=2.2, solid_capstyle='round', zorder=2)
                        marker = ('D' if method in {DENDRO_METHOD, DENDRO_FILTERED_METHOD}
                                  else 's' if method.endswith('L2') else 'o')
                        ax.plot(result['mean'], yy, marker=marker, color=color, markersize=6.7,
                            markeredgecolor='white', markeredgewidth=.65, linestyle='none', zorder=3)
                    else:
                        ax.text(.5, yy, 'N/A', transform=ax.get_yaxis_transform(),
                            ha='center', va='center', fontsize=9, color='#777777')
                    ax.text(1.035, yy, f'{int(result.n_success)}/{int(result.n_attempted)}',
                        transform=ax.get_yaxis_transform(), va='center', fontsize=9, color='#444444')
                    if method not in block['comparators']:
                        continue
                    comparison = pair_indexed.loc[key]
                    enough = (comparison.n_target_success >= 5 and comparison.n_comparator_success >= 5
                              and comparison.n_paired >= 5)
                    if enough:
                        subset = point_groups[key].sort_values('seed')
                        pair_ax.scatter(subset.difference, yy+rng.uniform(-.055, .055, len(subset)),
                            s=8, color=color, alpha=.55, linewidths=0, zorder=2)
                        pair_ax.plot(comparison.median_difference, yy, marker='D', color=color,
                            markersize=6, markeredgecolor='white', markeredgewidth=.65, linestyle='none', zorder=3)
                    p_label = f'{comparison.p_bh:.2g}' if enough and np.isfinite(comparison.p_bh) else 'NA'
                    pair_ax.text(1.08, yy, f'{int(comparison.n_paired)} | {p_label}',
                        transform=pair_ax.get_yaxis_transform(), va='center', fontsize=8.5, color='#444444')
            ax.text(1.035, 1.015, 'n', transform=ax.transAxes, fontsize=9, color='#444444')
            pair_ax.text(1.08, 1.015, 'n | pBH', transform=pair_ax.transAxes, fontsize=8.5, color='#444444')
            if r == len(rows)-1:
                ax.set_xlabel('Rooted triplet distance' if endpoint == 'triplet_distance' else 'Unrooted nRF', fontsize=11, labelpad=9)
                pair_ax.set_xlabel('Δ Triplet' if endpoint == 'triplet_distance' else 'Δ nRF', fontsize=11, labelpad=9)
        top += panel_height+.69
    sizes = sorted({size for _, _, _, sizes in rows for size in sizes})
    base_color = np.asarray(to_rgb(METHODS[TARGET_METHOD][1]))
    handles = [Line2D([], [], marker='o', color=tuple(base_color*(1-shades[size])+shades[size]),
        markersize=6, linewidth=2.2, label=f'{int(size)} cells') for size in sizes]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .10/height),
               ncol=len(handles), fontsize=11, frameon=False, handlelength=1.5)
    _save(fig, output, stem, DPI)


def save_organized_comparison(data, settings, title, output, endpoints=None, comparators=None, prefix=''):
    output.mkdir(parents=True, exist_ok=True)
    endpoints = list(ENDPOINTS) if endpoints is None else endpoints
    methods = [method for method, _ in settings]
    data = data[data.method_id.isin(methods)].copy()
    summary = summarize_sensitivity(data, endpoints)
    if comparators is None:
        comparators = [method for method in methods if method != TARGET_METHOD]
    paired, per_seed = paired_effects(data, comparators=comparators, endpoints=endpoints)
    for table, name in [(summary, 'summary'), (paired, 'paired_effect_summary'), (per_seed, 'paired_differences')]:
        table.assign(component=title).to_csv(output / f'{prefix}{name}.csv', index=False)
    return dict(data=data, settings=settings, title=title, summary=summary,
                paired=paired, per_seed=per_seed, comparators=comparators)


def plot_organized_results():
    setup_plot_style()
    benchmark = read_results(RESULTS)
    sensitivity = read_results(SENSITIVITY_RESULTS)
    matched = read_results(DENDRO_RESULTS)
    output = OUTPUT_DIR / 'organized'
    groups = organized_groups()
    # Sensitivity supplies the shared default and continuous results.
    data = pd.concat([sensitivity, benchmark[benchmark.method_id.isin(SR_ABLATIONS)]], ignore_index=True)
    topics = [
        ('01_Benchmark', 'Benchmark', [('BENCHMARK', benchmark,
            [(method, METHODS[method][0]) for method in MAIN_METHODS], COMPARATORS)]),
        ('01_Benchmark', 'Benchmark / DENDRO-retained cells', [('DENDRO', matched,
            [(method, METHODS[method][0]) for method in MATCHED_METHODS],
            COMPARATORS+[DENDRO_FILTERED_METHOD, DENDRO_METHOD])]),
        ('02_State', 'State', [('STATE_LOW', data, groups['STATE_LOW'], None)]),
        ('03_Distance', 'Distance', [('DISTANCE', data, groups['DISTANCE'], None)]),
        ('04_Imputation', 'Imputation', [('IMPUTATION', data, groups['IMPUTATION'], None)]),
        ('05_WLS', 'WLS', [('WLS', data, groups['WLS'], None)]),
        ('06_Germline_filtering', 'Germline filtering', [('GERMLINE_FILTERING', data, groups['GERMLINE_FILTERING'], None)]),
        ('07_Depth', 'Depth thresholds', [('MIN_TOTAL_DEPTH', data, groups['MIN_TOTAL_DEPTH'], None),
            ('MIN_CALLABLE_DEPTH', data, groups['MIN_CALLABLE_DEPTH'], None)]),
    ]
    for folder, title, sections in topics:
        destination = output / folder
        endpoints = ['triplet_distance'] if folder == '05_WLS' else list(ENDPOINTS)
        blocks = []
        for component, selected, settings, comparators in sections:
            prefix = ('DENDRO_' if component == 'DENDRO' else
                      component.lower()+'_' if len(sections) > 1 else '')
            blocks.append(save_organized_comparison(selected, settings, component, destination,
                endpoints=endpoints, comparators=comparators, prefix=prefix))
        prefix = 'DENDRO_' if sections[0][0] == 'DENDRO' else ''
        for endpoint in endpoints:
            metric = 'triplet' if endpoint == 'triplet_distance' else 'nRF'
            plot_organized_comparison(blocks, endpoint, destination, prefix+metric+'_main', title)
            plot_organized_comparison(blocks, endpoint, destination, prefix+metric+'_state_permutation',
                title+' | State permutation', control=True)
    native = read_results(OUTPUT_DIR / 'dendro_results.csv')
    retention = [summarize_dendro(selected).assign(germline_background=background, callability_scenario=callability)
        for (background, callability), selected in native.groupby(EXPERIMENT_KEYS, sort=False)]
    pd.concat(retention, ignore_index=True).to_csv(output / '01_Benchmark' / 'dendro_retention_summary.csv', index=False)
    labels = dict(germline_fp='Somatic sites removed (FP)', germline_fn='Germline sites retained (FN)',
                  germline_fpr='False positive rate', germline_fnr='False negative rate')
    settings = groups['GERMLINE_FILTERING']
    germline = summarize_sensitivity(data[data.method_id.isin([method for method, _ in settings])], labels, filtering=True)
    germline.to_csv(output / '06_Germline_filtering' / 'filtering_summary.csv', index=False)
    protocol = dict(groups=groups, callability_columns=['weak', 'empirical'],
        layout='Single page per metric and view; probabilities in rows; sample sizes within each method; paired right of accuracy.',
        accuracy_width_inches=2.285, paired_width_inches=.84,
        method_center_spacing=.75, sample_size_offsets={100: -.24, 200: 0., 500: .24},
        method_spacing_scale=.6, callability_gap_scale=.8,
        state_control='Existing 1e-8 / 200-cell state-permuted inputs and corresponding main observations.',
        paired_rule='Default minus comparator; pair seeds within background, callability and condition.',
        paired_tests='One BH family per component, background and callability across its endpoints and conditions; depth axes remain separate.',
        minimum_successes=5, paired_unavailable='Blank plot; actual paired n and NA for pBH.',
        wls_rule='Weight sensitivity: triplet only. No WLS also changes the NJ backbone input.',
        filtering_rule='Change one threshold at a time; other defaults stay fixed.',
        resource_statistics=False)
    (output / 'plot_manifest.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(f'Saved organized figures and statistics to {output}')


def main():
    if sum([BENCHMARK_ONLY, SENSITIVITY_ONLY, TIME_MEM, ORGANIZED]) > 1:
        raise ValueError('Use only one of --benchmark_only, --sensitivity_only, --time_mem or --organized.')
    print(f'GEI-NJ {gei_nj.__version__}: {Path(gei_nj.__file__).resolve()}')
    if OUTPUT_DIR.exists():
        previous = OUTPUT_DIR.with_name(OUTPUT_DIR.name + datetime.now().strftime('_previous_%Y%m%d_%H%M%S'))
        OUTPUT_DIR.rename(previous)
        print(f'Previous results saved to {previous}')
    if ORGANIZED:
        run_benchmark()
        run_sensitivity()
        plot_organized_results()
    elif TIME_MEM:
        run_time_mem()
        plot_resources()
    else:
        if not SENSITIVITY_ONLY:
            run_benchmark()
            plot_results()
        if not BENCHMARK_ONLY:
            run_sensitivity()
            plot_sensitivity_results()
    archive = shutil.make_archive(str(OUTPUT_DIR), 'gztar',
        root_dir=OUTPUT_DIR.parent, base_dir=OUTPUT_DIR.name)
    print(f'Saved results archive to {archive}')


if __name__ == '__main__':
    main()
