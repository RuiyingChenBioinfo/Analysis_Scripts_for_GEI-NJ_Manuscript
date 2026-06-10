# This script simulates diploid somatic mutation accumulation across cell divisions.
# It generates sparse cell-by-mutation matrices under different mutation probabilities
# and saves each simulated dataset as an AnnData .h5ad file.

import numpy as np
import scipy.sparse as sp
import pandas as pd
import anndata as ad

# Diploid encoding: 0 = WT/WT, 0.5 = MUT/WT, 1 = MUT/MUT
n_obs = 1 # One cell for the founder generation 0
n_vars = int(1e8)
n_generations = 12
p_ups = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

def divide_cell_with_mutations(mut0, mut1, n_vars, p_up):
    #mut0, mut1: sets of mutated site indices on two homologous chromosomes.
    #Returns: ((childA0, childA1), (childB0, childB1))
    #New mutations arise during replication on one chromatid and end up in only one daughter.
    childA0, childA1 = set(mut0), set(mut1)
    childB0, childB1 = set(mut0), set(mut1)

    for parent_mut, A, B in ((mut0, childA0, childB0), (mut1, childA1, childB1)):
        n_zero = n_vars - len(parent_mut)
        k_up = np.random.poisson(n_zero * p_up)

        new_pos = set()
        for _ in range(k_up):
            pos = np.random.randint(0, n_vars, dtype=np.int64)
            while (pos in parent_mut) or (pos in new_pos):
                pos = np.random.randint(0, n_vars, dtype=np.int64)
            new_pos.add(int(pos))

        for pos in new_pos:
            (A, B)[np.random.randint(0, 2)].add(pos)

    return (childA0, childA1), (childB0, childB1)

for p_up in p_ups:

    print("current p_up =", p_up)
    np.random.seed(42)
    cell_mut = [(set(), set()) for _ in range(n_obs)]
    lineage = [[i] for i in range(n_obs)]

    rows, cols, data = [], [], []
    obs_cell_id, obs_generation, obs_lineage = [], [], []

    global_row = 0
    for idx, lin in enumerate(lineage):
        obs_cell_id.append(f"cell_0_{idx}")
        obs_generation.append(0)
        obs_lineage.append("-".join(map(str, lin)))
        global_row += 1

    for gen in range(1, n_generations + 1):
        new_cell_mut, new_lineage = [], []

        for idx, ((mut0, mut1), lin) in enumerate(zip(cell_mut, lineage)):
            (childA0, childA1), (childB0, childB1) = divide_cell_with_mutations(
                mut0, mut1, n_vars, p_up
            )

            for daughter_idx, (c0, c1) in enumerate(((childA0, childA1), (childB0, childB1))):
                row_idx = global_row
                global_row += 1

                both = c0 & c1
                hemi = c0 ^ c1

                for c in both:
                    rows.append(row_idx); cols.append(c); data.append(1.0)
                for c in hemi:
                    rows.append(row_idx); cols.append(c); data.append(0.5)

                obs_cell_id.append(f"cell_{gen}_{idx * 2 + daughter_idx}")
                obs_generation.append(gen)
                new_lin = lin + [idx * 2 + daughter_idx]
                obs_lineage.append("-".join(map(str, new_lin)))

                new_cell_mut.append((set(c0), set(c1)))
                new_lineage.append(new_lin)

        cell_mut = new_cell_mut
        lineage = new_lineage

    X_sparse = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(global_row, n_vars),
        dtype=np.float32,
    )

    obs = pd.DataFrame(
        {"cell_id": obs_cell_id, "generation": obs_generation, "lineage": obs_lineage}
    )

    adata = ad.AnnData(X=X_sparse, obs=obs)
    fname = f"./diploid_{n_generations}_generations_{n_obs}cells_{n_vars:.0e}mutations_{p_up:.0e}.h5ad"
    adata.write(fname)
    print(f"Saved: {fname}")