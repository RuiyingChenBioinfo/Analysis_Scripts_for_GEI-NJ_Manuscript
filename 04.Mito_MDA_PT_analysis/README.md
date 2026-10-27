# MDA_PT mitochondrial SNV analysis

**run_MDA_PT_comparison.py**  
Compares four tree reconstruction configurations using the author-filtered MDA_PT mitochondrial SNV dataset:

- GEI-NJ with binary encoding (ALT-ratio threshold = 0.01) or continuous SNV ratios.
- DENDRO with default filtering or without SNV prefiltering (cell quality control retained).

The inferred trees are evaluated against experimentally defined guide-barcode (GBC) clone labels using pair-weighted dendrogram purity, clone-weighted dendrogram purity, and same-clone pair average precision.

## Input

The author-filtered `afm_filtered.h5ad` file from the [MiTo dataset (Zenodo)](https://zenodo.org/records/17425550) is expected at:

`source_data/data/lineage_inference/UPMGA/MDA_PT/4efe3d4d1f/afm_filtered.h5ad`

The script requires GEI-NJ, RapidNJ, Rscript, and the R package DENDRO. Adjust the paths defined at the beginning of the script if necessary.

## Run

`python run_MDA_PT_comparison.py`

## Output

Results are saved in `output/`, including reconstructed trees in Newick format, comparison metrics (`comparison_metrics.csv`), PDF/PNG figures, and an interactive GEI-NJ tree–PCA HTML viewer.
