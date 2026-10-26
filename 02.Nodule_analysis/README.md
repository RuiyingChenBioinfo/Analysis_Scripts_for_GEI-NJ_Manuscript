**run_samtools-UR2UB.sh**  
Filters the nodule BAM and constructs `UB` tags from `UR`, `Cx`, and `Cy` tags for downstream molecule-aware SNV analysis.

**run_GATK_for_GEI-NJ_selectCBs.sh**  
Selects spatial bins using metadata barcodes, performs per-bin SNV calling and filtering with GATK, and generates the depth and ALT-ratio matrices (`sc_depth.h5ad` and `sc_ratio.h5ad`) for GEI-NJ.

**GEI-NJ_analysis_on_nodule_data.ipynb**  
Processes the SNV matrices with GEI-NJ, reconstructs the NJ tree, exports rectangular, triangular, and circular tree plots, and creates an interactive circular tree–spatial coordinate viewer.

**Nodule_analysis.Plot_Marker_Gene_Expression.R**  
Plots the expression of selected marker genes across annotated nodule cell types using a Seurat dot plot.

**Nodule_analysis.Run_CytoTRACE.R**  
Runs CytoTRACE on the nodule expression matrix and visualizes developmental potential scores with spatial coordinates and cell-type annotations.

**Nodule_analysis.Run_scTOUR.ipynb**  
Runs scTour using 1,000 or 2,000 highly variable genes and visualizes the inferred vector fields and pseudotime. The original `Peripheral_tissues` label corresponds to `Mixed` in the manuscript.

**Nodule_analysis.Run_Monocle2.R**  
Performs Monocle 2 trajectory inference and plots the inferred trajectories colored by pseudotime, state, and cell type.

**Nodule_analysis.Run_Expression_similarity_clustering.ipynb**  
Calculates cell-type-averaged gene expression, performs hierarchical clustering, and visualizes expression similarities in a heatmap.
