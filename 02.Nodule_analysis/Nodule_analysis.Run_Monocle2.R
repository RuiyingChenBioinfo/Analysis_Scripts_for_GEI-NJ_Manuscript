library(Seurat)
library(monocle)
library(Matrix)
library(dplyr)
library(ggplot2)

packageVersion("monocle") # 2.30.1

seurat_obj <- readRDS("S17_crop.rds")
seurat_obj

DefaultAssay(seurat_obj) <- "RNA"

expr_matrix <- GetAssayData(seurat_obj, assay = "RNA", slot = "counts")
expr_matrix <- as(expr_matrix, "sparseMatrix")

# Make sure metadata rows match expression matrix columns
pd_data <- seurat_obj@meta.data[colnames(expr_matrix), , drop = FALSE]
pd <- new("AnnotatedDataFrame", data = pd_data)

gene_annotation <- data.frame(
  gene_short_name = rownames(expr_matrix),
  row.names = rownames(expr_matrix)
)

fd <- new("AnnotatedDataFrame", data = gene_annotation)

cds <- newCellDataSet(
  expr_matrix,
  phenoData = pd,
  featureData = fd,
  lowerDetectionLimit = 0.1,
  expressionFamily = negbinomial.size()
)

cds <- estimateSizeFactors(cds)
cds <- estimateDispersions(cds)

cds <- detectGenes(cds, min_expr = 0.1)

disp_table <- dispersionTable(cds)

ordering_genes <- subset(
  disp_table,
  mean_expression >= 0.1 &
    dispersion_empirical >= dispersion_fit
)$gene_id

cds <- setOrderingFilter(cds, ordering_genes)

plot_ordering_genes(cds)

cds <- reduceDimension(cds,max_components = 2, method = "DDRTree")

cds <- orderCells(cds, reverse = T) # This is set to make Infected_Zone as the terminal

col <- c('Early_nodule_cortex'='#a6d96a', 'Infected_Zone'='#f8766d', 'Meristems'='#336699', 'Vascular_tissue'='#fdb863','Peripheral_tissues'='#9970ab')
pdf("Nodule_Monocle2_plot.pdf",6,5)
plot_cell_trajectory(cds, color_by = "Pseudotime")
plot_cell_trajectory(cds, color_by = "State")
plot_cell_trajectory(cds, color_by = "final_sublineage_annotation") + scale_color_manual(values = col)
dev.off()

saveRDS(cds, "Nodule_after_orderCells.rds")
