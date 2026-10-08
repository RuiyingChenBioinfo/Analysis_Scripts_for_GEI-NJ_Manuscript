memory.limit(10000000000)

#### CytoTRACE ####
library(CytoTRACE)
library(ggplot2)
library(Seurat)
library(dplyr)

obj <- readRDS("Nodule_SeuratObject.rds")
table(obj$Celltype)

#run CytoTRACE
expr_matrix <- as.matrix(obj@assays$RNA@counts)
result <- CytoTRACE(expr_matrix, enableFast = FALSE, ncores = 6) 

anno <- paste(obj$Celltype)
anno <- as.character(anno)
names(anno) <- rownames(obj@meta.data)

plotCytoTRACE(result, emb = data.frame(obj@meta.data[,c("x","y")]), phenotype = anno)