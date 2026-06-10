library(Seurat)
library(ggplot2)
library(png)
library(dplyr)
library(reshape2)
library(tibble)
library(ggpubr)
library(ggsci)

rds <- readRDS("Processed_tomato_10X_CMBinfo.rds")

rds$Clade <- rds$CMB_info

rds$Clade <- gsub("Vascular_tissue_cmb_0", "Clade_inn1_noh", rds$Clade)
rds$Clade <- gsub("Inner_callus_cmb_1", "Clade_inn1", rds$Clade)

rds$Clade <- gsub("Shoot_primordia_cmb_5", "Others", rds$Clade)
rds$Clade <- gsub("Outgrowth_shoot_cmb_0", "Others", rds$Clade)
rds$Clade <- gsub("Epidermis_cmb_1", "Others", rds$Clade)

rds$Clade <- gsub("Epidermis_cmb_0", "Others", rds$Clade)
rds$Clade <- gsub("Epidermis_cmb_2", "Others", rds$Clade)

rds$Clade <- gsub("Shoot_primordia_cmb_1", "Others", rds$Clade)
rds$Clade <- gsub("Shoot_primordia_cmb_3", "Others", rds$Clade)
rds$Clade <- gsub("Shoot_primordia_cmb_4", "Others", rds$Clade)

rds$Clade <- gsub("Outgrowth_shoot_cmb_1", "Others", rds$Clade)
rds$Clade <- gsub("Outgrowth_shoot_cmb_2", "Others", rds$Clade)
rds$Clade <- gsub("Outgrowth_shoot_cmb_4", "Others", rds$Clade)

rds$Clade <- gsub("Inner_callus_cmb_0", "Clade_inn2", rds$Clade)
rds$Clade <- gsub("Inner_callus_cmb_2", "Clade_inn2", rds$Clade)
rds$Clade <- gsub("Inner_callus_cmb_3", "Clade_inn2", rds$Clade)
rds$Clade <- gsub("Vascular_tissue_cmb_2", "Others", rds$Clade)
rds$Clade <- gsub("Vascular_tissue_cmb_3", "Clade_inn2_noh", rds$Clade)
rds$Clade <- gsub("Vascular_tissue_cmb_1", "Clade_inn2_noh", rds$Clade)

rds$Clade <- gsub("Epidermis_cmb_3", "Others", rds$Clade)
rds$Clade <- gsub("Shoot_primordia_cmb_0", "Others", rds$Clade)
rds$Clade <- gsub("Shoot_primordia_cmb_2", "Others", rds$Clade)
rds$Clade <- gsub("Outgrowth_shoot_cmb_3", "Others", rds$Clade)

pdf("DimPlot_clade_10x_highlight_Cladeinn1inn2_newv2.pdf",10,14)
SpatialDimPlot(rds, group.by = "Clade", image.alpha = 0, pt.size.factor = 1.5, stroke = NA) + 
  scale_fill_manual(values = c("#e86856","#f2bdbd","#f5bf42","#fae5b4","#d8d8d8"))+
   theme_classic()
dev.off()

### Find DEGs in Clade inn1/inn2 ---------------------------------

rds <- subset(rds, Clade != "Others")
rds <- subset(rds, annotation == "Inner_callus")
Idents(rds) <- "Clade"

DEGS <- FindAllMarkers(rds, only.pos = T, min.pct = 0.1)
DEGS <- DEGS[which(DEGS$p_val_adj<0.05),]
write.csv(DEGS, "DEG_Clade_inn1inn2_NoOthers_InnerOnly_new_sig.csv", quote = F)