library(Seurat)
library(ggplot2)

rds <- readRDS("Nodule_SeuratObject.rds")

gl <- c("LOC130711631","LOC130747009","LOC130748837","LOC130748875", # Meristems
        "LOC130709486","LOC130722436", # Infected zone
        "LOC130725973","LOC130729049", # Early nodule cortex
        "LOC130724377","LOC130732407","LOC130717234") # Vascular tissue

rds$Celltype <- factor(rds$Celltype, levels = c("Meristems","Infected_Zone","Early_nodule_cortex","Vascular_tissue","Mixed"))

pdf("Nodule_Marker_gene_Exp_DotPlot_new.pdf", 7.5, 3)
DotPlot(rds, group.by = "Celltype", features = gl, scale.by = "size") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_color_gradientn(colours = colorRampPalette(c("#DBDCE1","#ffa55c","#ff745c","#ba061c"))(100))
dev.off()