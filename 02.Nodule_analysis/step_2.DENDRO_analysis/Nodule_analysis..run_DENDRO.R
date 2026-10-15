#!/usr/bin/env Rscript

# Paths and settings.
INPUT_FILE <- "../01.Data_new/02.DENDRO_input/DENDRO_input.rda"
MAPPING_FILE <- "S17_linemut_units_5celltypes_mixed.csv"
OUTPUT_DIR <- "DENDRO_results"
R_LIBRARY <- "~/Litianle/DENDRO_test/R_library"
SEED <- 20261014L
K <- NULL  # NULL: automatic ICD elbow; integer: manual override.
K_MAX <- 10L  # Upper scan limit in automatic mode.

CELL_TYPE_COLORS <- c(
    Early_nodule_cortex = "#A6D96A",
    Infected_Zone = "#F8766D",
    Meristems = "#336699",
    Vascular_tissue = "#FDB863",
    Mixed = "#9970AB"
)
CELL_FATE <- c(
    Early_nodule_cortex = "Uninfected fate",
    Infected_Zone = "Infected fate",
    Meristems = "Others",
    Vascular_tissue = "Uninfected fate",
    Mixed = "Others"
)
FATE_COLORS <- c(
    "Uninfected fate" = "#F4AD32",
    "Infected fate" = "#F8766D",
    "Others" = "#dedede"
)
CLUSTER_COLORS <- c("#D79A5B", "#78A85B", "#5B96B4", "#9970AB",
                    "#F8766D", "#66C2A5", "#E6AB02", "#A6761D",
                    "#F18BBF", "#666666")

.libPaths(c(path.expand(R_LIBRARY), .libPaths()))
suppressPackageStartupMessages(library(DENDRO))
set.seed(SEED)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
capture.output(sessionInfo(), file = file.path(OUTPUT_DIR, "sessionInfo.txt"))

e <- new.env()
load(INPUT_FILE, envir = e)
X <- e$X; N <- e$N; Z <- e$Z; Info <- e$Info
stopifnot(is.matrix(X), is.matrix(N), is.matrix(Z),
          identical(dimnames(X), dimnames(N)),
          identical(dimnames(X), dimnames(Z)),
          identical(rownames(X), rownames(Info)),
          all(is.finite(X)), all(is.finite(N)),
          all(X >= 0), all(X <= N),
          all(is.na(Z) == (N == 0)),
          all(Z[!is.na(Z)] %in% c(0, 1)))
cat("DENDRO version:", as.character(packageVersion("DENDRO")), "\n")
cat("Input:", nrow(N), "SNVs x", ncol(N), "bins\n")

# Mapping: barcode,cell_type; no header.
mapping <- read.csv(MAPPING_FILE, header = FALSE, colClasses = "character",
                    strip.white = TRUE, fileEncoding = "UTF-8-BOM")
stopifnot(ncol(mapping) == 2L)
names(mapping) <- c("barcode", "cell_type")
mapping$barcode <- trimws(mapping$barcode)
mapping$cell_type <- trimws(mapping$cell_type)
mapping$cell_type[tolower(mapping$cell_type) == "mixed"] <- "Mixed"
stopifnot(!anyDuplicated(mapping$barcode))

# Keep X=N=0 and Z=NA at uncovered entries throughout.
pdf(file.path(OUTPUT_DIR, "01_QC.pdf"), width = 10, height = 5)
qc <- tryCatch(FilterCellMutation(X, N, Z, Info = Info), finally = dev.off())
stopifnot(is.matrix(qc$N), nrow(qc$N) >= 2L, ncol(qc$N) >= 3L)
saveRDS(qc, file.path(OUTPUT_DIR, "qc_input.rds"))
if (exists("conversion_metadata", envir = e, inherits = FALSE))
    saveRDS(e$conversion_metadata, file.path(OUTPUT_DIR, "conversion_metadata.rds"))

coverage <- data.frame(
    barcode = colnames(N),
    covered_sites = colSums(N > 0),
    total_allele_templates = colSums(N),
    ALT_called_sites = colSums(Z == 1, na.rm = TRUE),
    retained_after_QC = colnames(N) %in% colnames(qc$N)
)
write.csv(coverage, file.path(OUTPUT_DIR, "cell_coverage.csv"), row.names = FALSE)
writeLines(colnames(qc$N), file.path(OUTPUT_DIR, "retained_barcodes.txt"))
cell_type <- mapping$cell_type[match(colnames(qc$N), mapping$barcode)]
stopifnot(!anyNA(cell_type), all(cell_type %in% names(CELL_TYPE_COLORS)))

# Coverage diagnostics use the same estimable loci as DENDRO.dist.
vaf <- qc$X / qc$N
mu <- rowMeans(vaf, na.rm = TRUE)
variance <- apply(vaf, 1, var, na.rm = TRUE)
alpha <- ((1 - mu) * mu / variance - 1) * mu
beta <- ((1 - mu) * mu / variance - 1) * (1 - mu)
usable <- is.finite(alpha) & is.finite(beta) & alpha > 0 & beta > 0
writeLines(rownames(qc$N)[usable], file.path(OUTPUT_DIR, "estimable_loci.txt"))
overlap <- crossprod(1.0 * (qc$N[usable, , drop = FALSE] > 0))
saveRDS(overlap, file.path(OUTPUT_DIR, "pairwise_covered_loci.rds"))
cat("After QC:", nrow(qc$N), "SNVs x", ncol(qc$N), "bins\n")
cat("Estimable SNVs:", sum(usable), "\n")
cat("Zero-overlap pairs:", sum(overlap[lower.tri(overlap)] == 0), "\n")

# Official distance and Ward clustering; no namespace patch.
d <- DENDRO.dist(qc$X, qc$N, qc$Z)
stopifnot(all(is.finite(as.vector(d))))
saveRDS(d, file.path(OUTPUT_DIR, "DENDRO_distance.rds"))
write.csv(as.matrix(d), file.path(OUTPUT_DIR, "DENDRO_distance.csv"))
correlation <- cor(as.vector(d), overlap[lower.tri(overlap)])
write.csv(data.frame(Pearson_distance_vs_shared_coverage = correlation),
          file.path(OUTPUT_DIR, "distance_coverage_correlation.csv"), row.names = FALSE)
hc <- DENDRO.cluster(d, plot = FALSE)
saveRDS(hc, file.path(OUTPUT_DIR, "hierarchical_clustering.rds"))

# Match the phylo conversion used by DENDRO.cluster.
tree <- ape::as.phylo(as.dendrogram(hc))
saveRDS(tree, file.path(OUTPUT_DIR, "DENDRO_bin_tree.rds"))
ape::write.tree(tree, file = file.path(OUTPUT_DIR, "DENDRO_bin_tree.nwk"), digits = 17)
patristic <- ape::cophenetic.phylo(tree)
write.csv(patristic, file.path(OUTPUT_DIR, "DENDRO_bin_tree_patristic.csv"))

# K selection does not alter the bin tree.
automatic_k <- is.null(K)
stopifnot(length(K_MAX) == 1L, K_MAX >= 3L, K_MAX == as.integer(K_MAX))
if (!automatic_k)
    stopifnot(length(K) == 1L, K >= 2L, K <= ncol(qc$N), K == as.integer(K))
icd <- DENDRO.icd(d, hc, kmax = min(max(K_MAX, K), ncol(qc$N)), plot = FALSE)
k_values <- seq_along(icd)
icd_drop <- icd[1L] - tail(icd, 1L)
elbow_score <- rep(NA_real_, length(icd))
if (is.finite(icd_drop) && icd_drop > 0) {
    x <- (k_values - 1) / (length(icd) - 1)
    y <- (icd - tail(icd, 1L)) / icd_drop
    elbow_score <- 1 - x - y
}
write.csv(data.frame(K = k_values, ICD = as.numeric(icd), elbow_score = elbow_score),
          file.path(OUTPUT_DIR, "ICD.csv"), row.names = FALSE)
pdf(file.path(OUTPUT_DIR, "02_ICD.pdf"), width = 7, height = 5)
plot(k_values, icd, type = "b", pch = 16, xlab = "K", ylab = "ICD")
if (automatic_k) {
    interior <- k_values[-c(1L, length(icd))]
    if (!all(is.finite(elbow_score[interior])) ||
        max(elbow_score[interior]) <= sqrt(.Machine$double.eps)) {
        dev.off()
        stop("No positive ICD elbow found; inspect 02_ICD.pdf and set K manually.")
    }
    # Largest deviation below the normalized endpoint chord; first tie wins.
    K <- interior[which.max(elbow_score[interior])]
}
k_method <- if (automatic_k) "automatic ICD elbow" else "manual override"
abline(v = K, lty = 2, col = "#D55E00")
points(K, icd[K], pch = 19, col = "#D55E00", cex = 1.3)
legend("topright", legend = paste0("K = ", K, " (", k_method, ")"),
       col = "#D55E00", lty = 2, pch = 19, bty = "n")
dev.off()
groups <- cutree(hc, k = K)
cluster_colors <- if (K <= length(CLUSTER_COLORS)) CLUSTER_COLORS[seq_len(K)] else
    grDevices::hcl.colors(K, "Dark 3")
barcodes <- tree$tip.label
type_name <- cell_type[match(barcodes, colnames(qc$N))]
fate_name <- unname(CELL_FATE[type_name])
cluster_id <- as.integer(groups[barcodes])
annotation <- data.frame(
    barcode = barcodes, cluster = cluster_id,
    cell_type = type_name, cell_fate = fate_name,
    cluster_color = cluster_colors[cluster_id],
    cell_type_color = unname(CELL_TYPE_COLORS[type_name]),
    cell_fate_color = unname(FATE_COLORS[fate_name])
)
write.csv(annotation, file.path(OUTPUT_DIR, "bin_annotations.csv"), row.names = FALSE)
write.csv(as.data.frame.matrix(table(cluster = cluster_id,
          cell_type = factor(type_name, levels = names(CELL_TYPE_COLORS)))),
          file.path(OUTPUT_DIR, "cluster_celltype_counts.csv"))
write.csv(as.data.frame.matrix(table(cluster = cluster_id,
          cell_fate = factor(fate_name, levels = names(FATE_COLORS)))),
          file.path(OUTPUT_DIR, "cluster_cellfate_counts.csv"))

plot_with_bars <- function() {
    n <- length(barcodes)
    par(mar = c(1, 1, 3, 26), xpd = NA)
    ape::plot.phylo(tree, type = "phylogram", direction = "rightwards",
        main = paste0("DENDRO | ", n, " bins | K = ", K),
        show.tip.label = FALSE, show.node.label = FALSE,
        edge.color = "black", edge.width = 0.9, no.margin = FALSE)
    plotted <- get("last_plot.phylo", envir = get(".PlotPhyloEnv", asNamespace("ape")))
    tip_y <- plotted$yy[seq_len(n)]
    tip_x <- plotted$xx[seq_len(n)]
    xspan <- diff(range(plotted$xx))
    root_id <- setdiff(tree$edge[, 1], tree$edge[, 2])
    # The decorative stem is not part of the saved tree.
    segments(plotted$xx[root_id] - xspan * 0.04, plotted$yy[root_id],
             plotted$xx[root_id], plotted$yy[root_id], col = "black", lwd = 0.9)

    bar_width <- xspan * 0.042
    bar_gap <- xspan * 0.014
    left <- max(tip_x) + xspan * 0.035 + (0:1) * (bar_width + bar_gap)
    right <- left + bar_width
    legend_x <- right[2L] + xspan * 0.055
    bar_colors <- as.matrix(annotation[, c("cell_type_color", "cell_fate_color")])
    for (i in seq_len(n)) for (j in 1:2)
        rect(left[j], tip_y[i] - 0.5, right[j], tip_y[i] + 0.5,
             col = bar_colors[i, j], border = NA)
    leaf_order <- order(tip_y)

    legend_cex <- if (K <= 10L) 0.76 else 0.62
    type_n <- table(factor(type_name, levels = names(CELL_TYPE_COLORS)))
    present <- type_n > 0
    lg2 <- legend(legend_x, max(tip_y),
        legend = paste0(names(type_n)[present], " (n=", as.integer(type_n[present]), ")"),
        fill = CELL_TYPE_COLORS[present], border = NA, bty = "n", cex = legend_cex,
        title = "Cell type", xjust = 0, yjust = 1)
    fate_n <- table(factor(fate_name, levels = names(FATE_COLORS)))
    present <- fate_n > 0
    legend(legend_x, lg2$rect$top - lg2$rect$h - n * 0.025,
        legend = paste0(names(fate_n)[present], " (n=", as.integer(fate_n[present]), ")"),
        fill = FATE_COLORS[present], border = NA, bty = "n", cex = legend_cex,
        title = "Cell fate", xjust = 0, yjust = 1)

    shown <- annotation[leaf_order, ]
    shown$plotted_y <- tip_y[leaf_order]
    shown
}

stem <- file.path(OUTPUT_DIR, paste0("DENDRO_bin_tree_K", K, "_three_bars"))
height <- max(7, min(18, length(barcodes) * 0.045))
pdf(paste0(stem, ".pdf"), width = 15, height = height, useDingbats = FALSE)
shown <- tryCatch(plot_with_bars(), finally = dev.off())
png(paste0(stem, ".png"), width = 3750, height = round(height * 250), res = 250)
tryCatch(invisible(plot_with_bars()), finally = dev.off())
write.csv(shown, paste0(stem, "_leaf_annotations.csv"), row.names = FALSE)

writeLines(c(
    paste("Seed:", SEED), paste("Selected K:", K),
    paste("K selection:", k_method),
    paste("ICD scan: K = 1 to", length(icd)),
    "Automatic K maximizes positive deviation below the normalized endpoint chord.",
    "Only interior K values are eligible; ties select the smallest K.",
    "The elbow rule is a script-added heuristic, not a DENDRO built-in optimizer.",
    "K affects cluster annotation only; the full bin tree and its distances are unchanged.",
    "Official QC, DENDRO.dist and ward.D defaults are used without a package patch.",
    "Uncovered entries remain X=N=0 and Z=NA; no genotype imputation is performed.",
    "Only official QC removes bins; zero-overlap pairs are reported, not removed.",
    "The saved bin tree is the Ward dendrogram converted by the native DENDRO plotting method.",
    "The dendrogram root is a clustering root, not an inferred biological ancestor.",
    "Tree patristic distances sum saved branch lengths; they differ from DENDRO input distances.",
    "Cell type and Cell fate labels are used only for annotation.",
    "The short plotted root stem is decorative and excluded from tree exports.",
    "Cluster-level NJ reconstruction is not used for the bin-level tree."
), file.path(OUTPUT_DIR, "analysis_notes.txt"))
cat("Selected K:", K, "; cluster sizes:", tabulate(cluster_id, nbins = K), "\n")
cat("Distance/shared-coverage Pearson correlation:", correlation, "\n")
cat("K selection:", k_method, "; see 02_ICD.pdf and ICD.csv.\n")
cat("Saved bin tree, patristic distances and annotated plots to:", OUTPUT_DIR, "\n")
