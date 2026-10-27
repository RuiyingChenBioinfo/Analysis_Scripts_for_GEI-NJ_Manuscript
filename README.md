# Analysis Scripts for the GEI-NJ Manuscript

This repository contains the scripts and notebooks used for the simulation study and real-data analyses presented in the GEI-NJ manuscript. The GEI-NJ Python package is available separately at [GEI-NJ](https://github.com/RuiyingChenBioinfo/GEI-NJ).

## Repository contents

**[01.Simulation_analysis](./01.Simulation_analysis)**  
Generates simulated SNV datasets under heterogeneous callability conditions and performs benchmarking, ablation studies, sensitivity analyses, and runtime and memory evaluations.

**[02.Nodule_analysis](./02.Nodule_analysis)**  
Processes *Lotus japonicus* nodule spatial transcriptomic data, performs SNV calling and GEI-NJ tree reconstruction, and includes complementary gene expression, clustering, and trajectory analyses.

**[03.Lotus_blasp](./03.Lotus_blasp)**  
Performs BLASTP-based comparisons between *Lotus japonicus* and *Arabidopsis thaliana* proteins, including the identification of candidate genes matching chloroplast- or mitochondrion-encoded proteins.

**[04.Mito_MDA_PT_analysis](./04.Mito_MDA_PT_analysis)**  
Compares GEI-NJ binary and continuous modes with DENDRO using the MDA_PT mitochondrial SNV dataset and evaluates the inferred trees against experimentally defined clonal labels.

## Usage

See the scripts and available README files in each subdirectory for analysis details and execution instructions. External input datasets and dependencies are required, and some file paths may need to be adjusted for the local environment.
