**run_blastp.sh**  
Runs BLASTP to compare *Lotus japonicus* proteins against the *Arabidopsis thaliana* protein database and saves the alignment results.

**run_Lotus_pro_to_loc.sh**  
Maps *Lotus* protein accessions to gene IDs using the GFF3 annotation and adds these gene IDs to the BLASTP results.

**run_filter_ATMGATCG_loc_best_hit.sh**  
Selects one best hit per *Lotus* gene, then extracts genes whose best-hit subject IDs begin with ATCG or ATMG.

**Lotus_vs_Arabidopsis.loc_best_hit.tsv**  
Contains the selected best BLASTP hit for each *Lotus* gene represented in the search results.

**Lotus_vs_Arabidopsis.loc_best_hit.ATCG_ATMG.tsv**  
Contains 218 candidate *Lotus* genes whose best hits match proteins encoded by the *Arabidopsis* chloroplast or mitochondrial genome.
