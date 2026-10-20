# Validation of GEI-NJ 0.2.0.4

Two review rounds were completed on 2026-10-20. These are local checks; the included GitHub Actions workflow has not yet run on a published commit.

## Round 1: implementation and regression checks

- 27 pytest cases passed with the real RapidNJ executable.
- Covered dense/sparse inputs, aligned names, zero-depth NaN warnings, rejection of invalid observed ratios/depths, state boundaries, raw-ratio recovery, depth filtering, and exact TSV exclusions.
- Germline decisions matched the benchmark at 2,000 random sites under nine threshold combinations (18,000 decisions).
- E-value threshold decisions were also checked against independent numerical integration.
- An additive eight-cell fixture recovered the known root with WLS powers 0, 1, 2, and 3. Its unrooted splits and pairwise distances were preserved.
- Both plot modes preserved the returned root, topology, and Newick branch lengths; topology-mode tips were aligned.

## Round 2: independent and end-to-end checks

- Independently checked 100 random five-tip WLS cases against bounded numerical optimization, including negative NJ lengths projected to zero. The maximum loss difference was 5.68e-14. Root-to-tip residuals and projected backbone distances were also checked.
- On a 200-cell, 515-site simulated dataset, the package matched the benchmark's retained sites, processed matrix, rooted Newick, and WLS diagnostics in all 16 combinations of germline filtering on/off, two-state/continuous representation, and weight powers 0/1/2/3.
- Checked public API documentation, local documentation links, version metadata, and the supplied six-locus TSV.
- Built source and wheel distributions and verified installation and tests from the wheel.

## Naming and storage update

- Rechecked the renamed API and layer lifecycle: X retains original ratios; assigned states are temporary; tree construction reads the zero-filled output layer.
- 30 pytest cases passed with real RapidNJ, including custom layer names, in-place processing, and rebuilding from the preserved ratios.
- Compared the uploaded package and this update on the same 200-cell, 515-site dataset. Retained sites, processed matrices, rooted Newick, and WLS diagnostics matched in all 32 combinations of germline filtering on/off, two-state/continuous representation, callable depths 1/3, and WLS powers 0/1/2/3.
- Confirmed that germline evidence, RapidNJ execution, WLS optimization, and plotting algorithms are unchanged apart from parameter names.

## Local environment

Python 3.12; anndata 0.10.9; Biopython 1.88; NumPy 2.3.5; SciPy 1.17.0; pandas 2.2.3; Matplotlib 3.10.8; pytest 9.1.1.

RapidNJ commit: `ed2d36e219d9db16778b941b5054c0fd021b528a`.

The upstream repository used as the package starting point was commit `1efc04d95db299b15800230f3e51e8c063e17858` of `RuiyingChenBioinfo/GEI-NJ`.

Python 3.9 is configured in CI but was not locally tested. Version-specific DOI archiving remains to be completed as described in RELEASE.md.
