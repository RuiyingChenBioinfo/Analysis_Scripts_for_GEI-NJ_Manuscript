# Changelog

## 0.2.0.4

- Use SNV filtering, two-state or continuous ALT ratios, zero-filling, L1 distances, RapidNJ, and WLS rooting.
- Add candidate homozygous ALT germline filtering and optional exclusion by a headerless TSV locus list. Use the `_germline` suffix for its three evidence thresholds.
- Validate matched inputs, depth counts, ratio ranges, and missing-value encoding. Preserve missing zero-depth entries until zero-filling.
- Use `assign_state()` for the default two-state representation or continuous ALT ratio states.
- Keep original ratios in `X`, total depths in `total_depth`, and the tree input in `zero_filled_state`. Remove the temporary `state` layer after zero-filling.
- Place the synthetic all-reference profile on the fixed real-cell NJ tree using constrained WLS.
- Draw aligned-tip topology views by default; retain branch-length views and Newick export.
- Replace the former PCA-based workflow and its processing and tree-building functions.
- Align package versions, add unit tests and CI, and document release archiving.
