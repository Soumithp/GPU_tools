# ColabFold Sample Output

Example protein structure predictions from `test_data/proteins.fasta`.

## Files Generated

For each protein, ColabFold generates:
- `{name}_relaxed_rank_001_*.pdb` - Best predicted structure (AMBER relaxed)
- `{name}_unrelaxed_rank_0*.pdb` - Alternative models (ranks 2-5)
- `{name}_scores_rank_001*.json` - Confidence metrics (pLDDT)
- `{name}_pae_rank_001*.png` - Predicted aligned error plot
- `{name}_coverage.png` - MSA coverage visualization

## Best Model

The file ending in `_relaxed_rank_001_*.pdb` is the highest-confidence prediction after energy minimization.

## Confidence Interpretation

**pLDDT scores:**
- > 90: Very high confidence (experimentally validated quality)
- 70-90: High confidence (good model)
- 50-70: Moderate confidence (likely correct fold)
- < 50: Low confidence (unreliable regions)

## Visualization

PDB files can be viewed in:
- **PyMOL** (professional)
- **ChimeraX** (free, powerful)
- **Mol* Viewer** (web-based)
- **Protein Imager** (simple online tool)

## Generated

- **Platform:** Google Colab
- **GPU:** Tesla T4 (16GB)
- **Model:** AlphaFold2 via ColabFold
- **Date:** February 2026