# Interactive Drug-Gene Risk Association Network

Sankey plot showing drug-gene associations in hepatocellular carcinoma. 
Filtered by paired t-test (FDR < 0.01) with concordant risk direction.

## Versions

- `sankey_original.R` — Original version built in R
- `sankey_drug_gene_network.html` — Updated version built in Python using Plotly

The Python version adds interactive drug filtering, hover tooltips with 
t-statistics and p-values, and a cleaner layout.

## Data

Source: Pharmacogenomic analysis of 8 candidate drugs against HCC risk-associated 
gene expression signatures. 177 significant drug-gene pairs from 2,419 tested.

## Live Demo

[View interactive plot](https://soumithp.github.io/GPU_tools/)