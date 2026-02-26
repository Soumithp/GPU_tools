# Boltz-2 - Biomolecular Structure & Affinity Prediction

## Overview

Boltz-2 is a biomolecular foundation model from MIT that predicts 3D structures of proteins, RNA, DNA, and small molecule complexes. It is the first open-source model to approach AlphaFold3 accuracy, and the first deep learning model to approach physics-based free-energy perturbation (FEP) accuracy for binding affinity prediction — while running 1000x faster.

**Key capabilities:**
- Protein monomer and complex structure prediction
- Protein-ligand co-folding with binding affinity estimation
- RNA/DNA structure prediction
- Multi-chain complex modeling

## Tool Summary

| Feature | Details |
|---------|---------|
| **Model** | Boltz-2 (biomolecular foundation model) |
| **GPU Required** | Yes (NVIDIA, ~8 GB VRAM minimum) |
| **Input** | YAML files defining protein sequences, ligands (SMILES/CCD), RNA/DNA |
| **Output** | mmCIF/PDB structures, confidence scores, affinity predictions |
| **Time** | ~2-10 min per protein on T4 GPU |
| **License** | MIT (free for academic and commercial use) |

## Repository Contents

```
boltz/
├── README.md
├── environment.yml
├── boltz_structure.def
├── notebooks/
│   └── boltz_structure_prediction.ipynb
├── test_data/
│   └── proteins.fasta
└── results/
    └── sample_output/
```

## Quick Start (Google Colab)

1. Open `notebooks/boltz_structure_prediction.ipynb` in Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells
4. Upload your FASTA file when prompted
5. Download results zip

## Input Format

### FASTA (for single-chain proteins)
```
>sp|P69905|HBA_HUMAN Hemoglobin subunit alpha
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH...
```
The notebook auto-converts FASTA → Boltz YAML format.

### YAML (native Boltz format, for complexes/ligands)
```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK...
  - ligand:
      id: B
      smiles: "CC(=O)Oc1ccccc1C(=O)O"
```

## Output Files

| File | Description |
|------|-------------|
| `*_model_0.cif` | Predicted 3D structure (mmCIF format) |
| `confidence_*.json` | pLDDT and PAE confidence scores |
| `affinity_*.json` | Binding affinity predictions (if ligand included) |

## Conda Environment Setup

```bash
conda env create -f environment.yml
conda activate boltz_env
boltz predict input.yaml --out_dir results/ --use_msa_server
```

## Singularity Container (HPC)

```bash
singularity build boltz.sif boltz_structure.def

singularity exec --nv boltz.sif boltz predict \
    input.yaml --out_dir results/ --use_msa_server

# SLURM example
sbatch --gres=gpu:1 --mem=32G --wrap="\
    singularity exec --nv boltz.sif boltz predict \
    input/ --out_dir results/ --use_msa_server"
```

## Downstream Applications

- **Drug discovery:** Protein-ligand binding affinity ranking
- **Protein engineering:** Structure-guided mutagenesis
- **Complex modeling:** Multi-chain protein assemblies
- **Comparison:** Benchmark against AlphaFold3 predictions

## Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 8 GB | 16+ GB |
| RAM | 16 GB | 32 GB |
| Disk | 5 GB (weights) | 10 GB |
| CUDA | 12.1+ | 12.x |

## References

- Passaro, S. et al. (2025). Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction. *bioRxiv*. doi: 10.1101/2025.06.14.659707
- Wohlwend, J. et al. (2024). Boltz-1: Democratizing Biomolecular Interaction Modeling. *bioRxiv*. doi: 10.1101/2024.11.19.624167
- [Boltz GitHub](https://github.com/jwohlwend/boltz)