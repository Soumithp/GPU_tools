# RFdiffusion - De Novo Protein Structure Generation & Binder Design

## Overview

RFdiffusion is a generative AI method from the Baker Lab (UW) that designs entirely new protein structures using diffusion models. Starting from random noise, it "denoises" into physically realistic protein backbones.

**Key capabilities:**
- **Unconditional generation:** Create novel protein folds from scratch
- **Protein binder design:** Design proteins that bind specific targets (cancer antigens, cytokines, viral proteins)
- **Motif scaffolding:** Build proteins around functional motifs
- **Symmetric assemblies:** Generate cyclic, dihedral, and tetrahedral complexes

## Tool Summary

| Feature | Details |
|---------|---------|
| **Model** | RFdiffusion (based on RoseTTAFold architecture) |
| **GPU Required** | Yes (NVIDIA, ~8 GB VRAM minimum) |
| **Input** | Protein length (unconditional) or target PDB + hotspot residues (binder design) |
| **Output** | PDB backbone structures (glycine-only — sequences designed separately) |
| **Time** | ~10-60 seconds per design on T4/V100 GPU |
| **License** | BSD (open source) |

## Repository Contents

```
rfdiffusion/
├── README.md                            # This file
├── environment.yml                      # Conda environment specification
├── rfdiffusion_design.def               # Singularity container definition
├── slurm_examples/
│   ├── run_monomer.sh                   # SLURM: unconditional generation
│   └── run_binder.sh                    # SLURM: binder design
├── test_data/
│   └── proteins.fasta                   # Reference sequences
└── results/
    └── sample_output/                   # Example output from test runs
        ├── analysis_summary.txt
        └── *.pdb (generated backbones)
```

## Running RFdiffusion

### Option 1: Google Colab (Quickest Start)

The official RFdiffusion Colab notebook by Sergey Ovchinnikov is the fastest way to test:

**→ [Open Official RFdiffusion Colab](https://colab.research.google.com/github/sokrypton/ColabDesign/blob/v1.1.1/rf/examples/diffusion.ipynb)**

1. Open the link above
2. Set runtime to **T4 GPU**
3. Run all cells
4. Configure your design (see "Contig Syntax" below)
5. Download generated PDBs

### Option 2: Singularity Container (HPC — Recommended for Production)

This is the recommended approach for shared HPC clusters where users cannot install software system-wide. The container bundles all dependencies including the notoriously difficult SE3-Transformer and DGL installations.

```bash
# Build container (one-time, requires fakeroot or sudo)
singularity build rfdiffusion.sif rfdiffusion_design.def

# Unconditional monomer generation (150 residues, 10 designs)
singularity exec --nv rfdiffusion.sif python \
    /opt/RFdiffusion/scripts/run_inference.py \
    inference.output_prefix=output/monomer \
    inference.num_designs=10 \
    'contigmap.contigs=[150-150]'

# Binder design against a target PDB
singularity exec --nv rfdiffusion.sif python \
    /opt/RFdiffusion/scripts/run_inference.py \
    inference.output_prefix=output/binder \
    inference.input_pdb=target.pdb \
    inference.num_designs=10 \
    'contigmap.contigs=[A1-150/0 80-120]' \
    'ppi.hotspot_res=[A30,A33,A34]'
```

### Option 3: Conda Environment (Local/HPC without Singularity)

```bash
# Clone RFdiffusion
git clone https://github.com/RosettaCommons/RFdiffusion.git
cd RFdiffusion

# Create conda environment
conda env create -f env/SE3nv.yml
conda activate SE3nv

# Install SE3-Transformer
cd env/SE3Transformer
pip install --no-cache-dir -r requirements.txt
python setup.py install
cd ../..

# Install RFdiffusion module
pip install -e .

# Download model weights
mkdir -p models && cd models
wget http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt
cd ..

# Test
python ./scripts/run_inference.py \
    inference.output_prefix=test_output/test \
    inference.num_designs=1 \
    'contigmap.contigs=[100-100]'
```

## Installation Challenges & Solutions

RFdiffusion has well-documented dependency challenges. The authors note they can only provide a CUDA 11.1 environment yml, and users must adapt to their own GPU/driver setup. Key issues include:

| Challenge | Solution in This Repo |
|-----------|----------------------|
| SE3-Transformer requires specific PyTorch + CUDA combo | Singularity container pins exact versions |
| DGL library has CUDA-version-specific wheels | Container uses matching `dgl==1.0.2+cu118` |
| Hydra config system requires exact directory structure | Container bakes in correct paths |
| Model weights are ~1.5 GB total | Container downloads during build; cached at `/opt/RFdiffusion/models` |
| Different GPU drivers on different cluster nodes | Singularity `--nv` flag handles driver mounting |

**This is exactly why containerization matters.** Without the Singularity container, each new user would need to debug these issues from scratch. With it, they run one command.

## Contig Syntax (How to Specify Designs)

RFdiffusion uses a "contig" string to define what to generate:

### Unconditional Monomer
```
contigmap.contigs=[150-150]         # Fixed 150 residues
contigmap.contigs=[100-200]         # Random length 100-200
```

### Binder Design
```
# Target chain A (residues 1-150), then binder of 80-120 residues
contigmap.contigs=[A1-150/0 80-120]
ppi.hotspot_res=[A30,A33,A34]       # Surface residues for binding
inference.input_pdb=target.pdb
```

### Motif Scaffolding
```
# Scaffold around residues 394-408 of chain A
contigmap.contigs=[10-40/A394-408/10-40]
```

## SLURM Job Examples

### Unconditional Generation
```bash
#!/bin/bash
#SBATCH --job-name=rfdiff_monomer
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=rfdiff_%j.log

singularity exec --nv rfdiffusion.sif python \
    /opt/RFdiffusion/scripts/run_inference.py \
    inference.output_prefix=designs/monomer \
    inference.num_designs=100 \
    'contigmap.contigs=[100-200]'
```

### Binder Design Campaign
```bash
#!/bin/bash
#SBATCH --job-name=rfdiff_binder
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=binder_%j.log

singularity exec --nv rfdiffusion.sif python \
    /opt/RFdiffusion/scripts/run_inference.py \
    inference.output_prefix=designs/binder \
    inference.input_pdb=target.pdb \
    inference.num_designs=100 \
    'contigmap.contigs=[A1-150/0 70-120]' \
    'ppi.hotspot_res=[A30,A33,A34]' \
    inference.ckpt_override_path=/opt/RFdiffusion/models/Complex_base_ckpt.pt
```

## Output Files

| File | Description |
|------|-------------|
| `*_N.pdb` | Generated backbone structure (design N) |
| `*_N.trb` | Metadata dictionary (contig info, scores) |

**Important:** RFdiffusion outputs are **backbone-only** (glycine residues). The complete design pipeline is:

```
RFdiffusion (backbone)  →  ProteinMPNN (sequence)  →  AlphaFold2 (validation)  →  GROMACS (stability)
```

## Downstream Applications

- **Cancer therapy:** Design binders targeting tumor surface proteins (HER2, PD-L1, EGFR)
- **Autoimmune disease:** Design binders to block inflammatory cytokines (TNF-α, IL-6, IL-17)
- **Infectious disease:** Design binders for viral spike proteins
- **Enzyme design:** Scaffold catalytic motifs into stable proteins
- **Biosensors:** Create binding proteins for diagnostic applications

## Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 8 GB | 16+ GB |
| RAM | 8 GB | 16 GB |
| Disk | 2 GB (weights) | 5 GB |
| CUDA | 11.1+ (conda) / 11.8+ (container) | 12.x |
| Time/design | ~10 sec (monomer) | ~60 sec (binder) |

## References

- Watson, J.L. et al. (2023). De novo design of protein structure and function with RFdiffusion. *Nature*, 620, 1089-1100. doi: 10.1038/s41586-023-06415-8
- [RFdiffusion GitHub](https://github.com/RosettaCommons/RFdiffusion)
- [Official Colab Notebook](https://colab.research.google.com/github/sokrypton/ColabDesign/blob/v1.1.1/rf/examples/diffusion.ipynb)
- [RFdiffusion Docker Image](https://github.com/RosettaCommons/RFdiffusion/pkgs/container/rfdiffusion)