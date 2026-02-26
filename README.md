# 🧬  GPU-Accelerated Bioinformatics Tools

A comprehensive, ready-to-deploy collection of GPU-accelerated bioinformatics tools for structural biology and genomics research. Each tool includes a **Conda environment**, **Singularity container definition**, **Google Colab notebook**, **sample data**, and **documentation** — enabling reproducible, portable deployment on any HPC cluster with NVIDIA GPUs.

> **Built by Soumith** | Computational Biologist | Demonstrating end-to-end GPU tool packaging, containerization, and scientific software administration.

---

## Why This Repository?

Installing GPU-accelerated scientific software on shared HPC clusters is one of the biggest pain points in computational biology. Dependency conflicts, CUDA version mismatches, and undocumented build steps waste researcher time and block science.

**This repository solves that by providing:**

- ✅ **Singularity containers** — portable, reproducible, no root access needed
- ✅ **Conda environments** — exact dependency specifications for every tool
- ✅ **Working notebooks** — tested on Google Colab with real biological data
- ✅ **Sample inputs/outputs** — so users can verify their installation works
- ✅ **SLURM job examples** — ready for HPC job submission
- ✅ **Clear documentation** — input formats, output formats, downstream applications

A new lab member or collaborator can go from zero to running any of these tools in minutes, not days.

---

## 🛠️ Tools Included

### Genomics

| Tool | Description | Input | Output | GPU Use |
|------|-------------|-------|--------|---------|
| [**Dorado**](dorado/) | Oxford Nanopore basecalling | FAST5/POD5 raw signals | FASTQ/BAM sequences | Neural network inference |

### Protein Language Models

| Tool | Description | Input | Output | GPU Use |
|------|-------------|-------|--------|---------|
| [**ESM2**](esm2/) | Meta's protein language model (650M params) | FASTA sequences | Per-residue embeddings, similarity analysis | Transformer inference |

### Structure Prediction

| Tool | Description | Input | Output | GPU Use |
|------|-------------|-------|--------|---------|
| [**ColabFold**](colabfold/) | AlphaFold2 + MMseqs2 structure prediction | FASTA sequences | PDB structures, pLDDT/PAE confidence scores | AlphaFold2 neural network |
| [**Boltz-2**](boltz/) | Biomolecular structure & binding affinity | FASTA/YAML (proteins, ligands, RNA) | mmCIF structures, affinity predictions | Diffusion model inference |

### Protein Design

| Tool | Description | Input | Output | GPU Use |
|------|-------------|-------|--------|---------|
| [**RFdiffusion**](rfdiffusion/) | De novo protein binder & backbone design | Target PDB + hotspot residues | Novel PDB backbones & binder complexes | Diffusion model inference |

### Molecular Dynamics

| Tool | Description | Input | Output | GPU Use |
|------|-------------|-------|--------|---------|
| [**GROMACS**](gromacs/) | Molecular dynamics simulation | PDB structure | Trajectories, RMSD, Rg, energy analysis | Non-bonded force calculations |

---

## 🔗 Integrated Pipeline

These tools connect into a complete computational biology workflow:

```
DNA/RNA Sequencing              Protein Analysis & Design
──────────────────              ─────────────────────────

FAST5/POD5 signals              Protein sequence (FASTA)
       │                               │
       ▼                         ┌─────┴──────┐
   ┌────────┐                    ▼            ▼
   │ Dorado │              ┌──────────┐  ┌──────────┐
   │ (GPU)  │              │   ESM2   │  │ColabFold │
   └───┬────┘              │  (GPU)   │  │  (GPU)   │
       │                   └────┬─────┘  └────┬─────┘
       ▼                        │              │
  FASTQ/BAM                Embeddings     PDB Structure
  (sequences)              (analysis)          │
                                         ┌─────┴──────┐
                                         ▼            ▼
                                   ┌──────────┐  ┌──────────┐
                                   │ Boltz-2  │  │ GROMACS  │
                                   │  (GPU)   │  │  (GPU)   │
                                   └────┬─────┘  └────┬─────┘
                                        │              │
                                   Structures     Dynamics &
                                   + Affinity     Stability
                                        │
                                        ▼
                                  ┌───────────┐
                                  │RFdiffusion│
                                  │   (GPU)   │
                                  └─────┬─────┘
                                        │
                                  Novel Protein
                                   Binders &
                                   Designs
```

**Example real-world workflow:**
1. **Sequence a tumor sample** → Dorado basecalling
2. **Identify a cancer surface protein** → ESM2 embeddings for functional annotation
3. **Predict its 3D structure** → ColabFold / Boltz-2
4. **Design a therapeutic binder against it** → RFdiffusion
5. **Simulate binder stability** → GROMACS molecular dynamics

---

## 🚀 Quick Start

### On Google Colab (no local GPU needed)

1. Navigate to any tool's `notebooks/` directory
2. Open the `.ipynb` file in Google Colab
3. Set runtime to **T4 GPU**
4. Run all cells → upload your data → download results

### On HPC with Singularity

```bash
# Clone the repo
git clone https://github.com/Soumithp/GPU_tools.git
cd GPU_tools

# Build any container (example: ESM2)
cd esm2
singularity build esm2.sif esm2_embedding.def

# Run with GPU passthrough
singularity exec --nv esm2.sif python run_esm2.py \
    --input test_data/proteins.fasta --output results/

# Submit via SLURM
sbatch --gres=gpu:1 --mem=16G --wrap="\
    singularity exec --nv esm2.sif python run_esm2.py \
    --input input.fasta --output results/"
```

### With Conda

```bash
cd esm2
conda env create -f environment.yml
conda activate esm2_env
```

---

## 📁 Repository Structure

Each tool follows a consistent layout:

```
tool_name/
├── README.md              # Documentation, usage, references
├── environment.yml        # Conda environment specification
├── tool_name.def          # Singularity container definition
├── notebooks/
│   └── *.ipynb            # Google Colab notebook (tested)
├── test_data/
│   └── sample_input.*     # Sample input for testing
└── results/
    └── sample_output/     # Example output to verify installation
```

---

## 💻 Tested Environments

| Environment | Status | Notes |
|-------------|--------|-------|
| Google Colab (T4 GPU) | ✅ Tested | All notebooks verified |
| NVIDIA A100 / V100 | ✅ Compatible | Via Singularity containers |
| SLURM HPC clusters | ✅ Compatible | SBATCH examples in each README |
| macOS (M-series, CPU) | ⚠️ Limited | Conda envs work; no GPU acceleration |

---

## 📊 Skills Demonstrated

- **Container building:** Singularity `.def` files with CUDA integration, multi-stage builds
- **Conda environment management:** Reproducible `environment.yml` with pinned versions and channel priority
- **GPU software deployment:** PyTorch+CUDA, JAX+CUDA, GROMACS+CUDA across different frameworks
- **Scientific workflow design:** Input validation, batch processing, automated visualization
- **Documentation:** Clear READMEs with I/O specs, SLURM examples, downstream applications
- **Version control:** Consistent Git repository structure across all tools

---

## 📚 References

| Tool | Publication |
|------|------------|
| Dorado | [Oxford Nanopore Technologies](https://github.com/nanoporetech/dorado) |
| ESM2 | Lin et al. (2023) *Science* 379(6637) doi:10.1126/science.ade2574 |
| ColabFold | Mirdita et al. (2022) *Nature Methods* 19(6) doi:10.1038/s41592-022-01488-1 |
| Boltz-2 | Passaro et al. (2025) *bioRxiv* doi:10.1101/2025.06.14.659707 |
| RFdiffusion | Watson et al. (2023) *Nature* 620 doi:10.1038/s41586-023-06415-8 |
| GROMACS | Abraham et al. (2015) *SoftwareX* 1-2 doi:10.1016/j.softx.2015.06.001 |

---

## 📬 Contact

**Soumith Paritala** — Computational Biologist  
Specializing in multi-omics data analysis and GPU-accelerated bioinformatics pipeline development.