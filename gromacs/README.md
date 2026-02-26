# GROMACS - GPU-Accelerated Molecular Dynamics Simulation

## Overview

GROMACS is one of the most widely used molecular dynamics (MD) simulation packages in computational biology. It simulates Newtonian equations of motion for systems with hundreds to millions of particles, with exceptional GPU acceleration for non-bonded interactions.

This tool demonstrates a complete MD simulation workflow: system preparation → energy minimization → equilibration → production MD → analysis.

## Tool Summary

| Feature | Details |
|---------|---------|
| **Software** | GROMACS 2024.x |
| **GPU Acceleration** | Yes (NVIDIA CUDA for non-bonded calculations) |
| **Input** | PDB structure file (.pdb) |
| **Output** | Trajectories (.xtc), structures (.gro), energy (.edr), analysis plots |
| **Force Field** | AMBER99SB-ILDN (default in this workflow) |
| **Water Model** | TIP3P |

## Repository Contents

```
gromacs/
├── README.md
├── environment.yml
├── gromacs_md.def
├── notebooks/
│   └── gromacs_md_simulation.ipynb
├── test_data/
│   └── 1AKI.pdb                    # Lysozyme crystal structure
└── results/
    └── sample_output/
        ├── analysis_summary.txt
        ├── md_analysis.png
        ├── rmsd.xvg
        ├── gyrate.xvg
        └── *.mdp (parameter files)
```

## Quick Start (Google Colab)

1. Open `notebooks/gromacs_md_simulation.ipynb` in Google Colab
2. Set runtime to **GPU** (optional — GROMACS runs on CPU too, GPU speeds up ~5-10x)
3. Run all cells
4. Upload a PDB file or use the built-in lysozyme (1AKI) example
5. Download results

## Input Format

### PDB File
Download from [RCSB PDB](https://www.rcsb.org/) or use output from ColabFold/Boltz-2.
```
ATOM      1  N   LYS A   1      26.260  25.529  32.148  1.00 47.52           N
ATOM      2  CA  LYS A   1      26.987  26.125  33.282  1.00 41.38           C
...
```

**Important:** If you only have a FASTA sequence, first predict the structure using ColabFold or Boltz-2 from this repo, then use the output PDB for MD simulation.

## Simulation Workflow

```
PDB file
  ↓ gmx pdb2gmx (generate topology)
  ↓ gmx editconf (define box)
  ↓ gmx solvate (add water)
  ↓ gmx genion (add ions, neutralize)
  ↓ gmx mdrun (energy minimization)
  ↓ gmx mdrun (NVT equilibration - 100 ps)
  ↓ gmx mdrun (NPT equilibration - 100 ps)
  ↓ gmx mdrun (production MD - 500 ps demo)
  ↓ Analysis (RMSD, Rg, temperature plots)
```

## Output Files

| File | Description |
|------|-------------|
| `md.gro` | Final structure after simulation |
| `md.xtc` | Compressed trajectory (all frames) |
| `md.tpr` | Binary run input (for analysis tools) |
| `md.edr` | Energy data file |
| `rmsd.xvg` | Root mean square deviation over time |
| `gyrate.xvg` | Radius of gyration over time |
| `md_analysis.png` | RMSD, Rg, and temperature plots |

## Conda Environment

```bash
conda env create -f environment.yml
conda activate gromacs_env
gmx --version
```

## Singularity Container (HPC)

```bash
# Pull official NVIDIA GROMACS container
singularity pull docker://nvcr.io/hpc/gromacs:2024.3
# OR build from definition
singularity build gromacs.sif gromacs_md.def

# Run with GPU
singularity exec --nv gromacs.sif gmx mdrun -deffnm md -nb gpu

# SLURM job
sbatch --gres=gpu:1 --mem=16G --wrap="\
    singularity exec --nv gromacs.sif gmx mdrun \
    -deffnm md -nb gpu -pin on -ntomp 4"
```

## Downstream Applications

- **Drug binding:** Simulate protein-ligand interactions post-docking
- **Conformational dynamics:** Study protein flexibility and domain motions
- **Free energy:** Calculate binding free energies (FEP, TI)
- **Stability analysis:** Compare wild-type vs mutant protein stability
- **Pipeline integration:** Structure prediction (ColabFold/Boltz) → MD simulation → Analysis

## Resource Requirements

| Resource | CPU Only | GPU Accelerated |
|----------|----------|-----------------|
| Time (1 ns) | ~2-4 hours | ~15-30 min |
| RAM | 4 GB | 8 GB |
| GPU VRAM | N/A | 2+ GB |
| Disk | 1-10 GB per ns | Same |

## References

- Abraham, M.J. et al. (2015). GROMACS: High performance molecular simulations through multi-level parallelism. *SoftwareX*, 1-2, 19-25.
- [GROMACS Documentation](https://manual.gromacs.org/)
- [GROMACS Tutorials](https://tutorials.gromacs.org/)
- [NVIDIA GROMACS Container](https://catalog.ngc.nvidia.com/orgs/hpc/containers/gromacs)