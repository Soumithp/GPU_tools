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

## Example: Lysozyme MD Simulation

Tested locally on macOS (Apple Silicon M4, CPU-only). For GPU runs, use `gmx mdrun -nb gpu`.

**Setup:**
```bash
conda activate gromacs_env
cd gromacs/notebooks
python setup_simulation.py    # downloads PDB + creates parameter files
cd sim
```

**Run simulation steps (one at a time in terminal):**
```bash
# Generate topology (select option 6 for AMBER99SB-ILDN)
gmx pdb2gmx -f 1AKI.pdb -o processed.gro -water tip3p -ignh

# Define box and solvate
gmx editconf -f processed.gro -o boxed.gro -c -d 1.0 -bt dodecahedron
gmx solvate -cp boxed.gro -cs spc216.gro -o solvated.gro -p topol.top

# Add ions
gmx grompp -f ions.mdp -c solvated.gro -p topol.top -o ions.tpr -maxwarn 1
echo "SOL" | gmx genion -s ions.tpr -o ions.gro -p topol.top -pname NA -nname CL -neutral

# Energy minimization (~1 min)
gmx grompp -f em.mdp -c ions.gro -p topol.top -o em.tpr
gmx mdrun -v -deffnm em

# NVT equilibration — 100 ps (~5-10 min on CPU)
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr
gmx mdrun -v -deffnm nvt

# NPT equilibration — 100 ps (~5-10 min)
gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 1
gmx mdrun -v -deffnm npt

# Production MD — 200 ps (~10-15 min on CPU)
gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr
gmx mdrun -v -deffnm md
```

**Analysis:**
```bash
echo "Backbone Backbone" | gmx rms -s md.tpr -f md.xtc -o rmsd.xvg -tu ps
echo "Protein" | gmx gyrate -s md.tpr -f md.xtc -o gyrate.xvg
echo "Temperature" | gmx energy -f nvt.edr -o temperature.xvg
echo "Potential" | gmx energy -f em.edr -o potential.xvg

cd ..
python plot_results.py
```

**Results:** Backbone RMSD stabilized at ~0.07 nm. Radius of gyration held steady at ~1.43 nm. Temperature maintained at 300 ± 4 K. See `results/sample_output/gromacs_analysis.png`.



## References

- Abraham, M.J. et al. (2015). GROMACS: High performance molecular simulations through multi-level parallelism. *SoftwareX*, 1-2, 19-25.
- [GROMACS Documentation](https://manual.gromacs.org/)
- [GROMACS Tutorials](https://tutorials.gromacs.org/)
- [NVIDIA GROMACS Container](https://catalog.ngc.nvidia.com/orgs/hpc/containers/gromacs)