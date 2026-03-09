# Conda Environment Testing

Ran all env installs on my MacBook Air M4 with Miniconda.
Removed `defaults` channel, using only conda-forge (+ bioconda/pytorch/nvidia where needed).

No NVIDIA GPU locally so CUDA stuff shows False — that's expected. GPU runs done on Colab.

---

## GROMACS — works

```
(gromacs_env) $ gmx --version | head -3
  :-) GROMACS - gmx, 2026.0-conda_forge (-:
Executable:   /Users/soumithparitala/miniconda3/envs/gromacs_env/bin.ARM_NEON_ASIMD/gmx

(gromacs_env) $ python -c "import numpy; print('numpy:', numpy.__version__)"
numpy: 2.2.6
(gromacs_env) $ python -c "import scipy; print('scipy:', scipy.__version__)"
scipy: 1.15.2
(gromacs_env) $ python -c "import matplotlib; print('matplotlib:', matplotlib.__version__)"
matplotlib: 3.10.8
(gromacs_env) $ python -c "from Bio import SeqIO; print('biopython: OK')"
biopython: OK
```

Everything installed first try. Interesting that conda-forge has GROMACS 2026.0 already.

**Real data test:** Ran a full MD simulation on hen egg-white lysozyme (PDB: 1AKI, 129 residues). AMBER99SB-ILDN force field, TIP3P water model, 200 ps production run at 300 K. Backbone RMSD stabilized around 0.07 nm, Rg stayed steady at ~1.43 nm. Temperature held at 300 ± 4 K. All expected behavior for a stable globular protein.

## Dorado — works

```
(dorado_env) $ samtools --version | head -1
samtools 1.23
(dorado_env) $ minimap2 --version
2.30-r1287
(dorado_env) $ python -c "import numpy; print('numpy:', numpy.__version__)"
numpy: 2.2.6
(dorado_env) $ python -c "from Bio import SeqIO; print('biopython: OK')"
biopython: OK
```

Dorado binary is distributed standalone by ONT. This env covers downstream analysis tools.

**Real data test:** Ran Dorado v1.4.0 on ONT official basecalling demo dataset (~1.1k reads, R10.4.1 chemistry). Tested four features: standard basecalling (sup model), modified base detection (5mCG_5hmCG), basecall + alignment (built-in minimap2), and dorado summary for per-read QC. Generated 6-panel quality report showing read length distribution, quality scores, GC content, and per-position quality.

## ESM2 — works

```
(esm2_env) $ python -c "import torch; print('PyTorch:', torch.__version__)"
PyTorch: 2.10.0
(esm2_env) $ python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
CUDA available: False
(esm2_env) $ python -c "import esm; print('fair-esm: OK')"
fair-esm: OK
(esm2_env) $ python -c "import sklearn; print('scikit-learn: OK')"
scikit-learn: OK
```

CUDA False on Mac — expected. On HPC with GPU it picks up automatically.

**Real data test:** Ran ESM2-650M on five Drosophila Hedgehog signaling pathway proteins (Hedgehog, Smoothened, Cubitus interruptus, Patched, Engrailed). Generated per-residue embeddings, pairwise cosine similarity matrix, PCA clustering, and predicted contact maps. Ran locally on Mac CPU in about 3 minutes.

## Boltz-2 — works

```
(boltz_env) $ python -c "import torch; print('PyTorch:', torch.__version__)"
PyTorch: 2.10.0
(boltz_env) $ python -c "import boltz; print('boltz:', boltz.__version__)"
boltz: 2.2.1
(boltz_env) $ python -c "import yaml; print('pyyaml: OK')"
pyyaml: OK
```

**Real data test:** Ran Boltz-2 on full-length Drosophila Hedgehog protein (471 aa, UniProt Q02936). Sequence fetched directly from UniProt REST API. Note: Boltz requires simple FASTA headers — standard UniProt headers with pipe characters cause parsing errors. Generated predicted structure and confidence scores.

## ColabFold — works

```
(colabfold_env) $ python -c "import colabfold; print('colabfold: OK')"
colabfold: OK
(colabfold_env) $ colabfold_batch --help 2>&1 | head -3
usage: colabfold_batch [-h] [--msa-only]
                       [--msa-mode {mmseqs2_uniref_env,mmseqs2_uniref_env_envpair,mmseqs2_uniref,single_sequence}]
(colabfold_env) $ python -c "import matplotlib; print('matplotlib: OK')"
matplotlib: OK
```

**Real data test:** Predicted structure of full-length Drosophila Hedgehog protein (471 aa, UniProt Q02936) using the official ColabFold notebook (AlphaFold2-ptm, 5 models, 3 recycles). Best model had mean pLDDT in the expected range — N-terminal signaling domain showed high confidence, signal peptide region showed lower confidence as expected for disordered regions. Generated pLDDT per-residue plots, PAE heatmap, model comparison, contact maps, and per-domain Rg analysis.

## RFdiffusion — Singularity only

Didn't try conda for this one. The authors themselves say they can only guarantee CUDA 11.1 and users need to customize for their setup. SE3-Transformer + DGL + specific PyTorch/CUDA combos make a portable conda env impractical.

Singularity container def is in `rfdiffusion/rfdiffusion_design.def`. Ran binder design and unconditional generation through the official Colab notebook — output PDBs are in `rfdiffusion/results/sample_output/`.