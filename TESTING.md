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

Note: dorado itself isn't on conda — ONT distributes it as a standalone binary. This env covers the downstream analysis tools (samtools, minimap2, pod5 for format conversion).

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

CUDA False because no NVIDIA GPU on Mac — on an HPC node with GPU it picks it up automatically. Ran the full embedding workflow on Colab with T4, works fine.

## Boltz-2 — works

```
(boltz_env) $ python -c "import torch; print('PyTorch:', torch.__version__)"
PyTorch: 2.10.0
(boltz_env) $ python -c "import boltz; print('boltz:', boltz.__version__)"
boltz: 2.2.1
(boltz_env) $ python -c "import yaml; print('pyyaml: OK')"
pyyaml: OK
```

Clean install. Prediction needs GPU at runtime.

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

This one took the longest to solve dependencies (~5 min) but installed without errors. The colabfold_batch CLI is available which is what you'd use for batch predictions on HPC.

## RFdiffusion — Singularity only

Didn't try conda for this one. The authors themselves say they can only guarantee CUDA 11.1 and users need to customize for their setup. SE3-Transformer + DGL + specific PyTorch/CUDA combos make a portable conda env impractical.

Singularity container def is in `rfdiffusion/rfdiffusion_design.def`. Ran binder design and unconditional generation through the official Colab notebook — output PDBs are in `rfdiffusion/results/sample_output/`.