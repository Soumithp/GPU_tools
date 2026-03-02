# Environment Testing Results

Tested locally on macOS Sequoia (Apple Silicon M4, 2025 MacBook Air)
Using Miniconda with `defaults` channel removed, `channel_priority: strict`

No NVIDIA GPU on this machine — CUDA tests show `False` as expected.
GPU-accelerated runs were done separately on Google Colab (T4 GPU).



## GROMACS 

# script for the test
conda activate gromacs_env
gmx --version | head -3
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import scipy; print('scipy:', scipy.__version__)"
python -c "import matplotlib; print('matplotlib:', matplotlib.__version__)"
python -c "from Bio import SeqIO; print('biopython: OK')"
conda deactivate

# Output
  :-) GROMACS - gmx, 2026.0-conda_forge (-:

Executable:   /Users/soumithparitala/miniconda3/envs/gromacs_env/bin.ARM_NEON_ASIMD/gmx
numpy: 2.2.6
scipy: 1.15.2
matplotlib: 3.10.8
biopython: OK

Env created and all imports working.

## DORADO 

# script for testing 
conda activate dorado_env
samtools --version | head -1
minimap2 --version
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "from Bio import SeqIO; print('biopython: OK')"
conda deactivate

# Output
samtools 1.23
2.30-r1287
numpy: 2.2.6
biopython: OK

Samtools and minimap2 available. 

## ESM2
# script for testing 
conda activate esm2_env
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import esm; print('fair-esm: OK')"
python -c "import sklearn; print('scikit-learn: OK')"
conda deactivate

# Output 
PyTorch: 2.10.0
CUDA available: False
fair-esm: OK
scikit-learn: OK

PyTorch installed (CPU on Mac). fair-esm imports clean.
CUDA shows False on Mac — expected. Tested with GPU on Colab notebook.

## boltz
# script for testing
conda activate boltz_env
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import boltz; print('boltz:', boltz.__version__)"
python -c "import yaml; print('pyyaml: OK')"
conda deactivate

# Output
PyTorch: 2.10.0
boltz: 2.2.1
pyyaml: OK

Boltz imports successfully. Prediction requires GPU at runtime.

## COLABFOLD
# script for testing
conda activate colabfold_env
python -c "import colabfold; print('colabfold: OK')"
colabfold_batch --help 2>&1 | head -3
python -c "import matplotlib; print('matplotlib: OK')"
conda deactivate

# Output
Matplotlib is building the font cache; this may take a moment.
usage: colabfold_batch [-h] [--msa-only]
                       [--msa-mode {mmseqs2_uniref_env,mmseqs2_uniref_env_envpair,mmseqs2_uniref,single_sequence}]
matplotlib: OK

colabfold env created 

