# ColabFold: Protein Structure Prediction

Fast and accessible protein structure prediction using AlphaFold2 via ColabFold, optimized for Google Colab and HPC environments.

## Overview

**ColabFold** is a streamlined implementation of AlphaFold2, the revolutionary AI system for protein structure prediction. This tool makes state-of-the-art structure prediction accessible without requiring massive computational infrastructure.

### What It Does
```
Input:  Protein sequence (FASTA)
        ↓
Process: MSA search + AlphaFold2 neural network (GPU)
        ↓
Output: 3D atomic structure (PDB file) + confidence scores
```

### Key Features

- **Fast:** 10-40x faster than original AlphaFold2
- **Accurate:** Same neural network as AlphaFold2
- **Accessible:** Runs on free Google Colab GPU
- **Complete:** Includes MSA generation, structure prediction, and relaxation

### Performance

- **Model:** AlphaFold2-multimer (latest weights)
- **Speed:** 2-5 minutes per protein (100-300 aa) on Tesla T4
- **Accuracy:** Comparable to experimental structures (RMSD < 2Å for high-confidence predictions)
- **Confidence:** pLDDT scores (0-100, higher = better)

---

## System Requirements

### For Google Colab (Recommended for Testing)

- **Free tier:** Tesla T4 GPU (16GB VRAM)
- **Pro tier:** A100 GPU (40GB VRAM) - faster, handles longer proteins
- **Time limits:** ~12 hours continuous runtime
- **Storage:** Temporary (download results after each session)

### For Local/HPC Deployment

#### Hardware

- **GPU:** NVIDIA with CUDA compute capability ≥ 7.0
  - Minimum: RTX 3060 (12GB VRAM) for proteins <400 aa
  - Recommended: A100 (40GB VRAM) for proteins up to 1000 aa
  - Multi-GPU: Supported for batch processing
- **CPU:** 8+ cores (for MSA search)
- **RAM:** 32+ GB system memory
- **Storage:** 
  - ~4 GB for model parameters
  - ~10-100 MB per predicted structure

#### Software

- **OS:** Linux (Ubuntu 20.04+, tested on Ubuntu 22.04)
- **CUDA:** 12.0+
- **Python:** 3.8-3.10
- **Dependencies:** JAX, OpenMM, HHsuite, Kalign

---

## Installation Methods

### Option 1: Google Colab (No Installation Required)

Use the Jupyter notebook in `notebooks/colabfold_structure_prediction.ipynb`:

1. Upload to Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Run all cells
4. Upload your FASTA file
5. Download predicted structures

**Advantages:**
- ✅ No setup required
- ✅ Free GPU access
- ✅ Always up-to-date

**Limitations:**
- ⚠️ Session time limits (~12 hours)
- ⚠️ Must download results (no persistent storage)
- ⚠️ Protein length limited by VRAM (~400-600 aa on T4)

---

### Option 2: Conda Environment (Local/HPC)
```bash
# Create environment
conda env create -f environment.yml

# Activate
conda activate colabfold_gpu

# Download AlphaFold2 parameters (one-time, ~4GB)
python -m colabfold.download

# Verify
python -c "import colabfold; print('ColabFold ready')"
python -c "import jax; print(f'JAX devices: {jax.devices()}')"
```

---

### Option 3: Singularity Container (HPC Production)
```bash
# Build container (requires Linux with sudo or --fakeroot)
sudo singularity build colabfold.sif colabfold.def

# Or using fakeroot
singularity build --fakeroot colabfold.sif colabfold.def

# Test
singularity exec --nv colabfold.sif \
    python3 -c "import jax; print(jax.devices())"
```

**Note:** Container building requires Linux. The `.def` file is provided for HPC deployment.

---

## Quick Start

### 1. Prepare Input FASTA
```
>protein1
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG

>protein2
MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVE
```

**Recommendations:**
- One protein per sequence for single-chain predictions
- Multiple sequences for complex/multimer predictions
- Keep sequences <600 aa for Colab, <1500 aa for local GPU

---

### 2. Run Prediction

**Using Colab:**
1. Upload notebook to https://colab.research.google.com/
2. Select GPU runtime
3. Run all cells
4. Upload your FASTA file
5. Wait 2-10 minutes per protein
6. Download results

**Using command line (local/HPC):**
```bash
# Activate environment
conda activate colabfold_gpu

# Run prediction
colabfold_batch input.fasta output_dir/ \
    --num-models 5 \
    --num-recycle 3 \
    --use-gpu-relax \
    --amber

# Options explained:
#   --num-models 5        Generate 5 models, rank by confidence
#   --num-recycle 3       AlphaFold recycling iterations
#   --use-gpu-relax       AMBER relaxation on GPU (faster)
#   --amber               Energy minimize structures
```

**Using container:**
```bash
singularity exec --nv colabfold.sif \
    colabfold_batch input.fasta output_dir/ \
    --num-models 5 \
    --amber
```

---

### 3. Examine Results
```bash
ls output_dir/

# For each protein, you'll find:
# protein1_relaxed_rank_001_*.pdb       # Best model (highest confidence)
# protein1_unrelaxed_rank_002_*.pdb     # Alternative models
# protein1_scores_rank_001*.json        # Confidence metrics
# protein1_pae_rank_001*.png            # Predicted aligned error
# protein1_coverage.png                 # MSA coverage plot
```

**Key files:**
- `*_relaxed_rank_001*.pdb` - **Use this one!** (best prediction)
- `*_scores*.json` - Contains pLDDT confidence scores
- `*_pae*.png` - Shows prediction uncertainty

---

## Understanding Results

### pLDDT Confidence Scores

**What is pLDDT?**
- Predicted Local Distance Difference Test
- Ranges from 0-100
- Per-residue confidence metric

**Interpretation:**

| pLDDT Score | Confidence Level | Meaning |
|-------------|------------------|---------|
| **> 90** | Very high | Comparable to experimental structures |
| **70-90** | High | Generally accurate backbone |
| **50-70** | Moderate | Likely correct fold, details uncertain |
| **< 50** | Low | Unreliable, likely disordered region |

**Viewing pLDDT:**
- Encoded in B-factor column of PDB file
- Color-coded in visualization tools (blue = high, red = low)

---

### PAE (Predicted Aligned Error)

**What is PAE?**
- Shows confidence in relative positions between residues
- Dark blue = confident, light/yellow = uncertain
- Useful for identifying domains and assessing complex structures

**Interpretation:**
- **Blue blocks on diagonal:** Well-predicted domains
- **Blue off-diagonal:** Confident domain-domain orientation
- **Yellow/green:** Uncertain relative positioning

---

### Model Ranking

ColabFold generates 5 models and ranks them by:
1. **pTM score** (predicted TM-score) - overall confidence
2. **pLDDT** - local confidence

**Best model:** `*_relaxed_rank_001*.pdb` has highest combined score.

---

## Example Workflow

### Predict Structure of Insulin
```bash
# Create FASTA file
cat > insulin.fasta << EOF
>insulin_human
MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAE
DLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN
EOF

# Run prediction
colabfold_batch insulin.fasta insulin_results/ \
    --num-models 5 \
    --amber

# Check results
ls insulin_results/
# insulin_human_relaxed_rank_001_*.pdb  ← Best model
# insulin_human_scores_rank_001*.json   ← Confidence

# View confidence
cat insulin_results/insulin_human_scores_rank_001*.json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['plddt'])"
# Output: 92.3  ← Very high confidence!
```

---

## Visualization

### Online Viewers (No Installation)

1. **Mol\* Viewer**
   - https://molstar.org/viewer/
   - Upload PDB file
   - Professional quality, fast

2. **PDB Redo Viewer**
   - https://pdb-redo.eu/
   - Upload and auto-validate

3. **Protein Imager**
   - https://3dproteinimaging.com/protein-imager/
   - Simple, quick visualization

---

### Desktop Software (Professional)

**PyMOL (Free for academic use):**
```bash
# Install
conda install -c conda-forge pymol-open-source

# Open structure
pymol insulin_results/insulin_human_relaxed_rank_001*.pdb

# Color by pLDDT (B-factor)
PyMOL> spectrum b, blue_white_red, minimum=50, maximum=90
PyMOL> show cartoon
```

**ChimeraX (Free):**
- Download: https://www.cgl.ucsf.edu/chimerax/
- Drag and drop PDB file
- Color → By Attribute → B-factor

---

## Sample Data & Results

### Test Dataset

- **File:** `test_data/proteins.fasta`
- **Proteins:**
  - Insulin (110 aa) - hormone
  - Lysozyme (148 aa) - antibacterial enzyme

### Expected Performance (Google Colab T4)

| Protein | Length | Time | pLDDT | Quality |
|---------|--------|------|-------|---------|
| Insulin | 110 aa | ~3 min | ~92 | Very high |
| Lysozyme | 148 aa | ~4 min | ~94 | Very high |

**Sample outputs:** See `results/sample_output/` for actual predictions.

---

## Advanced Usage

### Multimer Prediction (Protein Complexes)
```bash
# Create FASTA with multiple chains
cat > complex.fasta << EOF
>proteinA
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQ
>proteinB
KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNA
EOF

# Predict complex
colabfold_batch complex.fasta complex_output/ \
    --num-models 5 \
    --model-type alphafold2_multimer_v3
```

---

### Custom MSA (Skip MMseqs2 Search)

If you have pre-computed MSA:
```bash
# Use custom A3M file
colabfold_batch input.fasta output/ \
    --msa-mode custom \
    --custom-msa your_alignment.a3m
```

---

### Template-Based Modeling
```bash
# Use specific PDB template
colabfold_batch input.fasta output/ \
    --templates \
    --custom-template template.pdb
```

---

## Troubleshooting

### Out of Memory

**Symptoms:**
```
jax._src.traceback_util.UnfilteredStackTrace: RuntimeError: RESOURCE_EXHAUSTED
```

**Solutions:**

1. **Reduce sequence length:**
   - Predict domains separately
   - Remove disordered regions

2. **Use fewer models:**
```bash
   colabfold_batch input.fasta output/ --num-models 1
```

3. **Reduce recycles:**
```bash
   colabfold_batch input.fasta output/ --num-recycle 1
```

4. **Upgrade to larger GPU** (Colab Pro with A100)

---

### Slow MSA Generation

**Symptoms:**
- Stuck at "Generating MSA..."
- Takes >20 minutes

**Solutions:**

1. **Use MMseqs2 web server** (faster):
```bash
   colabfold_batch input.fasta output/ --msa-mode mmseqs2_server
```

2. **Use precomputed MSA:**
   - Search UniRef/BFD separately
   - Use `--custom-msa` flag

---

### Low Confidence Predictions

**Symptoms:**
- pLDDT < 70
- Yellow/red regions in structure

**Possible causes:**
1. **Intrinsically disordered regions** - expected, not a failure
2. **Novel fold** - limited templates
3. **Poor MSA** - few homologs

**Solutions:**
- Check MSA depth: `*_coverage.png`
- Try different MSA database
- Accept that some regions are genuinely disordered

---

### JAX/CUDA Errors

**Symptoms:**
```
RuntimeError: Unable to initialize backend 'cuda'
```

**Solutions:**

1. **Check CUDA installation:**
```bash
   nvidia-smi
   nvcc --version
```

2. **Reinstall JAX with correct CUDA:**
```bash
   pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

3. **Set environment variables:**
```bash
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   export CUDA_VISIBLE_DEVICES=0
```

---

## Performance Optimization

### Speed Tips

1. **Use MMseqs2 web server** - Faster MSA generation
2. **Reduce models** - `--num-models 1` for quick predictions
3. **Skip relaxation** - Remove `--amber` flag (faster but less refined)
4. **Batch processing** - Process multiple proteins in one job
5. **Use A100 GPU** - 3-5x faster than T4

### Resource Management

**VRAM usage by protein length:**

| Length (aa) | VRAM Required | Example GPU |
|-------------|---------------|-------------|
| < 200 | 8 GB | RTX 3060 |
| 200-400 | 12-16 GB | Tesla T4 |
| 400-800 | 24 GB | RTX 3090, A5000 |
| 800-1500 | 40 GB | A100 |
| > 1500 | 80 GB | A100 80GB |

---

## Deployment on HPC

### SLURM Job Script
```bash
#!/bin/bash
#SBATCH --job-name=colabfold
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Load modules
module load singularity

# Set paths
CONTAINER=/shared/containers/colabfold.sif
INPUT=/scratch/proteins.fasta
OUTPUT=/scratch/colabfold_results/

# Create output directory
mkdir -p $OUTPUT

# Run ColabFold
singularity exec --nv $CONTAINER \
    colabfold_batch $INPUT $OUTPUT \
    --num-models 5 \
    --num-recycle 3 \
    --use-gpu-relax \
    --amber

echo "Complete: $OUTPUT"
```

### Resource Estimation

For capacity planning:

- **CPU:** 8 cores (MSA search is parallel)
- **RAM:** 32-64 GB
- **VRAM:** 16-40 GB depending on protein length
- **Time:** 3-10 minutes per protein (100-400 aa)
- **Storage:** ~50-200 MB per protein (all outputs)

---

## Comparison: ColabFold vs AlphaFold2

| Feature | AlphaFold2 | ColabFold |
|---------|------------|-----------|
| **Accuracy** | ★★★★★ | ★★★★★ (same) |
| **Speed** | Slow (1-2 hours) | **Fast** (5-10 min) |
| **Setup** | Complex | **Easy** |
| **Database** | 2.2 TB BFD+UniClust | **MMseqs2 web** (no download) |
| **GPU required** | Yes | Yes |
| **Free option** | No | **Yes** (Colab) |
| **Best for** | Local clusters | **Quick predictions, Colab** |

**Bottom line:** ColabFold uses AlphaFold2's neural network but with optimized MSA generation, making it 10-40x faster without sacrificing accuracy.

---

## Citations

### AlphaFold2
```
Jumper, J., Evans, R., Pritzel, A. et al. 
Highly accurate protein structure prediction with AlphaFold. 
Nature 596, 583–589 (2021).
```

### ColabFold
```
Mirdita, M., Schütze, K., Moriwaki, Y., Heo, L., Ovchinnikov, S., Steinegger, M.
ColabFold: making protein folding accessible to all.
Nature Methods 19, 679–682 (2022).
```

---

## References & Resources

### Official Documentation
- [ColabFold GitHub](https://github.com/sokrypton/ColabFold)
- [AlphaFold2 Paper](https://www.nature.com/articles/s41586-021-03819-2)
- [ColabFold Paper](https://www.nature.com/articles/s41592-022-01488-1)

### Related Tools
- **AlphaFold Database** - Pre-computed structures for 200M+ proteins
- **ESMFold** - Meta's alternative (faster, slightly lower accuracy)
- **RoseTTAFold** - Baker lab's structure prediction

### Tutorials
- [ColabFold Tutorial](https://github.com/sokrypton/ColabFold/blob/main/README.md)
- [Protein Structure Analysis](https://www.rcsb.org/)

---

## Limitations

### What ColabFold CAN Do

✅ Predict single-chain protein structures  
✅ Predict protein complexes (multimers)  
✅ Model point mutations  
✅ Generate confidence estimates  
✅ Handle proteins up to ~1500 aa (with enough VRAM)

### What ColabFold CANNOT Do

❌ Predict structures with ligands/cofactors (unless in template)  
❌ Model post-translational modifications  
❌ Predict membrane protein orientations  
❌ Account for cellular environment effects  
❌ Predict dynamics (only static structures)

---

## Version History

- **v1.5.5** (Current)
  - AlphaFold2-multimer v3 support
  - Improved MSA generation
  - AMBER relaxation on GPU
  - Template search optimization

---

## Contact & Support

**Repository maintained by:** Soumith Paritala  
**Email:** soumith.p6@gmail.com  
**Purpose:** Demonstration of GPU tool deployment for research computing infrastructure

For ColabFold-specific issues:
- [GitHub Issues](https://github.com/sokrypton/ColabFold/issues)
- [ColabFold Google Group](https://groups.google.com/g/colabfold)

---

*Last updated: February 2026*  
*Tested on: Google Colab Tesla T4, NVIDIA A100*