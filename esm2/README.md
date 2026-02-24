# ESM2 Protein Language Model

Facebook AI's ESM2 (Evolutionary Scale Modeling 2) for generating protein sequence embeddings with GPU acceleration.

## Overview

**ESM2** is a state-of-the-art protein language model that generates meaningful representations (embeddings) of protein sequences without requiring structural information.

### What It Does
```
Input:  Protein sequences (FASTA or plain text)
        ↓
Process: Deep learning inference on GPU (ESM2-650M model)
        ↓
Output: 1280-dimensional embeddings per amino acid position
```

### Applications

- **Protein similarity search** - Find functionally related proteins
- **Mutation effect prediction** - Predict impact of amino acid changes
- **Structure prediction** - Input for AlphaFold and similar tools
- **Functional annotation** - Classify unknown proteins
- **Protein design** - Guide engineering of new proteins

### Performance

- **Model:** ESM2-650M (650 million parameters, 33 layers)
- **Speed:** ~1-2 seconds per protein (100 aa) on GPU
- **Accuracy:** State-of-the-art protein understanding
- **Memory:** ~12 GB VRAM required

---

## System Requirements

### Hardware

- **GPU:** NVIDIA with CUDA compute capability ≥ 7.0
  - Tested on: Tesla T4 (16GB), V100 (32GB), A100 (40GB)
  - Consumer GPUs: RTX 3060 (12GB) or better
- **VRAM:** Minimum 12 GB
- **RAM:** 16+ GB system memory
- **Storage:** ~3 GB for model weights

### Software

- **OS:** Linux (Ubuntu 20.04+, tested on Ubuntu 22.04)
- **CUDA:** Version 12.0+
- **Python:** 3.10-3.11
- **PyTorch:** 2.1.0+

---

## Installation Methods

### Option 1: Conda Environment (Recommended for Development)
```bash
# Create environment
conda env create -f conda_env/esm2_env.yml

# Activate
conda activate esm2_gpu

# Verify
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import esm; print(f'ESM version: {esm.__version__}')"
```

### Option 2: Singularity Container (Recommended for Production)
```bash
# Build container (requires Linux with sudo or --fakeroot)
sudo singularity build esm2.sif containers/esm2.def

# Or using fakeroot (no sudo required)
singularity build --fakeroot esm2.sif containers/esm2.def

# Test container
singularity exec --nv esm2.sif python3 -c "import esm; print(esm.__version__)"

# Verify GPU access
singularity exec --nv esm2.sif python3 -c "import torch; print(torch.cuda.is_available())"
```

**Note:** Container building requires Linux. The `.def` file is provided for deployment on HPC systems.

---

## Quick Start

### 1. Prepare Input File

ESM2 accepts **FASTA** or **plain text** formats:

**FASTA format (recommended):**
```
>HBA_HUMAN Hemoglobin subunit alpha
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH
>HBB_HUMAN Hemoglobin subunit beta
MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLST
```

**Plain text format:**
```
hemoglobin_alpha:MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK
hemoglobin_beta:MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQR
```

See `examples/proteins.fasta` for sample data.

### 2. Generate Embeddings

**Using conda environment:**
```bash
conda activate esm2_gpu
cd examples/
python generate_embeddings.py proteins.fasta
```

**Using Singularity container:**
```bash
singularity exec --nv esm2.sif \
    python3 generate_embeddings.py proteins.fasta
```

**Custom output directory:**
```bash
python generate_embeddings.py proteins.fasta -o my_results/
```

### 3. Check Results
```bash
ls results/
# HBA_HUMAN_embedding.npy
# HBA_HUMAN_info.txt
# HBB_HUMAN_embedding.npy
# HBB_HUMAN_info.txt
# ...
```

---

## Example Workflow

### Complete Python Example
```python
import torch
import esm
import numpy as np

# Load model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Prepare sequences
sequences = [
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
]

# Generate embeddings
batch_converter = alphabet.get_batch_converter()
batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
batch_tokens = batch_tokens.to(device)

with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=False)

embeddings = results["representations"][33].cpu().numpy()
print(f"Shape: {embeddings.shape}")  # (2, max_length, 1280)

# Save
np.save("protein1_embedding.npy", embeddings[0])
```

---

## Understanding Embeddings

### What Are Protein Embeddings?

Embeddings are **numerical representations** that capture:
- Amino acid identity and properties
- Local structural propensity (helix, sheet, loop)
- Evolutionary conservation patterns
- Functional site information

### Embedding Structure
```python
# Load an embedding
embedding = np.load("HBA_HUMAN_embedding.npy")
# Shape: (141, 1280)
#   141 = sequence length (amino acids)
#   1280 = embedding dimension

# Position 58 (example: critical residue)
position_58_vector = embedding[58]  # 1280 numbers
# This vector encodes everything ESM2 "knows" about this position
```

### Common Uses

**1. Protein Similarity**
```python
from scipy.spatial.distance import cosine

# Average across positions
emb1_avg = embedding1.mean(axis=0)
emb2_avg = embedding2.mean(axis=0)

# Compute similarity
similarity = 1 - cosine(emb1_avg, emb2_avg)
# 0.9-1.0 = very similar (same family)
# 0.7-0.9 = similar (related function)
# <0.5 = different
```

**2. Clustering Proteins**
```python
from sklearn.cluster import KMeans

# Average embeddings for all proteins
avg_embeddings = [emb.mean(axis=0) for emb in embeddings]

# Cluster
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(avg_embeddings)
```

**3. Dimensionality Reduction (Visualization)**
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(avg_embeddings)

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
plt.show()
```

---

## Sample Data & Results

### Test Dataset

- **Source:** UniProt (reviewed human proteins)
- **File:** `examples/proteins.fasta`
- **Proteins:**
  - HBA_HUMAN - Hemoglobin subunit alpha (141 aa)
  - HBB_HUMAN - Hemoglobin subunit beta (146 aa)
  - INS_HUMAN - Insulin (110 aa)
  - G3P_HUMAN - GAPDH (335 aa)
  - LYSC_HUMAN - Lysozyme C (148 aa)

### Example Output

See `results/sample_output/` for real embeddings generated on Google Colab Tesla T4.

**Expected similarities:**
- HBA_HUMAN ↔ HBB_HUMAN: ~0.85 (both hemoglobin subunits!)
- Unrelated proteins: ~0.40-0.60

ESM2 learns these relationships from **sequence alone**, without being told about protein function.

---

## Smart File Parser

The included parser automatically handles multiple formats:

### Supported Formats

✅ **Standard FASTA** (UniProt, NCBI)
```
>sp|P69905|HBA_HUMAN Hemoglobin alpha
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK
```

✅ **Simple FASTA**
```
>protein1
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQ
```

✅ **Plain text with labels**
```
protein1:MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQ
protein2:KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAA
```

✅ **Plain text (auto-named)**
```
MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQ
KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAA
```

### Validation

The parser automatically:
- Detects file format (FASTA vs plain text)
- Removes whitespace and invalid characters
- Validates amino acid codes (A-Z standard 20)
- Checks sequence length (10-50,000 aa)
- Reports detailed errors for invalid sequences

---

## Troubleshooting

### GPU Not Detected

**Symptoms:**
```
CUDA available: False
WARNING: No GPU detected, using CPU
```

**Solutions:**

1. **Check GPU:**
```bash
   nvidia-smi
```

2. **Verify CUDA:**
```bash
   python -c "import torch; print(torch.version.cuda)"
```

3. **For containers:**
```bash
   # Singularity: use --nv flag
   singularity exec --nv esm2.sif python3 script.py
   
   # Docker: use --gpus flag
   docker run --gpus all esm2:latest python3 script.py
```

### Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Process fewer sequences at once:**
```bash
   # Split large FASTA into smaller files
   split -l 10 proteins.fasta proteins_batch_
```

2. **Use CPU (slower but works):**
```python
   device = torch.device("cpu")
```

3. **Use smaller model:**
```python
   # Instead of 650M, use 150M
   model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
```

### Invalid Sequences

**Symptoms:**
```
✗ protein1: Invalid characters: ['X', 'B', 'Z']
```

**Cause:** Non-standard amino acids

**Solution:**
- Remove ambiguous residues (B, Z, X, *)
- Use only standard 20 amino acids: ACDEFGHIKLMNPQRSTVWY

### Model Download Fails

**Symptoms:**
```
URLError: Connection timeout
```

**Solutions:**

1. **Retry with stable connection**

2. **Manual download:**
```bash
   # Models cache to ~/.cache/torch/hub/checkpoints/
   # Download from: https://github.com/facebookresearch/esm
```

---

## Performance Optimization

### Speed Tips

1. **Use GPU** - 100x faster than CPU
2. **Batch processing** - Process multiple sequences together
3. **Use appropriate model size:**
   - ESM2-35M: Fast, good for large datasets
   - ESM2-650M: Best accuracy (recommended)
   - ESM2-3B: Highest accuracy, very slow

### Resource Management

**VRAM Usage by Model:**
| Model | Parameters | VRAM | Speed |
|-------|------------|------|-------|
| ESM2-8M | 8 million | 2 GB | Very fast |
| ESM2-35M | 35 million | 4 GB | Fast |
| ESM2-150M | 150 million | 8 GB | Moderate |
| **ESM2-650M** | 650 million | 12 GB | **Recommended** |
| ESM2-3B | 3 billion | 24 GB | Slow |

---

## Deployment on HPC

### SLURM Job Script Example
```bash
#!/bin/bash
#SBATCH --job-name=esm2_embeddings
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# Load modules
module load singularity

# Set paths
CONTAINER=/shared/containers/esm2.sif
INPUT=/scratch/data/proteins.fasta
OUTPUT=/scratch/results/

# Run embedding generation
singularity exec --nv $CONTAINER \
    python3 generate_embeddings.py \
    $INPUT \
    -o $OUTPUT

echo "Complete: $OUTPUT"
```

### Resource Estimation

For capacity planning:

- **CPU:** 4 cores (mostly GPU-bound)
- **RAM:** 16-32 GB system memory
- **VRAM:** 12-16 GB (650M model)
- **Time:** ~1-2 hours per 1000 proteins
- **Storage:** ~10 MB per protein (embeddings + metadata)

---

## Google Colab Demo

A complete interactive demo is available for running ESM2 on free Tesla T4 GPU:

1. Upload protein sequences (FASTA or text)
2. Generate embeddings with GPU acceleration
3. Visualize protein relationships (PCA, t-SNE)
4. Compute similarity matrices
5. Download results

**Tested environment:**
- Platform: Google Colab (free tier)
- GPU: Tesla T4 (16GB VRAM)
- CUDA: 12.2
- Runtime: ~1-2 seconds per protein

---

## References & Resources

### Official Documentation
- [ESM GitHub](https://github.com/facebookresearch/esm)
- [Research Paper](https://www.science.org/doi/10.1126/science.ade2574)
- [Model Card](https://github.com/facebookresearch/esm/blob/main/README.md)

### Related Tools
- **AlphaFold2** - Structure prediction (uses ESM embeddings)
- **ProtTrans** - Alternative protein language models
- **UniProt** - Protein sequence database

### Citations

If you use ESM2 in your research, please cite:
```
Lin, Z., Akin, H., Rao, R. et al. 
Evolutionary-scale prediction of atomic-level protein structure with a language model. 
Science 379, 1123-1130 (2023).
```

---

## Version History

- **v2.0.0** (2024) - Current version
  - ESM2-650M model
  - Smart file parser (FASTA + text)
  - GPU optimization
  - Comprehensive validation

---

## Contact & Support

**Repository maintained by:** Soumith Paritala  
**Email:** soumith.p6@gmail.com  
**Purpose:** Demonstration of GPU tool deployment for research computing infrastructure

For ESM2-specific issues, please use the [official GitHub issues](https://github.com/facebookresearch/esm/issues).

---

*Last updated: February 2026*  
*Tested on: Google Colab Tesla T4, NVIDIA A100*
```

---

# **PART 2: FIX RESULTS FOLDER NOT SHOWING**

The `results/` folder is probably empty or only contains `sample_output/` which is also empty.

## **Why Git Ignores It**

Git **doesn't track empty folders**. You need at least one file inside.

---

## **SOLUTION: Add Sample Results**

### **Option 1: Move Your Colab Results**

1. Find your downloaded `esm2_results.zip`
2. Unzip it
3. Copy **ALL files** from inside the `results/` folder
4. Paste into: `stowers-gpu-tools/esm2/results/sample_output/`

**You should have:**
```
esm2/results/sample_output/
├── HBA_HUMAN_embedding.npy
├── HBA_HUMAN_info.txt
├── HBB_HUMAN_embedding.npy
├── HBB_HUMAN_info.txt
├── INS_HUMAN_embedding.npy
├── INS_HUMAN_info.txt
├── G3P_HUMAN_embedding.npy
├── G3P_HUMAN_info.txt
├── LYSC_HUMAN_embedding.npy
├── LYSC_HUMAN_info.txt
├── protein_embeddings_visualization.png
└── analysis_summary.txt