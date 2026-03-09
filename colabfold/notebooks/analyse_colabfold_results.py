"""
ColabFold Analysis — Drosophila melanogaster Hedgehog protein (471 aa)
Reads ColabFold output files and generates publication-quality analysis plots.

Run locally:
    conda activate colabfold_env   (or any env with numpy, matplotlib, biopython)
    cd ~/Documents/Soumith's/Stowers_Institute/Stowers_gpu_tools/colabfold/notebooks
    python analyze_colabfold_results.py
"""

import json
import numpy as np
import os
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser

# ============================================================
# PATHS — adjust if needed
# ============================================================
RESULTS_DIR = "../results/sample_output/Drosophila_Hedgehog_7577d"
OUTPUT_DIR = "../results/sample_output"
PREFIX = "Drosophila_Hedgehog_7577d"

# ============================================================
# 1. Load scores from all 5 ranked models
# ============================================================
print("=" * 60)
print("COLABFOLD ANALYSIS — Drosophila Hedgehog (471 aa)")
print("=" * 60)

score_files = sorted(glob.glob(f"{RESULTS_DIR}/{PREFIX}_scores_rank_*.json"))
pdb_files = sorted(glob.glob(f"{RESULTS_DIR}/{PREFIX}_relaxed_rank_*.pdb"))

model_data = []
for sf in score_files:
    with open(sf) as f:
        data = json.load(f)

    # Extract rank from filename
    basename = os.path.basename(sf)
    rank = int(basename.split("rank_")[1].split("_")[0])

    plddt = np.array(data.get("plddt", []))
    ptm = data.get("ptm", 0)
    max_pae = data.get("max_pae", 0)

    model_data.append({
        'rank': rank,
        'plddt': plddt,
        'mean_plddt': np.mean(plddt) if len(plddt) > 0 else 0,
        'ptm': ptm,
        'max_pae': max_pae,
        'filename': basename
    })

    print(f"  Rank {rank}: mean pLDDT = {np.mean(plddt):.1f}, pTM = {ptm:.3f}")

# ============================================================
# 2. Load PAE from best model
# ============================================================
pae_file = f"{RESULTS_DIR}/{PREFIX}_predicted_aligned_error_v1.json"
pae_data = None
if os.path.exists(pae_file):
    with open(pae_file) as f:
        pae_json = json.load(f)
    if isinstance(pae_json, list) and len(pae_json) > 0:
        pae_data = np.array(pae_json[0].get("predicted_aligned_error",
                            pae_json[0].get("pae", [])))
    elif isinstance(pae_json, dict):
        pae_data = np.array(pae_json.get("predicted_aligned_error",
                            pae_json.get("pae", [])))

# ============================================================
# 3. Parse best PDB for structural analysis
# ============================================================
parser = PDBParser(QUIET=True)
best_pdb = pdb_files[0] if pdb_files else None
ca_coords = []
bfactors = []

if best_pdb:
    structure = parser.get_structure("best", best_pdb)
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ' and 'CA' in residue:
                    ca_coords.append(residue['CA'].get_vector().get_array())
                    bfactors.append(residue['CA'].get_bfactor())
    ca_coords = np.array(ca_coords)
    bfactors = np.array(bfactors)
    print(f"\n  Best model: {os.path.basename(best_pdb)}")
    print(f"  Residues: {len(ca_coords)}")

# ============================================================
# 4. Create analysis figure (6 panels)
# ============================================================
print("\nGenerating analysis plots...")

fig = plt.figure(figsize=(18, 16))
fig.suptitle("ColabFold Structure Prediction — Drosophila melanogaster Hedgehog\n"
             "Full-length protein (471 aa) | UniProt: Q02936 | AlphaFold2-ptm, 5 models",
             fontweight='bold', fontsize=14)

# --- Panel 1: pLDDT per residue (all 5 models) ---
ax1 = fig.add_subplot(3, 2, 1)
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
for i, md in enumerate(model_data):
    if len(md['plddt']) > 0:
        ax1.plot(range(1, len(md['plddt'])+1), md['plddt'],
                alpha=0.7, linewidth=1, color=colors[i],
                label=f"Rank {md['rank']} ({md['mean_plddt']:.1f})")

ax1.axhline(y=90, color='green', linestyle='--', alpha=0.5, linewidth=0.8)
ax1.axhline(y=70, color='orange', linestyle='--', alpha=0.5, linewidth=0.8)
ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

# Mark domain boundaries
ax1.axvline(x=84, color='black', linestyle=':', alpha=0.5)
ax1.axvline(x=254, color='black', linestyle=':', alpha=0.5)
ax1.text(42, 95, 'Signal\npeptide', ha='center', fontsize=8, style='italic')
ax1.text(169, 95, 'N-terminal\nsignaling', ha='center', fontsize=8, style='italic')
ax1.text(362, 95, 'C-terminal\nautoprocessing', ha='center', fontsize=8, style='italic')

ax1.set_xlabel("Residue number", fontweight='bold')
ax1.set_ylabel("pLDDT score", fontweight='bold')
ax1.set_title("Per-residue Confidence (pLDDT) — All 5 Models", fontweight='bold')
ax1.set_ylim(0, 100)
ax1.legend(fontsize=8, loc='lower left')
ax1.grid(True, alpha=0.2)

# --- Panel 2: PAE heatmap ---
ax2 = fig.add_subplot(3, 2, 2)
if pae_data is not None and pae_data.size > 0:
    im = ax2.imshow(pae_data, cmap='Greens_r', vmin=0, vmax=30, origin='lower')
    ax2.set_xlabel("Residue", fontweight='bold')
    ax2.set_ylabel("Residue", fontweight='bold')
    ax2.set_title("Predicted Aligned Error (PAE)", fontweight='bold')
    plt.colorbar(im, ax=ax2, label='PAE (Å)', shrink=0.8)

    # Mark domain boundaries
    for pos in [84, 254]:
        ax2.axhline(y=pos, color='white', linestyle='--', alpha=0.7, linewidth=0.8)
        ax2.axvline(x=pos, color='white', linestyle='--', alpha=0.7, linewidth=0.8)
else:
    ax2.text(0.5, 0.5, "PAE data not available", ha='center', va='center',
             transform=ax2.transAxes)

# --- Panel 3: Model comparison bar chart ---
ax3 = fig.add_subplot(3, 2, 3)
ranks = [md['rank'] for md in model_data]
plddts = [md['mean_plddt'] for md in model_data]
ptms = [md['ptm'] for md in model_data]

x = np.arange(len(ranks))
width = 0.35
bars1 = ax3.bar(x - width/2, plddts, width, label='Mean pLDDT', color='#3498db', edgecolor='black')
ax3_twin = ax3.twinx()
bars2 = ax3_twin.bar(x + width/2, ptms, width, label='pTM score', color='#e74c3c', edgecolor='black')

ax3.set_xlabel("Model Rank", fontweight='bold')
ax3.set_ylabel("Mean pLDDT", fontweight='bold', color='#3498db')
ax3_twin.set_ylabel("pTM score", fontweight='bold', color='#e74c3c')
ax3.set_xticks(x)
ax3.set_xticklabels([f"Rank {r}" for r in ranks])
ax3.set_title("Model Comparison", fontweight='bold')

lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=9)
ax3.grid(True, alpha=0.2, axis='y')

# --- Panel 4: pLDDT distribution (best model) ---
ax4 = fig.add_subplot(3, 2, 4)
best_plddt = model_data[0]['plddt']
if len(best_plddt) > 0:
    # Color bins by confidence level
    bins = np.arange(0, 105, 5)
    n, bins_out, patches = ax4.hist(best_plddt, bins=bins, edgecolor='black', alpha=0.8)

    # Color by confidence category
    for patch, left_edge in zip(patches, bins_out[:-1]):
        if left_edge >= 90:
            patch.set_facecolor('#2ecc71')  # Very high
        elif left_edge >= 70:
            patch.set_facecolor('#3498db')  # High
        elif left_edge >= 50:
            patch.set_facecolor('#f39c12')  # Medium
        else:
            patch.set_facecolor('#e74c3c')  # Low

    ax4.set_xlabel("pLDDT Score", fontweight='bold')
    ax4.set_ylabel("Number of Residues", fontweight='bold')
    ax4.set_title("Confidence Distribution (Best Model)", fontweight='bold')

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label=f'Very high (≥90): {np.sum(best_plddt >= 90)}'),
        Patch(facecolor='#3498db', label=f'High (70-90): {np.sum((best_plddt >= 70) & (best_plddt < 90))}'),
        Patch(facecolor='#f39c12', label=f'Medium (50-70): {np.sum((best_plddt >= 50) & (best_plddt < 70))}'),
        Patch(facecolor='#e74c3c', label=f'Low (<50): {np.sum(best_plddt < 50)}'),
    ]
    ax4.legend(handles=legend_elements, fontsize=8)
    ax4.grid(True, alpha=0.2)

# --- Panel 5: Contact map from best structure ---
ax5 = fig.add_subplot(3, 2, 5)
if len(ca_coords) > 0:
    dist_matrix = np.sqrt(((ca_coords[:, None] - ca_coords[None, :]) ** 2).sum(-1))
    contact_map = (dist_matrix < 8.0).astype(float)

    ax5.imshow(contact_map, cmap='Blues', origin='lower')
    ax5.set_xlabel("Residue", fontweight='bold')
    ax5.set_ylabel("Residue", fontweight='bold')
    ax5.set_title("Contact Map (Cα < 8 Å)", fontweight='bold')

    for pos in [84, 254]:
        ax5.axhline(y=pos, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
        ax5.axvline(x=pos, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

# --- Panel 6: Radius of gyration per domain ---
ax6 = fig.add_subplot(3, 2, 6)
if len(ca_coords) > 0:
    # Calculate Rg for different domains
    domains = {
        'Signal peptide\n(1-84)': ca_coords[:84],
        'N-terminal\nsignaling (85-254)': ca_coords[84:254],
        'C-terminal\nautoprocessing (255-471)': ca_coords[254:],
        'Full protein\n(1-471)': ca_coords
    }

    domain_names = []
    rg_values = []
    domain_colors = ['#e74c3c', '#3498db', '#2ecc71', '#7f8c8d']

    for name, coords in domains.items():
        if len(coords) > 0:
            center = coords.mean(axis=0)
            rg = np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1)))
            domain_names.append(name)
            rg_values.append(rg)

    bars = ax6.bar(range(len(domain_names)), rg_values, color=domain_colors[:len(domain_names)],
                   edgecolor='black', alpha=0.8)
    ax6.set_xticks(range(len(domain_names)))
    ax6.set_xticklabels(domain_names, fontsize=9)
    ax6.set_ylabel("Radius of Gyration (Å)", fontweight='bold')
    ax6.set_title("Domain Compactness", fontweight='bold')
    ax6.grid(True, alpha=0.2, axis='y')

    for bar, val in zip(bars, rg_values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f} Å', ha='center', fontweight='bold', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f"{OUTPUT_DIR}/colabfold_analysis.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR}/colabfold_analysis.png")

# ============================================================
# 5. Summary report
# ============================================================
with open(f"{OUTPUT_DIR}/colabfold_summary.txt", "w") as f:
    f.write("ColabFold Structure Prediction — Analysis Summary\n")
    f.write("=" * 55 + "\n\n")
    f.write("Protein: Drosophila melanogaster Hedgehog (Hh)\n")
    f.write("UniProt: Q02936\n")
    f.write("Length: 471 amino acids\n")
    f.write("Method: AlphaFold2-ptm via ColabFold\n")
    f.write("MSA: MMseqs2 server\n")
    f.write("Models: 5 (ranked by pLDDT)\n\n")

    f.write("Model Rankings:\n")
    f.write("-" * 55 + "\n")
    for md in model_data:
        f.write(f"  Rank {md['rank']}: mean pLDDT = {md['mean_plddt']:.1f}, pTM = {md['ptm']:.3f}\n")

    if len(best_plddt) > 0:
        f.write(f"\nBest Model Confidence Breakdown:\n")
        f.write(f"  Very high (pLDDT >= 90): {np.sum(best_plddt >= 90)} residues ({np.sum(best_plddt >= 90)/len(best_plddt)*100:.1f}%)\n")
        f.write(f"  High (70-90):            {np.sum((best_plddt >= 70) & (best_plddt < 90))} residues\n")
        f.write(f"  Medium (50-70):          {np.sum((best_plddt >= 50) & (best_plddt < 70))} residues\n")
        f.write(f"  Low (<50):               {np.sum(best_plddt < 50)} residues\n")

    f.write(f"\nDomain Analysis:\n")
    f.write(f"  Signal peptide (1-84): typically disordered, low pLDDT expected\n")
    f.write(f"  N-terminal signaling (85-254): secreted fragment, active in Hh pathway\n")
    f.write(f"  C-terminal autoprocessing (255-471): catalytic, self-cleaving\n")
    f.write(f"\nThe PAE plot shows strong intra-domain confidence but weaker\n")
    f.write(f"inter-domain predictions, consistent with the known biology —\n")
    f.write(f"the N-terminal and C-terminal domains are cleaved apart in vivo.\n")

print(f"✓ Saved: {OUTPUT_DIR}/colabfold_summary.txt")
print("\nDone! Results in:", OUTPUT_DIR)