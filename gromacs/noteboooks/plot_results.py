import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def parse_xvg(filename):
    x, y = [], []
    with open(filename) as f:
        for line in f:
            if line.startswith(("#", "@")):
                continue
            parts = line.split()
            if len(parts) >= 2:
                x.append(float(parts[0]))
                y.append(float(parts[1]))
    return np.array(x), np.array(y)

output_dir = "../results/sample_output"
os.makedirs(output_dir, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("GROMACS MD Simulation — Lysozyme (PDB: 1AKI)\n"
             "200 ps production run, AMBER99SB-ILDN, TIP3P water, 300 K",
             fontweight='bold', fontsize=14)

# RMSD
t, rmsd = parse_xvg("sim/rmsd.xvg")
axes[0,0].plot(t, rmsd, color='#2c3e50', linewidth=0.8)
axes[0,0].set_xlabel("Time (ps)")
axes[0,0].set_ylabel("RMSD (nm)")
axes[0,0].set_title("Backbone RMSD", fontweight='bold')
mean_rmsd = np.mean(rmsd[len(rmsd)//2:])
axes[0,0].axhline(y=mean_rmsd, color='red', linestyle='--',
                   label=f'Mean (2nd half): {mean_rmsd:.3f} nm')
axes[0,0].legend()

# Radius of Gyration
t, rg = parse_xvg("sim/gyrate.xvg")
axes[0,1].plot(t, rg, color='#27ae60', linewidth=0.8)
axes[0,1].set_xlabel("Time (ps)")
axes[0,1].set_ylabel("Rg (nm)")
axes[0,1].set_title("Radius of Gyration", fontweight='bold')
mean_rg = np.mean(rg[len(rg)//2:])
axes[0,1].axhline(y=mean_rg, color='red', linestyle='--',
                   label=f'Mean: {mean_rg:.3f} nm')
axes[0,1].legend()

# Potential Energy
t, pe = parse_xvg("sim/potential.xvg")
axes[1,0].plot(t, pe, color='#e74c3c', linewidth=0.8)
axes[1,0].set_xlabel("Step")
axes[1,0].set_ylabel("Energy (kJ/mol)")
axes[1,0].set_title("Energy Minimization", fontweight='bold')

# Temperature
t, temp = parse_xvg("sim/temperature.xvg")
axes[1,1].plot(t, temp, color='#e67e22', linewidth=0.5, alpha=0.7)
axes[1,1].set_xlabel("Time (ps)")
axes[1,1].set_ylabel("Temperature (K)")
axes[1,1].set_title("Temperature (NVT)", fontweight='bold')
axes[1,1].axhline(y=300, color='red', linestyle='--', label='Target: 300 K')
axes[1,1].legend()

for ax in axes.flat:
    ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(f"{output_dir}/gromacs_analysis.png", dpi=300, bbox_inches='tight')
print(f"✓ Figure saved to {output_dir}/gromacs_analysis.png")

# Also save a text summary
with open(f"{output_dir}/simulation_summary.txt", "w") as f:
    f.write("GROMACS MD Simulation Summary\n")
    f.write("=" * 40 + "\n\n")
    f.write("System: Lysozyme (PDB: 1AKI)\n")
    f.write("Force field: AMBER99SB-ILDN\n")
    f.write("Water model: TIP3P\n")
    f.write("Temperature: 300 K\n")
    f.write("Production run: 200 ps\n\n")
    f.write(f"Backbone RMSD (mean, 2nd half): {mean_rmsd:.3f} nm\n")
    f.write(f"Radius of gyration (mean): {mean_rg:.3f} nm\n")
    f.write(f"Temperature stability: {np.mean(temp):.1f} ± {np.std(temp):.1f} K\n")

print("✓ Summary saved")