import os
import urllib.request

# Create working directory inside notebooks
os.makedirs("sim", exist_ok=True)
os.chdir("sim")

# Download a real PDB: Lysozyme from hen egg white
# This is the standard GROMACS tutorial protein — small, fast, well-tested
pdb_id = "1AKI"
url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

print("Downloading PDB structure...")
urllib.request.urlretrieve(url, f"{pdb_id}.pdb")
print(f"✓ Downloaded {pdb_id}.pdb")

# Count atoms
with open(f"{pdb_id}.pdb") as f:
    atoms = [l for l in f if l.startswith("ATOM")]
    print(f"  Atoms: {len(atoms)}")

# Create MDP parameter files

# Energy minimization
with open("em.mdp", "w") as f:
    f.write("""; Energy Minimization
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000
nstlist     = 1
cutoff-scheme = Verlet
ns_type     = grid
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
""")

# NVT equilibration (100 ps)
with open("nvt.mdp", "w") as f:
    f.write("""; NVT Equilibration (100 ps)
define       = -DPOSRES
integrator   = md
nsteps       = 50000
dt           = 0.002
nstxout-compressed = 500
nstlog       = 500
nstenergy    = 500
continuation = no
constraint_algorithm = lincs
constraints  = h-bonds
lincs_iter   = 1
lincs_order  = 4
cutoff-scheme = Verlet
ns_type      = grid
nstlist      = 10
rcoulomb     = 1.0
rvdw         = 1.0
coulombtype  = PME
pme_order    = 4
fourierspacing = 0.16
tcoupl       = V-rescale
tc-grps      = Protein Non-Protein
tau_t        = 0.1     0.1
ref_t        = 300     300
pcoupl       = no
pbc          = xyz
gen_vel      = yes
gen_temp     = 300
gen_seed     = -1
""")

# NPT equilibration (100 ps)
with open("npt.mdp", "w") as f:
    f.write("""; NPT Equilibration (100 ps)
define       = -DPOSRES
integrator   = md
nsteps       = 50000
dt           = 0.002
nstxout-compressed = 500
nstlog       = 500
nstenergy    = 500
continuation = yes
constraint_algorithm = lincs
constraints  = h-bonds
lincs_iter   = 1
lincs_order  = 4
cutoff-scheme = Verlet
ns_type      = grid
nstlist      = 10
rcoulomb     = 1.0
rvdw         = 1.0
coulombtype  = PME
pme_order    = 4
fourierspacing = 0.16
tcoupl       = V-rescale
tc-grps      = Protein Non-Protein
tau_t        = 0.1     0.1
ref_t        = 300     300
pcoupl       = Parrinello-Rahman
pcoupltype   = isotropic
tau_p        = 2.0
ref_p        = 1.0
compressibility = 4.5e-5
pbc          = xyz
gen_vel      = no
""")

# Production MD (200 ps — short demo)
with open("md.mdp", "w") as f:
    f.write("""; Production MD (200 ps)
integrator   = md
nsteps       = 100000
dt           = 0.002
nstxout-compressed = 500
nstlog       = 1000
nstenergy    = 500
continuation = yes
constraint_algorithm = lincs
constraints  = h-bonds
lincs_iter   = 1
lincs_order  = 4
cutoff-scheme = Verlet
ns_type      = grid
nstlist      = 10
rcoulomb     = 1.0
rvdw         = 1.0
coulombtype  = PME
pme_order    = 4
fourierspacing = 0.16
tcoupl       = V-rescale
tc-grps      = Protein Non-Protein
tau_t        = 0.1     0.1
ref_t        = 300     300
pcoupl       = Parrinello-Rahman
pcoupltype   = isotropic
tau_p        = 2.0
ref_p        = 1.0
compressibility = 4.5e-5
pbc          = xyz
gen_vel      = no
""")

# ions.mdp
with open("ions.mdp", "w") as f:
    f.write("""; Ions
integrator = steep
nsteps = 50000
emtol = 1000.0
emstep = 0.01
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb = 1.0
rvdw = 1.0
pbc = xyz
""")

print("✓ Created em.mdp, nvt.mdp, npt.mdp, md.mdp, ions.mdp")
print("\nNow run the GROMACS commands from the terminal (see instructions)")


print("""
=== GROMACS COMMANDS (run in terminal, one at a time) ===

cd sim

# 1. Generate topology (select force field 6 = AMBER99SB-ILDN)
gmx pdb2gmx -f 1AKI.pdb -o processed.gro -water tip3p -ignh

# 2. Define box
gmx editconf -f processed.gro -o boxed.gro -c -d 1.0 -bt dodecahedron

# 3. Solvate
gmx solvate -cp boxed.gro -cs spc216.gro -o solvated.gro -p topol.top

# 4. Prepare ions
gmx grompp -f ions.mdp -c solvated.gro -p topol.top -o ions.tpr -maxwarn 1

# 5. Add ions to neutralize
echo "SOL" | gmx genion -s ions.tpr -o ions.gro -p topol.top -pname NA -nname CL -neutral

# 6. Energy minimization
gmx grompp -f em.mdp -c ions.gro -p topol.top -o em.tpr
gmx mdrun -v -deffnm em

# 7. NVT equilibration (100 ps, ~5-10 min)
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr
gmx mdrun -v -deffnm nvt

# 8. NPT equilibration (100 ps, ~5-10 min)
gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 1
gmx mdrun -v -deffnm npt

# 9. Production MD (200 ps, ~10-15 min)
gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr
gmx mdrun -v -deffnm md

# 10. Analysis
echo "Backbone Backbone" | gmx rms -s md.tpr -f md.xtc -o rmsd.xvg -tu ps
echo "Protein" | gmx gyrate -s md.tpr -f md.xtc -o gyrate.xvg
echo "Temperature" | gmx energy -f nvt.edr -o temperature.xvg
echo "Potential" | gmx energy -f em.edr -o potential.xvg
""")