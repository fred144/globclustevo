import numpy as np

sfc_file = "../sim_log_files/fs07_refine/logSFC_1Dprof"

birth_z = []
radii_pc = []
rad_bins = []
shell_dens = []
shell_mass = []
with open(sfc_file) as f:
    for line in f:
        first_word = line.split(" ")[0]
        if first_word == "coarse_step,":
            for l in line.split(" "):
                if (len(l) == 12) and (l != first_word):
                    z_form = l
                    birth_z.append(z_form)
        if first_word == "R":
            rad_pc = line.split(" ")[4][:-1]
            radii_pc.append(rad_pc)
        if first_word == "rbin(:):":
            r_pc = np.array(list(filter(None, list(line.split(" ")[6:])))).astype(
                "float64"
            )
            rad_bins.append(r_pc)
        if first_word == "RHOshell(:):":
            rho = np.array(list(filter(None, list(line.split(" ")[2:])))).astype(
                "float64"
            )
            shell_dens.append(rho)
        if first_word == "Mshell(:):":
            mass = np.array(list(filter(None, list(line.split(" ")[4:])))).astype(
                "float64"
            )
            shell_mass.append(mass)
#%%
import matplotlib.pyplot as plt

plt.style.use("default")
plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "font.size": 7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
    }
)

fig, ax = plt.subplots(figsize=(4, 4), dpi=400)
for i, r in enumerate(rad_bins):
    mask = shell_dens[i] > 0
    ax.plot(r[mask], shell_dens[i][mask], alpha=0.5)
    ax.set(xscale="log", yscale="log")
