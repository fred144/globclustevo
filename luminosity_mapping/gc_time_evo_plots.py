

# plt.figure(figsize = (8,8), dpi=200)
# plt.hist(core_masses, bins=np.geomspace(core_masses.min(), core_masses.max(),10), histtype='step',  fill=False)
# plt.xscale('log')

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html

# colors = np.random.uniform(size=masses.size)
# biggest_gc = np.max(core_diameter)
# map to differnt sizes for better plotting
# core_diameter_per_size = (500 * core_diameter) / biggest_gc

# fig, ax = plt.subplots(figsize=(8, 8), dpi=200)

# scatter = ax.scatter(
#     ages,
#     masses,
#     c=colors,
#     s=core_diameter_per_size,
#     cmap="Set3",
#     alpha=0.6,
#     linewidths=2,
#     )

# # remap to actual sizes for legend
# legend_properties = dict(
#     prop="sizes",
#     num=4,
#     color="white",
#     fmt=" {x:.2f}",
#     func=lambda r: (r * biggest_gc) / 500,
#     )
# legend = ax.legend(
#     *scatter.legend_elements(**legend_properties),
#     loc="upper right",
#     title="$d_{core}$ (pc)",
#     title_fontsize=16,
#     fontsize=15,
#     )

# ax.set_yscale("log")
# ax.set_title(r"$t_{{sim}} = {} Myr$".format(time))
# ax.set_ylabel(r"Total GC Mass ($M_{\odot}$)", fontsize=16)
# ax.set_xlabel(r"Age (Myr)", fontsize=16)
# fig.savefig(folder_name + "/scatter.png", dpi=300)
# # fig.close()