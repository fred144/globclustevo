import numpy as np
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/23528477/share-axes-in-matplotlib-for-only-part-of-the-subplots

f7_fof_dat = np.loadtxt("../fof_time_series/fs070_fof_series_results.txt")
f7_pro_dat = np.loadtxt(
    "../../gc_profiles/profile_runs/fs07_refine/time_series_run_stats.txt"
)

f3_fof_dat = np.loadtxt("../fof_time_series/fs035_fof_series_results.txt")
f3_pro_dat = np.loadtxt(
    "../../gc_profiles/profile_runs/fs035_ms10/hi_fidelity_time_series_run_stats.txt"
)

fof_f7_t_myr = np.round(f7_fof_dat[:, 0], 2)
fof_f7_m_gc = f7_fof_dat[:, 2]
fof_f7_m_field = f7_fof_dat[:, 3]

pro_f7_t_myr = f7_pro_dat[:, 1][:-1]
pro_f7_m_gc = f7_pro_dat[:, 6][:-1]
pro_f7_m_field = f7_pro_dat[:, 5][:-1] - pro_f7_m_gc

fof_f3_t_myr = np.round(f3_fof_dat[:, 0], 2)
fof_f3_m_gc = f3_fof_dat[:, 2]
fof_f3_m_field = f3_fof_dat[:, 3]

pro_f3_t_myr = f3_pro_dat[:, 1]
pro_f3_m_gc = f3_pro_dat[:, 6]
pro_f3_m_field = f3_pro_dat[:, 5] - pro_f3_m_gc
#%%
with plt.rc_context(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.size": 12,
    }
):

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), dpi=300, sharex=True)
    ax[0].plot(fof_f7_t_myr, pro_f7_m_gc / fof_f7_m_gc)

    ax[0].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")
    ax[0].set_title("$f_{*}$ = 0.70")

    ax[1].plot(fof_f3_t_myr, pro_f3_m_gc / fof_f3_m_gc)

    ax[1].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")
    ax[1].set_title("$f_{*}$ = 0.35")

    fig.text(
        0.04, 0.5, "Profiler GC Mass / FOF GC Mass", va="center", rotation="vertical"
    )
    plt.subplots_adjust(hspace=0.5)
#%%
with plt.rc_context(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.size": 12,
    }
):
    mask = fof_f7_m_field > 10
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), dpi=300, sharex=True)
    ax[0].plot(fof_f7_t_myr[mask], pro_f7_m_field[mask] / fof_f7_m_field[mask])

    ax[0].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")
    ax[0].set_title("$f_{*}$ = 0.70")
    mask = fof_f3_m_field > 10
    ax[1].plot(fof_f3_t_myr[mask], pro_f3_m_field[mask] / fof_f3_m_field[mask])

    ax[1].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")
    ax[1].set_title("$f_{*}$ = 0.35")

    fig.text(
        0.04, 0.5, "Profiler GC Field / FOF GC Field", va="center", rotation="vertical"
    )
    plt.subplots_adjust(hspace=0.5)
