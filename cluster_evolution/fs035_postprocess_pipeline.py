"""
Script that gets the data. Meant to be the most direct translation of the scripts
built up thus far. Not optimized at all.
"""

import sys

# sys.path.insert(
#     1,
#     "/scratch/zt1/project/ricotti-prj/user/fgarcia4/master/lib/python3.7/site-packages",
# )
# sys.path.insert(1, "/homes/fgarcia4/py-virtual-envs/master/lib/python3.7/site-packages")
sys.path.append("..")  # makes sure that importing the modules work

from modules.luminosity.lum_functions import unpack_pop_ii_data
from modules.macros import code_age_to_myr, common_filter_snapshots
from gc_profiles.snapshot_profiler import run_profiler
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import yt
import os
import glob
from modules.macros import filter_snapshots
from yt.extensions.astro_analysis.halo_analysis import HaloCatalog
from scipy import stats

yt.enable_parallelism()

# warnings.simplefilter(action="ignore", category=RuntimeWarning)
print("=============================================================================")
print("fs035_ms10 post processing")
print("=============================================================================")
# ==============================================================================
simulation_run = "fs035_ms10"
finder_profiler_run = "fof_best"
processor_number = 0  # 0 means one process unparalleled
start_step = 1469
end_step = 1575
step = 1
absolute_start = 154


particle_data_proc = True
halo_finder_proc = True
gc_profiler_proc = True
fof_info_proc = True
t_series_proc = True
# ===================================local test=================================
# datadir = os.path.relpath("../../cosm_test_data/refine")
# parent_folder = "../rendering"
# sequence_folder = "test_frames"
# ===================================dt2 paths=================================
# datadir = os.path.expanduser(
#     "/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/{}"  # lustre data path
# ).format(simulation_run)
datadir = os.path.expanduser(
    "/scratch/dt2/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/{}"
).format(simulation_run)
# ===================================save path=================================

pop_2_save = "../particle_data/pop_2_data/{}".format(simulation_run)
sfc_save = "../particle_data/sfc_data/{}".format(simulation_run)
psc_save = "../particle_data/psc_data/{}".format(simulation_run)

halo_data_directory = r"../halo_data/{}/{}".format(simulation_run, finder_profiler_run)
profiler_run_save_path = "../gc_profiles/profile_runs/{}/{}".format(
    simulation_run, finder_profiler_run
)
tseries_save_path = "../sci_plots/fof_time_series/{}_fof_best_{}_{}.txt".format(
    simulation_run, absolute_start, end_step
)

if not os.path.exists(pop_2_save):
    print("# Creating new sequence directory", pop_2_save)
    os.makedirs(pop_2_save)
if not os.path.exists(sfc_save):
    print("# Creating new sequence directory", sfc_save)
    os.makedirs(sfc_save)
if not os.path.exists(psc_save):
    print("# Creating new sequence directory", psc_save)
    os.makedirs(psc_save)


# read fields explicitly, not recognized by YT from this ver of RAMSES
cell_fields = [
    "Density",
    "x-velocity",
    "y-velocity",
    "z-velocity",
    "Pressure",
    "Metallicity",
    # "dark_matter_density",
    "xHI",
    "xHII",
    "xHeII",
    "xHeIII",
]
epf = [
    ("particle_family", "b"),
    ("particle_tag", "b"),
    ("particle_birth_epoch", "d"),
    ("particle_metallicity", "d"),
]

# =============================================================================
#!!! particle data post processing
# =============================================================================

if particle_data_proc is True:
    print(
        "============================================================================="
    )
    print("Extracting Particle Data")
    print(
        "============================================================================="
    )
    for loop_num, output_num in enumerate(range(start_step, end_step + 1)):
        print("#________________________________________________________________")
        infofile = os.path.abspath(
            datadir + f"/output_{output_num:05}/info_{output_num:05}.txt"
        )
        print("# reading in", infofile)

        # read in RAMSES data set
        ds = yt.load(infofile, fields=cell_fields, extra_particle_fields=epf)

        # get time-dependent params.
        redshft = ds.current_redshift
        current_hubble = ds.hubble_constant
        current_time = float(ds.current_time.in_units("Myr"))

        # extract all data fields
        ad = ds.all_data()

        # get SFC/PSC positions and other important fields,
        # need to modify definitions to get these sinks
        pos_sfcs = np.array(ad["SFC", "particle_position"])
        pos_pscs = np.array(ad["PSC", "particle_position"])

        # read POPII star info
        star_id = np.array(ad["star", "particle_identity"])
        be_star = ad["star", "particle_birth_epoch"]
        x_pos = np.array(ad["star", "particle_position_x"])
        y_pos = np.array(ad["star", "particle_position_y"])
        z_pos = np.array(ad["star", "particle_position_z"])

        # center based on star position distribution
        x_center = np.mean(x_pos)
        y_center = np.mean(y_pos)
        z_center = np.mean(z_pos)
        plt_ctr = np.array([x_center, y_center, z_center])
        plt_ctr_in_pc = np.array(ds.arr(plt_ctr, "code_length").to("pc"))

        # translate points to center
        x_pos = x_pos - plt_ctr[0]
        y_pos = y_pos - plt_ctr[1]
        z_pos = z_pos - plt_ctr[2]

        pos_sfcs_recentered = pos_sfcs - plt_ctr
        pos_pscs_recentered = pos_pscs - plt_ctr

        # particle clumps by age; converts code age to relative ages
        unique_birth_epochs = code_age_to_myr(
            ad["star", "particle_birth_epoch"], current_hubble, unique_age=True
        )
        # calculate the age of the universe when the first star was born
        # using the logSFC as a reference point for redshift when the first star
        # was born. Every age is relative to this. Due to our mods of ramses.
        log_sfc = np.loadtxt("../sim_log_files/{}/logSFC".format(simulation_run))
        birth_start = np.round_(
            float(ds.cosmology.t_from_z(log_sfc[0, 2]).in_units("Myr")), 0
        )

        # all the birth epochs of the stars
        converted_unfiltered = code_age_to_myr(
            ad["star", "particle_birth_epoch"], current_hubble, unique_age=False
        )

        # ==========================luminosity mappping data extraction==============

        abs_birth_epochs = np.round(converted_unfiltered + birth_start, 3)  #!
        current_ages = np.round(current_time, 3) - np.round(abs_birth_epochs, 3)
        extra_info = np.array(
            [
                np.concatenate(
                    (np.array([current_time, redshft]), plt_ctr, plt_ctr_in_pc)
                )
            ]
        )
        extra_info.resize(np.size(current_ages))
        star_info = np.array(
            [
                star_id,
                current_ages,
                ds.arr(x_pos, "code_length").to("pc"),
                ds.arr(y_pos, "code_length").to("pc"),
                ds.arr(z_pos, "code_length").to("pc"),
                ds.arr(ad["star", "particle_mass"], "code_mass").to("msun"),
                extra_info,
            ]
        )

        # =========================star positions save=================================

        star_info = np.array(star_info).T
        save_time = str(format(current_time, ".2f")).replace(".", "_")
        save_name = "{}/pos_{:05d}_{}_myr.txt".format(pop_2_save, output_num, save_time)
        header = (
            "\t\tID"
            "\t\tCurrentAges[MYr]"
            "\t\tX[pc]"
            "\t\tY[pc]\t\t"
            "Z[pc]\t\t"
            "mass[Msun]"
            "\t\tt_sim[Myr], z, ctr(code), ctr(pc)"
        )
        np.savetxt(save_name, X=star_info, header=header)
        print("# saved:", save_name)

        # =========================== psc sfc save==================================

        psc_kazu_radii = np.abs(
            ds.arr(ad["PSC", "particle_metallicity"], "code_length").to("pc")
        )
        sfc_kazu_radii = np.abs(
            ds.arr(ad["SFC", "particle_metallicity"], "code_length").to("pc")
        )
        pos_pscs = ds.arr(pos_pscs_recentered, "code_length").to("pc")
        pos_sfcs = ds.arr(pos_sfcs_recentered, "code_length").to("pc")

        # particle tags, see if unique
        psc_tag = np.array(ad["PSC", "particle_index"])
        sfc_tag = np.array(ad["SFC", "particle_index"])

        # save paths
        psc_path = "{}/psc_{:05d}_{}_myr.txt".format(psc_save, output_num, save_time)
        sfc_path = "{}/sfc_{:05d}_{}_myr.txt".format(sfc_save, output_num, save_time)
        # x(pc), y(pc), z(pc),radii at birth (pc), particle tag
        psc_save_data = np.concatenate(
            (pos_pscs, psc_kazu_radii[:, None], psc_tag[:, None]), axis=1
        )
        sfc_save_data = np.concatenate(
            (pos_sfcs, sfc_kazu_radii[:, None], sfc_tag[:, None]), axis=1
        )
        test_particale_header = "x(pc), y(pc), z(pc),radii at birth (pc), particle tag"
        print("# saved:", psc_path)
        print("# saved:", sfc_path)
        np.savetxt(psc_path, X=psc_save_data, header=test_particale_header)
        np.savetxt(sfc_path, X=sfc_save_data, header=test_particale_header)


# =============================================================================
#!!! halo run post procesing
# =============================================================================

# local_snapshots = filter_snapshots(r"../../cosm_test_data/fs035_ms10", 500, 500, 1)
if halo_finder_proc is True:
    print(
        "============================================================================="
    )
    print("RUNING HALO FINDER")
    print(
        "============================================================================="
    )
    datadir = os.path.expanduser(
        "/lustre/fgarcia4/ramses/dwarf/data/cluster_evolution/{}"  # lustre data path
    ).format(simulation_run)
    local_snapshots = filter_snapshots(datadir, start_step, end_step, 1)

    # if post processing isn't done alongside catalogue
    # fof_run_snapshots = filter_snapshots("../halo_data/fs07_refine/fof_best", 113, 918, 1)
    # pop_2_dataset = filter_snapshots(
    #     "../particle_data/pop_2_data/fs07_refine", 113, 918, 1
    # )
    # reduced_h5_list = common_filter_snapshots(fof_run_snapshots, local_snapshots)
    # reduced_pop2_list = common_filter_snapshots(pop_2_dataset, reduced_h5_list)
    # reduced_local_list = common_filter_snapshots(local_snapshots, reduced_h5_list)

    for i, file_name in enumerate(local_snapshots):
        snapshot_num_string = file_name.split("_")[-1]
        ds_file_name = os.path.join(
            file_name, "info_{}.txt".format(snapshot_num_string)
        )
        print("> Reading", file_name)

        ds = yt.load(ds_file_name, fields=cell_fields, extra_particle_fields=epf)
        ad = ds.all_data()
        hc = HaloCatalog(
            data_ds=ds,
            finder_method="fof",
            finder_kwargs={
                "ptype": "star",
                "padding": 0.0001,
                "link": 0.00001,  # "best"
                # "link": 0.0000025, # "fof"
                "dm_only": False,
            },
            output_dir="../halo_data/{}/{}/".format(
                simulation_run, finder_profiler_run
            ),
        )

        hc.create()

        # read POPII star info
        x_pos = np.array(ad["star", "particle_position_x"])
        y_pos = np.array(ad["star", "particle_position_y"])
        z_pos = np.array(ad["star", "particle_position_z"])
        star_id = np.array(ad["star", "particle_identity"])

        # center based on star position distribution
        x_center = np.mean(x_pos)
        y_center = np.mean(y_pos)
        z_center = np.mean(z_pos)
        plt_ctr = np.array([x_center, y_center, z_center])

        # some post processing

        # read in the newly created hdf5 catalogue

        fof_catalogue = "../halo_data/{}/{}/info_{}/info_{}.{}.h5".format(
            simulation_run,
            finder_profiler_run,
            snapshot_num_string,
            snapshot_num_string,
            processor_number,
        )
        cata_h5 = h5.File(fof_catalogue, "r")

        # need to read in using yt for virial radius for some reason unknown units in catalogue
        cata_yt = yt.load(fof_catalogue)

        # make a halo catalogue for yt overplot
        halo_cat_plotting = HaloCatalog(halos_ds=cata_yt)
        halo_cat_plotting.load()

        cata_yt = cata_yt.all_data()

        pop2_data = np.loadtxt(
            glob.glob(
                "../particle_data/pop_2_data/{}/pos_{}_*.txt".format(
                    simulation_run, snapshot_num_string
                )
            )[0]
        )
        ctr_at_pc = pop2_data[5:8, 6]
        ctr_at_code_length = pop2_data[2:5, 6]
        star_ids = pop2_data[:, 0]
        x_pos = pop2_data[:, 2]
        y_pos = pop2_data[:, 3]
        z_pos = pop2_data[:, 4]

        # get the halo centers

        halo_id = np.array(cata_h5["particle_identifier"])

        halo_x = np.array(cata_h5["particle_position_x"])
        halo_y = np.array(cata_h5["particle_position_y"])
        halo_z = np.array(cata_h5["particle_position_z"])

        halo_x = np.array(ds.arr(halo_x, "code_length").to("pc")) - ctr_at_pc[0]
        halo_y = np.array(ds.arr(halo_y, "code_length").to("pc")) - ctr_at_pc[1]
        halo_z = np.array(ds.arr(halo_z, "code_length").to("pc")) - ctr_at_pc[2]

        # get halo virial radii
        halo_vir_rad = np.array(ds.arr(cata_yt["all", "virial_radius"], "cm").to("pc"))

        cat_pc = np.vstack((halo_id, halo_x, halo_y, halo_z, halo_vir_rad)).T
        cat_save_name = "../halo_data/{}/{}/info_{}/catalogue_{}.txt".format(
            simulation_run,
            finder_profiler_run,
            snapshot_num_string,
            snapshot_num_string,
        )
        header = (
            "halo_id \t x_coord [pc] \t y_coord [pc] \t z_coord [pc] \t vir_rad [pc]"
        )
        np.savetxt(cat_save_name, X=cat_pc, header=header)

        # get particles belonging to each halo
        num_stars_in_halo = np.array(cata_h5["particle_number"])
        start_of_new_halo = np.array(cata_h5["particle_index_start"])
        halo_star_ids = np.array(cata_h5["particles/ids"])

        cata_h5.close()
        for i, (new_h, h_id) in enumerate(zip(start_of_new_halo, halo_id), start=1):

            if i == np.size(
                start_of_new_halo
            ):  # cheeky over ride once it reaches end of list
                star_ids_inside = halo_star_ids[new_h:]
            else:
                star_ids_inside = halo_star_ids[new_h : start_of_new_halo[i]]

            # translate stars, taking into account centers
            gc_mask = np.isin(star_ids, star_ids_inside)
            gc_x = x_pos[gc_mask]
            gc_y = y_pos[gc_mask]
            gc_z = z_pos[gc_mask]

            # take the x,y,z of individual clusters and subtract center of halo cluster
            # this makes them all centered at the origin (0,0,0)
            gc_stars = np.vstack((gc_x, gc_y, gc_z)).T - cat_pc[:, 1:-1][i - 1]
            gc_stars = np.column_stack((star_ids_inside, gc_stars))
            header = "star id \t star_x_coords [pc] \t star_y_coords [pc] \t star_z_coords [pc] "

            save_name = "../halo_data/{}/{}/info_{}/gc_vir_{}.txt".format(
                simulation_run,
                finder_profiler_run,
                snapshot_num_string,
                str(int(h_id)).zfill(3),
            )
            np.savetxt(save_name, X=gc_stars, header=header)

        # optionally plot the halod finder results and put in the catalogue directory
        # width = (400, "pc")
        # p = yt.ProjectionPlot(ds, "z", "density", width=width, center=ctr_at_code_length)
        # p.annotate_particles(
        #     width=width, ptype="star", alpha=0.5, p_size=0.2, marker=".", col="red"
        # )
        # p.annotate_halos(
        #     halo_cat_plotting,
        #     width=width,
        # )
        # p.set_cmap("density", "copper")
        # p["gas", "density"].axes.scatter(
        #     halo_x,
        #     halo_y,
        #     color="green",
        #     alpha=1,
        #     marker="x",
        #     linewidths=0.1,
        #     s=2,
        # )

        # # p.set_figure_size(5)
        # p.save(
        #     "../halo_data/{}/{}/info_{}/annotated.png".format(
        #         simulation_run, finder_profiler_run, snapshot_num_string
        #     ),
        #     mpl_kwargs={"bbox_inches": "tight", "dpi": 200, "pad_inches": 0.1},
        # )

# =============================================================================
#!!! bsc profiler post processor
# =============================================================================
if gc_profiler_proc is True:
    print(
        "============================================================================="
    )
    print("RUNING PROFILER")
    print(
        "============================================================================="
    )
    plt.rcParams.update({"figure.max_open_warning": 0})

    def fof_profiler(pop2_data_set, halo_data_directory, run_save_path, gc_radii):
        # declare lists that will be filled as the profiler is ran,
        # one data point per snpashot-- time series over an entire sim.

        if not os.path.exists(run_save_path):
            print("# Creating new profiler run directory", run_save_path)
            os.makedirs(run_save_path)

        snapshot = []
        gc_counts_in_snap = []
        gc_masses_in_snap = []
        gc_lums_in_snap = []
        gc_core_masses_in_snap = []
        total_counts_in_snap = []
        total_masses_in_snap = []
        total_lums_in_snap = []
        time_myr_in_snap = []
        redshift_in_snap = []

        # loop over snapshots
        for pop_data, hc_data in zip(pop2_data_set, halo_data_directory):
            pop_data = pop_data.replace("\\", "/")

            snapshot_num = pop_data.split("/")[-1].split("_")[1]

            pop2_data = np.loadtxt(pop_data)

            # load the processed catalogue, recentered and in pc
            catalogue_text_file = os.path.join(
                hc_data, "catalogue_{}.txt".format(snapshot_num)
            )
            halo_data = np.loadtxt(catalogue_text_file)

            # centers with label in the first columns
            try:
                ctrs_pc = halo_data[:, :-1]
                if gc_radii == "vir_rad":
                    profile_gcs_up_to = halo_data[:, -1]
                elif isinstance(gc_radii, int):
                    profile_gcs_up_to = gc_radii
                else:
                    print("> gc radius not supported")
                    break
            except:
                # triggered if there is only one in the catalogue
                ctrs_pc = halo_data[:-1]
                if gc_radii == "vir_rad":
                    profile_gcs_up_to = halo_data[-1]
                elif isinstance(gc_radii, int):
                    profile_gcs_up_to = gc_radii
                else:
                    print("> gc radius not supported")
                    break

            # put all verbose output into a text file
            # orig_stdout = sys.stdout
            # sys.stdout = open(os.path.join(run_save_path, "log.txt"), "w")

            (
                gc_full_masses,
                gc_lums,
                core_radii,
                core_masses,
                r_trunc,
                ages,
                time,
                particles_in_this_snapshot,
                pop_2_ids_in_gcs,
                pop_2_ids_in_fitted_gcs,  # profiler run dictated
                scaled_stellar_lums,  # for all stars, UV [erg s^-1 Angstrom-^1]
            ) = run_profiler(
                star_file_path=pop_data,
                parent_save_path=profiler_run_save_path,
                proj_width=400,  # pc
                gc_radii=profile_gcs_up_to,  # radii to be used to extract clusters
                lum_map_bins=4000,  # bad resolution so that you can see it better
                centers=ctrs_pc,
                particle_filter=hc_data,
            )
            # modify pop2 by tacking on luminosities
            pop2_data = np.insert(pop2_data, 2, scaled_stellar_lums, axis=1)
            # print(pop_2_ids_in_gcs)
            # print(pop_2_ids_in_fitted_gcs)
            mask = np.isin(pop2_data[:, 0], pop_2_ids_in_gcs)
            pop2_data_field = pop2_data[~mask]
            pop2_data_bound = pop2_data[mask]

            mask_only_fitted = np.isin(pop2_data[:, 0], pop_2_ids_in_fitted_gcs)
            fitted_pop2_data_field = pop2_data[~mask_only_fitted]
            fitted_pop2_data_bound = pop2_data[mask_only_fitted]

            gc_counts_in_snap.append(np.sum(particles_in_this_snapshot))
            # masses within the vir_radius, full cluster
            gc_masses_in_snap.append(np.sum(gc_full_masses))
            gc_core_masses_in_snap.append(np.sum(core_masses))  # Msun
            # total light in gcs, profiler determined
            gc_lums_in_snap.append(np.sum(gc_lums))

            # idxs adjust due to insertion of luminosity data
            total_masses_in_snap.append(np.sum(pop2_data[:, 6]))  # Msun
            total_lums_in_snap.append(np.sum(scaled_stellar_lums))
            total_counts_in_snap.append(pop2_data.shape[0])
            time_myr_in_snap.append(time)
            redshift_in_snap.append(pop2_data[1, 7])
            snapshot.append(int(snapshot_num))

            pop_2_header = (
                "\t\tID"
                "\t\tCurrentAges[Myr]"
                "\t CurrentLums[erg s^-1 Angstrom-^1]"
                "\t\tX[pc] "
                "Y[pc] "
                "Z[pc] "
                "\t\tmass[Msun]"
                "\t\tt_sim[Myr], z, ctr(code), ctr(pc)"
            )
            # save, regardless of profiler outcome
            field_save_path = os.path.join(hc_data, "field_stars.txt")
            clust_save_path = os.path.join(hc_data, "bound_stars.txt")
            # save only those that weer succesfully fitted
            fitted_field_save_path = os.path.join(hc_data, "fitted_field_stars.txt")
            fitted_clust_save_path = os.path.join(hc_data, "fitted_bound_stars.txt")
            # save the feild / bound stars as seen by the halo finder,
            # doesn't care if fitted or not
            np.savetxt(
                fname=field_save_path,
                X=pop2_data_field,
                header=pop_2_header,
            )
            np.savetxt(
                fname=clust_save_path,
                X=pop2_data_bound,
                header=pop_2_header,
            )

            np.savetxt(
                fname=fitted_field_save_path,
                X=fitted_pop2_data_field,
                header=pop_2_header,
            )
            np.savetxt(
                fname=fitted_clust_save_path,
                X=fitted_pop2_data_bound,
                header=pop_2_header,
            )

            # undo put all verbose output into a text file
            # sys.stdout.close()
            # sys.stdout = orig_stdout

            # profiler run stats, one row per snapshot, save per loop for crashes
            save_gc_counts = np.array(gc_counts_in_snap)
            save_gc_masses = np.array(gc_masses_in_snap)
            save_gc_lums = np.array(gc_lums_in_snap)
            save_gc_core_masses = np.array(gc_core_masses_in_snap)
            save_total_counts = np.array(total_counts_in_snap)
            save_total_masses = np.array(total_masses_in_snap)
            save_total_lums = np.array(total_lums_in_snap)
            save_snapshot = np.array(snapshot)
            save_time_myr = np.array(time_myr_in_snap)
            save_redshift = np.array(redshift_in_snap)

            prof_run_header = (
                "\t Snapshotnum"
                "\t Time[MYR]"
                "\t redshift"
                "\t total_counts"
                "\t gc_counts"
                "\t total_masses[Msun]"
                "\t gc_masses[Msun]"
                "\t gc_core_masses[Msun]"
                "\t total_lums[erg s^-1 Angstrom-^1]"
                "\t gc_lums[erg s^-1 Angstrom-^1]"
            )

            output = np.vstack(
                (
                    save_snapshot,
                    save_time_myr,
                    save_redshift,
                    save_total_counts,
                    save_gc_counts,
                    save_total_masses,
                    save_gc_masses,
                    save_gc_core_masses,
                    save_total_lums,
                    save_gc_lums,
                )
            ).T

            save_name = os.path.join(
                run_save_path,
                "time_series_run_stats_{}_to_{}.txt".format(
                    save_snapshot.min(), save_snapshot.max()
                ),
            )
            for f in glob.glob(os.path.join(run_save_path, "*.txt")):
                os.remove(f)
            np.savetxt(fname=save_name, X=output, header=prof_run_header)

    # pop2_data_directory = r"../particle_data/pop_2_data/fs07_refine"
    # halo_data_directory = r"../halo_data/fs07_refine/fof_best"

    # save_path = "./profile_runs/fs07_refine/fof_best_v3"

    # strt = 113
    # end = 918
    # step = 1

    pop2 = filter_snapshots(pop_2_save, start_step, end_step, step)
    halos = filter_snapshots(halo_data_directory, start_step, end_step, step)

    pop2 = common_filter_snapshots(pop2, halos)

    fof_profiler(
        pop2_data_set=pop2,
        halo_data_directory=halos,
        run_save_path=profiler_run_save_path,
        gc_radii="vir_rad",
    )


if fof_info_proc is True:
    print(
        "============================================================================="
    )
    print("MAKING FOF RESULT INFO FILE")
    print(
        "============================================================================="
    )
    pop2 = filter_snapshots(pop_2_save, start_step, end_step, step)
    halo_ds = filter_snapshots(halo_data_directory, start_step, end_step, step)

    # TODO: make a an independent profiler and lum map three panel. etc.
    for p2, ds in zip(pop2, halo_ds):
        output_num = int(ds.split("/")[-1].split("_")[-1])

        _, scaled_stellar_lums, masses, ages, tz, ids = unpack_pop_ii_data(
            p2, return_ids=True, return_z=True
        )
        t_myr = tz[0]
        redshift = tz[1]

        halos = sorted(glob.glob(os.path.join(ds, "gc_vir_*.txt")))

        gc_ages_per_snapshot = []
        gc_masses_per_snapshot = []
        gc_lums_per_snapshot = []

        for halo in halos:

            halo_data = np.loadtxt(halo)

            halo_num = int(halo.split("/")[-1].split("_")[-1].split(".")[0])
            gc_mask = np.isin(ids, halo_data[:, 0])
            gc_lums = scaled_stellar_lums[gc_mask]
            gc_mass = masses[gc_mask]
            gc_ages = ages[gc_mask] / 1e6

            gc_masses_per_snapshot.append(np.sum(gc_mass))
            gc_lums_per_snapshot.append(np.sum(gc_lums))
            gc_ages_per_snapshot.append(float(stats.mode(gc_ages)[0]))

        gc_ages_per_snapshot = np.array(gc_ages_per_snapshot)
        gc_masses_per_snapshot = np.array(gc_masses_per_snapshot)
        gc_lums_per_snapshot = np.array(gc_lums_per_snapshot)

        t_myr = np.array([t_myr])
        redshift = np.array([redshift])
        t_myr.resize(np.size(gc_lums_per_snapshot))
        redshift.resize(np.size(gc_lums_per_snapshot))
        output = np.vstack(
            (
                t_myr,
                redshift,
                gc_ages_per_snapshot,
                gc_masses_per_snapshot,
                gc_lums_per_snapshot,
            )
        ).T

        header = "t_myr, z, gc_ages_per_snapshot, gc_masses_per_snapshot, gc_lums_per_snapshot"
        info_save_path = os.path.join(ds, "fof_info.txt")
        np.savetxt(fname=info_save_path, X=output, header=header)
        print("> saved", info_save_path)


if t_series_proc is True:
    print(
        "============================================================================="
    )
    print("GETTING TIME SERIES INFO")
    print(
        "============================================================================="
    )
    # halo_data_directory = r"../../halo_data/fs035_ms10/fof_best"
    # pop2_data_directory = r"../../particle_data/pop_2_data/fs035_ms10"
    # save_pth = "../sci_plots/fof_time_series/fs035_fof_best_154_1316.txt"
    # strt = 154
    # end = 1316
    # step = 1
    pop2 = filter_snapshots(pop_2_save, absolute_start, end_step, step)
    halos_ds = filter_snapshots(halo_data_directory, absolute_start, end_step, step)
    pop2_ds = common_filter_snapshots(pop2, halos_ds)
    # fof bound and unbound

    t = []
    rdshift = []

    fof_bound_mass = []
    fof_field_mass = []

    prof_bound_mass = []
    prof_field_mass = []

    fof_bound_lumi = []
    fof_field_lumi = []

    prof_bound_lumi = []
    prof_field_lumi = []

    for (ds, p2) in zip(halos_ds, pop2_ds):
        print("> including", p2)
        p_ii = np.loadtxt(p2)
        t_myr, z = p_ii[
            :2, 6
        ]  # old pop 2 data, no luminosities, adjust idx accordingly
        # pure friends of friends algorithm
        bound = np.loadtxt(os.path.join(ds, "bound_stars.txt"))
        field = np.loadtxt(os.path.join(ds, "field_stars.txt"))
        # exclude if not fitted
        bound_fitted = np.loadtxt(os.path.join(ds, "fitted_bound_stars.txt"))
        field_fitted = np.loadtxt(os.path.join(ds, "fitted_field_stars.txt"))
        try:
            lum_bound = np.sum(bound[:, 2])
            m_bound = np.sum(bound[:, 6])

        except:
            lum_bound = 0
            m_bound = 0

        try:
            lum_field = np.sum(field[:, 2])
            m_field = np.sum(field[:, 6])
        except:
            lum_field = 0
            m_field = 0

        try:
            lum_bound_fitted = np.sum(bound_fitted[:, 2])
            m_bound_fitted = np.sum(bound_fitted[:, 6])
        except:
            lum_bound_fitted = 0
            m_bound_fitted = 0

        try:
            lum_field_fitted = np.sum(field_fitted[:, 2])
            m_field_fitted = np.sum(field_fitted[:, 6])
        except:
            lum_field_fitted = 0
            m_field_fitted = 0

        t.append(t_myr)
        rdshift.append(z)

        fof_bound_mass.append(m_bound)
        fof_field_mass.append(m_field)
        prof_bound_mass.append(m_bound_fitted)
        prof_field_mass.append(m_field_fitted)

        fof_bound_lumi.append(lum_bound)
        fof_field_lumi.append(lum_field)
        prof_bound_lumi.append(lum_bound_fitted)
        prof_field_lumi.append(lum_field_fitted)

    # x, y, z = np.loadtxt(os.path.join(ds, "field_stars.txt"))
    t = np.array(t)
    rdshift = np.array(rdshift)

    fof_bound_mass = np.array(fof_bound_mass)
    fof_field_mass = np.array(fof_field_mass)

    prof_bound_mass = np.array(prof_bound_mass)
    prof_field_mass = np.array(prof_field_mass)

    fof_bound_lumi = np.array(fof_bound_lumi)
    fof_field_lumi = np.array(fof_field_lumi)

    prof_bound_lumi = np.array(prof_bound_lumi)
    prof_field_lumi = np.array(prof_field_lumi)

    data = np.vstack(
        (
            t,
            rdshift,
            fof_bound_mass,
            fof_field_mass,
            fof_bound_lumi,
            fof_field_lumi,
            prof_bound_mass,
            prof_field_mass,
            prof_bound_lumi,
            prof_field_lumi,
        )
    ).T

    h = (
        "\t\t\t time \t\t\t\t redshift \t\t\t"
        "\tfof_bound_mass \t\t fof_field_mass \t\t"
        "\tfof_bound_lumi \t\t fof_field_lumi \t\t"
        "\tprof_bound_mass \t\t profiler_field_mass \t\t"
        "\tprof_bound_lumi \t\t prof_field_lumi \t\t"
    )

    np.savetxt(tseries_save_path, X=data, header=h)
