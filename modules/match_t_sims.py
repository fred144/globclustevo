import sys

sys.path.append("..")
import numpy as np
from modules.macros import filter_snapshots


def find_matching_time(sequence, look_up_sequence, orig_seq_out_num=False):
    r"""
    given a sequence, return files from look_up sequence with the same time
    Example:

        fs070 = filter_snapshots("../particle_data/pop_2_data/fs07_refine", 113, 1000, 1)
        fs035 = filter_snapshots("../particle_data/pop_2_data/fs035_ms10", 154, 917, 1)
        f3_matched = find_matching_time(fs035, fs070)
    """
    look_up_sequence_output_nums = []
    look_up_sequence_output_times = []
    for file_name in look_up_sequence:
        try:
            out_num = int(file_name.split("/")[-1].split("_")[1])
        except:
            out_num = int(file_name.split("\\")[-1].split("_")[1])
        sim_time = np.loadtxt(file_name, max_rows=2)[0, 6]
        look_up_sequence_output_nums.append(out_num)
        look_up_sequence_output_times.append(sim_time)
    look_up_sequence_output_nums = np.array(look_up_sequence_output_nums)
    look_up_sequence_output_times = np.array(look_up_sequence_output_times)

    sequence_output_nums = []
    sequence_output_times = []
    for file_name in sequence:
        sim_time = np.loadtxt(file_name, max_rows=2)[0, 6]
        # sim_time = float(file_name.split("/")[-1][10:16].replace("_", "."))
        sequence_output_nums.append(out_num)
        sequence_output_times.append(sim_time)
    sequence_output_nums = np.array(sequence_output_nums)
    sequence_output_times = np.array(sequence_output_times)

    # since fs70 has progressed more, use it as a lookup table to match fs035
    residuals = np.abs(
        look_up_sequence_output_times - sequence_output_times[:, np.newaxis]
    )
    closest_match_idxs = np.argmin(residuals, axis=1)

    look_up_sequence_same_time = look_up_sequence_output_nums[closest_match_idxs]

    filter_list = list(map(str, list(look_up_sequence_same_time)))
    filter_list = [f.zfill(5) for f in filter_list]

    common_items = []
    for num in filter_list:
        for file in look_up_sequence:
            if num in file:
                common_items.append(file)
                break

    if orig_seq_out_num is True:

        orig_sequence_output_nums = []
        orig_sequence_output_times = []
        for file_name in sequence:
            try:
                out_num = int(file_name.split("/")[-1].split("_")[1])
            except:
                out_num = int(file_name.split("\\")[-1].split("_")[1])
            sim_time = np.loadtxt(file_name, max_rows=2)[0, 6]
            orig_sequence_output_nums.append(out_num)
            orig_sequence_output_times.append(sim_time)
        orig_sequence_output_nums = np.array(orig_sequence_output_nums)
        orig_sequence_output_times = np.array(orig_sequence_output_times)

        return common_items, look_up_sequence_same_time, orig_sequence_output_nums
    else:

        return common_items, look_up_sequence_same_time


def get_snapshots(snapshot_file_list, get_list, verbose=True):

    filter_list = list(map(str, list(get_list)))
    filter_list = [f.zfill(5) for f in filter_list]
    filtered_lst = []
    not_found = []
    # print(filter_list)
    for num in filter_list:
        # print(num)
        for file in snapshot_file_list:
            # print(snapshot_file_list)
            found = False
            if num in file:
                # print(file)
                filtered_lst.append(file)
                found = True
                break

        if found is False:
            not_found.append(num)

    if len(filtered_lst) != len(filter_list) and verbose is True:
        print("> Filter done, but missing some snapshots requested")
        print(not_found)
    return filtered_lst


fs070 = filter_snapshots("../particle_data/pop_2_data/fs07_refine", 113, 1000, 1)
fs035 = filter_snapshots("../particle_data/pop_2_data/fs035_ms10", 154, 917, 1)
_, f7_matched_nums = find_matching_time(fs035, fs070)
f7_generic = get_snapshots(
    filter_snapshots("../halo_data/fs07_refine/fof_best", 113, 917, 1),
    f7_matched_nums,
)
