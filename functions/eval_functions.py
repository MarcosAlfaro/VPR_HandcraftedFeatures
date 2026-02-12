import pickle
import numpy as np

def get_distance_threshold(env):
    if env in ["FR_A", "FR_B", "SA_A", "SA_B"]:
        d_threshold = 0.5
    elif env in ["atrium", "hall", "piatrium"]:
        d_threshold = 10
    elif env == "concourse":
        d_threshold = 5
    else:
        d_threshold = 1
    return d_threshold


def compute_recalls_from_pkl(pkl_path):

    with open(pkl_path, "rb") as f:
        results_list = pickle.load(f)

    recall_at_1, recall_at_n = 0, 0

    for result in results_list:
        correct_index = result["correct_index"]
        if correct_index != -1:
            recall_at_n += 1
            if correct_index == 0:
                recall_at_1 += 1

    recall_at_1 *= 100/len(results_list)
    recall_at_n *= 100/len(results_list)

    return recall_at_1, recall_at_n

def get_avg_results_env_cold(r1_list, rn_list):
    avg_r1 = sum(r1_list) / len(r1_list)
    avg_rn = sum(rn_list) / len(rn_list)
    return avg_r1, avg_rn

def get_avg_results_env_360loc(r1_list, rn_list):
    if len(r1_list) == 3:
        avg_r1_day = (r1_list[0] + r1_list[1]) / 2
        avg_rn_day = (rn_list[0] + rn_list[1]) / 2
        avg_r1_night = r1_list[2]
        avg_rn_night = rn_list[2]
    else:
        avg_r1_day = (r1_list[0] + r1_list[1]) / 2
        avg_rn_day = (rn_list[0] + rn_list[1]) / 2
        avg_r1_night = (r1_list[2] + r1_list[3]) / 2
        avg_rn_night = (rn_list[2] + rn_list[3]) / 2
    avg_r1 = np.average([avg_r1_day, avg_r1_night])
    avg_rn = np.average([avg_rn_day, avg_rn_night])
    return avg_r1_day, avg_rn_day, avg_r1_night, avg_rn_night, avg_r1, avg_rn


def get_glob_results_cold(r1_list, rn_list):
    glob_r1 = sum(r1_list[0:3] + r1_list[4:6] + r1_list[7:9] + r1_list[10:13]) / 10
    glob_rn = sum(rn_list[0:3] + rn_list[4:6] + rn_list[7:9] + rn_list[10:13]) / 10
    return glob_r1, glob_rn

def get_glob_results_360loc(r1_list, rn_list):
    glob_r1 = np.average([r1_list[6], r1_list[12], r1_list[19], r1_list[25]])
    glob_rn = np.average([rn_list[6], rn_list[12], rn_list[19], rn_list[25]])
    r1_list = r1_list[4:7] + r1_list[10:13] + r1_list[17:20] + r1_list[23:26]
    rn_list = rn_list[4:7] + rn_list[10:13] + rn_list[17:20] + rn_list[23:26]
    return glob_r1, glob_rn, r1_list, rn_list
