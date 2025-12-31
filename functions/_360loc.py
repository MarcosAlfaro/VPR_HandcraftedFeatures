import os
import numpy as np
from config import PARAMS


dataset_dir = PARAMS._360loc_path


def read_coords_txt(env="atrium", ilum="daytime_360_0", imgSet="database"):
    
    if imgSet == "database":
        txtName = f'{dataset_dir}{env}/pose/360_mapping_gt.txt'
    else:
        txtName = f'{dataset_dir}{env}/pose/query_gt_360_{ilum.split("time")[0]}.txt'

    with open(txtName, 'r') as file:
        lines = file.readlines()
    coords = {}
    for line in lines:
        line = line.split()
        imgPath, seqIdx = line[0], line[0].split("/")[-3].split("_")[-1]
        if seqIdx != ilum[-1] and imgSet != "database":
            continue
        coordX, coordY, coordZ = float(line[1]), float(line[2]), float(line[3])
        coords[imgPath] = (coordX, coordY, coordZ)

    return coords


def build_header_results_csv(params):
    header = params + ["R@1 atrium day", "R@1 atrium night", "R@1 atrium avg",
                       "R@1 concourse day", "R@1 concourse night", "R@1 concourse avg",
                       "R@1 hall day", "R@1 hall night", "R@1 hall avg",
                       "R@1 piatrium day", "R@1 piatrium night", "R@1 piatrium avg",
                       "R@1 Global Avg.",
                       "R@N atrium day", "R@N atrium night", "R@N atrium avg",
                       "R@N concourse day", "R@N concourse night", "R@N concourse avg",
                       "R@N hall day", "R@N hall night", "R@N hall avg",
                       "R@N piatrium day", "R@N piatrium night", "R@N piatrium avg",
                       "R@N Global Avg."]
    return header

def build_row_results_csv(params, r1_list, rn_list):
    row = params + [f"{x:.2f}" for x in r1_list] + [f"{x:.2f}" for x in rn_list]
    return row

envs_360loc = ["atrium", "concourse", "hall", "piatrium"]
trainSeq_360loc = ["daytime_360_0", "daytime_360_1", "daytime_360_0", "daytime_360_2"]
condIlum_360loc = [["daytime_360_1", "daytime_360_2", "nighttime_360_1", "nighttime_360_2"],
                     ["daytime_360_0", "daytime_360_2", "nighttime_360_0"],
                     ["daytime_360_1", "daytime_360_2", "nighttime_360_1", "nighttime_360_2"],
                    ["daytime_360_0", "daytime_360_1", "nighttime_360_0"]]






    