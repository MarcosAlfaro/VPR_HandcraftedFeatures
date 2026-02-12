import os
import numpy as np
from config import PARAMS


dataset_dir = PARAMS.rawseeds_path


def read_coords_csv(env="27a"):
    
    csvName = f'{dataset_dir}{env}/sampled_Bicocca_2009-02-{env}_locations.csv'

    coords = {}
    with open(csvName, 'r') as file:
        # read the csv file and extract the image name and the coordinates (not the orientation)
        # the headers are: file,x,y,orientation
        # the output are the coordinates in a dictionary with the image path as key
        lines = file.readlines()
        for i, line in enumerate(lines):
            if i == 0:  # skip header
                continue
            parts = line.strip().split(',')
            if len(parts) >= 3:
                image_path = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                coords[image_path] = (x, y)

    return coords


def build_header_results_csv(params):
    header = params + ["R@1 25a", "R@1 25b", "R@1 26a", "R@1 26b", "R@1 Global Avg.",
                       "R@N 25a", "R@N 25b", "R@N 26a", "R@N 26b", "R@N Global Avg."]
    return header

def build_row_results_csv(params, r1_list, rn_list):
    row = params + [f"{x:.2f}" for x in r1_list] + [f"{x:.2f}" for x in rn_list]
    return row







    