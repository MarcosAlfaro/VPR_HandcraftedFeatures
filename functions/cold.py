import os
import numpy as np

def get_coords_from_image_path(imageDir):
    idxX, idxY, idxA = imageDir.index('_x'), imageDir.index('_y'), imageDir.index('_a')
    x, y = float(imageDir[idxX + 2:idxY]), float(imageDir[idxY + 2:idxA])
    return x, y

def get_all_paths_from_seq(seqDir, writer):
    # read all the files from all the folders at seqDir and write them to the csv
    # don't use the rooms variable in order to be able to use this function for any sequence
    img_paths, coords = [], []
    for root, _, files in os.walk(seqDir):
        for file in sorted(files):
            if file.endswith('.jpeg') or file.endswith('.png'):
                relDir = os.path.relpath(root, seqDir)
                imagePath = os.path.join(relDir, file)
                x, y = get_coords_from_image_path(file)
                writer.writerow([imagePath, x, y])
                img_paths.append(imagePath)
                coords.append(np.array([x, y]))
    return img_paths, coords


def get_cond_ilum(env):
    if env in ["FR_A", "SA_B"]:
        condIlum = ['Cloudy', 'Night', 'Sunny']
    elif env == "FR_B":
        condIlum = ['Cloudy', 'Sunny']
    elif env == "SA_A":
        condIlum = ['Cloudy', 'Night']
    else:
        raise ValueError("Environment not available. Valid environments: FR_A, FR_B, SA_A, SA_B")
    return condIlum


def build_header_results_csv(params):
    header = params + ["R@1 FR-A Cl", "R@1 FR-A Ni", "R@1 FR-A Su", "R@1 FR-A Avg.",
                       "R@1 FR-B Cl", "R@1 FR-B Su", "R@1 FR-B Avg.",
                       "R@1 SA-A Cl", "R@1 SA-A Ni", "R@1 SA-A Avg.",
                       "R@1 SA-B Cl", "R@1 SA-B Ni", "R@1 SA-B Su", "R@1 SA-B Avg.",
                       "R@1 Global Avg.",
                       "R@N FR-A Cl", "R@N FR-A Ni", "R@N FR-A Su", "R@N FR-A Avg.",
                       "R@N FR-B Cl", "R@N FR-B Su", "R@N FR-B Avg.",
                       "R@N SA-A Cl", "R@N SA-A Ni", "R@N SA-A Avg.",
                       "R@N SA-B Cl", "R@N SA-B Ni", "R@N SA-B Su", "R@N SA-B Avg.",
                       "R@N Global Avg."]
    return header

def build_row_results_csv(params, r1_list, rn_list):
    row = params + [f"{x:.2f}" for x in r1_list] + [f"{x:.2f}" for x in rn_list]
    return row

envs_COLD = ["FR_A", "FR_B", "SA_A", "SA_B"]






    