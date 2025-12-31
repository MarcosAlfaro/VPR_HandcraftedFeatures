import csv
from sklearn.neighbors import KDTree
from functions.cold import get_cond_ilum, get_all_paths_from_seq, envs_COLD
from functions.train_functions import select_triplet_sample
from functions.misc_functions import create_path
from config import PARAMS

csvDir = create_path(f"{PARAMS.csv_path}train_eval_COLD/")
datasetDir = PARAMS.cold_path


def train(epochLength, tree, env="FR_A"):
    with open(f'{csvDir}Train_{env}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ImgAnc", "ImgPos", "ImgNeg"])
        for _ in range(epochLength):
            imgAnc, imgPos, imgNeg = select_triplet_sample(imgsList, coordsList, tree, PARAMS.r_pos, PARAMS.r_neg)
            writer.writerow([imgAnc, imgPos, imgNeg])
    return


def validation(env="FR_A"):
    ds_dir = f'{datasetDir}{env}/Validation'
    csv_path = f'{csvDir}Validation_{env}.csv'
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "CoordX", "CoordY"])
        _, _ = get_all_paths_from_seq(ds_dir, writer)
    return

def test(env="FR_A", il="Cloudy"):
    ds_dir = f'{datasetDir}{env}/Test{il}'
    csv_path = f'{csvDir}Test_{env}_{il}.csv'
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "CoordX", "CoordY"])
        _, _ = get_all_paths_from_seq(ds_dir, writer)
    return


def visual_model(env="FR_A"):
    ds_dir = f'{datasetDir}{env}/Train'
    csv_path = f'{csvDir}VisualModel_{env}.csv'
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "CoordX", "CoordY"])
        images, coords = get_all_paths_from_seq(ds_dir, writer)
    return images, coords


envs = ["FR_A", "FR_B", "SA_A", "SA_B"]
for env in envs_COLD:
    imgsList, coordsList = visual_model(env=env)
    for ilum in get_cond_ilum(env):
        test(il=ilum, env=env)
    if env == "FR_A":
        treeVM = KDTree(coordsList, leaf_size=2)
        train(epochLength=PARAMS.train_length, tree=treeVM)
        validation()
