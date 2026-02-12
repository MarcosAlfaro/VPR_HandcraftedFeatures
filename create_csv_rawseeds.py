import csv
from sklearn.neighbors import KDTree
from functions.rawseeds import read_coords_csv
from functions.train_functions import select_triplet_sample
from functions.misc_functions import create_path
from config import PARAMS
import numpy as np

csvDir = create_path(f"{PARAMS.csv_path}train_eval_rawseeds/")
datasetDir = PARAMS.rawseeds_path


def write_csv(env="25a"):

    imageSet = "test" if env in ["25a", "25b", "26a", "26b"] else "database"

    csvName = f'{csvDir}/{imageSet}_{env}.csv'

    coordsDict = read_coords_csv(env=env)
    coords = []
    for img in coordsDict:
        coords.append(np.array(coordsDict[img]))
    imgsList = list(coordsDict.keys())

    with open(csvName, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "CoordX", "CoordY"])
        for img in imgsList:
            coordX, coordY = coordsDict[img]
            writer.writerow([img.split("/")[-1], coordX, coordY])
    return 



def write_train_csv(env="27a", rPos=1, rNeg=1):
    with open(f'{csvDir}/train_{env}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ImgAnc", "ImgPos", "ImgNeg"])

        coordsDict = read_coords_csv(env=env)
        coordsList = []
        for img in coordsDict:
            coordsList.append(np.array(coordsDict[img]))
        imgsList = list(coordsDict.keys())
        tree = KDTree(coordsList, leaf_size=2)

        for _ in range(PARAMS.train_length):
            anc, pos, neg = select_triplet_sample(imgsList, coordsList, tree, rPos, rNeg)
            writer.writerow([anc.split("/")[-1], pos.split("/")[-1], neg.split("/")[-1]])
    return


envs = ["25a", "25b", "26a", "26b", "27a"]
for env in envs:
    write_csv(env=env)
    if env in ["27a"]:
        write_train_csv(env=env, rPos=0.7, rNeg=0.7)
        

