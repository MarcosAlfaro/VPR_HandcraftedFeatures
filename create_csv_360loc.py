import os
import csv
import numpy as np
from sklearn.neighbors import KDTree
from config import PARAMS
from functions._360loc import read_coords_txt, envs_360loc, trainSeq_360loc, condIlum_360loc
from functions.train_functions import select_triplet_sample
from functions.misc_functions import create_path
# from exp5_select_train_examples import coarse_loc_protocol_1, fine_loc_protocol_1, fine_loc_protocol_2



datasetDir = PARAMS._360loc_path
csvDir = create_path(f"{PARAMS.csv_path}train_eval_360LOC")


# this function has to write a .csv file with the following columns: image path, coordX, coordY, coordZ
def write_csv(env="atrium", imgSet="database", ilum="daytime"):

    csvName = f'{csvDir}/{imgSet}_{env}_{ilum}.csv'

    coordsDict = read_coords_txt(env=env, ilum=ilum, imgSet=imgSet)
    coords = []
    for img in coordsDict:
        coords.append(np.array(coordsDict[img]))
    imgsList = list(coordsDict.keys())

    with open(csvName, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "CoordX", "CoordY", "CoordZ"])
        for img in imgsList:
            coordX, coordY, coordZ = coordsDict[img]
            writer.writerow([img.split("/")[-1], coordX, coordY, coordZ])
    return 



def write_train_csv(env="atrium", ilum="daytime", rPos=1, rNeg=1):
    with open(f'{csvDir}/train_{env}_{ilum}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ImgAnc", "ImgPos", "ImgNeg"])

        coordsDict = read_coords_txt(env=env, ilum=ilum, imgSet="database")
        coordsList = []
        for img in coordsDict:
            coordsList.append(np.array(coordsDict[img]))
        imgsList = list(coordsDict.keys())
        tree = KDTree(coordsList, leaf_size=2)

        for _ in range(PARAMS.train_length):
            anc, pos, neg = select_triplet_sample(imgsList, coordsList, tree, rPos, rNeg)
            writer.writerow([anc.split("/")[-1], pos.split("/")[-1], neg.split("/")[-1]])
    return


for envs in envs_360loc:
    for ilum in condIlum_360loc[envs_360loc.index(envs)]:
        write_csv(env=envs, imgSet="test", ilum=ilum)
    train_seq = trainSeq_360loc[envs_360loc.index(envs)]
    write_csv(env=envs, imgSet="database", ilum=train_seq)
    write_train_csv(env=envs, ilum=train_seq, rPos=PARAMS.r_pos_360loc, rNeg=PARAMS.r_neg_360loc)
