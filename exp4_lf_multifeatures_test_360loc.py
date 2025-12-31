import torch
from torch.utils.data import DataLoader
import os
import csv
import datasets_360loc
from functions.models import load_model
from functions.misc_functions import create_path, select_device
from functions._360loc import condIlum_360loc, trainSeq_360loc, envs_360loc, build_row_results_csv, build_header_results_csv
from functions.eval_functions import get_avg_results_env_360loc, get_glob_results_360loc, compute_recalls_from_pkl
from functions.img_process_functions import select_tf
from eval.tests import build_vm_lf_multifeatures, get_predictions
from config import PARAMS

device = select_device()
csvDir = create_path(f"{PARAMS.csv_path}Results/")
tf = select_tf(model=PARAMS.model)


with open(csvDir + "EXP03_LF_360Loc.csv", 'a', newline='') as file:
    writer = csv.writer(file)
    #writer.writerow(build_header_results_csv(["Preprocess Method", "LF Method"]))

    features = ["RGB", "GRAYSCALE", "MAGNITUDE", "ANGLE", "HUE"]
    lf = "concat"
    savedModelsDir = f"{PARAMS.saved_models_path}/EXP03_360Loc/"

    models = []
    for feature in features:
        state_dict_path = None
        #state_dict_path = f"{savedModelsDir}{feature}/net.pth"
        net_feature = load_model(model=PARAMS.model, backbone=PARAMS.backbone, embedding_size=PARAMS.embedding_size, 
                                 state_dict_path=state_dict_path, device=device)
        net_feature.eval()
        models.append(net_feature)

    
    rowCSV = [features, lf]
    recall_at_1, recall_at_n = [], []

    with torch.no_grad():

        for env in envs_360loc:

            condIlum = condIlum_360loc[envs_360loc.index(env)]
            trainSeq = trainSeq_360loc[envs_360loc.index(env)]

            vmDataset = datasets_360loc.Database_multifeatures(env=env, features=features, enc="vitl", tf=tf, il=trainSeq)
            vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)

            print(f"\nTest Environment: {env}\n")

            descVM, coordsVM, treeCoords = build_vm_lf_multifeatures(models=models, features=features, dataloader=vmDataloader)

            for ilum in condIlum:
                print(f"Test {ilum}")
                idxIlum = condIlum.index(ilum)

                testDataset = datasets_360loc.Test_multifeatures(il=ilum, env=env, features=features, enc="vitl", tf=tf)
                testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)
                
                pkl_path = f"PKL_FILES/LF_Multifeatures/{"_".join(features)}_{lf}/{env}_{ilum}.pkl"
                if not os.path.exists(pkl_path) or PARAMS.override == True:
                    create_path(os.path.dirname(pkl_path))
                    get_predictions(pkl_path=pkl_path, model=models, testDataloader=testDataloader, 
                                    descriptorsVM=descVM, features=features,
                                    treeCoords=treeCoords, coordsVM= coordsVM, env=env, lf_method=lf, device=device)
                r1, rn = compute_recalls_from_pkl(pkl_path=pkl_path)
                recall_at_1.append(r1)
                recall_at_n.append(rn)
                print(f"Env: {env}, ilum: {ilum}, R@1 = {r1}, R@1% = {rn}")

            avg_r1_day, avg_rn_day, avg_r1_night, avg_rn_night, avg_r1, avg_rn = get_avg_results_env_360loc(recall_at_1[len(recall_at_1)-len(condIlum):], 
                                                                                                            recall_at_n[len(recall_at_n)-len(condIlum):])
            recall_at_1.extend([avg_r1_day, avg_r1_night, avg_r1])
            recall_at_n.extend([avg_rn_day, avg_rn_night, avg_rn])
            print(f"Env: {env} Average, R@1 = {avg_r1}, R@N = {avg_rn}\n")

        glob_r1, glob_rn, recall_at_1, recall_at_n = get_glob_results_360loc(recall_at_1, recall_at_n)
        print(f"Global Average R@1 = {glob_r1}, R@N = {glob_rn}\n")
        recall_at_1.append(glob_r1)
        recall_at_n.append(glob_rn)
    rowCSV = build_row_results_csv(rowCSV, recall_at_1, recall_at_n)
    writer.writerow(rowCSV)

