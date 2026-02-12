import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import csv
import datasets_rawseeds
from functions.models import load_model
from functions.misc_functions import create_path, select_device
from functions.rawseeds import build_row_results_csv, build_header_results_csv
from functions.eval_functions import get_avg_results_env_cold, get_glob_results_cold, compute_recalls_from_pkl
from functions.img_process_functions import select_tf
from eval.tests import build_vm_lf_multifeatures, get_predictions
from config import PARAMS


device = select_device()
csvDir = create_path(f"{PARAMS.csv_path}Results/")
tf = select_tf(model=PARAMS.model)

with open(csvDir + "EXP03_LF_rawseeds.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(build_header_results_csv(["Preprocess Method", "LF Method", "Trained"]))

    features = ["RGB", "MAGNITUDE"]
    lf = "concat"
    savedModelsDir = f"{PARAMS.saved_models_path}EXP03_RAWSEEDS/"

    models = []
    for feature in features:
        state_dict_path = None
        #state_dict_path = f"{savedModelsDir}{feature}/net.pth"
        net_feature = load_model(model=PARAMS.model, backbone=PARAMS.backbone, embedding_size=PARAMS.embedding_size, 
                                 state_dict_path=state_dict_path, device=device)
        net_feature.eval()
        models.append(net_feature)
    
    rowCSV = ['_'.join(features), lf, "Yes" if state_dict_path is not None else "No"]
    recall_at_1, recall_at_n = [], []

    with torch.no_grad():
        envs_RAWSEEDS = ["25a", "25b", "26a", "26b"]
        for env in envs_RAWSEEDS:

            vmDataset = datasets_rawseeds.Database_multifeatures(env="27a", features=features, tf=tf)
            vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)

            print(f"\nTest Environment: {env}\n")

            descVM, coordsVM, treeCoords = build_vm_lf_multifeatures(models=models, features=features, dataloader=vmDataloader)


            testDataset = datasets_rawseeds.Test_multifeatures(env=env, features=features, tf=tf)
            testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

            pkl_path = f"PKL_FILES/LF/{'_'.join(features)}_{lf}/{env}.pkl"
            if not os.path.exists(pkl_path) or PARAMS.override == True:
                create_path(os.path.dirname(pkl_path))
                get_predictions(pkl_path=pkl_path, model=models, testDataloader=testDataloader, 
                                descriptorsVM=descVM, features=features,
                                treeCoords=treeCoords, coordsVM= coordsVM, env=env, lf_method=lf, device=device)
            r1, rn = compute_recalls_from_pkl(pkl_path=pkl_path)
            recall_at_1.append(r1)
            recall_at_n.append(rn)
            print(f"Env: {env}, R@1 = {r1}, R@1% = {rn}")

        glob_r1, glob_rn = np.average(np.array(recall_at_1)), np.average(np.array(recall_at_n))
        print(f"Global Average R@1 = {glob_r1}, R@N = {glob_rn}\n")
        recall_at_1.append(glob_r1)
        recall_at_n.append(glob_rn)
    rowCSV = build_row_results_csv(rowCSV, recall_at_1, recall_at_n)
    writer.writerow(rowCSV)
