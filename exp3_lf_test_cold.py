import torch
from torch.utils.data import DataLoader
import os
import csv
import datasets_cold
from functions.models import load_model
from functions.misc_functions import create_path, select_device
from functions.cold import get_cond_ilum, build_row_results_csv, build_header_results_csv, envs_COLD
from functions.eval_functions import get_avg_results_env_cold, get_glob_results_cold, compute_recalls_from_pkl
from functions.img_process_functions import select_tf
from eval.tests import build_vm_lf, get_predictions
from config import PARAMS


device = select_device()
csvDir = create_path(f"{PARAMS.csv_path}Results/")
tf = select_tf(model=PARAMS.model)

with open(csvDir + "EXP03_LF_COLD.csv", 'a', newline='') as file:
    writer = csv.writer(file)
    #writer.writerow(build_header_results_csv(["Preprocess Method", "LF Method"]))

    preprocessing_method = "ANGLE"
    lf_methods = ["sum"]
    savedModelsDir = f"{PARAMS.saved_models_path}/EXP03_COLD/"

    for lf in lf_methods:

        print(f"Preprocess Method: {preprocessing_method}, LF Method: {lf}\n")
        state_dict_path_RGB, state_dict_path_depth = None, None
        #state_dict_path_RGB, state_dict_path_depth = f"{savedModelsDir}RGB/net.pth", f"{savedModelsDir}{prep_method}/net.pth"
        net_RGB = load_model(model=PARAMS.model, backbone=PARAMS.backbone, embedding_size=PARAMS.embedding_size, 
                         state_dict_path=state_dict_path_RGB, device=device)
        net_RGB.eval()
        state_dict_path_depth = None
        net_depth = load_model(model=PARAMS.model, backbone=PARAMS.backbone, embedding_size=PARAMS.embedding_size, 
                         state_dict_path=state_dict_path_depth, device=device)
        net_depth.eval()
        
        rowCSV = [preprocessing_method, lf]
        recall_at_1, recall_at_n = [], []

        with torch.no_grad():

            for env in envs_COLD:

                condIlum = get_cond_ilum(env)

                vmDataset = datasets_cold.Database_LF(env=env, input_type=preprocessing_method, enc="vitl", tf=tf)
                vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)

                print(f"\nTest Environment: {env}\n")
 
                descVM_RGB, descVM_depth, coordsVM, treeCoords = build_vm_lf(model_RGB=net_RGB, model_f=net_depth, dataloader=vmDataloader)

                for ilum in condIlum:
                    print(f"Test {ilum}")
                    idxIlum = condIlum.index(ilum)

                    testDataset = datasets_cold.Test_LF(il=ilum, env=env, input_type=preprocessing_method, enc="vitl", tf=tf)
                    testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

                    pkl_path = f"PKL_FILES/LF/{preprocessing_method}_{lf}/{env}_{ilum}.pkl"
                    if not os.path.exists(pkl_path) or PARAMS.override == True:
                        create_path(os.path.dirname(pkl_path))
                        get_predictions(pkl_path=pkl_path, model=[net_RGB, net_depth], testDataloader=testDataloader, 
                                        descriptorsVM=[descVM_RGB, descVM_depth],
                                        treeCoords=treeCoords, coordsVM= coordsVM, env=env, lf_method=lf, device=device)
                    r1, rn = compute_recalls_from_pkl(pkl_path=pkl_path)
                    recall_at_1.append(r1)
                    recall_at_n.append(rn)
                    print(f"Env: {env}, ilum: {ilum}, R@1 = {r1}, R@1% = {rn}")

                avg_r1, avg_rn = get_avg_results_env_cold(recall_at_1[len(recall_at_1)-len(condIlum):], 
                                                          recall_at_n[len(recall_at_n)-len(condIlum):])
                recall_at_1.append(avg_r1)
                recall_at_n.append(avg_rn)
                print(f"Env: {env} Average, R@1 = {avg_r1}, R@N = {avg_rn}\n")

            glob_r1, glob_rn = get_glob_results_cold(recall_at_1, recall_at_n)
            print(f"Global Average R@1 = {glob_r1}, R@N = {glob_rn}\n")
            recall_at_1.append(glob_r1)
            recall_at_n.append(glob_rn)
        rowCSV = build_row_results_csv(rowCSV, recall_at_1, recall_at_n)
        writer.writerow(rowCSV)
