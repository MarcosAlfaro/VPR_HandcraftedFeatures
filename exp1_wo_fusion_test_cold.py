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
from eval.tests import build_vm, get_predictions
from config import PARAMS

device = select_device()
csvDir = create_path(f"{PARAMS.csv_path}Results/")
tf = select_tf(model=PARAMS.model)


with open(csvDir + "EXP01_DepthPreprocessingCOLD.csv", 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(build_header_results_csv(["Model","Backbone", "Desc.size", "Trained", "Modality"]))

    input_types = ["RGB", "GRAYSCALE", "HUE", "MAGNITUDE", "ANGLE"]
    input_types = ["RGB"]

    for input_type in input_types:

        print(f"Image modality: {input_type}\n")
        state_dict_path = None
        state_dict_path = f"{PARAMS.saved_models_path}EXP03_COLD/{input_type}/net.pth"
        net = load_model(model=PARAMS.model, backbone=PARAMS.backbone, embedding_size=PARAMS.embedding_size, 
                         state_dict_path=state_dict_path, device=device)
        net.eval()

        rowCSV = [PARAMS.model, PARAMS.backbone, PARAMS.embedding_size, "Yes" if state_dict_path is None else "No", input_type]
        recall_at_1, recall_at_n = [], []

        with torch.no_grad():

            for env in envs_COLD:

                condIlum = get_cond_ilum(env)

                vmDataset = datasets_cold.Database_wo_Fusion(env=env, input_type=input_type, tf=tf, enc=PARAMS.enc)
                vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)

                print(f"Test Environment: {env}\n")

                descriptorsVM, coordsVM, treeCoords = build_vm(model=net,  dataloader=vmDataloader)

                for ilum in condIlum:
                    print(f"Test {ilum}")
                    idxIlum = condIlum.index(ilum)

                    testDataset = datasets_cold.Test_wo_Fusion(il=ilum, env=env, input_type=input_type, tf=tf, enc=PARAMS.enc)
                    testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

                    pkl_path = f"PKL_FILES/no_fusion/{input_type}/{env}_{ilum}.pkl"
                    if not os.path.exists(pkl_path) or PARAMS.override == True:
                        create_path(os.path.dirname(pkl_path))
                        get_predictions(pkl_path=pkl_path, model=net, testDataloader=testDataloader, descriptorsVM=descriptorsVM,
                                        treeCoords=treeCoords, coordsVM= coordsVM, env=env, lf_method=None, device=device)

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
