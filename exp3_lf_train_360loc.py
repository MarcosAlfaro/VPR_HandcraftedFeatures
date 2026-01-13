import torch
import os
import csv
import sys
import numpy as np
from torch.utils.data import DataLoader
from sklearn.neighbors import KDTree
import functions.losses as losses
import datasets_360loc
from functions.models import load_model
from functions.misc_functions import create_path, select_device
from functions._360loc import condIlum_360loc
from functions.eval_functions import compute_recalls_from_pkl
from functions.img_process_functions import select_tf
from eval.tests import build_vm, get_predictions
from functions.train_functions import make_deterministic
from config import PARAMS


device = select_device()
csvDir = create_path(f"{PARAMS.csv_path}Train/")
tf = select_tf(model=PARAMS.model)
baseModelDir = create_path(f"{PARAMS.saved_models_path}EXP03_360Loc/")


"""NETWORK TRAINING"""

with open(f"{csvDir}/EXP03_LF_360Loc.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Method", "Iteration", "R@1 Day1", "R@1 Day2", "R@1 Night1", "R@1 Night2", "R@1 Avg."])

    input_types = ["RGB"]
    
    print("Training Late Fusion 360Loc database")

    for input_type in input_types:

        vmDataset = datasets_360loc.Database_wo_Fusion(input_type=input_type, env="atrium", il="daytime_360_0", tf=tf)
        vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)

        if input_types.index(input_type) == 0:
            coordsVM = []
            for i, vmData in enumerate(vmDataloader, 0):
                _, coords = vmData
                coordsVM.append(coords.detach().numpy()[0])
            treeCoordsVM = KDTree(coordsVM, leaf_size=2)

        criterion = losses.get_loss(PARAMS.loss)
        if criterion == -1:
            sys.exit()

        trainDataset = datasets_360loc.Train_LF(input_type=input_type, env="atrium", il="daytime_360_0", tf=tf)
        trainDataloader = DataLoader(trainDataset, shuffle=False, num_workers=0, batch_size=PARAMS.batch_size)

        netDir = create_path(f"{baseModelDir}{input_type}/")
        net = load_model(model=PARAMS.model, backbone=PARAMS.backbone, embedding_size=PARAMS.embedding_size, 
                         state_dict_path=None, device=device)
        net.aggregation.requires_grad_(False)
        net.backbone.requires_grad_(True)

        # Mantener semillas fijas
        make_deterministic(42)
        
        print(f"\n\nInput type: {input_type}\n")

        bestRecall = 0

        optimizer = torch.optim.SGD(net.parameters(), lr=PARAMS.lr, momentum=0.9)

        for i, data in enumerate(trainDataloader, 0):

            anc, pos, neg = data[0].float().to(device), data[1].float().to(device), data[2].float().to(device)

            optimizer.zero_grad()

            output1, output2, output3 = net(anc), net(pos), net(neg)

            loss = criterion(output1, output2, output3, PARAMS.margin)
            loss.backward()

            optimizer.step()

            if i % int(len(trainDataloader) / PARAMS.num_validations) == 0 and i > 0:
                print(f"It{i}, Loss:{loss}")

                net.eval()

                with torch.no_grad():

                    condIlum = condIlum_360loc[0]
                    recall_at_1, recall_at_n = [], []

                    descriptorsVM, coordsVM, treeCoords = build_vm(model=net, dataloader=vmDataloader)
                    for ilum in condIlum:
                        idxIlum = condIlum.index(ilum)

                        testDataset = datasets_360loc.Test_wo_Fusion(il=ilum, env="atrium", input_type=input_type, tf=tf, enc=PARAMS.enc)
                        testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

                        pkl_path = f"PKL_FILES/no_fusion/{input_type}/atrium_{ilum}_val.pkl"
                        if not os.path.exists(pkl_path):
                            create_path(os.path.dirname(pkl_path))
                        get_predictions(pkl_path=pkl_path, model=net, testDataloader=testDataloader, 
                                        descriptorsVM=descriptorsVM, env="atrium",
                                        treeCoords=treeCoords, coordsVM= coordsVM, device=device)
                        r1, rn = compute_recalls_from_pkl(pkl_path=pkl_path)
                        recall_at_1.append(r1)
                        recall_at_n.append(rn)
                        print(f"Env: atrium, ilum: {ilum}, R@1 = {r1}, R@1% = {rn}")

                    avg_recall_at_1, avg_recall_at_n = np.average(np.array(recall_at_1)), np.average(np.array(recall_at_n))
                    print(f"Env: atrium, R@1 = {avg_recall_at_1}, R@1% = {avg_recall_at_n}\n")

                    if avg_recall_at_1 > bestRecall:
                        bestRecall = avg_recall_at_1
                        netName = os.path.join(netDir, f"net.pth")
                        torch.save(net.state_dict(), netName)
                        print("Modelo guardado")
                        writer.writerow([input_type, str(i + 1), recall_at_1[0], recall_at_1[1], recall_at_1[2], avg_recall_at_1])

                net.backbone.requires_grad_(True)

        print(f"Training finished, Best Recall: {bestRecall}")
        net.eval()
        condIlum = condIlum_360loc[0]
        recall_at_1, recall_at_n = [], []
        with torch.no_grad():
            descriptorsVM, coordsVM, treeCoords = build_vm(model=net, dataloader=vmDataloader)
            for ilum in condIlum:
                idxIlum = condIlum.index(ilum)

                testDataset = datasets_360loc.Test_wo_Fusion(il=ilum, input_type=input_type, tf=tf)
                testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

                get_predictions(pkl_path=pkl_path, model=net, testDataloader=testDataloader, 
                                descriptorsVM=descriptorsVM, env="atrium",
                                treeCoords=treeCoords, coordsVM= coordsVM, device=device)
                r1, rn = compute_recalls_from_pkl(pkl_path=pkl_path)
                recall_at_1.append(r1)
                recall_at_n.append(rn)
                print(f"Env: atrium, ilum: {ilum}, R@1 = {r1}, R@1% = {rn}")

            avg_recall_at_1, avg_recall_at_n = np.average(np.array(recall_at_1)), np.average(np.array(recall_at_n))
            print(f"Env: atrium, R@1 = {avg_recall_at_1}, R@1% = {avg_recall_at_n}\n")

            if avg_recall_at_1 > bestRecall:
                bestRecall = avg_recall_at_1
                netName = os.path.join(netDir, f"net.pth")
                torch.save(net.state_dict(), netName)
                print("Modelo guardado")
                writer.writerow([input_type, str(i + 1), recall_at_1[0], recall_at_1[1], recall_at_1[2], avg_recall_at_1])
