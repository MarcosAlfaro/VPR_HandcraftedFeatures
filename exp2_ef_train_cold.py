import torch
import os
import csv
import sys
import numpy as np
from torch.utils.data import DataLoader
from sklearn.neighbors import KDTree
import functions.losses as losses
from functions.models import load_model_ef
import datasets_cold
from functions.misc_functions import create_path, select_device
from functions.cold import get_cond_ilum
from functions.eval_functions import compute_recalls_from_pkl
from functions.img_process_functions import select_tf
from eval.tests import build_vm, get_predictions
from functions.train_functions import make_deterministic
from config import PARAMS


device = select_device()
csvDir = create_path(f"{PARAMS.csv_path}Train/")
tf = select_tf(model=PARAMS.model)
baseModelDir = create_path(f"{PARAMS.saved_models_path}EXP02_COLD/")


"""NETWORK TRAINING"""

with open(f"{csvDir}/EXP02_EF_COLD.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Method", "Iteration", "R@1 Cloudy", "R@1 Night", "R@1 Sunny", "R@1 Avg."])

    ef_methods = ["4_channels", "3_channels_RF_GF_BF", "3_channels_RGF", "3_channels_RG_BF"]
    input_type = "MAGNITUDE"

    print("Training Early Fusion COLD database\n")

    for method in ef_methods:

        vmDataset = datasets_cold.Database_EF(ef_method=method, tf=tf, input_type=input_type)
        vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)

        if ef_methods.index(method) == 0:
            coordsVM = []
            for i, vmData in enumerate(vmDataloader, 0):
                _, coords = vmData
                coordsVM.append(coords.detach().numpy()[0])
            treeCoordsVM = KDTree(coordsVM, leaf_size=2)

        criterion = losses.get_loss(PARAMS.loss)
        if criterion == -1:
            sys.exit()

        trainDataset = datasets_cold.Train_EF(ef_method=method, input_type=input_type, tf=tf)
        trainDataloader = DataLoader(trainDataset, shuffle=False, num_workers=0, batch_size=PARAMS.batch_size)

        netDir = create_path(f"{baseModelDir}{method}/")
        net = load_model_ef(pretrained_model=PARAMS.model, num_channels=int(method[0]), weightDir=None).to(device)
        net.backbone.requires_grad_(True)
        net.aggregation.requires_grad_(False)

        # Mantener semillas fijas
        make_deterministic(42)
        

        print(f"\nInput type: {method}\n")

        bestRecall = 0

        optimizer = torch.optim.SGD(net.parameters(), lr=PARAMS.lr, momentum=0.9)

        testDataloaders = []
        condIlum = get_cond_ilum("FR_A")
        for ilum in condIlum:
            testDataset = datasets_cold.Test_EF(il=ilum, env="FR_A", ef_method=method, input_type=input_type, tf=tf)
            testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)
            testDataloaders.append(testDataloader)

        for i, data in enumerate(trainDataloader, 0):

            anc, pos, neg = data[0].float().to(device), data[1].float().to(device), data[2].float().to(device)

            optimizer.zero_grad()
            output1, output2, output3 = net(anc), net(pos), net(neg)

            loss = criterion(output1, output2, output3, PARAMS.margin)
            loss.backward()

            optimizer.step()

            if i % int(len(trainDataloader) / PARAMS.num_validations) == 0 and i > 0:
    
                print(f"\nIt{i}, Loss:{loss}")

                net.eval()

                with torch.no_grad():
                    recall_at_1, recall_at_n = [], []

                    descriptorsVM, coordsVM, treeCoords = build_vm(model=net, dataloader=vmDataloader)
                    for ilum in condIlum:
                        idxIlum = condIlum.index(ilum)

                        pkl_path = f"PKL_FILES/EF/{method}/FR_A_{ilum}_val.pkl"
                        if not os.path.exists(pkl_path):
                            create_path(os.path.dirname(pkl_path))
                        get_predictions(pkl_path=pkl_path, model=net, testDataloader=testDataloaders[idxIlum], 
                                        descriptorsVM=descriptorsVM,
                                        treeCoords=treeCoords, coordsVM= coordsVM, device=device)
                        r1, rn = compute_recalls_from_pkl(pkl_path=pkl_path)
                        recall_at_1.append(r1)
                        recall_at_n.append(rn)
                        print(f"Env: FR-A, ilum: {ilum}, R@1 = {r1}, R@1% = {rn}")

                    avg_recall_at_1, avg_recall_at_n = np.average(np.array(recall_at_1)), np.average(np.array(recall_at_n))
                    print(f"Env: FR_A, R@1 = {avg_recall_at_1}, R@1% = {avg_recall_at_n}\n")

                    if avg_recall_at_1 > bestRecall:
                        bestRecall = avg_recall_at_1
                        netName = os.path.join(netDir, f"net.pth")
                        torch.save(net.state_dict(), netName)
                        print("Model saved")
                        writer.writerow([method, str(i + 1), recall_at_1[0], recall_at_1[1], recall_at_1[2], avg_recall_at_1])

                net.backbone.requires_grad_(True)

        print(f"Training finished, Best Recall: {bestRecall}")
        net.eval()
        recall_at_1, recall_at_n = [], []
        with torch.no_grad():
            descriptorsVM, coordsVM, treeCoords = build_vm(model=net, dataloader=vmDataloader)
            for ilum in condIlum:
                idxIlum = condIlum.index(ilum)

                get_predictions(pkl_path=pkl_path, model=net, testDataloader=testDataloaders[idxIlum], 
                                        descriptorsVM=descriptorsVM, 
                                        treeCoords=treeCoords, coordsVM= coordsVM, device=device)
                r1, rn = compute_recalls_from_pkl(pkl_path=pkl_path)
                recall_at_1.append(r1)
                recall_at_n.append(rn)
                print(f"Env: FR-A, ilum: {ilum}, R@1 = {r1}, R@1% = {rn}")

            avg_recall_at_1, avg_recall_at_n = np.average(np.array(recall_at_1)), np.average(np.array(recall_at_n))
            print(f"Env: FR_A, R@1 = {avg_recall_at_1}, R@1% = {avg_recall_at_n}\n")

            if avg_recall_at_1 > bestRecall:
                bestRecall = avg_recall_at_1
                netName = os.path.join(netDir, f"net.pth")
                torch.save(net.state_dict(), netName)
                print("Model saved")
                writer.writerow([method, str(i + 1), recall_at_1[0], recall_at_1[1], recall_at_1[2], avg_recall_at_1])