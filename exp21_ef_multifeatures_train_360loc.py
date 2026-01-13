import torch
import os
import csv
import sys
import numpy as np
from torch.utils.data import DataLoader
from sklearn.neighbors import KDTree
import functions.losses as losses
from functions.models import load_model_ef
import datasets_360loc
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
baseModelDir = create_path(f"{PARAMS.saved_models_path}EXP02_360LOC/")
input_type = "MAGNITUDE"


"""NETWORK TRAINING"""

with open(f"{csvDir}/EF_360Loc.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Method", "Iteration", "R@1 Cloudy", "R@1 Night", "R@1 Sunny", "R@1 Avg."])

    featuresList = [
        ["RGB", "GRAYSCALE"],
        ["RGB", "MAGNITUDE"],
        ["RGB", "ANGLE"],
        ["RGB", "HUE"],
        ["RGB", "GRAYSCALE", "MAGNITUDE"],
        ["RGB", "GRAYSCALE", "MAGNITUDE", "ANGLE"],
        ["RGB", "GRAYSCALE", "MAGNITUDE", "ANGLE", "HUE"]
    ]

    print("Training Early Fusion 360Loc database\n")

    for features in featuresList:
        vmDataset = datasets_360loc.Database_multifeatures(features=features, tf=tf, env="atrium", il="daytime_360_0")
        vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)

        if featuresList.index(features) == 0:
            coordsVM = []
            for i, vmData in enumerate(vmDataloader, 0):
                _, coords = vmData
                coordsVM.append(coords.detach().numpy()[0])
            treeCoordsVM = KDTree(coordsVM, leaf_size=2)

        criterion = losses.get_loss(PARAMS.loss)
        if criterion == -1:
            sys.exit()

        trainDataset = datasets_360loc.Train_EF_multifeatures(features=features, tf=tf)
        trainDataloader = DataLoader(trainDataset, shuffle=False, num_workers=0, batch_size=PARAMS.batch_size)

        netDir = create_path(os.path.join(baseModelDir, "_".join(features)))

        net = load_model_ef(pretrained_model=PARAMS.model, num_channels=2+len(features), weightDir=None).to(device)
        net.aggregation.requires_grad_(False)
        net.backbone.requires_grad_(True)

        # Mantener semillas fijas
        make_deterministic(42)

        print(f"\nInput type: {features}\n")

        bestRecall = 0

        optimizer = torch.optim.SGD(net.parameters(), lr=PARAMS.lr, momentum=0.9)

        testDataloaders = []
        condIlum = condIlum_360loc[0]
        for ilum in condIlum:
            testDataset = datasets_360loc.Test_multifeatures(il=ilum, env="atrium", features=features, tf=tf)
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
                print(f"It{i}, Loss:{loss}")

                net.eval()

                with torch.no_grad():

                    recall_at_1, recall_at_n = [], []
                    descriptorsVM, coordsVM, treeCoords = build_vm(model=net, dataloader=vmDataloader)

                    for ilum in condIlum:
                        idxIlum = condIlum.index(ilum)

                        testDataset = datasets_360loc.Test_multifeatures(il=ilum, env="atrium", features=features)
                        testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

                        pkl_path = f"PKL_FILES/EF/{'_'.join(features)}/atrium_{ilum}_val.pkl"
                        if not os.path.exists(pkl_path):
                            create_path(os.path.dirname(pkl_path))
                        get_predictions(pkl_path=pkl_path, model=net, testDataloader=testDataloaders[idxIlum], 
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
                        writer.writerow(["_".join(features), str(i + 1), recall_at_1[0], recall_at_1[1], recall_at_1[2], avg_recall_at_1])

                net.backbone.requires_grad_(True)

        print(f"Training finished, Best Recall: {bestRecall}")
        net.eval()
        recall_at_1, recall_at_n = [], []
        with torch.no_grad():
            descriptorsVM, coordsVM, treeCoords = build_vm(model=net, dataloader=vmDataloader)
            for ilum in condIlum:
                idxIlum = condIlum.index(ilum)

                get_predictions(pkl_path=pkl_path, model=net, testDataloader=testDataloaders[idxIlum], 
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
                writer.writerow(["_".join(features), str(i + 1), recall_at_1[0], recall_at_1[1], recall_at_1[2], avg_recall_at_1])