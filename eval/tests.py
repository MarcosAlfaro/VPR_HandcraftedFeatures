import pickle
import torch
import numpy as np
from sklearn.neighbors import KDTree
import torch.nn.functional as F
from functions.eval_functions import get_distance_threshold
from config import PARAMS
#from utils.functions import get_img_path_from_index
import time



def build_vm(model, dataloader, device="cuda"):
    coordsVM, descriptorsVM = [], []
    model.eval()
    with torch.no_grad():
        for _, vmData in enumerate(dataloader, 0):
            img, imgCoords = vmData
            img, imgCoords = vmData[0].float().to(device), vmData[1].detach().numpy()[0]
            output = model(img)
            descriptorsVM.append(output)
            coordsVM.append(imgCoords)
    descriptorsVM = torch.squeeze(torch.stack(descriptorsVM)).to(device)
    treeCoordsVM = KDTree(coordsVM, leaf_size=2)
    model.train(True)
    return descriptorsVM, coordsVM, treeCoordsVM


def build_vm_lf(model_RGB, model_f, dataloader, device="cuda"):
    coordsVM, desc_RGB, desc_f = [], [], []
    model_RGB.eval()
    model_f.eval()
    with torch.no_grad():
        for _, vmData in enumerate(dataloader, 0):
            img_RGB, img_f, imgCoords = vmData[0].float().to(device), vmData[1].float().to(device), vmData[2].detach().numpy()[0]
            out_RGB, out_f = model_RGB(img_RGB), model_f(img_f) # .cpu().detach().numpy()[0]
            desc_RGB.append(out_RGB)
            desc_f.append(out_f)
            coordsVM.append(imgCoords)
    desc_RGB = torch.stack(desc_RGB).squeeze(1).to(device)
    desc_f = torch.stack(desc_f).squeeze(1).to(device)
    #para PCA
    #treeDescVM = KDTree(descriptorsVM, leaf_size=2)
    treeCoordsVM = KDTree(coordsVM, leaf_size=2)

    return desc_RGB, desc_f, coordsVM, treeCoordsVM


def build_vm_lf_multifeatures(models, features, dataloader, device="cuda"):
    coordsVM = []
    [models[i].eval() for i in range(len(models))]
    with torch.no_grad():
        descVM = torch.zeros((len(features), len(dataloader), PARAMS.embedding_size)).to(device)
        for i, vmData in enumerate(dataloader, 0):
            img, imgCoords = vmData[0].float().to(device), vmData[1].detach().numpy()[0]
            img_RGB = img[:,0:3,:,:]
            start_time = time.time()
            for f in range(len(features)):
                if features[f] == "RGB":
                    out_f = models[0](img_RGB)
                else:
                    img_f = img[:,3+f-1:3+f,:,:]
                    img_f = torch.cat((img_f, img_f, img_f), dim=1)
                    out_f = models[f](img_f) if len(models) > 1 else models[0](img_f)
                descVM[f, i] = out_f + descVM[f, i]
            end_time = time.time()
            print(f"Processed image {i} in {end_time - start_time:.4f} seconds")
            coordsVM.append(imgCoords)

    treeCoordsVM = KDTree(coordsVM, leaf_size=2)

    return descVM, coordsVM, treeCoordsVM


# create a new function in which you do the following:
# you perform the same test as in the previous function, but using only one feature per time
# you have to store, for each image, the indexes of the K retrieved images
# this function must have, among its parameters, the feature that you want to use and the value of K



def get_predictions(pkl_path, model, testDataloader, descriptorsVM, treeCoords, coordsVM, features=None, lf_method=None, env="FR_A", device="cuda"):

    d_threshold = get_distance_threshold(env)
    results_list = []

    with torch.no_grad():

        for i, data in enumerate(testDataloader, 0):

            if features is not None:
                img, coordsImgTest = data[0].float().to(device), data[1].detach().numpy()[0]
                img_RGB = img[:,0:3,:,:]
                out = torch.zeros((len(features), PARAMS.embedding_size)).to(device)
                for f in range(len(features)):
                    if features[f] == "RGB":
                        out[f] = model[0](img_RGB)
                    else:
                        img_f = img[:,3+f-1:3+f,:,:]
                        img_f = torch.cat((img_f, img_f, img_f), dim=1)
                        if len(model) > 1:
                            out[f] = model[f](img_f)
                        else:
                            out[f] = model[0](img_f)

                out, descVM = late_fusion(out, descriptorsVM, features, lf_method=lf_method)
            else:
                img_RGB, coordsImgTest = data[0].float().to(device), data[1].detach().numpy()[0]
                out = model(img_RGB)
                descVM = descriptorsVM

            idxDesc = torch.argsort(F.pairwise_distance(out, descVM), descending=False).cpu().numpy() #[0]

            _, idxGeom = treeCoords.query(coordsImgTest.reshape(1, -1), k=len(descVM))
            idxMinReal = idxGeom[0][0]

            n = int(np.floor(len(coordsVM)/100))
            for k in range(n):
                coordsPredictedImg = coordsVM[idxDesc[k]]
                errorDist = np.linalg.norm(coordsImgTest - coordsPredictedImg)
                if errorDist <= d_threshold:
                    break
                k += 1

            # Guardar información de esta imagen
            results_list.append({
                "query_index": i,
                "retrieved_indices": idxDesc[:n].tolist(),  # Top n recuperadas
                "real_index": int(idxMinReal),
                "correct_index": int(k) if k < n else -1,  # -1 si no encontró dentro de top-n
                "query_coords": coordsImgTest.tolist(),
                "real_coords": coordsVM[int(idxMinReal)].tolist(),
                "retrieved_coords": [coordsVM[idx].tolist() for idx in idxDesc[:n]]
            })

            # Guardar todos los resultados en UN SOLO archivo pickle
    with open(pkl_path, "wb") as f:
        pickle.dump(results_list, f)

    print(f"Results saved to {pkl_path}")


    return



def late_fusion(out, descriptorsVM, features, lf_method="concat"):
    if lf_method == "concat":
        for f in range(descriptorsVM.shape[0]):
            if f == 0:
                out_f, descVM_f = out[f].unsqueeze(0), descriptorsVM[f]
            else:
                out_f = torch.cat((out_f, out[f].unsqueeze(0)), dim=-1)
                descVM_f = torch.cat((descVM_f, descriptorsVM[f]), dim=-1)
        out, descVM = out_f, descVM_f
        return out, descVM
    elif lf_method == "sum":
        for f in range(descriptorsVM.shape[0]):
            if f == 0:
                out_f, descVM_f = out[f], descriptorsVM[f]
            else:
                out_f = out_f + out[f]
                descVM_f = descVM_f + descriptorsVM[f]
        out, descVM = out_f, descVM_f
        return out, descVM
    elif lf_method == "sum_prod":
        for f in range(descriptorsVM.shape[0]):
            if f == 0:
                out_f, descVM_f = out[f], descriptorsVM[f]
            else:
                out_f = out_f + out[f] * out[f]
                descVM_f = descVM_f + descriptorsVM[f] * descriptorsVM[f]
        out, descVM = out_f, descVM_f
        return out, descVM
    elif lf_method == "confidence":
        confidenceScores = np.zeros((len(features), 1))
        for f in range(len(features)):
            distances = np.array(sorted(F.pairwise_distance(out[f], descriptorsVM[f,:,:]).cpu().detach().numpy()))
            invDistances = 1 / (distances + 1e-8)  # Avoid division by zero
            confidenceScores[f] = invDistances[0]/sum(invDistances)
        # confidenceScores = torch.tensor(confidenceScores).to(device)
        confidenceScores = torch.tensor(confidenceScores / sum(confidenceScores)).to(PARAMS.device)
        
        # # Method 1
        weightedDescriptorsVM = torch.zeros_like(descriptorsVM)
        for f in range(out.shape[0]):
            weightedDescriptorsVM[f,:,:] = descriptorsVM[f,:,:] * confidenceScores[f]
        weightedDescriptorsVM = torch.sum(weightedDescriptorsVM, dim=0)
        weightedOut = torch.sum(out * confidenceScores, dim=0, keepdim=True)
        
        return weightedOut, weightedDescriptorsVM






# quiero que crees una función en la que se comparen los resultados almacenados en los archivos .pkl entre el método de late fusion y el baseline
# tienes que calcular, para cada entorno y secuencia, el R@1, el R@1%, el número de veces que aciertan ambos métodos,
# el número de veces que acierta solo el método late fusion, el número de veces que acierta solo el baseline y el número de veces que fallan ambos métodos (usually just one point)
def compare_lf_baseline_results(env, ilum, k=1):
    
    # Cargar los resultados desde los archivos pickle
    with open(f"PKL_FILES/LF/{env}_{ilum}.pkl", "rb") as f:
        lf_results = pickle.load(f)

    with open(f"PKL_FILES/BASELINE/{env}_{ilum}.pkl", "rb") as f:
        baseline_results = pickle.load(f)

    assert len(lf_results) == len(baseline_results), "Los archivos .pkl deben tener el mismo número de entradas."

    both_correct = 0
    only_lf_correct = 0
    only_baseline_correct = 0
    both_incorrect = 0

    for lf_result, baseline_result in zip(lf_results, baseline_results):
        lf_correct_index = lf_result["correct_index"]
        baseline_correct_index = baseline_result["correct_index"]

        lf_correct = lf_correct_index != -1 and lf_correct_index < k
        baseline_correct = baseline_correct_index != -1 and baseline_correct_index < k

        if lf_correct and baseline_correct:
            both_correct += 1
            """
            if both_correct == 10 and env == "atrium" and ilum == "daytime1":
                query_path = get_img_path_from_index('360LOC', True, env, ilum, lf_result['query_index'])
                db_path = get_img_path_from_index('360LOC', False, env, ilum, lf_result['real_index'])
                retrieved_baseline_paths = get_img_path_from_index('360LOC', False, env, ilum, baseline_result['retrieved_indices'][0])
                retrieved_lf_paths = get_img_path_from_index('360LOC', False, env, ilum, lf_result['retrieved_indices'][0])
                print(f"Query: {query_path}, DB real: {db_path}, Ret. Baseline: {retrieved_baseline_paths}, Ret. LF: {retrieved_lf_paths}")
            """
        elif lf_correct and not baseline_correct:
            only_lf_correct += 1
            # de forma aleatoria, imprime el índice de la imagen y las coordenadas de la imagen query
            """
            if only_lf_correct % 10 == 0:
                query_path = get_img_path_from_index('360LOC', True, env, ilum, lf_result['query_index'])
                db_path = get_img_path_from_index('360LOC', False, env, ilum, lf_result['real_index'])
                retrieved_baseline_paths = get_img_path_from_index('360LOC', False, env, ilum, baseline_result['retrieved_indices'][0])
                retrieved_lf_paths = get_img_path_from_index('360LOC', False, env, ilum, lf_result['retrieved_indices'][0])
                print(f"Query: {query_path}, DB real: {db_path}, Ret. Baseline: {retrieved_baseline_paths}, Ret. LF: {retrieved_lf_paths}")
                #print(f"Imagen query (LF correcto): {lf_result['query_index']}, Imagen DB real: {lf_result['real_index']}, Retrieved Baseline: {baseline_result['retrieved_indices'][0]}, Retrieved LF: {lf_result['retrieved_indices'][0]}")
                """
        elif not lf_correct and baseline_correct:
            only_baseline_correct += 1
        else:
            both_incorrect += 1

        if not lf_correct:
            query_coords, retrieved_coords = lf_result['query_coords'], lf_result['retrieved_coords'][0]
            distance = np.linalg.norm(np.array(query_coords) - np.array(retrieved_coords))
            if distance > 80:
                query_path = get_img_path_from_index('360LOC', True, env, ilum, lf_result['query_index'])
                db_path = get_img_path_from_index('360LOC', False, env, ilum, lf_result['real_index'])
                retrieved_lf_paths = get_img_path_from_index('360LOC', False, env, ilum, lf_result['retrieved_indices'][0])
                #print(f"Query coords: {query_coords}, Retrieved coords: {retrieved_coords}")
                print(f"Query: {query_path}, DB real: {db_path}, Retrieved LF: {retrieved_lf_paths}, distance: {distance}m")

    return {
        "both_correct": both_correct,
        "only_lf_correct": only_lf_correct,
        "only_baseline_correct": only_baseline_correct,
        "both_incorrect": both_incorrect
    }










