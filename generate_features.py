import os
import cv2
from PIL import Image
from config import PARAMS
import numpy as np


dataset_dir = PARAMS.rawseeds_path
envs = ["25a", "25b", "26a", "26b", "27a"]


def generate_grayscale(folder_path):
    for img_name in os.listdir(f"{folder_path}/PANO/"):
        if img_name.endswith('.jpeg'):
            img_path = os.path.join(folder_path, "PANO", img_name)
            img = Image.open(img_path).convert('RGB')
            gray_img = img.convert('L')
            gray_img_path = os.path.join(folder_path, f"GRAYSCALE")
            os.makedirs(gray_img_path, exist_ok=True)
            # save gray img as .npy file
            gray = np.array(gray_img)/255.0
            npy_path = img_path.replace("PANO", "GRAYSCALE").replace(".jpeg", ".npy")
            np.save(npy_path, gray)          
    return


def generate_hue(folder_path):
    for img_name in os.listdir(f"{folder_path}/PANO/"):
        if img_name.endswith('.jpeg'):
            img_path = os.path.join(folder_path, "PANO", img_name)
            img = Image.open(img_path).convert('RGB')
            hsv_img = img.convert('HSV')
            hue, _, _ = hsv_img.split()
            hue_img_path = os.path.join(folder_path, f"HUE")
            os.makedirs(hue_img_path, exist_ok=True)
            # save hue img as .npy file
            hue.save("hue_temp.jpeg")
            hue = np.array(hue)/255.0
            npy_path = img_path.replace("PANO", "HUE").replace(".jpeg", ".npy")
            np.save(npy_path, hue)
    return


for env in envs:
    env_path  = f"{dataset_dir}{env}/"
    generate_grayscale(folder_path=env_path)
    generate_hue(folder_path=env_path)
