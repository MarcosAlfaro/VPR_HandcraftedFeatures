"""
THIS PROGRAM CONTAINS ALL THE CLASSES THAT CREATE THE REQUIRED IMAGE SETS TO DO A TRAINING, VALIDATION OR TEST
These classes will be called by training and test programmes
"""
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import os
from config import PARAMS
import functions.img_process_functions as img_proc
from PIL import Image


csvDir = f"{PARAMS.csv_path}train_eval_rawseeds/"

dataset_dir = f"{PARAMS.rawseeds_path}"


def process_image(image, rgb=True, eq=True, inv=True, sh=True, color_rep=None, tf=transforms.ToTensor()):
    save_img = False
    if "/atrium" in image and "0200" in image and "daytime_360_0" in image:
        save_img = False
        feature = image.split("/")[-2]
    image = img_proc.load_image(image, rgb=rgb)    
    if save_img:
        img = Image.fromarray(image)
        img.save(f"FIGURES/EXAMPLES_FEATURES/test_image_{feature}.png")   
    if not rgb:
        image = img_proc.equalize_image(image) if eq else image
        image = img_proc.inverse_image(image) if inv else image
        image = img_proc.sharpen_image(image) if sh else image
        image = img_proc.apply_colormap(image, color_rep) if color_rep is not None else image
        if save_img:
            img = Image.fromarray(image)
            img.save(f"FIGURES/EXAMPLES_FEATURES/test_image_{feature}.png")  

    image = img_proc.tf_image(image, tf=tf)
    image = img_proc.normalize_image(image) if rgb else image
    
    return image



class Train_EF_multifeatures(Dataset):

    def __init__(self, features=["RGB", "GRAYSCALE", "MAGNITUDE", "ANGLE", "HUE"], env="27a", tf=transforms.ToTensor()):
        
        self.features, self.env, self.tf = features, env, tf

        CSV_file = pd.read_csv(f'{csvDir}/train_{self.env}.csv')
        self.imgsAnc, self.imgsPos, self.imgsNeg = CSV_file['ImgAnc'], CSV_file['ImgPos'], CSV_file['ImgNeg']

        self.color_rep = None
    
        self.rgb_dir = f"{dataset_dir}{self.env}/PANO/"

    def __getitem__(self, index):

        imgAnc, imgPos, imgNeg = self.imgsAnc[index], self.imgsPos[index], self.imgsNeg[index]
        
        imgAnc_RGB, imgPos_RGB, imgNeg_RGB = f"{self.rgb_dir}{imgAnc}".replace(".png", ".jpeg"), f"{self.rgb_dir}{imgPos}".replace(".png", ".jpeg"), f"{self.rgb_dir}{imgNeg}".replace(".png", ".jpeg")
        
        anc = process_image(image=imgAnc_RGB, rgb=True, eq=False, inv=False, sh=False, color_rep=False, tf=self.tf)
        pos = process_image(image=imgPos_RGB, rgb=True, eq=False, inv=False, sh=False, color_rep=False, tf=self.tf)
        neg = process_image(image=imgNeg_RGB, rgb=True, eq=False, inv=False, sh=False, color_rep=False, tf=self.tf)

        for f in self.features:
            if f != "RGB":
                imgAnc_f = f"{self.rgb_dir.replace('PANO', f) }{imgAnc}".replace(".jpeg", ".npy")
                imgPos_f = f"{self.rgb_dir.replace('PANO', f) }{imgPos}".replace(".jpeg", ".npy")
                imgNeg_f = f"{self.rgb_dir.replace('PANO', f) }{imgNeg}".replace(".jpeg", ".npy")

                anc_f = process_image(image=imgAnc_f, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=self.color_rep, tf=self.tf)
                pos_f = process_image(image=imgPos_f, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=self.color_rep, tf=self.tf)
                neg_f = process_image(image=imgNeg_f, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=self.color_rep, tf=self.tf)

                anc = torch.cat((anc, anc_f), dim=0)
                pos = torch.cat((pos, pos_f), dim=0)
                neg = torch.cat((neg, neg_f), dim=0)

        return anc, pos, neg

    def __len__(self):
        return len(self.imgsAnc)
    


class Train_LF(Dataset):

    def __init__(self, input_type="RGB", env="27a", tf=transforms.ToTensor()):
        
        self.input_type, self.env, self.tf = input_type, env, tf

        CSV_file = pd.read_csv(f'{csvDir}/train_{self.env}.csv')
        self.imgsAnc, self.imgsPos, self.imgsNeg = CSV_file['ImgAnc'], CSV_file['ImgPos'], CSV_file['ImgNeg']
        
        self.rgb = True if self.input_type=="RGB" else False
        self.img_path = f"{dataset_dir}{self.env}/PANO/" if self.input_type=="RGB" else f"{dataset_dir}{self.env}/{self.input_type}/"

    def __getitem__(self, index):

        imgAnc, imgPos, imgNeg = self.imgsAnc[index], self.imgsPos[index], self.imgsNeg[index]
        imgAnc, imgPos, imgNeg = f"{self.img_path}{imgAnc}".replace(".png", ".jpeg"), f"{self.img_path}{imgPos}".replace(".png", ".jpeg"), f"{self.img_path}{imgNeg}".replace(".png", ".jpeg")

        if self.input_type != "RGB":
            imgAnc, imgPos, imgNeg = imgAnc.replace(".jpeg", ".npy"), imgPos.replace(".jpeg", ".npy"), imgNeg.replace(".jpeg", ".npy")
        
        anc = process_image(image=imgAnc, rgb=self.rgb, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
        pos = process_image(image=imgPos, rgb=self.rgb, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
        neg = process_image(image=imgNeg, rgb=self.rgb, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
        
        if self.input_type != "RGB":
            anc, pos, neg = torch.cat((anc, anc, anc), dim=0), torch.cat((pos, pos, pos), dim=0), torch.cat((neg, neg, neg), dim=0)

        return anc, pos, neg

    def __len__(self):
        return len(self.imgsAnc) 




class Test_LF(Dataset):

    def __init__(self, input_type="MAGNITUDE", env="27a", tf=transforms.ToTensor()):

        self.input_type, self.env, self.tf = input_type, env, tf

        CSV_file = pd.read_csv(f'{csvDir}/test_{self.env}.csv')
        self.imgList, self.coordX, self.coordY = CSV_file['Img'], CSV_file['CoordX'], CSV_file['CoordY']
        self.rgb_dir  = f"{dataset_dir}{self.env}/PANO/"
        self.feature_dir = self.rgb_dir.replace("PANO", self.input_type)

    def __getitem__(self, index):

        imgPath, coords = self.imgList[index], img_proc.load_coords(self.coordX[index], self.coordY[index])

        img_RGB, img_depth = f"{self.rgb_dir}{imgPath}".replace(".png", ".jpeg"), f"{self.feature_dir}{imgPath}".replace(".png", ".npy")
        img_RGB = process_image(image=img_RGB, rgb=True, eq=False, inv=False, sh=False, tf=self.tf)

        img_depth = img_depth.replace(".jpeg", ".npy")
        img_depth = process_image(image=img_depth, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
        img_depth = torch.cat((img_depth, img_depth, img_depth), dim=0) 
        img_depth = img_proc.normalize_image(img_depth)

        return img_RGB, img_depth, coords

    def __len__(self):
        return len(self.imgList)

    

class Database_LF(Dataset):

    def __init__(self, input_type="MAGNITUDE", env="27a", tf=transforms.ToTensor()):

        self.input_type, self.env, self.tf = input_type, env, tf

        CSV_file = pd.read_csv(f'{csvDir}/database_{self.env}.csv')
        self.imgList, self.coordX, self.coordY = CSV_file['Img'], CSV_file['CoordX'], CSV_file['CoordY']
        self.rgb_dir  = f"{dataset_dir}{self.env}/PANO/"
        self.feature_dir = self.rgb_dir.replace("PANO", self.input_type)

    def __getitem__(self, index):

        imgPath, coords = self.imgList[index], img_proc.load_coords(self.coordX[index], self.coordY[index])

        img_RGB, img_depth = f"{self.rgb_dir}{imgPath}".replace(".png", ".jpeg"), f"{self.feature_dir}{imgPath}".replace(".png", ".npy")
        img_RGB = process_image(image=img_RGB, rgb=True, eq=False, inv=False, sh=False, tf=self.tf)
        img_depth = process_image(image=img_depth, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
        img_depth = torch.cat((img_depth, img_depth, img_depth), dim=0) 
        
        return img_RGB, img_depth, coords

    def __len__(self):
        return len(self.imgList)
    


"""LATE FUSION (MORE THAN ONE FEATURE)"""

class Database_multifeatures(Dataset):

    def __init__(self, features=["GRAYSCALE", "MAGNITUDE", "ANGLE", "HUE"], env="27a", tf=transforms.ToTensor()):

        self.features, self.env, self.tf = features, env, tf

        CSV_file = pd.read_csv(f'{csvDir}/database_{self.env}.csv')
        self.imgList, self.coordX, self.coordY = CSV_file['Img'], CSV_file['CoordX'], CSV_file['CoordY']
        self.rgb_dir  = f"{dataset_dir}{self.env}/PANO/"
        #self.feature_dir = self.rgb_dir.replace("image_resized", self.input_type)

    def __getitem__(self, index):

        imgPath, coords = self.imgList[index], img_proc.load_coords(self.coordX[index], self.coordY[index])

        img = f"{self.rgb_dir}{imgPath}".replace(".png", ".jpeg")
        img = process_image(image=img, rgb=True, eq=False, inv=False, sh=False, tf=self.tf)

        for feature in self.features:
            if feature == "RGB":
                continue
            img_f = self.rgb_dir.replace("PANO", feature) + imgPath
            img_f = img_f.replace(".png", ".npy")
            img_f = process_image(image=img_f, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
            img = torch.cat((img, img_f), dim=0)
        return img, coords

    def __len__(self):
        return len(self.imgList)
    

class Test_multifeatures(Dataset):

    def __init__(self, features=["GRAYSCALE", "MAGNITUDE", "ANGLE", "HUE"], env="25a", tf=transforms.ToTensor()):

        self.features, self.env, self.tf = features, env, tf

        CSV_file = pd.read_csv(f'{csvDir}/test_{self.env}.csv')
        self.imgList, self.coordX, self.coordY = CSV_file['Img'], CSV_file['CoordX'], CSV_file['CoordY']
        self.rgb_dir  = f"{dataset_dir}{self.env}/PANO/"
        #self.feature_dir = self.rgb_dir.replace("image_resized", self.input_type)

    def __getitem__(self, index):

        imgPath, coords = self.imgList[index], img_proc.load_coords(self.coordX[index], self.coordY[index])

        img = f"{self.rgb_dir}{imgPath}".replace(".png", ".jpeg")
        img = process_image(image=img, rgb=True, eq=False, inv=False, sh=False, tf=self.tf)

        for feature in self.features:
            if feature == "RGB":
                continue
            img_f = self.rgb_dir.replace("PANO", feature) + imgPath
            img_f = img_f.replace(".png", ".npy")
            img_f = process_image(image=img_f, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
            img = torch.cat((img, img_f), dim=0)
        return img, coords

    def __len__(self):
        return len(self.imgList)








class Test_wo_Fusion(Dataset):

    def __init__(self,input_type=None, env="25a", tf=transforms.ToTensor()):

        self.input_type, self.env, self.tf = input_type, env, tf

        CSV_file = pd.read_csv(f'{csvDir}/test_{self.env}.csv')
        self.imgList, self.coordX, self.coordY = CSV_file['Img'], CSV_file['CoordX'], CSV_file['CoordY']

        self.rgb_dir = f"{dataset_dir}{self.env}/PANO/"
        self.feature_dir = self.rgb_dir.replace("PANO", self.input_type)

    def __getitem__(self, index):

        imgPath, coords = self.imgList[index], img_proc.load_coords(self.coordX[index], self.coordY[index])

        if self.input_type is None or self.input_type=="RGB":
            img = f"{self.rgb_dir}{imgPath}".replace(".png", ".jpeg")
            img = process_image(image=img, rgb=True, eq=False, inv=False, sh=False, color_rep=None, tf=self.tf)
        else:
            img = f"{self.feature_dir}{imgPath}"
            img = img.replace(".png", ".npy")
            img = process_image(image=img, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
            img = torch.cat((img, img, img), dim=0) 
        return img, coords

    def __len__(self):
        return len(self.imgList)


class Database_wo_Fusion(Dataset):

    def __init__(self, input_type=None, env="27a", tf=transforms.ToTensor()):

        self.input_type, self.env, self.tf = input_type, env, tf

        CSV_file = pd.read_csv(f'{csvDir}/database_{self.env}.csv')
        self.imgList, self.coordX, self.coordY = CSV_file['Img'], CSV_file['CoordX'], CSV_file['CoordY']

        self.rgb_dir = f"{dataset_dir}{self.env}/PANO/"
        self.feature_dir = self.rgb_dir.replace("PANO", self.input_type)

    def __getitem__(self, index):

        imgPath, coords = self.imgList[index], img_proc.load_coords(self.coordX[index], self.coordY[index])

        if self.input_type is None or self.input_type=="RGB":
            img = f"{self.rgb_dir}{imgPath}".replace(".png", ".jpeg") 
            img = process_image(image=img, rgb=True, eq=False, inv=False, sh=False, color_rep=None, tf=self.tf)
        else:
            img = f"{self.feature_dir}{imgPath}"
            img = img.replace(".png", ".npy")
            img = process_image(image=img, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
            img = torch.cat((img, img, img), dim=0) 
        return img, coords

    def __len__(self):
        return len(self.imgList)
    

class Train_MLP(Dataset):

    def __init__(self, env="atrium", il="daytime_360_0", tf=transforms.ToTensor()):

        self.tf, self.env, self.il = tf, env, il

        CSV_file = pd.read_csv(csvDir + f'/train_{self.env}_{self.il}.csv')

        self.imgsAnc, self.imgsPos, self.imgsNeg = CSV_file['ImgAnc'], CSV_file['ImgPos'], CSV_file['ImgNeg']
        self.rgb_dir = f"{dataset_dir}{self.env}/mapping/{self.il}/image_resized/"
        self.depth_dir = self.rgb_dir.replace("image_resized", "vitl/GRAY")
        

    def __getitem__(self, index):

        imgAnc, imgPos, imgNeg = self.imgsAnc[index], self.imgsPos[index], self.imgsNeg[index]

        imgAnc_RGB, imgPos_RGB, imgNeg_RGB = f"{self.rgb_dir}{imgAnc}", f"{self.rgb_dir}{imgPos}", f"{self.rgb_dir}{imgNeg}"
        imgAnc_depth, imgPos_depth, imgNeg_depth = f"{self.depth_dir}{imgAnc}", f"{self.depth_dir}{imgPos}", f"{self.depth_dir}{imgNeg}"

        anc_RGB = process_image(image=imgAnc_RGB, tf=self.tf, rgb=True, eq=False, inv=False, sh=False)
        pos_RGB = process_image(image=imgPos_RGB, tf=self.tf, rgb=True, eq=False, inv=False, sh=False)
        neg_RGB = process_image(image=imgNeg_RGB, tf=self.tf, rgb=True, eq=False, inv=False, sh=False)

        anc_d = process_image(image=imgAnc_depth, tf=self.tf, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep)
        pos_d = process_image(image=imgPos_depth, tf=self.tf, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep)
        neg_d = process_image(image=imgNeg_depth, tf=self.tf, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep)
        return anc_RGB, pos_RGB, neg_RGB, anc_d, pos_d, neg_d

    def __len__(self):
        return len(self.imgsAnc)
