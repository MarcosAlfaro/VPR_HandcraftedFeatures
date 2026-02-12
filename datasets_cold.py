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



csvDir = f"{PARAMS.csv_path}train_eval_COLD/"

rgb_dir = f"{PARAMS.cold_path}"
features_dir = f"{PARAMS.cold_path}FEATURES/"


def process_image(image, rgb=True, eq=True, inv=True, sh=True, color_rep=None, tf=transforms.ToTensor()):

    image = img_proc.load_image(image, rgb=rgb)       
    if not rgb:
        image = img_proc.equalize_image(image) if eq else image
        image = img_proc.inverse_image(image) if inv else image
        image = img_proc.sharpen_image(image) if sh else image
        image = img_proc.apply_colormap(image, color_rep) if color_rep is not None else image

    image = img_proc.tf_image(image, tf=tf)
    image = img_proc.normalize_image(image) if rgb else image

    return image


""" EARLY FUSION"""


class Train_EF(Dataset):

    def __init__(self, enc="vitl", ef_method=None, input_type="MAGNITUDE", env="FR_A", tf=transforms.ToTensor()):

        self.enc, self.ef_method, self.input_type, self.env, self.tf = enc, ef_method, input_type, env, tf
        self.color_rep = PARAMS.color_rep if self.ef_method in ["6_channels", "3_channels_RF_GF_BF"] else None

        CSV_file = pd.read_csv(f'{csvDir}/Train_{env}.csv')
        self.imgsAnc, self.imgsPos, self.imgsNeg = CSV_file['ImgAnc'], CSV_file['ImgPos'], CSV_file['ImgNeg']
        self.rgb_dir, self.depth_dir = f"{rgb_dir}{self.env}/Train/", f"{features_dir}{self.input_type}/{self.env}/Train/"

    def __getitem__(self, index):

        imgAnc, imgPos, imgNeg = self.imgsAnc[index], self.imgsPos[index], self.imgsNeg[index]

        imgAnc_RGB, imgPos_RGB, imgNeg_RGB = f"{self.rgb_dir}{imgAnc}", f"{self.rgb_dir}{imgPos}", f"{self.rgb_dir}{imgNeg}"
        imgAnc_depth, imgPos_depth, imgNeg_depth = f"{self.depth_dir}{imgAnc}", f"{self.depth_dir}{imgPos}", f"{self.depth_dir}{imgNeg}"
        imgAnc_depth, imgPos_depth, imgNeg_depth = imgAnc_depth.replace(".jpeg", ".npy"), imgPos_depth.replace(".jpeg", ".npy"), imgNeg_depth.replace(".jpeg", ".npy")

        anc_RGB = process_image(image=imgAnc_RGB, rgb=True, eq=False, inv=False, sh=False, color_rep=False, tf=self.tf)
        pos_RGB = process_image(image=imgPos_RGB, rgb=True, eq=False, inv=False, sh=False, color_rep=False, tf=self.tf)
        neg_RGB = process_image(image=imgNeg_RGB, rgb=True, eq=False, inv=False, sh=False, color_rep=False, tf=self.tf)

        anc_depth = process_image(image=imgAnc_depth, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=self.color_rep, tf=self.tf)
        pos_depth = process_image(image=imgPos_depth, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=self.color_rep, tf=self.tf)
        neg_depth = process_image(image=imgNeg_depth, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=self.color_rep, tf=self.tf)

        if "3" in self.ef_method:
            if "RGF" in self.ef_method:
                anc = torch.cat((anc_RGB[0].unsqueeze(0), anc_RGB[1].unsqueeze(0), anc_depth[0].unsqueeze(0)), dim=0)
                pos = torch.cat((pos_RGB[0].unsqueeze(0), pos_RGB[1].unsqueeze(0), pos_depth[0].unsqueeze(0)), dim=0)
                neg = torch.cat((neg_RGB[0].unsqueeze(0), neg_RGB[1].unsqueeze(0), neg_depth[0].unsqueeze(0)), dim=0)
            elif "RG_BF" in self.ef_method:
                anc = torch.cat((anc_RGB[0].unsqueeze(0), anc_RGB[1].unsqueeze(0), ((anc_RGB[2]+anc_depth[0])/2).unsqueeze(0)), dim=0)
                pos = torch.cat((pos_RGB[0].unsqueeze(0), pos_RGB[1].unsqueeze(0), ((pos_RGB[2]+pos_depth[0])/2).unsqueeze(0)), dim=0)
                neg = torch.cat((neg_RGB[0].unsqueeze(0), neg_RGB[1].unsqueeze(0), ((neg_RGB[2]+neg_depth[0])/2).unsqueeze(0)), dim=0)
            elif "RF_GF_BF" in self.ef_method:
                anc = torch.cat((((anc_RGB[0]+anc_depth[0])/2).unsqueeze(0), ((anc_RGB[1]+anc_depth[0])/2).unsqueeze(0), ((anc_RGB[2]+anc_depth[0])/2).unsqueeze(0)), dim=0)
                pos = torch.cat((((pos_RGB[0]+pos_depth[0])/2).unsqueeze(0), ((pos_RGB[1]+pos_depth[0])/2).unsqueeze(0), ((pos_RGB[2]+pos_depth[0])/2).unsqueeze(0)), dim=0)
                neg = torch.cat((((neg_RGB[0]+neg_depth[0])/2).unsqueeze(0), ((neg_RGB[1]+neg_depth[0])/2).unsqueeze(0), ((neg_RGB[2]+neg_depth[0])/2).unsqueeze(0)), dim=0)
        else:
            anc, pos, neg = torch.cat((anc_RGB, anc_depth), dim=0), torch.cat((pos_RGB, pos_depth), dim=0), torch.cat((neg_RGB, neg_depth), dim=0)

        return anc, pos, neg

    def __len__(self):
        return len(self.imgsAnc)
    

class Test_EF(Dataset):

    def __init__(self, env="FR_A", enc="vitl", il="Cloudy", ef_method="6_channels", input_type="MAGNITUDE", tf=transforms.ToTensor()):

        self.enc, self.ef_method, self.input_type, self.env, self.il, self.tf = enc, ef_method, input_type, env, il, tf
        self.color_rep = PARAMS.color_rep if self.ef_method in ["6_channels", "3_channels_RF_GF_BF"] else None

        CSV_file = pd.read_csv(f'{csvDir}/Test_{env}_{self.il}.csv')
        self.imgList, self.coordX, self.coordY = CSV_file['Img'], CSV_file['CoordX'], CSV_file['CoordY']
        self.rgb_dir, self.depth_dir = f"{rgb_dir}{self.env}/Test{self.il}/", f"{features_dir}{self.input_type}/{self.env}/Test{self.il}/"    

    def __getitem__(self, index):

        imgPath, coords = self.imgList[index], img_proc.load_coords(self.coordX[index], self.coordY[index])
        img_RGB, img_depth = f"{self.rgb_dir}{imgPath}", f"{self.depth_dir}{imgPath}".replace(".jpeg", ".npy")    

        img_RGB = process_image(image=img_RGB, rgb=True, eq=False, inv=False, sh=False, color_rep=False, tf=self.tf)   
        img_depth = process_image(image=img_depth, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=self.color_rep, tf=self.tf)

        if "3" in self.ef_method:
            if "RGF" in self.ef_method:
                img = torch.cat((img_RGB[0].unsqueeze(0), img_RGB[1].unsqueeze(0), img_depth[0].unsqueeze(0)), dim=0)
            elif "RG_BF" in self.ef_method:
                img = torch.cat((img_RGB[0].unsqueeze(0), img_RGB[1].unsqueeze(0), ((img_RGB[2]+img_depth[0])/2).unsqueeze(0)), dim=0)
            elif "RF_GF_BF" in self.ef_method:
                img = torch.cat((((img_RGB[0]+img_depth[0])/2).unsqueeze(0), ((img_RGB[1]+img_depth[0])/2).unsqueeze(0), ((img_RGB[2]+img_depth[0])/2).unsqueeze(0)), dim=0)
        else:
            img = torch.cat((img_RGB, img_depth), dim=0)

        return img, coords

    def __len__(self):
        return len(self.imgList)



class Database_EF(Dataset):

    def __init__(self, enc="vitl", ef_method="6_channels", input_type="MAGNITUDE", env="FR_A", tf=transforms.ToTensor()):

        self.enc, self.ef_method, self.input_type, self.env, self.tf = enc, ef_method, input_type, env, tf
        self.color_rep = PARAMS.color_rep if self.ef_method in ["6_channels", "3_channels_RF_GF_BF"] else None

        CSV_file = pd.read_csv(f'{csvDir}/VisualModel_{env}.csv')
        self.imgList, self.coordX, self.coordY = CSV_file['Img'], CSV_file['CoordX'], CSV_file['CoordY']
        self.rgb_dir, self.depth_dir = f"{rgb_dir}{self.env}/Train/", f"{features_dir}{self.input_type}/{self.env}/Train/"   

    def __getitem__(self, index):

        imgPath, coords = self.imgList[index], img_proc.load_coords(self.coordX[index], self.coordY[index])
        img_RGB, img_depth = f"{self.rgb_dir}{imgPath}", f"{self.depth_dir}{imgPath}".replace(".jpeg", ".npy")

        img_RGB = process_image(image=img_RGB, rgb=True, eq=False, inv=False, sh=False, color_rep=False, tf=self.tf)   
        img_depth = process_image(image=img_depth, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=self.color_rep, tf=self.tf)

        if "3" in self.ef_method:
            if "RGF" in self.ef_method:
                img = torch.cat((img_RGB[0].unsqueeze(0), img_RGB[1].unsqueeze(0), img_depth[0].unsqueeze(0)), dim=0)
            elif "RG_BF" in self.ef_method:
                img = torch.cat((img_RGB[0].unsqueeze(0), img_RGB[1].unsqueeze(0), ((img_RGB[2]+img_depth[0])/2).unsqueeze(0)), dim=0)
            elif "RF_GF_BF" in self.ef_method:
                img = torch.cat((((img_RGB[0]+img_depth[0])/2).unsqueeze(0), ((img_RGB[1]+img_depth[0])/2).unsqueeze(0), ((img_RGB[2]+img_depth[0])/2).unsqueeze(0)), dim=0)
        else:
            img = torch.cat((img_RGB, img_depth), dim=0)

        return img, coords

    def __len__(self):
        return len(self.imgList)







"""   LATE FUSION """

class Train_LF(Dataset):

    def __init__(self, enc="vitl", input_type="RGB", env="FR_A", tf=transforms.ToTensor()):

        self.enc, self.input_type, self.env, self.tf = enc, input_type, env, tf

        CSV_file = pd.read_csv(f'{csvDir}Train_{env}.csv')
        self.imgsAnc, self.imgsPos, self.imgsNeg = CSV_file['ImgAnc'], CSV_file['ImgPos'], CSV_file['ImgNeg']

        self.rgb = True if self.input_type=="RGB" else False
        self.img_path = f"{rgb_dir}{self.env}/Train/" if self.input_type=="RGB" else f"{features_dir}{self.input_type}/{self.env}/Train/"
    
    def __getitem__(self, index):

        imgAnc, imgPos, imgNeg = self.imgsAnc[index], self.imgsPos[index], self.imgsNeg[index]
        imgAnc, imgPos, imgNeg = f"{self.img_path}{imgAnc}", f"{self.img_path}{imgPos}", f"{self.img_path}{imgNeg}"

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

    def __init__(self, env="FR_A", enc="vitl", il="Cloudy", input_type="DEPTH_COLOR", tf=transforms.ToTensor()):

        self.enc, self.input_type, self.env, self.il, self.tf = enc, input_type, env, il, tf

        CSV_file = pd.read_csv(f'{csvDir}/Test_{self.env}_{self.il}.csv')
        self.imgList, self.coordX, self.coordY = CSV_file['Img'], CSV_file['CoordX'], CSV_file['CoordY']
        self.rgb_dir, self.depth_dir = f"{rgb_dir}{self.env}/Test{self.il}/", f"{features_dir}{self.input_type}/{self.env}/Test{self.il}/"
    
    def __getitem__(self, index):

        imgPath, coords = self.imgList[index], img_proc.load_coords(self.coordX[index], self.coordY[index])

        img_RGB, img_depth = f"{self.rgb_dir}{imgPath}", f"{self.depth_dir}{imgPath}"
        img_RGB = process_image(image=img_RGB, rgb=True, eq=False, inv=False, sh=False, tf=self.tf)

        img_depth = img_depth.replace(".jpeg", ".npy")
        img_depth = process_image(image=img_depth, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
        img_depth = torch.cat((img_depth, img_depth, img_depth), dim=0) 

        return img_RGB, img_depth, coords

    def __len__(self):
        return len(self.imgList)

    
class Database_LF(Dataset):

    def __init__(self, env="FR_A", enc="vitl", input_type="DEPTH_COLOR", tf=transforms.ToTensor()):

        self.enc, self.input_type, self.env, self.tf = enc, input_type, env, tf

        CSV_file = pd.read_csv(f'{csvDir}/VisualModel_{self.env}.csv')
        self.imgList, self.coordX, self.coordY = CSV_file['Img'], CSV_file['CoordX'], CSV_file['CoordY']
        self.rgb_dir, self.depth_dir = f"{rgb_dir}{self.env}/Train/", f"{features_dir}{self.input_type}/{self.env}/Train/"

    def __getitem__(self, index):

        imgPath, coords = self.imgList[index], img_proc.load_coords(self.coordX[index], self.coordY[index])

        img_RGB, img_depth = f"{self.rgb_dir}{imgPath}", f"{self.depth_dir}{imgPath}"
        img_RGB = process_image(image=img_RGB, rgb=True, eq=False, inv=False, sh=False, tf=self.tf)
        img_depth = img_depth.replace(".jpeg", ".npy")
        img_depth = process_image(image=img_depth, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
        img_depth = torch.cat((img_depth, img_depth, img_depth), dim=0) 
        return img_RGB, img_depth, coords

    def __len__(self):
        return len(self.imgList)
    



"""LATE FUSION (MORE THAN ONE FEATURE)"""

class Train_EF_multifeatures(Dataset):

    def __init__(self, enc="vitl", features=None, env="FR_A", tf=transforms.ToTensor()):

        self.enc, self.features, self.env, self.tf = enc, features, env, tf
        self.color_rep = None

        CSV_file = pd.read_csv(f'{csvDir}/Train_{env}.csv')
        self.imgsAnc, self.imgsPos, self.imgsNeg = CSV_file['ImgAnc'], CSV_file['ImgPos'], CSV_file['ImgNeg']
        self.rgb_dir = f"{rgb_dir}{self.env}/Train/"

    def __getitem__(self, index):

        imgAnc, imgPos, imgNeg = self.imgsAnc[index], self.imgsPos[index], self.imgsNeg[index]

        imgAnc_RGB, imgPos_RGB, imgNeg_RGB = f"{self.rgb_dir}{imgAnc}", f"{self.rgb_dir}{imgPos}", f"{self.rgb_dir}{imgNeg}"

        anc = process_image(image=imgAnc_RGB, rgb=True, eq=False, inv=False, sh=False, color_rep=False, tf=self.tf)
        pos = process_image(image=imgPos_RGB, rgb=True, eq=False, inv=False, sh=False, color_rep=False, tf=self.tf)
        neg = process_image(image=imgNeg_RGB, rgb=True, eq=False, inv=False, sh=False, color_rep=False, tf=self.tf)

        for feature in self.features:
            if feature == "RGB":
                continue
            imgAnc_f = features_dir + feature + "/" + self.env + "/Train/" + imgAnc
            imgPos_f = features_dir + feature + "/" + self.env + "/Train/" + imgPos
            imgNeg_f = features_dir + feature + "/" + self.env + "/Train/" + imgNeg
            imgAnc_f = imgAnc_f.replace(".jpeg", ".npy")
            imgPos_f = imgPos_f.replace(".jpeg", ".npy")
            imgNeg_f = imgNeg_f.replace(".jpeg", ".npy")

            anc_f = process_image(image=imgAnc_f, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
            pos_f = process_image(image=imgPos_f, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
            neg_f = process_image(image=imgNeg_f, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)

            anc = torch.cat((anc, anc_f), dim=0)
            pos = torch.cat((pos, pos_f), dim=0)
            neg = torch.cat((neg, neg_f), dim=0)

        return anc, pos, neg

    def __len__(self):
        return len(self.imgsAnc)

class Database_multifeatures(Dataset):

    def __init__(self, enc="vitl", features=["GRAYSCALE", "MAGNITUDE", "ANGLE", "HUE"], env="FR_A", il="Cloudy", tf=transforms.ToTensor()):

        self.enc, self.features, self.env, self.il, self.tf = enc, features, env, il, tf

        CSV_file = pd.read_csv(f'{csvDir}/VisualModel_{self.env}.csv')
        self.imgList, self.coordX, self.coordY = CSV_file['Img'], CSV_file['CoordX'], CSV_file['CoordY']
        self.rgb_dir = f"{rgb_dir}{self.env}/Train/"

    def __getitem__(self, index):

        imgPath, coords = self.imgList[index], img_proc.load_coords(self.coordX[index], self.coordY[index])

        img = f"{self.rgb_dir}{imgPath}"
        img = process_image(image=img, rgb=True, eq=False, inv=False, sh=False, tf=self.tf)

        for feature in self.features:
            if feature == "RGB":
                continue
            img_f = features_dir + feature + "/" + self.env + "/Train/" + imgPath
            img_f = img_f.replace(".jpeg", ".npy")
            img_f = process_image(image=img_f, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
            img = torch.cat((img, img_f), dim=0)
        return img, coords

    def __len__(self):
        return len(self.imgList)
    

class Test_multifeatures(Dataset):

    def __init__(self, enc="vitl", features=["GRAYSCALE", "MAGNITUDE", "ANGLE", "HUE"], env="FR_A", il="Cloudy", tf=transforms.ToTensor()):

        self.enc, self.features, self.env, self.il, self.tf = enc, features, env, il, tf

        CSV_file = pd.read_csv(f'{csvDir}/Test_{self.env}_{self.il}.csv')
        self.imgList, self.coordX, self.coordY = CSV_file['Img'], CSV_file['CoordX'], CSV_file['CoordY']
        self.rgb_dir = f"{rgb_dir}{self.env}/Test{self.il}/"

    def __getitem__(self, index):

        imgPath, coords = self.imgList[index], img_proc.load_coords(self.coordX[index], self.coordY[index])

        img = f"{self.rgb_dir}{imgPath}"
        img = process_image(image=img, rgb=True, eq=False, inv=False, sh=False, tf=self.tf)

        for feature in self.features:
            if feature == "RGB":
                continue
            img_f = features_dir + feature + "/" + self.env + "/Test" + self.il + "/" + imgPath
            img_f = img_f.replace(".jpeg", ".npy")
            img_f = process_image(image=img_f, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
            img = torch.cat((img, img_f), dim=0)
        return img, coords

    def __len__(self):
        return len(self.imgList)
    





"""\wo FUSION, \wo NET ADAPTATION"""


class Test_wo_Fusion(Dataset):

    def __init__(self, env="FR_A", enc="vitl", il="Cloudy", input_type="RGB", tf=transforms.ToTensor()):

        self.enc, self.input_type, self.env, self.il, self.tf = enc, input_type, env, il, tf

        CSV_file = pd.read_csv(f'{csvDir}/Test_{self.env}_{self.il}.csv')
        self.imgList, self.coordX, self.coordY = CSV_file['Img'], CSV_file['CoordX'], CSV_file['CoordY']

        self.rgb_dir, self.depth_dir  = f"{rgb_dir}{self.env}/Test{self.il}/", f"{features_dir}{self.input_type}/{self.env}/Test{self.il}/"

    def __getitem__(self, index):

        imgPath, coords = self.imgList[index], img_proc.load_coords(self.coordX[index], self.coordY[index])

        if self.input_type is None or self.input_type=="RGB":
            img = f"{self.rgb_dir}{imgPath}" 
            img = process_image(image=img, rgb=True, eq=False, inv=False, sh=False, color_rep=None, tf=self.tf)
        else:
            img = f"{self.depth_dir}{imgPath}"
            img = img.replace(".jpeg", ".npy")
            img = process_image(image=img, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
            img = torch.cat((img, img, img), dim=0) 
        return img, coords

    def __len__(self):
        return len(self.imgList)
    

class Database_wo_Fusion(Dataset):

    def __init__(self, env="FR_A", enc="vitl", input_type="RGB", tf=transforms.ToTensor()):

        self.enc, self.input_type, self.env, self.tf = enc, input_type, env, tf

        CSV_file = pd.read_csv(f'{csvDir}/VisualModel_{self.env}.csv')
        self.imgList, self.coordX, self.coordY = CSV_file['Img'], CSV_file['CoordX'], CSV_file['CoordY']
        self.rgb_dir, self.depth_dir  = f"{rgb_dir}{self.env}/Train/", f"{features_dir}{self.input_type}/{self.env}/Train/"

    def __getitem__(self, index):

        imgPath, coords = self.imgList[index], img_proc.load_coords(self.coordX[index], self.coordY[index])

        if self.input_type is None or self.input_type=="RGB":
            img = f"{self.rgb_dir}{imgPath}" 
            img = process_image(image=img, rgb=True, eq=False, inv=False, sh=False, color_rep=None, tf=self.tf)
        else:
            img = f"{self.depth_dir}{imgPath}"
            img = img.replace(".jpeg", ".npy")
            img = process_image(image=img, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
            img = torch.cat((img, img, img), dim=0) 
        return img, coords

    def __len__(self):
        return len(self.imgList)





class Train_MLP(Dataset):

    def __init__(self, env="FR_A", tf=transforms.ToTensor()):

        self.tf, self.env = tf, env

        CSV_file = pd.read_csv(csvDir + f'/Train_{env}.csv')

        self.imgsAnc, self.imgsPos, self.imgsNeg = CSV_file['ImgAnc'], CSV_file['ImgPos'], CSV_file['ImgNeg']
        self.rgb_dir, self.depth_dir = f"{rgb_dir}{self.env}/Train/", f"{depth_dir}{self.env}/Train/"
        

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
