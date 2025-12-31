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


csvDir = f"{PARAMS.csv_path}train_eval_360LOC/"

dataset_dir = f"{PARAMS._360loc_path}"


def process_image(image, rgb=True, eq=True, inv=True, sh=True, color_rep=None, tf=transforms.ToTensor()):
    save_img = False
    if "/atrium" in image and "0200" in image and "daytime_360_0" in image:
        save_img = True
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

    return image



class Train_EF(Dataset):

    def __init__(self, enc="vitl", ef_method=None, input_type="MAGNITUDE", env="atrium", il="daytime_360_0", tf=transforms.ToTensor()):
        
        self.enc, self.ef_method, self.input_type, self.env, self.il, self.tf = enc, ef_method, input_type, env, il, tf

        CSV_file = pd.read_csv(f'{csvDir}/train_{self.env}_{self.il}.csv')
        self.imgsAnc, self.imgsPos, self.imgsNeg = CSV_file['ImgAnc'], CSV_file['ImgPos'], CSV_file['ImgNeg']

        self.color_rep = PARAMS.color_rep if self.ef_method in ["6_channels", "3_channels_RH_GS_BV"] else None
    
        self.rgb_dir = f"{dataset_dir}{self.env}/mapping/{self.il}/image_resized/"
        self.depth_dir = self.rgb_dir.replace("image_resized", self.input_type)

    def __getitem__(self, index):

        imgAnc, imgPos, imgNeg = self.imgsAnc[index], self.imgsPos[index], self.imgsNeg[index]
        
        imgAnc_RGB, imgPos_RGB, imgNeg_RGB = f"{self.rgb_dir}{imgAnc}", f"{self.rgb_dir}{imgPos}", f"{self.rgb_dir}{imgNeg}"
        imgAnc_depth, imgPos_depth, imgNeg_depth = f"{self.depth_dir}{imgAnc}", f"{self.depth_dir}{imgPos}", f"{self.depth_dir}{imgNeg}"
        imgAnc_depth, imgPos_depth, imgNeg_depth = imgAnc_depth.replace(".jpg", ".npy"), imgPos_depth.replace(".jpg", ".npy"), imgNeg_depth.replace(".jpg", ".npy")

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

    def __init__(self, enc="vitl",  ef_method="6_channels", input_type="MAGNITUDE", env="atrium", il="daytime1", tf=transforms.ToTensor()):

        self.enc, self.ef_method, self.input_type, self.env, self.il, self.tf = enc, ef_method, input_type, env, il, tf
        self.color_rep = PARAMS.color_rep if self.ef_method in ["6_channels", "3_channels_RF_GF_BF"] else None

        CSV_file = pd.read_csv(f'{csvDir}/test_{env}_{self.il}.csv')
        self.imgList, self.coordX, self.coordY = CSV_file['Img'], CSV_file['CoordX'], CSV_file['CoordY']

        self.rgb_dir, self.depth_dir = f"{dataset_dir}{self.env}/query_360/{self.il}/image_resized/", f"{dataset_dir}{self.env}/query_360/{self.il}/{self.input_type}/"

    def __getitem__(self, index):

        imgPath, coords = self.imgList[index], img_proc.load_coords(self.coordX[index], self.coordY[index])
        img_RGB, img_depth = f"{self.rgb_dir}{imgPath}", f"{self.depth_dir}{imgPath}".replace(".jpg", ".npy") 

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

    def __init__(self, enc="vitl", ef_method="6_channels", input_type="MAGNITUDE", env="atrium", il="daytime_360_0", tf=transforms.ToTensor()):

        self.enc, self.ef_method, self.input_type, self.env, self.il, self.tf = enc, ef_method, input_type, env, il, tf
        self.color_rep = PARAMS.color_rep if self.ef_method in ["6_channels", "3_channels_RF_GF_BF"] else None

        CSV_file = pd.read_csv(f'{csvDir}database_{env}_{self.il}.csv')
        self.imgList, self.coordX, self.coordY = CSV_file['Img'], CSV_file['CoordX'], CSV_file['CoordY']

        self.rgb_dir, self.depth_dir = f"{dataset_dir}{self.env}/mapping/{self.il}/image_resized/", f"{dataset_dir}{self.env}/mapping/{self.il}/{self.input_type}/"

    def __getitem__(self, index):

        imgPath, coords = self.imgList[index], img_proc.load_coords(self.coordX[index], self.coordY[index])
        img_RGB, img_depth = f"{self.rgb_dir}{imgPath}", f"{self.depth_dir}{imgPath}".replace(".jpg", ".npy") 

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





class Train_LF(Dataset):

    def __init__(self, enc="vitl", input_type="RGB", env="atrium", il="daytime_360_0", tf=transforms.ToTensor()):
        
        self.enc, self.input_type, self.env, self.il, self.tf = enc, input_type, env, il, tf

        CSV_file = pd.read_csv(f'{csvDir}/train_{self.env}_{self.il}.csv')
        self.imgsAnc, self.imgsPos, self.imgsNeg = CSV_file['ImgAnc'], CSV_file['ImgPos'], CSV_file['ImgNeg']
        
        self.rgb = True if self.input_type=="RGB" else False
        self.img_path = f"{dataset_dir}{self.env}/mapping/{self.il}/image_resized/" if self.input_type=="RGB" else f"{dataset_dir}{self.env}/mapping/{self.il}/{self.input_type}/"

    def __getitem__(self, index):

        imgAnc, imgPos, imgNeg = self.imgsAnc[index], self.imgsPos[index], self.imgsNeg[index]
        imgAnc, imgPos, imgNeg = f"{self.img_path}{imgAnc}", f"{self.img_path}{imgPos}", f"{self.img_path}{imgNeg}"

        if self.input_type != "RGB":
            imgAnc, imgPos, imgNeg = imgAnc.replace(".jpg", ".npy"), imgPos.replace(".jpg", ".npy"), imgNeg.replace(".jpg", ".npy")

        anc = process_image(image=imgAnc, rgb=self.rgb, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
        pos = process_image(image=imgPos, rgb=self.rgb, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
        neg = process_image(image=imgNeg, rgb=self.rgb, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
        
        if self.input_type != "RGB":
            anc, pos, neg = torch.cat((anc, anc, anc), dim=0), torch.cat((pos, pos, pos), dim=0), torch.cat((neg, neg, neg), dim=0)

        return anc, pos, neg

    def __len__(self):
        return len(self.imgsAnc) 




class Test_LF(Dataset):

    def __init__(self, enc="vitl", input_type="MAGNITUDE", env="atrium", il="daytime_360_1", tf=transforms.ToTensor()):

        self.enc, self.input_type, self.env, self.il, self.tf = enc, input_type, env, il, tf

        CSV_file = pd.read_csv(f'{csvDir}/test_{self.env}_{self.il}.csv')
        self.imgList, self.coordX, self.coordY = CSV_file['Img'], CSV_file['CoordX'], CSV_file['CoordY']
        self.rgb_dir  = f"{dataset_dir}{self.env}/query_360/{self.il}/image_resized/"
        self.feature_dir = self.rgb_dir.replace("image_resized", self.input_type)

    def __getitem__(self, index):

        imgPath, coords = self.imgList[index], img_proc.load_coords(self.coordX[index], self.coordY[index])

        img_RGB, img_depth = f"{self.rgb_dir}{imgPath}", f"{self.feature_dir}{imgPath}"
        img_RGB = process_image(image=img_RGB, rgb=True, eq=False, inv=False, sh=False, tf=self.tf)

        img_depth = img_depth.replace(".jpg", ".npy")
        img_depth = process_image(image=img_depth, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
        img_depth = torch.cat((img_depth, img_depth, img_depth), dim=0) 

        return img_RGB, img_depth, coords

    def __len__(self):
        return len(self.imgList)

    

class Database_LF(Dataset):

    def __init__(self, enc="vitl", input_type="MAGNITUDE", env="atrium", il="daytime_360_0", tf=transforms.ToTensor()):

        self.enc, self.input_type, self.env, self.il, self.tf = enc, input_type, env, il, tf

        CSV_file = pd.read_csv(f'{csvDir}/database_{self.env}_{self.il}.csv')
        self.imgList, self.coordX, self.coordY = CSV_file['Img'], CSV_file['CoordX'], CSV_file['CoordY']
        self.rgb_dir  = f"{dataset_dir}{self.env}/mapping/{self.il}/image_resized/"
        self.feature_dir = self.rgb_dir.replace("image_resized", self.input_type)

    def __getitem__(self, index):

        imgPath, coords = self.imgList[index], img_proc.load_coords(self.coordX[index], self.coordY[index])

        img_RGB, img_depth = f"{self.rgb_dir}{imgPath}", f"{self.feature_dir}{imgPath}"
        img_RGB = process_image(image=img_RGB, rgb=True, eq=False, inv=False, sh=False, tf=self.tf)
        img_depth = img_depth.replace(".jpg", ".npy")
        img_depth = process_image(image=img_depth, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
        img_depth = torch.cat((img_depth, img_depth, img_depth), dim=0) 
        return img_RGB, img_depth, coords

    def __len__(self):
        return len(self.imgList)
    


"""LATE FUSION (MORE THAN ONE FEATURE)"""

class Database_LF_multifeatures(Dataset):

    def __init__(self, enc="vitl", features=["GRAYSCALE", "MAGNITUDE", "ANGLE", "HUE"], env="atrium", il="daytime_360_0", tf=transforms.ToTensor()):

        self.enc, self.features, self.env, self.il, self.tf = enc, features, env, il, tf

        CSV_file = pd.read_csv(f'{csvDir}/database_{self.env}_{self.il}.csv')
        self.imgList, self.coordX, self.coordY = CSV_file['Img'], CSV_file['CoordX'], CSV_file['CoordY']
        self.rgb_dir  = f"{dataset_dir}{self.env}/mapping/{self.il}/image_resized/"
        #self.feature_dir = self.rgb_dir.replace("image_resized", self.input_type)

    def __getitem__(self, index):

        imgPath, coords = self.imgList[index], img_proc.load_coords(self.coordX[index], self.coordY[index])

        img = f"{self.rgb_dir}{imgPath}"
        img = process_image(image=img, rgb=True, eq=False, inv=False, sh=False, tf=self.tf)

        for feature in self.features:
            if feature == "RGB":
                continue
            img_f = self.rgb_dir.replace("image_resized", feature) + imgPath
            img_f = img_f.replace(".jpg", ".npy")
            img_f = process_image(image=img_f, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
            img = torch.cat((img, img_f), dim=0)
        return img, coords

    def __len__(self):
        return len(self.imgList)
    

class Test_LF_multifeatures(Dataset):

    def __init__(self, enc="vitl", features=["GRAYSCALE", "MAGNITUDE", "ANGLE", "HUE"], env="atrium", il="daytime_360_1", tf=transforms.ToTensor()):

        self.enc, self.features, self.env, self.il, self.tf = enc, features, env, il, tf

        CSV_file = pd.read_csv(f'{csvDir}/test_{self.env}_{self.il}.csv')
        self.imgList, self.coordX, self.coordY = CSV_file['Img'], CSV_file['CoordX'], CSV_file['CoordY']
        self.rgb_dir  = f"{dataset_dir}{self.env}/query_360/{self.il}/image_resized/"
        #self.feature_dir = self.rgb_dir.replace("image_resized", self.input_type)

    def __getitem__(self, index):

        imgPath, coords = self.imgList[index], img_proc.load_coords(self.coordX[index], self.coordY[index])

        img = f"{self.rgb_dir}{imgPath}"
        img = process_image(image=img, rgb=True, eq=False, inv=False, sh=False, tf=self.tf)

        for feature in self.features:
            if feature == "RGB":
                continue
            img_f = self.rgb_dir.replace("image_resized", feature) + imgPath
            img_f = img_f.replace(".jpg", ".npy")
            img_f = process_image(image=img_f, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
            img = torch.cat((img, img_f), dim=0)
        return img, coords

    def __len__(self):
        return len(self.imgList)








class Test_wo_Fusion(Dataset):

    def __init__(self, enc="vitl",input_type=None, env="atrium", il="daytime1", tf=transforms.ToTensor()):

        self.enc, self.input_type, self.env, self.il, self.tf = enc, input_type, env, il, tf

        CSV_file = pd.read_csv(f'{csvDir}/test_{self.env}_{self.il}.csv')
        self.imgList, self.coordX, self.coordY = CSV_file['Img'], CSV_file['CoordX'], CSV_file['CoordY']

        self.rgb_dir = f"{dataset_dir}{self.env}/query_360/{self.il}/image_resized/"
        self.feature_dir = self.rgb_dir.replace("image_resized", self.input_type)

    def __getitem__(self, index):

        imgPath, coords = self.imgList[index], img_proc.load_coords(self.coordX[index], self.coordY[index])

        if self.input_type is None or self.input_type=="RGB":
            img = f"{self.rgb_dir}{imgPath}" 
            img = process_image(image=img, rgb=True, eq=False, inv=False, sh=False, color_rep=None, tf=self.tf)
        else:
            img = f"{self.feature_dir}{imgPath}"
            img = img.replace(".jpg", ".npy")
            img = process_image(image=img, rgb=False, eq=PARAMS.eq, inv=PARAMS.inv, sh=PARAMS.sh, color_rep=PARAMS.color_rep, tf=self.tf)
            img = torch.cat((img, img, img), dim=0) 
        return img, coords

    def __len__(self):
        return len(self.imgList)


class Database_wo_Fusion(Dataset):

    def __init__(self, enc="vitl", env="atrium", input_type=None, il="daytime_360_0", tf=transforms.ToTensor()):

        self.enc, self.input_type, self.env, self.il, self.tf = enc, input_type, env, il, tf

        CSV_file = pd.read_csv(f'{csvDir}/database_{self.env}_{self.il}.csv')
        self.imgList, self.coordX, self.coordY = CSV_file['Img'], CSV_file['CoordX'], CSV_file['CoordY']

        self.rgb_dir = f"{dataset_dir}{self.env}/mapping/{self.il}/image_resized/"
        self.feature_dir = self.rgb_dir.replace("image_resized", self.input_type)

    def __getitem__(self, index):

        imgPath, coords = self.imgList[index], img_proc.load_coords(self.coordX[index], self.coordY[index])

        if self.input_type is None or self.input_type=="RGB":
            img = f"{self.rgb_dir}{imgPath}" 
            img = process_image(image=img, rgb=True, eq=False, inv=False, sh=False, color_rep=None, tf=self.tf)
        else:
            img = f"{self.feature_dir}{imgPath}"
            img = img.replace(".jpg", ".npy")
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
