from PIL import Image
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms


def load_image(imagePath, rgb=True):
    if imagePath.endswith((".jpeg", ".jpg", ".png")):
        if rgb:
            return np.array(Image.open(imagePath)).astype(np.uint8)
        else:
            image = np.array(Image.open(imagePath)).astype(np.uint8)
            if image.shape[-1] == 3:
                return image[:,:,0]
            else:
                return image
    elif imagePath.endswith(".npy"):
        image = np.load(imagePath)
        # interpolate to uint8
        image = (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)
        return image
    else:
        return None


def load_coords(x, y):
    x, y = x.astype(np.float32), y.astype(np.float32)
    return np.array([x, y])


def equalize_image(image):
    if len(image.shape) == 2:  # Grayscale image
        return cv2.equalizeHist(image)
    else:
        raise ValueError("Unsupported image shape for equalization.")
    

def normalize_image(image):
    tf = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    image = tf(image)
    return image

def inverse_image(image):
    return 255 - np.array(image)

def sharpen_image(image):
    sharpening_kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
    return cv2.filter2D(np.array(image), -1, sharpening_kernel)

def apply_colormap(image, color_rep):
    if color_rep == "HSV":
        image = cv2.applyColorMap(image, cv2.COLORMAP_HSV)
    if color_rep == "JET":
        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    else:  # Default is RGB
        image = image
    return image


def select_tf(model):
    if model in ["anyloc", "salad"]:
        tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((518, 518))])
    elif model in ["mixvpr"]:
        tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((320, 320))])
    elif model in ["cosplace", "eigenplaces"]:
        tf = transforms.ToTensor()
    else:
        tf = None
    return tf

def tf_image(image, tf=None):
    if tf is not None:
        image = tf(Image.fromarray(image))
    return image


def replicate_channels(image):
    return torch.cat((image, image, image), dim=0)




