import random
import torch
import numpy as np


def make_deterministic(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_triplet_sample(imgs, coords, tree, rPos, rNeg):
    idxAnc = random.randrange(len(imgs))
    imgAnc = imgs[idxAnc]
    coordsAnc = coords[idxAnc]

    indices = tree.query_radius(coordsAnc.reshape(1, -1), r=rPos)[0]
    idxPos = random.choice(indices)
    while idxAnc == idxPos:
        idxPos = random.choice(indices)
    imgPos = imgs[idxPos]

    indices = tree.query_radius(coordsAnc.reshape(1, -1), r=rNeg)[0]
    idxNeg = random.randrange(len(imgs))
    while idxNeg in indices or idxAnc == idxNeg:
        idxNeg = random.randrange(len(imgs))
    imgNeg = imgs[idxNeg]
    return imgAnc, imgPos, imgNeg