import os
import torch


def select_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    return device



def create_path(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return directory


