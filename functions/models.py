import torch
import torch.nn as nn
from config import PARAMS
from MixVPR.main import VPRModel
from utils.losses import get_loss



def load_model(model="cosplace", backbone="VGG16", embedding_size=512, state_dict_path=None, device=PARAMS.device):
    if model == "cosplace":
        net = torch.hub.load(repo_or_dir="gmberton/cosplace", model="get_trained_model", 
                             backbone=backbone, fc_output_dim=embedding_size)
    elif model == "eigenplaces":
        net = torch.hub.load(repo_or_dir="gmberton/eigenplaces", model="get_trained_model", 
                             backbone=backbone, fc_output_dim=embedding_size)
    elif model == "salad":
        net = torch.hub.load("serizba/salad", "dinov2_salad")
        net.eval()
    elif model == "mixvpr":
        net = VPRModel(backbone_arch=backbone,
                       layers_to_crop=[4],
                       agg_arch='MixVPR',
                       agg_config={'in_channels': 1024,
                                   'in_h': 20,
                                   'in_w': 20,
                                  'out_channels': 1024,
                                   'mix_depth': 4,
                                   'mlp_ratio': 1,
                                   'out_rows': 4},
                       )
        state_dict = torch.load(
            '/home/arvc/Marcos/INVESTIGACION/0_SAVED_MODELS/0_OTHERS_WORK/MIXVPR/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt')
        net.load_state_dict(state_dict)
    else:
        raise ValueError(f"Model {model} not recognized.")
    net = net.float().to(device)
    if state_dict_path is not None:
        state_dict = torch.load(state_dict_path, map_location=device)
        net.load_state_dict(state_dict)
    return net


def load_model_ef(pretrained_model, num_channels=3, weightDir=None, device="cuda:0"):

    model = load_model(model=pretrained_model)
       
    with torch.no_grad():

        if num_channels < 3:
            raise ValueError("Number of channels must be at least 3")
        elif num_channels == 3:
            if weightDir is None:
                print("Using pretrained model")
                return model
            else:
                print(f"Loading weights from {weightDir}")
                state_dict = torch.load(weightDir, map_location=device, weights_only=False)
                model.load_state_dict(state_dict)
        else:
            state_dict = model.state_dict()

            model.backbone[0] = torch.nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=(3, 3),
                                                stride=(1, 1), padding=(1, 1))

            weights_conv1 = state_dict['backbone.0.weight'] # Copiar los pesos originales de RGB
            if num_channels == 4:
                state_dict['backbone.0.weight'] = torch.cat((weights_conv1, weights_conv1.mean(dim=1, keepdim=True)), dim=1)
                #state_dict['backbone.0.weight'] = torch.cat((weights_conv1, torch.zeros((64, 1, 3, 3))), dim=1)
            elif num_channels == 5:
                state_dict['backbone.0.weight'] = torch.cat((weights_conv1, weights_conv1.mean(dim=1, keepdim=True), weights_conv1.mean(dim=1, keepdim=True)), dim=1)
                #state_dict['backbone.0.weight'] = torch.cat((weights_conv1, weights_conv1, torch.zeros((64, 1, 3, 3))), dim=1)
            elif num_channels == 6:
                state_dict['backbone.0.weight'] = torch.cat((weights_conv1, weights_conv1), dim=1)
            elif num_channels == 7:
                state_dict['backbone.0.weight'] = torch.cat((weights_conv1, weights_conv1, weights_conv1.mean(dim=1, keepdim=True)), dim=1)
                #state_dict['backbone.0.weight'] = torch.cat((weights_conv1, torch.zeros_like(weights_conv1)), dim=1)
            if weightDir is not None:
                print(f"Loading weights from {weightDir}")
                state_dict = torch.load(weightDir, map_location=device, weights_only=False)
            model.load_state_dict(state_dict)

    return model



class MonolayerPerceptron(nn.Module):
    def __init__(self, in_dim=1024, out_dim=2048):
        super(MonolayerPerceptron, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLP_2layers(nn.Module):
    def __init__(self, in_dim=1024, mid_dim=4096, out_dim=1024):
        super(MLP_2layers, self).__init__()
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
