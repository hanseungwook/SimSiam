from .simsiam import SimSiam, SimSiamKD, SimSiamKDAnchor, SimSiamAdv, SimSiamJoint, SimSiamMI, SimSiamNoise
from .byol import BYOL
from .simclr import SimCLR, SimCLRJoint, SimCLRMI, SimCLRGram, SimCLRKL
from torchvision.models import resnet50, resnet18
import torch
from .backbones import *

def get_backbone(backbone, castrate=True):
    backbone = eval(f"{backbone}()")

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone

def get_backbone_kd(backbone_s, backbone_t, num_classes=10, castrate=True):
    backbone_s = eval(f"{backbone_s}()")
    backbone_t = eval(f"{backbone_t}(pretrained=True, num_classes={num_classes})")

    if castrate:
        backbone_s.output_dim = backbone_s.fc.in_features
        backbone_s.fc = torch.nn.Identity()
    
        backbone_t.output_dim = backbone_t.fc.in_features
        backbone_t.fc = torch.nn.Identity()


    return backbone_s, backbone_t


def get_model(model_cfg):    

    if model_cfg.name == 'simsiam':
        model =  SimSiam(get_backbone(model_cfg.backbone))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.name == 'simsiam_kd':
        model =  SimSiamKD(get_backbone_kd(model_cfg.backbone_s, model_cfg.backbone_t, model_cfg.num_classes))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.name == 'simsiam_kd_anchor':
        model =  SimSiamKDAnchor(get_backbone_kd(model_cfg.backbone_s, model_cfg.backbone_t, model_cfg.num_classes))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.name == 'simsiam_adv':
        model =  SimSiamAdv(get_backbone(model_cfg.backbone))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers) 
    elif model_cfg.name == 'simsiam_joint':
        model =  SimSiamJoint(get_backbone(model_cfg.backbone))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers) 
    elif model_cfg.name == 'simsiam_mi':
        model =  SimSiamMI(get_backbone(model_cfg.backbone), model_cfg.proj_dim)
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers) 
    elif model_cfg.name == 'simsiam_mi':
        model =  SimSiamNoise(get_backbone(model_cfg.backbone))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers) 
    elif model_cfg.name == 'byol':
        model = BYOL(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'simclr':
        model = SimCLR(get_backbone(model_cfg.backbone), model_cfg.proj_dim)
    elif model_cfg.name == 'simclr_mi':
        model = SimCLRMI(get_backbone(model_cfg.backbone), model_cfg.proj_dim)
    elif model_cfg.name == 'simclr_joint':
        model = SimCLRJoint(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'simclr_gram':
        model = SimCLRGram(get_backbone(model_cfg.backbone), model_cfg.proj_dim)
    elif model_cfg.name == 'simclr_kl':
        model = SimCLRKL(get_backbone(model_cfg.backbone), model_cfg.proj_dim)
    elif model_cfg.name == 'swav':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return model






