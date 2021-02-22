from .simsiam import SimSiam
from .byol import BYOL
from .simclr import SimCLR
from torchvision.models import resnet50, resnet18
import torch
from .backbones import resnet18_cifar_variant1, resnet18_cifar_variant2

def get_backbone(backbone, castrate=True):
    backbone = eval(f"{backbone}()")

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone

def get_backbone_kd(backbone, castrate=True):
    backbone_s = eval(f"{backbone}()")
    backbone_t = eval(f"{backbone}(pretrained=True)")

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
        model =  SimSiamKD(get_backbone_kd(model_cfg.backbone))
        if model_cfg.proj_layers is not None:
            model.projector.set_layers(model_cfg.proj_layers)
    elif model_cfg.name == 'byol':
        model = BYOL(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'simclr':
        model = SimCLR(get_backbone(model_cfg.backbone))
    elif model_cfg.name == 'swav':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return model






