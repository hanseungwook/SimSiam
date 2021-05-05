from .lars import LARS
from .lars_simclr import LARS_simclr
from .larc import LARC
import torch
from .lr_scheduler import LR_Scheduler


def get_optimizer(name, model, lr, momentum, weight_decay):

    predictor_prefix = ('module.backbone1', 'module.projector1')
    parameters_f = [{
        'name': 'encoder1',
        'params': [param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
        'lr': lr
    }]

    parameters_g = [{
        'name': 'encoder2',
        'params': [param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
        'lr': lr
    }]

    if name == 'lars':
        optimizer = LARS(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'sgd':
        optimizer_f = torch.optim.SGD(parameters_f, lr=lr, momentum=momentum, weight_decay=weight_decay)
        optimizer_g = torch.optim.SGD(parameters_g, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'lars_simclr': # Careful
        optimizer = LARS_simclr(model.named_modules(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'larc':
        optimizer = LARC(
            torch.optim.SGD(
                parameters,
                lr=lr, 
                momentum=momentum, 
                weight_decay=weight_decay
            ),
            trust_coefficient=0.001, 
            clip=False
        )
    else:
        raise NotImplementedError
    return optimizer_f, optimizer_g



