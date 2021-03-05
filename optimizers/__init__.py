from .lars import LARS
from .lars_simclr import LARS_simclr
from .larc import LARC
import torch
from .lr_scheduler import LR_Scheduler


def get_optimizer(name, model, lr, momentum, weight_decay):

    discriminator_prefix = ('module.discriminator', 'discriminator')
    parameters = [{
        'name': 'base',
        'params': [param for name, param in model.named_parameters() if not name.startswith(discriminator_prefix)],
        'lr': lr
    }]

    d_parameters = [{
        'name': 'discriminator',
        'params': [param for name, param in model.named_parameters() if name.startswith(discriminator_prefix)],
        'lr': lr
    }
    ,{
        'name': 'base',
        'params': [param for name, param in model.named_parameters() if not name.startswith(discriminator_prefix)],
        'lr': lr
    }
    ]

    optimizer, optimizer_e, optimizer_d = None, None, None
    if name == 'lars':
        optimizer = LARS(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    # Optimizer in adversarial setting
    elif name == 'sgd':
        optimizer_e = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
        optimizer_d = torch.optim.SGD(d_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    # elif name == 'sgd':
        # optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer_e = torch.optim.Adam(parameters, lr=lr)
        optimizer_d = torch.optim.Adam(d_parameters, lr=lr)
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

    return optimizer_e, optimizer_d



