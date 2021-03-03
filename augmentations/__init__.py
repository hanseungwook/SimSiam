from .simsiam_aug import SimSiamTransform
from .eval_aug import Transform_single
from .byol_aug import BYOL_transform
from .simclr_aug import SimCLRTransform

dataset_norm_dict = {
    'mnist': [[0.1307,], [0.3081,]],
    'stl10': [[0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616]],
    'cifar10': [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]],
    'cifar100': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
    'imagenet': [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]
}

dataset_img_size_dict = {
    'mnist': 28,
    'stl10': 96,
    'cifar10': 32,
    'cifar100': 32,
    'imagenet': 224
}


def get_aug(dataset, name='simsiam', image_size=224, train=True, train_classifier=None):
    image_size, norm = dataset_img_size_dict[dataset], dataset_norm_dict[dataset]

    if train==True:
        if name == 'simsiam' or name == 'simsiam_kd' or name == 'simsiam_adv' or name == 'simsiam_adv_mmd':
            augmentation = SimSiamTransform(image_size, mean_std=norm)
        elif name == 'simsiam_kd_anchor':
            augmentation = SimSiamTransform(image_size, mean_std=norm, anchor=True)
        elif name == 'byol':
            augmentation = BYOL_transform(image_size, mean_std=norm)
        elif name == 'simclr':
            augmentation = SimCLRTransform(image_size, mean_std=norm)
        else:
            raise NotImplementedError
    elif train==False:
        if train_classifier is None:
            raise Exception
        augmentation = Transform_single(image_size, train=train_classifier, normalize=norm)
    else:
        raise Exception
    
    return augmentation








