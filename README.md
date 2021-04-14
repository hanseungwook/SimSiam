# News
It's been two months and I think I've finally discovered the **True** reasons why Simsiam/BYOL avoids collapsed solutions using stop gradient and predictor!!! Follow me on [twitter](https://twitter.com/tianyu_hua) and stay tuned!


# SimSiam
A PyTorch implementation for the paper [**Exploring Simple Siamese Representation Learning**](https://arxiv.org/abs/2011.10566) by Xinlei Chen & Kaiming He



### Dependencies

If you don't have python 3 environment:
```
conda create -n simsiam python=3.8
conda activate simsiam
```
Then install the required packages:
```
pip install -r requirements.txt
```

### Run SimSiam

```
CUDA_VISIBLE_DEVICES=0 python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_cifar.yaml --ckpt_dir ~/.cache/ --hide_progress
```
The data folder `../Data/` should look like this:
```
➜  ~ tree ../Data/
├── cifar-10-batches-py
│   ├── batches.meta
│   ├── data_batch_1
│   ├── ...
└── stl10_binary
    ├── ...
```
```
Training: 100%|#################################################################| 800/800 [11:46:06<00:00, 52.96s/it, epoch=799, accuracy=90.3]
Model saved to /root/.cache/simsiam-cifar10-experiment-resnet18_cifar_variant1.pth
Evaluating: 100%|##########################################################################################################| 100/100 [08:29<00:00,  5.10s/it]
Accuracy = 90.83
Log file has been saved to ../logs/completed-simsiam-cifar10-experiment-resnet18_cifar_variant1(2)
```
To evaluate separately:
```
CUDA_VISIBLE_DEVICES=4 python linear_eval.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_cifar_eval.yaml --ckpt_dir ~/.cache/ --hide_progress --eval_from ~/simsiam-cifar10-experiment-resnet18_cifar_variant1.pth

creating file ../logs/in-progress_0111061045_simsiam-cifar10-experiment-resnet18_cifar_variant1
Evaluating: 100%|##########################################################################################################| 200/200 [16:52<00:00,  5.06s/it]
Accuracy = 90.87
```
![simsiam-cifar10-800e](simsiam-800e90.83acc.svg)

>`export DATA="/path/to/your/datasets/"` and `export LOG="/path/to/your/log/"` will save you the trouble of entering the folder name every single time!

### Run SimSiam_KD / SimSiam_KD_Anchor

SimSiam_KD is a version of SimSiam, in which we conduct knowledge distillation from a teacher to a student network via the contrastive learning objective.

SimSiam_Kd_Anchor is a version of SimSiam_KD, in which we use an anchor point (x) for the teacher network (instead of an augmented view of x).

SimSiam_KD can be run with Slurm (multi-GPU setting) with the following scripts:

For CIFAR-10, ResNet18 -> ResNet18
```
sbatch slurm/simsiam_kd_cifar10.slurms
```

For CIFAR-100, ResNet56 -> ResNet20
```
sbatch slurm/simsiam_kd_anchor_cifar100_r56_r20.slurm
```

### Run SimCLR

```
CUDA_VISIBLE_DEVICES=1 python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simclr_cifar.yaml --ckpt_dir ~/.cache/ --hide_progress
```

### Run BYOL
```
CUDA_VISIBLE_DEVICES=2 python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/byol_cifar.yaml --ckpt_dir ~/.cache/ --hide_progress
```

### TODO

- convert from data-parallel (DP) to distributed data-parallel (DDP)
- create PyPI package `pip install simsiam-pytorch`


If you find this repo helpful, please consider star so that I have the motivation to improve it.

### Branches

`adv`: Two-player formulation: discriminator estimating the ratio of joint / marginal and encoder maximizing it \
`adv-mmd`: Two-player formulation in which discriminator is replaced with a MMD estimator \
`joint`: Discrimator and encoder are treated as an end-to-end model estimating the ratio of joint / marginal (with symmetric loss added at z-level) \
`joint-simclr`: SimCLR's NT-Xent loss + MI loss via discriminator with Adam optimization dynamics merged in\
`joint-simclr-iter`: Step 1 (MI Estimation with respect to f + D) and Step 2 (SimCLR's NT-Xent loss + MI maximization with respect to f) \
`calc-mi`: Original models of SimSiam, SimCLR, with a discriminator added to training to estimate MI (no gradients from MI estimation)
`gram`: Experiments with Gram ratio loss and changed kernels
`noise`: SimSiam with SG replaced with noise added to the leg w/o predictor
`archive/joint-simclr-adam`: SimCLR's NT-Xent loss + MI loss via discriminator with Adam optimization dynamics (before being merged into `joint-simclr`)\
`e2e-baseline`: End-to-end model only estimating the ratio (two versions, in which the two distributions are created at x and z-levels) \


