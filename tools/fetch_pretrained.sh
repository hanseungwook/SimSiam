# fetch pre-trained teacher models

# Assume running from repo root
cd models/backbones/state_dicts/

wget http://shape2prog.csail.mit.edu/repo/wrn_40_2_vanilla/ckpt_epoch_240.pth
mv ckpt_epoch_240.pth cifar100/wrn_40_2.pth

wget http://shape2prog.csail.mit.edu/repo/resnet56_vanilla/ckpt_epoch_240.pth
mv ckpt_epoch_240.pth cifar100/resnet56.pth

wget http://shape2prog.csail.mit.edu/repo/resnet110_vanilla/ckpt_epoch_240.pth
mv ckpt_epoch_240.pth cifar100/resnet110.pth

wget http://shape2prog.csail.mit.edu/repo/resnet32x4_vanilla/ckpt_epoch_240.pth
mv ckpt_epoch_240.pth cifar100/resnet32x4.pth

wget http://shape2prog.csail.mit.edu/repo/vgg13_vanilla/ckpt_epoch_240.pth
mv ckpt_epoch_240.pth cifar100/vgg13.pth

wget http://shape2prog.csail.mit.edu/repo/ResNet50_vanilla/ckpt_epoch_240.pth
mv ckpt_epoch_240.pth cifar100/resnet50.pth

cd ../../../