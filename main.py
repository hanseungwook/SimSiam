import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from linear_eval import main as linear_eval
from datetime import datetime

def main(device, args):

    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(dataset=args.dataset_kwargs['dataset'], train=True, **args.aug_kwargs), 
            train=True,
            **args.dataset_kwargs),
        shuffle=True,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )
    memory_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_aug(dataset=args.dataset_kwargs['dataset'], train=False, train_classifier=False, **args.aug_kwargs), 
            train=True,
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset( 
            transform=get_aug(dataset=args.dataset_kwargs['dataset'], train=False, train_classifier=False, **args.aug_kwargs), 
            train=False,
            **args.dataset_kwargs),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )

    # Define model
    model = get_model(args.model)

    # Load model
    if args.load_model:
        print('Loading model', file=sys.stderr)
        checkpoint = torch.load(args.load_model.weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])

    # Move to device and DP
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    # define optimizer
    optimizer, optimizer_e, optimizer_d = get_optimizer(
        args.train.optimizer.name, model, 
        lr=args.train.base_lr, 
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay)

    # lr_scheduler = LR_Scheduler(
    #     optimizer,
    #     args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256, 
    #     args.train.num_epochs, args.train.base_lr*args.train.batch_size/256, args.train.final_lr*args.train.batch_size/256, 
    #     len(train_loader),
    #     constant_predictor_lr=True # see the end of section 4.2 predictor
    # )

    logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    best_accuracy = 0.0
    accuracy = 0
    pretrain = False

    # Start training
    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
    for epoch in global_progress:
        model.train()      

        if epoch < args.train.pretrain_epochs:
            pretrain = True
        else:
            pretrain = False
        
        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
        for idx, (images, labels) in enumerate(local_progress):
            images1 = images[0].to(device, non_blocking=True)
            images2 = images[1].to(device, non_blocking=True)

            # Pretrain both f + d with NT-Xent loss + MI estimation loss
            if pretrain:
                optimizer.zero_grad()
                data_dict = model.forward(images1, images2, sym_loss_weight=args.train.symmetric_loss_weight, logistic_loss_weight=args.train.logistic_loss_weight, est=True)
                loss = data_dict['loss_est'].mean()
                loss.backward()
                optimizer.step()

                local_progress.set_postfix({k:(v.mean() if isinstance(v, torch.Tensor) else v) for k, v in data_dict.items()})
                logger.update_scalers(data_dict)
            
            else:
                # Discriminator step: MI Estimation
                optimizer_d.zero_grad()
                data_dict_d = model.forward(images1, images2, sym_loss_weight=0.0, logistic_loss_weight=args.train.logistic_loss_weight, est=True)
                loss = data_dict_d['loss_est/total'].mean()
                loss.backward()
                optimizer_d.step()

                # Encoder step: MI Maximization
                if idx % args.train.max_step_int == 0:
                    optimizer_e.zero_grad()
                    data_dict_e = model.forward(images1, images2, sym_loss_weight=args.train.symmetric_loss_weight, logistic_loss_weight=args.train.logistic_loss_weight, est=False)
                    loss = data_dict_e['loss_m/total'].mean()
                    loss.backward()
                    optimizer_e.step()

                    data_dict_d.update(data_dict_e)
            
            # Scheduler step
            # lr_scheduler.step()
            # data_dict.update({'lr':lr_scheduler.get_lr()})
            
                local_progress.set_postfix({k:(v.mean() if isinstance(v, torch.Tensor) else v) for k, v in data_dict_d.items()})
                logger.update_scalers(data_dict_d)

        if args.train.knn_monitor and epoch % args.train.knn_interval == 0:
            backbone = model.module.backbone_s if (args.model.name == 'simsiam_kd' or args.model.name == 'simsiam_kd_anchor') else model.module.backbone
            accuracy = knn_monitor(backbone, memory_loader, test_loader, device, k=min(args.train.knn_k, len(memory_loader.dataset)), hide_progress=args.hide_progress) 

            # Save best model (evaluated by knn accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                model_path = os.path.join(args.ckpt_dir, f"{args.name}_best.pth")
                torch.save({
                    'epoch': epoch+1,
                    'state_dict':model.module.state_dict()
                }, model_path)

            # Save after pretrain
            if (epoch+1) == args.train.pretrain_epochs:
                model_path = os.path.join(args.ckpt_dir, f"{args.name}_{epoch+1}_pretrain.pth")
                torch.save({
                    'epoch': epoch+1,
                    'state_dict':model.module.state_dict()
                }, model_path)

            if (epoch+1) % 10 == 0:
                model_path = os.path.join(args.ckpt_dir, f"{args.name}_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch+1,
                    'state_dict':model.module.state_dict()
                }, model_path)
        
        epoch_dict = {"epoch":epoch, "accuracy":accuracy}
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)
    
    # Save checkpoint
    model_path = os.path.join(args.ckpt_dir, f"{args.name}_{datetime.now().strftime('%m%d%H%M%S')}.pth") # datetime.now().strftime('%Y%m%d_%H%M%S')
    torch.save({
        'epoch': epoch+1,
        'state_dict':model.module.state_dict()
    }, model_path)
    print(f"Model saved to {model_path}")
    with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
        f.write(f'{model_path}')

    if args.eval is not False:
        args.eval_from = model_path
        linear_eval(args)


if __name__ == "__main__":
    args = get_args()

    main(device=args.device, args=args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')

    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')














