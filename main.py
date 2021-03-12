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
from tools import AverageMeter, knn_monitor, Logger, LossScheduler, file_exist_check
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
    _, optimizer = get_optimizer(
        args.train.optimizer.name, model, 
        lr=args.train.base_lr, 
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay)

    # Define logistic loss scheduler
    loss_scheduler = LossScheduler(init_lw=args.train.logistic_loss_weight,
                                   min_lw=args.train.loss_scheduler.min_lw,
                                   decay_rate=args.train.loss_scheduler.decay_rate,
                                   decay_int=args.train.loss_scheduler.decay_int)    

    # lr_scheduler = LR_Scheduler(
    #     optimizer_e,
    #     args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256, 
    #     args.train.num_epochs, args.train.base_lr*args.train.batch_size/256, args.train.final_lr*args.train.batch_size/256, 
    #     len(train_loader),
    #     constant_predictor_lr=True # see the end of section 4.2 predictor
    # )

    logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    best_accuracy = 0.0
    accuracy = 0
    only_disc = False

    # Start training
    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
    for epoch in global_progress:
        model.train()      
        
        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
        for idx, (images, labels) in enumerate(local_progress):
            images1 = images[0].to(device, non_blocking=True)
            images2 = images[1].to(device, non_blocking=True)

            # Optimizer step
            optimizer.zero_grad()
            data_dict = model.forward(images1, images2, sym_loss_weight=args.train.symmetric_loss_weight, logistic_loss_weight=loss_scheduler.get_lw())
            loss = data_dict['loss'].mean() # ddp
            loss.backward()
            optimizer.step()

            data_dict.update({'sym_loss_weight': args.train.symmetric_loss_weight, 'logistic_loss_weight': loss_scheduler.get_lw()})
            loss_scheduler.step() # Step loss scheduler for logistic loss
            
            local_progress.set_postfix({k:v.mean() for k, v in data_dict.items()})
            logger.update_scalers(data_dict)

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

            if (epoch+1) % 10 == 0:
                model_path = os.path.join(args.ckpt_dir, f"{args.name}_latest.pth")
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














