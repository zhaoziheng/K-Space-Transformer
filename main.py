import os
import argparse
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import random
import torch.optim as optim
import math
import torch.backends.cudnn as cudnn

from dataset import TrainDataSet, ValidDataSet, TestDataSet
from Ktransformer import Transformer
from train import train

def str2bool(v):
    return v.lower() in ('true')

def set_seed(config):
  seed = config.seed
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  cudnn.benchmark = True
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  cudnn.benchmark = False
  cudnn.deterministic = True

# -------------------------------------------------------- 读取超参数
parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str,
                    help='File path to log each training, including best model, visualization, tensorboard log file ...') # ！
parser.add_argument('--checkpoint', default=None, help='Path to checkpoint')
parser.add_argument('--resume_train', type=str2bool, default='False',
                    help='Resume learning rate, epoch num of an interrupted training')
parser.add_argument('--gpu', type=str, default='0,1,2,3')

# ----------------- Learning Rate, Loss and Regularizations

parser.add_argument('--epoch_num', type=int, default=200)

parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--kspace_loss', type=str2bool, default='True', help='MSE loss in k-space')
parser.add_argument('--img_loss', type=str2bool, default='True', help='MSE loss in image domain')

parser.add_argument('--lr', type=float, default=5e-4, help='Maximum learning rate')

parser.add_argument('--lr_weights', nargs='+', type=float,
                    help='Loss weights for each LR decoder layers repectively(must align with num_LRcoder_layers)',
                    default=[0.3, 0.3, 0.3, 0.3])
parser.add_argument('--hr_weights', nargs='+', type=float,
                    help='Loss weights for each HR decoder layers repectively(must align with num_HRdecoder_layers)',
                    default=[0.3, 0.3, 0.3, 0.3, 0.3, 1.0])
parser.add_argument('--conv_weight', type=float, default=1.0,
                    help='Loss weight for image refinement module')
parser.add_argument('--pure_LR_training_epoch', nargs='+', type=int, default=50,
                    help='Training epoch for LR decoder only')
parser.add_argument('--pure_K_training_epoch', nargs='+', type=int, default=100,
                    help='Training epoch for k-space LR+HR decoder before image RM, must be large than pure_LR_training_epoch')

parser.add_argument('--l2norm', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--reassign_mask', type=int, default=1, help='Reassign undersampling masks to each sample every N epoch')

# ----------------- Model Structure

parser.add_argument('--d_model', type=int, default=256)
parser.add_argument('--n_head', type=int, default=4)
parser.add_argument('--num_encoder_layers', type=int, default=4)
parser.add_argument('--num_LRdecoder_layers', type=int, default=4)
parser.add_argument('--num_HRdecoder_layers', type=int, default=6)
parser.add_argument('--dim_feedforward', type=int, default=1024)

parser.add_argument('--hr_conv_channel', type=int, default=64)
parser.add_argument('--hr_conv_num', type=int, default=3)
parser.add_argument('--hr_kernel_size', type=int, default=3)

# ----------------- Dataset

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--valid_batch_size', type=int, default=16)

parser.add_argument('--lr_size', type=int, default=64, help='Resolution of LR decoder reconstruction')

parser.add_argument('--train_hr_data_path', type=str, help='Path to the k-space data', default='xxx/xxx.npy')
parser.add_argument('--train_lr_data_path', type=str, help='Path to the downsampled k-space data', default='xxx/xxx.npy')
parser.add_argument('--train_mask_path', type=str, help='Path to the undersampling masks', default='xxx/xxx.npy')

parser.add_argument('--valid_hr_data_path', type=str)
parser.add_argument('--valid_lr_data_path', type=str)
parser.add_argument('--valid_mask_path', type=str)

config = parser.parse_args()

set_seed(config)

train_path = config.train_hr_data_path
train_lr_path = config.train_lr_data_path
valid_path = config.valid_hr_data_path
valid_lr_path = config.valid_lr_data_path
train_mask_path = config.train_mask_path
valid_mask_path = config.valid_mask_path

def data_loader():
    print('Start Loading Dataset from %s, \nMask from %s' % (config.train_hr_data_path, config.train_mask_path))
    t1 = time.time()
    trainSet = TrainDataSet(train_path, train_lr_path, train_mask_path)
    trainLoader = DataLoader(trainSet, batch_size=config.batch_size, shuffle=True, num_workers=8)
    print('Train data num : %d' % len(trainSet))
    # val和valid用的是同一个mask
    validSet = ValidDataSet(valid_path, valid_lr_path, valid_mask_path)
    validLoader = DataLoader(validSet, batch_size=config.valid_batch_size, shuffle=True, num_workers=8)
    print('valid data num : %d ' % len(validSet))
    print('Sampled Ratio: %.4f ' % (trainSet.sampled_num/trainSet.unsampled_num))
    print('Dataset load time : %d \n' % (time.time() - t1))
    return trainSet, trainLoader, validSet, validLoader

trainSet, trainLoader, validSet, validLoader = data_loader()

model = Transformer(lr_size=config.lr_size,
                    d_model=config.d_model,
                    # Multi Head
                    nhead=config.n_head,
                    # Layer Number
                    num_LRdecoder_layers=config.num_LRdecoder_layers,
                    num_HRdecoder_layers=config.num_HRdecoder_layers,
                    num_encoder_layers=config.num_encoder_layers,
                    # MLP in Transformer Block
                    dim_feedforward=config.dim_feedforward,
                    # Refine Module CNN
                    HR_conv_channel=config.hr_conv_channel,
                    HR_conv_num=config.hr_conv_num,
                    HR_kernel_size=config.hr_kernel_size,
                    dropout=config.dropout,
                    activation="relu")

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
device = torch.device('cuda')
model = nn.DataParallel(model)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.l2norm)

def lambda_rule(epoch):
    return 0.5 * (1 + math.cos(math.pi * (epoch) / (config.epoch_num)))

lr_sch = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)  # cosine annealing

start_epoch = 1
best_val_psnr = best_val_ssim = 0.0
stage = 'LR'

if config.resume_train:
  checkpoint = torch.load(config.checkpoint)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  lr_sch.load_state_dict(checkpoint['lr_sch_state_dict'])
  best_val_psnr = checkpoint['best_val_psnr']
  best_val_ssim = checkpoint['best_val_ssim']
  stage = checkpoint['stage']
  start_epoch = int(checkpoint['epoch']) + 1
  print('Resume Training from %s, Epoch %d, Stage %s, Best_PSNR:%.2f' % (config.checkpoint, start_epoch, stage, best_val_psnr))
elif config.checkpoint:
  checkpoint = torch.load(config.checkpoint)
  model.load_state_dict(checkpoint['model_state_dict'])
  print('Load checkpoint from %s, Best_PSNR:.3f' % (config.checkpoint, checkpoint['best_val_psnr']))

print('Output file locate at : %s' % os.path.join('Log', config.output_dir))

train(model=model,
      optimizer=optimizer,
      lr_sch=lr_sch,
      start_epoch=start_epoch,
      best_valid_psnr=best_val_psnr,
      best_valid_ssim=best_val_ssim,
      device=device,
      config=config,
      trainSet=trainSet, trainLoader=trainLoader,
      validSet=validSet, validLoader=validLoader)


