import os
from numpy import random
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from utils import *
from fftc import ifft2c_new, fft2c_new
from pathlib import Path
import shutil

def train(model, optimizer, lr_sch, start_epoch,
          best_valid_psnr, best_valid_ssim,
          device, config,
          trainSet, trainLoader,
          validSet, validLoader):

  f_path = os.path.join('Log', config.output_dir, 'log.txt')
  log_path = os.path.join('Log', config.output_dir, 'log')
  Path(log_path).mkdir(parents=True, exist_ok=True)
  writer = SummaryWriter(log_path)

  validModelPath = os.path.join('Log', config.output_dir, '(refine)best_valid.pth')
  validModelPath_K = os.path.join('Log', config.output_dir, '(pure_k)best_valid.pth')
  validModelPath_LR = os.path.join('Log', config.output_dir, '(LR)best_valid.pth')
  checkpointModelPath = os.path.join('Log', config.output_dir, 'checkpoint.pth')

  validfig_path = os.path.join('Log', config.output_dir, 'valid_figs')
  Path(validfig_path).mkdir(parents=True, exist_ok=True)
  trainfig_path = os.path.join('Log', config.output_dir, 'train_figs')
  Path(trainfig_path).mkdir(parents=True, exist_ok=True)

  configDict = config.__dict__
  with open(f_path, 'a') as f:
    f.write('Configs :\n')
    for eachArg, value in configDict.items():
      f.write(eachArg + ' : ' + str(value) + '\n')
    f.write('\n')
    f.close()

  shutil.copy('train.py', os.path.join('Log', config.output_dir, 'train.py'))
  shutil.copy('main.py', os.path.join('Log', config.output_dir, 'main.py'))
  shutil.copy('dataset.py', os.path.join('Log', config.output_dir, 'dataset.py'))
  shutil.copy('Ktransformer.py', os.path.join('Log', config.output_dir, 'Ktransformer.py'))
  shutil.copy('layers.py', os.path.join('Log', config.output_dir, 'layers.py'))
  shutil.copy('multi_head_attn.py', os.path.join('Log', config.output_dir, 'multi_head_attn.py'))

  total = sum([param.nelement() for param in model.parameters()])
  print("Start Training, Number of model parameter: %.2fM" % (total / 1e6))

  lossCalculator = nn.MSELoss()

  train_batch_num = len(trainLoader)
  valid_batch_num = len(validLoader)

  num_hr_loss = config.num_HRdecoder_layers # deep supervision loss
  num_lr_loss = config.num_LRdecoder_layers

  for epoch in range(start_epoch, config.epoch_num+1):
    t1 = time.time()
    if config.reassign_mask and epoch % config.reassign_mask == 0:
      trainSet.reassign_mask()  # generate train data

    if epoch > config.pure_K_training_epoch:  # progressive training: LR Decoder --> HR Decoder --> K-Space Decoding + Image Refinement
      stage = 'RM'
      conv_weight = config.conv_weight
      hr_weights = config.hr_weights
      lr_weights = config.lr_weights
      eval_interval = 1
    elif epoch > config.pure_LR_training_epoch:
      stage = 'K'
      conv_weight = 0.0
      hr_weights = config.hr_weights
      lr_weights = config.lr_weights
      eval_interval = 5
    else:
      stage = 'LR'
      conv_weight = 0.0
      hr_weights = [0.0 for x in range(num_hr_loss)]
      lr_weights = config.lr_weights
      lr_weights[-1] = 1.0
      eval_interval = 10

    hr_Conv_loss = [0.0 for i in range(num_hr_loss)]  # deep supervision loss on refinement output
    k_hr_Conv_loss = [0.0 for i in range(num_hr_loss)]
    hr_unConv_loss = [0.0 for i in range(num_hr_loss)]  # deep supervision loss on k-space decoding output
    k_hr_unConv_loss = [0.0 for i in range(num_hr_loss)]
    lr_loss = [0.0 for i in range(num_lr_loss)]  # deep supervision loss on LR decoder layers
    k_lr_loss = [0.0 for i in range(num_lr_loss)]
    up_hr_psnr = 0.0  # evaluate the upsampled LR decoder output
    up_hr_ssim = 0.0
    train_unConv_ssim = train_Conv_ssim = 0.0 # psnr and ssim on HR decoder output
    train_unConv_psnr = train_Conv_psnr = 0.0

    model.train()
    print("learning rate: %e    stage: %s    " % (optimizer.param_groups[0]['lr'], stage))
    for batch in tqdm(trainLoader):
      sampled_k = batch['sampled_k'].to(device)  # [bs, input_len, c]
      sampled_pos_norm = batch['sampled_pos_norm'].to(device)  # [bs, input_len, 2]

      unsampled_pos_norm = batch['unsampled_pos_norm'].to(device)  # [bs, query_len, 2]
      unsampled_pos = batch['unsampled_pos']                       # [bs, query_len, 2]

      k_us = batch['k_us'].to(device)  # [bs, h, w, 2]
      i_gt = batch['i_gt'].to(device)  # [bs, h, w, 2]
      k_gt = batch['k_gt'].to(device)
      mask = batch['selected_mask'].to(device)

      LR_i_gt = batch['LR_i_gt'].to(device)  # [bs, h, w, 2]
      LR_k_gt = batch['LR_k_gt'].to(device)
      LR_pos_norm = batch['LR_pos_norm'].to(device)

      LR_unConv_results, Up_LR_i, Up_LR_k, \
      HR_unConv_results, HR_Conv_results = model(
                                            src=sampled_k,
                                            lr_pos=LR_pos_norm,
                                            src_pos=sampled_pos_norm,
                                            hr_pos=unsampled_pos_norm,
                                            k_us=k_us,
                                            unsampled_pos=unsampled_pos,
                                            up_scale=2,
                                            mask=mask,
                                            conv_weight=conv_weight,
                                            stage=stage)
      batch_loss = torch.tensor(0.0).to(device)
      # loss calculation
      # for LRD
      for i in range(len(LR_unConv_results)):
        if config.kspace_loss:
          tmp = lossCalculator(fft2c_new(LR_unConv_results[i]), LR_k_gt)
          batch_loss += lr_weights[i]*tmp
          k_lr_loss[i] += tmp.item()
        if config.img_loss:
          tmp = lossCalculator(LR_unConv_results[i], LR_i_gt)
          batch_loss += lr_weights[i]*tmp
          lr_loss[i] += tmp.item()
      # for HRD
      if stage != 'LR':
        for i in range(len(HR_Conv_results)):
          if config.kspace_loss:
            tmp2 = lossCalculator(fft2c_new(HR_unConv_results[i]), k_gt)
            k_hr_unConv_loss[i] += tmp2.item()
            batch_loss += hr_weights[i] * tmp2
          if config.img_loss:
            tmp2 = lossCalculator(HR_unConv_results[i], i_gt)
            hr_unConv_loss[i] += tmp2.item()
            batch_loss += hr_weights[i] * tmp2
        # for RM
        if stage == 'RM':
          for i in range(len(HR_Conv_results)):
            if config.kspace_loss:
              tmp1 = lossCalculator(fft2c_new(HR_Conv_results[i]), k_gt)
              k_hr_Conv_loss[i] += tmp1.item()
              batch_loss += hr_weights[i] * tmp1
            if config.img_loss:
              tmp1 = lossCalculator(HR_Conv_results[i], i_gt)
              hr_Conv_loss[i] += tmp1.item()
              batch_loss += hr_weights[i] * tmp1
      # bp
      optimizer.zero_grad()
      batch_loss.backward()
      optimizer.step()
      # other metric (on cpu)
      i_gt = i_gt.detach().cpu().numpy()
      Up_LR_i = Up_LR_i.detach().cpu().numpy()
      HR_Conv_predicted_i = HR_Conv_results[-1].detach().cpu().numpy()
      HR_unConv_predicted_i = HR_unConv_results[-1].detach().cpu().numpy()
      # for LRD
      up_hr_psnr += cal_psnr(Up_LR_i, i_gt)
      up_hr_ssim += cal_ssim(Up_LR_i, i_gt)
      # for HRD
      if stage != 'LR':
        train_unConv_psnr += cal_psnr(HR_unConv_predicted_i, i_gt)
        train_unConv_ssim += cal_ssim(HR_unConv_predicted_i, i_gt)
        if stage == 'RM':
          train_Conv_psnr += cal_psnr(HR_Conv_predicted_i, i_gt)
          train_Conv_ssim += cal_ssim(HR_Conv_predicted_i, i_gt)

    lr_sch.step()
    lr_loss = [x/train_batch_num for x in lr_loss]   # average loss per pixel over all samples
    hr_Conv_loss = [x/train_batch_num for x in hr_Conv_loss]
    hr_unConv_loss = [x / train_batch_num for x in hr_unConv_loss]
    k_lr_loss = [x/train_batch_num for x in lr_loss]   # average loss per pixel over all samples
    k_hr_Conv_loss = [x/train_batch_num for x in hr_Conv_loss]
    k_hr_unConv_loss = [x / train_batch_num for x in hr_unConv_loss]
    up_hr_psnr /= len(trainSet)
    up_hr_ssim /= len(trainSet)
    train_unConv_psnr /= len(trainSet)
    train_Conv_psnr /= len(trainSet)
    train_unConv_ssim /= len(trainSet)
    train_Conv_ssim /= len(trainSet)

    # flag metric in different stages
    if stage == 'LR':
      train_psnr = up_hr_psnr
      train_ssim = up_hr_ssim
    elif stage == 'K':
      train_psnr = train_unConv_psnr
      train_ssim = train_unConv_ssim
    elif stage == 'RM':
      train_psnr = train_Conv_psnr
      train_ssim = train_Conv_ssim

    # visualization log every 10 epoches
    if epoch % 10 == 0:
      LR_predicted_i = LR_unConv_results[-1]
      LR_predicted_k = fft2c_new(LR_predicted_i)
      HR_Conv_predicted_k = fft2c_new(torch.from_numpy(HR_Conv_predicted_i))  # cpu tensor
      HR_unConv_predicted_k = fft2c_new(torch.from_numpy(HR_unConv_predicted_i))  # cpu tensor

      zf_i = ifft2c_new(k_us, needShift=True)
      zf_i = zf_i.cpu().numpy()
      k_us = k_us.cpu().numpy()
      LR_predicted_i = LR_predicted_i.detach().cpu().numpy()
      LR_i_gt = LR_i_gt.cpu().numpy()
      LR_k_gt = LR_k_gt.cpu().numpy()
      LR_predicted_k = LR_predicted_k.detach().cpu().numpy()
      Up_LR_k = Up_LR_k.detach().cpu().numpy()
      k_gt = k_gt.detach().cpu().numpy()
      HR_Conv_predicted_k = HR_Conv_predicted_k.numpy()
      HR_unConv_predicted_k = HR_unConv_predicted_k.numpy()
      if zf_i.shape[0] < 5:
        index = np.arange(0, zf_i.shape[0])
      else:
        index = np.random.choice(zf_i.shape[0], 5, replace=False)
      rcd_image(up_hr_psnr, up_hr_ssim,
                train_Conv_psnr, train_unConv_psnr,
                train_Conv_ssim, train_unConv_ssim,
                zf_i[index],
                LR_predicted_i[index], LR_i_gt[index],
                Up_LR_i[index],
                HR_unConv_predicted_i[index], HR_Conv_predicted_i[index], i_gt[index],
                k_us[index],
                LR_predicted_k[index], LR_k_gt[index],
                Up_LR_k[index],
                HR_unConv_predicted_k[index], HR_Conv_predicted_k[index], k_gt[index],
                trainfig_path + '/' + str(epoch) + '.png')

    # before next valid
    # torch.cuda.empty_cache()

    with torch.no_grad():
      model.eval()
      if epoch % eval_interval == 0 or epoch==start_epoch :
        valid_up_hr_ssim = valid_up_hr_psnr = 0.0
        valid_unConv_psnr = valid_Conv_psnr = 0.0
        valid_unConv_ssim = valid_Conv_ssim = 0.0
        valid_unConv_loss = valid_Conv_loss = valid_lr_loss = 0.0
        k_valid_unConv_loss = k_valid_Conv_loss = k_valid_lr_loss = 0.0
        for batch in tqdm(validLoader):
          sampled_k = batch['sampled_k'].to(device)  # [bs, input_len, 2]
          sampled_pos_norm = batch['sampled_pos_norm'].to(device)  # [bs, input_len, 2]

          unsampled_pos = batch['unsampled_pos']            # [bs, query_len, 2]
          unsampled_pos_norm = batch['unsampled_pos_norm'].to(device)  # [bs, query_len, 2]

          k_us = batch['k_us'].to(device)
          i_gt = batch['i_gt'].to(device)
          k_gt = batch['k_gt'].to(device)
          mask = batch['selected_mask'].to(device)

          LR_i_gt = batch['LR_i_gt'].to(device)  # [bs, h, w, 2]
          LR_k_gt = batch['LR_k_gt'].to(device)
          LR_pos_norm = batch['LR_pos_norm'].to(device) # [bs, lh*lw, 2]

          LR_unConv_results, Up_LR_i, Up_LR_k, \
          HR_unConv_results, HR_Conv_results = model(src=sampled_k,
                                                lr_pos=LR_pos_norm,
                                                src_pos=sampled_pos_norm,
                                                hr_pos=unsampled_pos_norm,
                                                k_us=k_us,
                                                unsampled_pos=unsampled_pos,
                                                up_scale=2,
                                                mask=mask,
                                                conv_weight=conv_weight,
                                                stage=stage)
          # loss calculation
          LR_predicted_i = LR_unConv_results[-1]
          if config.kspace_loss:
            k_valid_lr_loss += lossCalculator(fft2c_new(LR_predicted_i), LR_k_gt).item()
          if config.img_loss:
            valid_lr_loss += lossCalculator(LR_predicted_i, LR_i_gt).item()
          if stage != 'LR':
            if config.kspace_loss:
              k_valid_unConv_loss += lossCalculator(fft2c_new(HR_unConv_results[-1]), k_gt).item()
            if config.img_loss:
              valid_unConv_loss += lossCalculator(HR_unConv_results[-1], i_gt).item()
            if stage =='RM':
              if config.kspace_loss:
                k_valid_Conv_loss += lossCalculator(fft2c_new(HR_Conv_results[-1]), k_gt).item()
              if config.img_loss:
                valid_Conv_loss += lossCalculator(HR_Conv_results[-1], i_gt).item()
          # other metric (on cpu)
          HR_Conv_predicted_i = HR_Conv_results[-1].cpu().numpy()
          HR_unConv_predicted_i = HR_unConv_results[-1].cpu().numpy()
          i_gt = i_gt.cpu().numpy()
          Up_LR_i = Up_LR_i.cpu().numpy()
          valid_up_hr_psnr += cal_psnr(Up_LR_i, i_gt)
          valid_up_hr_ssim += cal_ssim(Up_LR_i, i_gt)
          if stage != 'LR':
            valid_unConv_ssim += cal_ssim(HR_unConv_predicted_i, i_gt)
            valid_unConv_psnr += cal_psnr(HR_unConv_predicted_i, i_gt)
            if stage == 'RM':
              valid_Conv_psnr += cal_psnr(HR_Conv_predicted_i, i_gt)
              valid_Conv_ssim += cal_ssim(HR_Conv_predicted_i, i_gt)

        valid_lr_loss /= valid_batch_num
        valid_Conv_loss /= valid_batch_num
        valid_unConv_loss /= valid_batch_num
        k_valid_lr_loss /= valid_batch_num
        k_valid_Conv_loss /= valid_batch_num
        k_valid_unConv_loss /= valid_batch_num
        valid_Conv_psnr /= len(validSet)
        valid_unConv_psnr /= len(validSet)
        valid_Conv_ssim /= len(validSet)
        valid_unConv_ssim /= len(validSet)
        valid_up_hr_psnr /= len(validSet)
        valid_up_hr_ssim /= len(validSet)

        # flag metric in different stages
        if stage == 'LR':
          valid_psnr = valid_up_hr_psnr
          valid_ssim = valid_up_hr_ssim
          path = validModelPath_LR
        elif stage == 'K':
          valid_psnr = valid_unConv_psnr
          valid_ssim = valid_unConv_ssim
          path = validModelPath_K
        elif stage == 'RM':
          valid_psnr = valid_Conv_psnr
          valid_ssim = valid_Conv_ssim
          path = validModelPath

        if best_valid_psnr < valid_psnr:
          best_valid_psnr = valid_psnr
          best_valid_ssim = valid_ssim
          torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'lr_sch_state_dict': lr_sch.state_dict(),
              'best_valid_psnr': best_valid_psnr,
              'best_valid_ssim': best_valid_ssim,
              'train_psnr':train_psnr,
              'train_ssim':train_ssim,
              'stage':stage,
           }, path)

          zf_i = ifft2c_new(k_us, needShift=True)
          zf_i = zf_i.cpu().numpy()
          k_us = k_us.cpu().numpy()

          LR_predicted_i = LR_predicted_i.cpu().numpy()
          LR_i_gt = LR_i_gt.cpu().numpy()

          LR_k_gt = LR_k_gt.cpu().numpy()
          LR_predicted_k = fft2c_new(torch.from_numpy(LR_predicted_i)).numpy()

          Up_LR_k = Up_LR_k.cpu().numpy()

          k_gt = k_gt.detach().cpu().numpy()
          HR_Conv_predicted_k = fft2c_new(torch.from_numpy(HR_Conv_predicted_i)).numpy()
          HR_unConv_predicted_k = fft2c_new(torch.from_numpy(HR_unConv_predicted_i)).numpy()

          if zf_i.shape[0] < 5:
            index = np.arange(0, zf_i.shape[0])
          else:
            index = np.random.choice(zf_i.shape[0], 5, replace=False)

          rcd_image(valid_up_hr_psnr, valid_up_hr_ssim,
                    valid_Conv_psnr, valid_unConv_psnr,
                    valid_Conv_ssim, valid_unConv_ssim,
                    zf_i[index],
                    LR_predicted_i[index], LR_i_gt[index],
                    Up_LR_i[index],
                    HR_unConv_predicted_i[index], HR_Conv_predicted_i[index], i_gt[index],
                    k_us[index],
                    LR_predicted_k[index], LR_k_gt[index],
                    Up_LR_k[index],
                    HR_unConv_predicted_k[index], HR_Conv_predicted_k[index], k_gt[index],
                    validfig_path + '/' + str(epoch) + '.png')
        elif epoch % 10 == 0:
          # 在最好的valid或每隔10个epoch保存模型
          torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_sch_state_dict': lr_sch.state_dict(),
            'best_valid_psnr': best_valid_psnr,
            'best_valid_ssim': best_valid_ssim,
            'train_psnr': train_psnr,
            'train_ssim': train_ssim,
            'stage': stage,
          }, checkpointModelPath)

    if config.kspace_loss:
      writer.add_scalar('validloss/k-valid-LR-Loss', k_valid_lr_loss, epoch)
      writer.add_scalar('validloss/k-valid-K-Loss', k_valid_unConv_loss, epoch)
      writer.add_scalar('validloss/k-valid-RM-Loss:', k_valid_Conv_loss, epoch)
      for i in range(len(hr_Conv_loss)):
        writer.add_scalar('deeplossHR/k-(K)Layer-' + str(i), k_hr_Conv_loss[i], epoch)
        writer.add_scalar('deeplossHR/k-(RM)Layer-' + str(i), k_hr_unConv_loss[i], epoch)
      for i in range(len(lr_loss)):
        writer.add_scalar('deeplossLR/k-Layer-' + str(i), k_lr_loss[i], epoch)

    if config.img_loss:
      writer.add_scalar('validloss/valid-LR-Loss', valid_lr_loss, epoch)
      writer.add_scalar('validloss/valid-K-Loss', valid_unConv_loss, epoch)
      writer.add_scalar('validloss/valid-RM-Loss:', valid_Conv_loss, epoch)
      for i in range(len(hr_Conv_loss)):
        writer.add_scalar('deeplossHR/(K)Layer-' + str(i), hr_Conv_loss[i], epoch)
        writer.add_scalar('deeplossHR/(RM)Layer-' + str(i), hr_unConv_loss[i], epoch)
      for i in range(len(lr_loss)):
        writer.add_scalar('deeplossLR/Layer-' + str(i), lr_loss[i], epoch)


    writer.add_scalar('trainloss/train-UP-PSNR', up_hr_psnr, epoch)
    writer.add_scalar('trainloss/train-UP-SSIM', up_hr_ssim, epoch)
    writer.add_scalar('trainloss/train-RM-PSNR', train_Conv_psnr, epoch)
    writer.add_scalar('trainloss/train-K-PSNR', train_unConv_psnr, epoch)
    writer.add_scalar('trainloss/train-RM-SSIM', train_Conv_ssim, epoch)
    writer.add_scalar('trainloss/train-K-SSIM', train_unConv_ssim, epoch)

    writer.add_scalar('validloss/valid-UP-PSNR', valid_up_hr_psnr, epoch)
    writer.add_scalar('validloss/valid-UP-PSNR', valid_up_hr_ssim, epoch)
    writer.add_scalar('validloss/valid-K-PSNR', valid_unConv_psnr, epoch)
    writer.add_scalar('validloss/valid-RM-PSNR', valid_Conv_psnr, epoch)
    writer.add_scalar('validloss/valid-K-SSIM', valid_unConv_ssim, epoch)
    writer.add_scalar('validloss/valid-RM-SSIM', valid_Conv_ssim, epoch)
    writer.add_scalar('lr/lr', optimizer.param_groups[0]['lr'], epoch)

    t2 = time.time()

    with open(f_path, 'a') as f:
      out_str = ("Epoch:%d  Time:%.1f" % (epoch, t2 - t1))
      print(out_str)
      f.write(out_str + '\n')

      if config.img_loss:
        deep_loss = "HRD Layer Loss "
        for i in range(len(hr_Conv_loss)):
          deep_loss += "%d--%e" % (i, hr_unConv_loss[i])
          deep_loss += "(RM--%e)  " % (hr_Conv_loss[i])
        print(deep_loss)
        f.write(deep_loss + '\n')
      if config.kspace_loss:
        deep_loss = "HRD Layer K-Loss "
        for i in range(len(hr_Conv_loss)):
          deep_loss += "%d--%e" % (i, k_hr_unConv_loss[i])
          deep_loss += "(RM--%e)  " % (k_hr_Conv_loss[i])
        print(deep_loss)
        f.write(deep_loss + '\n')

      if config.img_loss:
        deep_loss = "LRD Layer Loss "
        for i in range(len(lr_loss)):
          deep_loss += "%d--%e  " % (i, lr_loss[i])
        print(deep_loss)
      if config.kspace_loss:
        deep_loss = "LRD Layer K-Loss "
        for i in range(len(lr_loss)):
          deep_loss += "%d--%e  " % (i, k_lr_loss[i])
        print(deep_loss)
        f.write(deep_loss + '\n')

      f.write(deep_loss + '\n')
      out_str = "trn-UP-P:%.4f    trn-K-P:%.4f    trn-RM-P:%.4f\n" \
                "valid-UP-P:%.4f    valid-K-P:%.4f    valid-RM-P:%.4f\n" \
                "Best-P:%.4f      Best-S:%.4f" % \
                (up_hr_psnr, train_unConv_psnr, train_Conv_psnr,
                 valid_up_hr_psnr, valid_unConv_psnr, valid_Conv_psnr,
                 best_valid_psnr, best_valid_ssim)
      print(out_str)
      f.write(out_str + '\n\n')

    # before next train
    # torch.cuda.empty_cache()
  with open(os.path.join('Log', config.output_dir, "log.txt"), "a") as f:
    f.write('Best PSNR %.4f and SSIM %.4f \n' % (best_valid_psnr, best_valid_ssim))
  shutil.move(os.path.join('Log', config.output_dir),
              os.path.join('Log', '(%.2f)%s' % (best_valid_psnr, config.output_dir)))
