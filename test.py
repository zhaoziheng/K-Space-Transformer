import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse

from dataset import TestDataSet
from Ktransformer import Transformer
from utils import *
from fftc import *

def str2bool(v):
  return v.lower() in ('true')

def test(config, output_dir, model, model_Path, conv_weight, device, testSet, testLoader):
    test_Conv_psnr = 0.0
    test_unConv_psnr = 0.0
    test_Conv_ssim = 0.0
    test_unConv_ssim = 0.0

    # Load the best parameters
    checkpoint = torch.load(model_Path)
    model.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
      model.eval()
      # validate valSet
      for batch in tqdm(testLoader):
        sampled_k = batch['sampled_k'].to(device)  # [bs, input_len, 2]
        sampled_pos_norm = batch['sampled_pos_norm'].to(device)  # [bs, input_len, 2]

        unsampled_pos = batch['unsampled_pos']  # [bs, query_len, 2]
        unsampled_pos_norm = batch['unsampled_pos_norm'].to(device)  # [bs, query_len, 2]

        k_us = batch['k_us'].to(device)
        i_gt = batch['i_gt'].to(device)
        mask = batch['selected_mask'].to(device)

        LR_pos_norm = batch['LR_pos_norm'].to(device)  # [bs, lh*lw, 2]

        LR_unConv_results, Up_LR_i, Up_LR_k, \
        HR_unConv_results, HR_Conv_results = model(src=sampled_k,
                                                   lr_pos=LR_pos_norm,
                                                   src_pos=sampled_pos_norm,
                                                   hr_pos=unsampled_pos_norm,
                                                   k_us=k_us,
                                                   unsampled_pos=unsampled_pos,
                                                   up_scale=2,
                                                   mask=mask,
                                                   conv_weight=conv_weight)

        HR_Conv_predicted_i = HR_Conv_results[-1].cpu().numpy()
        HR_unConv_predicted_i = HR_unConv_results[-1].cpu().numpy()
        i_gt = i_gt.cpu().numpy()

        tmp_Conv_psnr = cal_psnr(HR_Conv_predicted_i, i_gt)
        test_Conv_psnr += tmp_Conv_psnr

        tmp_unConv_psnr = cal_psnr(HR_unConv_predicted_i, i_gt)
        test_unConv_psnr += tmp_unConv_psnr

        tmp_Conv_ssim = cal_ssim(HR_Conv_predicted_i, i_gt)
        test_Conv_ssim += tmp_Conv_ssim

        tmp_unConv_ssim = cal_ssim(HR_unConv_predicted_i, i_gt)
        test_unConv_ssim += tmp_unConv_ssim

      test_Conv_psnr /= len(testSet)
      test_unConv_psnr /= len(testSet)
      test_Conv_ssim /= len(testSet)
      test_unConv_ssim /= len(testSet)

      out_str = "test-unConv-PSNR:%.4f  test-Conv-PSNR:%.4f  test-unConv-SSIM:%.4f  test-Conv-SSIM:%.4f\n" % \
                (test_unConv_psnr, test_Conv_psnr, test_unConv_ssim, test_Conv_ssim)

      print(out_str)
      if output_dir :
        with open(os.path.join('Log(Test)', config.output_dir, 'record.txt'), 'a') as f:
          f.write('Load Model From %s, Evaluate on %s, Mask %s\n' % (config.modelPath, config.dataset, config.mask))
          f.write(out_str)

if __name__ == '__main__':
  # -------------------------------------------------------- 读取超参数

  parser = argparse.ArgumentParser()

  # ----------------- Testing ID

  parser.add_argument('--output_dir', type=str, default=None, help="path to record the evaluation results")

  parser.add_argument('--gpu', type=str, default='0,1,2,3')

  # ----------------- Model Structure

  parser.add_argument('--modelPath', type=str, help="checkpoint to evaluate")

  parser.add_argument('--d_model', type=int, default=256)
  parser.add_argument('--n_head', type=int, default=4)
  parser.add_argument('--num_encoder_layers', type=int, default=4)
  parser.add_argument('--num_LRdecoder_layers', type=int, default=4)
  parser.add_argument('--num_HRdecoder_layers', type=int, default=6)
  parser.add_argument('--dim_feedforward', type=int, default=1024)

  parser.add_argument('--dropout', type=float, default=0.0)

  parser.add_argument('--HR_conv_channel', type=int, default=64)
  parser.add_argument('--HR_conv_num', type=int, default=3)
  parser.add_argument('--HR_kernel_size', type=int, default=3)
  parser.add_argument('--conv_weight', type=float, default=1.0)

  # ----------------- Dataset Control

  parser.add_argument('--batch_size', type=int, default=24)
  parser.add_argument('--hr_data_path', type=str, help='Path to the k-space data', default='xxx/xxx.npy')
  parser.add_argument('--lr_data_path', type=str, help='Path to the downsampled k-space data', default='xxx/xxx.npy')
  parser.add_argument('--mask_path', type=str, help='Path to the undersampling masks', default='xxx/xxx.npy')

  config = parser.parse_args()

  # Prepare Dataset
  testPath = config.hr_data_path
  testlrPath = config.lr_data_path
  test_maskPath = config.mask_path

  testSet = TestDataSet(testPath, testlrPath, test_maskPath)
  testLoader = DataLoader(testSet, batch_size=config.batch_size, shuffle=False, num_workers=8)

  # Prepare Model
  model = Transformer(lr_size=64,
                      d_model=config.d_model,
                      # Multi Head
                      nhead=config.n_head,
                      # Layer Number
                      num_LRdecoder_layers=config.num_LRdecoder_layers,
                      num_HRdecoder_layers=config.num_HRdecoder_layers,
                      num_encoder_layers=config.num_encoder_layers,
                      # MLP in Transformer Block
                      dim_feedforward=config.dim_feedforward,
                      # HR Conv
                      HR_conv_channel=config.HR_conv_channel,
                      HR_conv_num=config.HR_conv_num,
                      HR_kernel_size=config.HR_kernel_size,
                      dropout=config.dropout,
                      activation="relu")

  os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
  device = torch.device('cuda')
  model = nn.DataParallel(model)
  model.to(device)

  # ___________________________________________________________________________________________
  test(config, config.output_dir, model, config.modelPath, config.conv_weight, device, testSet, testLoader)
