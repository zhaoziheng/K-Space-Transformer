import torch
import numpy as np
import copy
import matplotlib.pylab as plt
from math import log10
from skimage.metrics import structural_similarity as SSIM

def array2matrix(h, w, c, sampled_postions, sampled_intensity, query_positions, query_result):
  """

  Args:
    sampled_position / sampled_intensity : [bs, sampled_len, 2/c]
    query_positions / query_result : [bs, query_Len, 2/c]

  Returns: 
    the filled matrix : [bs, h, w, c] / [bs, h/2, w, c]

  Note: sampled points will be left zero if not given
  """
  bs = query_positions.shape[0]

  kdata = torch.zeros((bs, h, w, c)).to(torch.device('cuda'))

  if sampled_postions != None:
    sampled_postions = copy.deepcopy(sampled_postions).permute(0, 2, 1)   # [bs, 2, sampled_len]
    sampled_postions = sampled_postions.tolist()

  query_positions = copy.deepcopy(query_positions).permute(0, 2, 1)     # [bs, 2, query_len]
  query_positions = query_positions.tolist()

  for i in range(bs):
    if sampled_postions != None:
      kdata[i, :, :, 0][sampled_postions[i]] = sampled_intensity[i, :, 0]   # real part
      kdata[i, :, :, 1][sampled_postions[i]] = sampled_intensity[i, :, 1]   # imag part
    kdata[i, :, :, 0][query_positions[i]] = query_result[i, :, 0]
    kdata[i, :, :, 1][query_positions[i]] = query_result[i, :, 1]
  return kdata

def fill_in_k(masked_k, query_pos, query_result):
  """

  Args:
    masked_k: [bs, h , w, c]
    query_pos: [bs, 2, query_len] array
    query_result: [bs, query_len, 2]

  Returns: 
    reconstructed k-space
  """
  k_tobe_filled = masked_k.clone()

  for i in range(masked_k.shape[0]):
    k_tobe_filled[i, :, :, 0][query_pos[i]] = query_result[i, :, 0]   # real part
    k_tobe_filled[i, :, :, 1][query_pos[i]] = query_result[i, :, 1]   # imag part

  return k_tobe_filled

# visualization function

def save_image(np_data, path, type='np'):
  if type == 'torch':
    data = np_data.numpy()
  else:
    data = np_data
  plt.figure(dpi=300, figsize=(1, 1))
  plt.imshow(np.abs(data[:, :, 0] + 1j * data[:, :, 1]), cmap="gray")
  plt.axis("off")
  plt.savefig(path)
  plt.close()

def save_k(np_data, path, type='np'):
  if type == 'torch':
    data = np_data.numpy()
  else:
    data = np_data
  plt.figure(dpi=300, figsize=(1, 1))
  plt.imshow(np.log(1 + np.abs(data[:, :, 0] + 1j * data[:, :, 1])), cmap="gray")
  plt.axis("off")
  plt.savefig(path)
  plt.close()
  
def rcd_image(up_psnr, up_ssim,
              Conv_psnr, unConv_psnr,
              Conv_ssim, unConv_ssim,
              zf_i, lr_pd_i, lr_gt_i,
              up_i, hr_unConv_pd_i, hr_Conv_pd_i, hr_gt_i,
              us_k, lr_pd_k, lr_gt_k,
              up_k, hr_unConv_pd_k, hr_Conv_pd_k, hr_gt_k, path):
  sample_num = zf_i.shape[0]
  index = 0
  plt.figure(dpi=300, figsize=(10, sample_num*3))
  plt.subplots_adjust(hspace=0.05, wspace=0.05)
  for i in range(0, sample_num):
    plt.subplot(sample_num*2, 10, index + 1)
    number = int(np.sum(np.abs(zf_i[i])))
    plt.imshow(np.abs(zf_i[i, :, :, 0] + 1j * zf_i[i, :, :, 1]), cmap="gray", interpolation=None)
    plt.title('UnderSampled:' + str(number), fontdict={'fontsize': 6})
    plt.axis("off")

    plt.subplot(sample_num*2, 10, index + 2)
    number = int(np.sum(np.abs(lr_pd_i[i])))
    plt.imshow(np.abs(lr_pd_i[i, :, :, 0] + 1j * lr_pd_i[i, :, :, 1]), cmap="gray", interpolation=None)
    plt.title('LR Predicted:'+str(number), fontdict={'fontsize': 6})
    plt.axis("off")

    plt.subplot(sample_num*2, 10, index + 3)
    number = int(np.sum(np.abs(lr_gt_i[i])))
    plt.imshow(np.abs(lr_gt_i[i, :, :, 0] + 1j * lr_gt_i[i, :, :, 1]), cmap="gray", interpolation=None)
    plt.title('LR GT:'+str(number), fontdict={'fontsize': 6})
    plt.axis("off")

    plt.subplot(sample_num * 2, 10, index + 4)
    error_map = (np.abs(lr_pd_i[i, :, :, 0] + 1j *lr_pd_i[i, :, :, 1]) - np.abs(lr_gt_i[i, :, :, 0] + 1j *lr_gt_i[i, :, :, 1]))   # [h, w]
    bound = min(abs(error_map.max().item()), abs(error_map.min().item()))
    error_map = np.clip(error_map, -bound, bound)
    plt.imshow(error_map, cmap='bwr', interpolation=None)
    plt.title('LR Error Map', fontdict={'fontsize': 6})
    plt.axis("off")

    plt.subplot(sample_num * 2, 10, index + 5)
    number = int(np.sum(np.abs(up_i[i])))
    plt.imshow(np.abs(up_i[i, :, :, 0] + 1j * up_i[i, :, :, 1]), cmap="gray", interpolation=None)
    plt.title('UpSampled Result:' + str(number), fontdict={'fontsize': 6})
    plt.axis("off")

    plt.subplot(sample_num * 2, 10, index + 6)
    number = int(np.sum(np.abs(hr_unConv_pd_i[i])))
    plt.imshow(np.abs(hr_unConv_pd_i[i, :, :, 0] + 1j * hr_unConv_pd_i[i, :, :, 1]), cmap="gray", interpolation=None)
    plt.title('HR unConv Predicted:' + str(number), fontdict={'fontsize': 6})
    plt.axis("off")

    plt.subplot(sample_num * 2, 10, index + 7)
    number = int(np.sum(np.abs(hr_Conv_pd_i[i])))
    plt.imshow(np.abs(hr_Conv_pd_i[i, :, :, 0] + 1j * hr_Conv_pd_i[i, :, :, 1]), cmap="gray", interpolation=None)
    plt.title('HR Conv Predicted:' + str(number), fontdict={'fontsize': 6})
    plt.axis("off")

    plt.subplot(sample_num * 2, 10, index + 8)
    number = int(np.sum(np.abs(hr_gt_i[i])))
    plt.imshow(np.abs(hr_gt_i[i, :, :, 0] + 1j * hr_gt_i[i, :, :, 1]), cmap="gray", interpolation=None)
    plt.title('HR GT:' + str(number), fontdict={'fontsize': 6})
    plt.axis("off")

    plt.subplot(sample_num * 2, 10, index + 9)
    error_map = (np.abs(hr_unConv_pd_i[i, :, :, 0] + 1j * hr_unConv_pd_i[i, :, :, 1]) - np.abs(hr_gt_i[i, :, :, 0] + 1j * hr_gt_i[i, :, :, 1]))  # [h, w]
    bound = min(abs(error_map.max().item()), abs(error_map.min().item()))
    error_map = np.clip(error_map, -bound, bound)
    plt.imshow(error_map, cmap='bwr', interpolation=None)
    plt.title('HR unConv Error Map', fontdict={'fontsize': 6})
    plt.axis("off")

    plt.subplot(sample_num * 2, 10, index + 10)
    error_map = (np.abs(hr_Conv_pd_i[i, :, :, 0] + 1j * hr_Conv_pd_i[i, :, :, 1]) - np.abs(
      hr_gt_i[i, :, :, 0] + 1j * hr_gt_i[i, :, :, 1])) # [h, w]
    bound = min(abs(error_map.max().item()), abs(error_map.min().item()))
    error_map = np.clip(error_map, -bound, bound)
    plt.imshow(error_map, cmap='bwr', interpolation=None)
    plt.title('HR Conv Error Map', fontdict={'fontsize': 6})
    plt.axis("off")

    plt.subplot(sample_num*2, 10, index + 11)
    number = int(np.sum(np.abs(us_k[i])))
    plt.imshow(np.log(1+np.abs(us_k[i, :, :, 0] + 1j * us_k[i, :, :, 1])), cmap="gray", interpolation=None)
    plt.title('UnderSampled:'+str(number), fontdict={'fontsize': 6})
    plt.axis("off")

    plt.subplot(sample_num*2, 10, index + 12)
    number = int(np.sum(np.abs(lr_pd_k[i])))
    plt.imshow(np.log(1+np.abs(lr_pd_k[i, :, :, 0] + 1j * lr_pd_k[i, :, :, 1])), cmap="gray", interpolation=None)
    plt.title('LR Predicted:'+str(number), fontdict={'fontsize': 6})
    plt.axis("off")

    plt.subplot(sample_num*2, 10, index + 13)
    number = int(np.sum(np.abs(lr_gt_k[i])))
    plt.imshow(np.log(1+np.abs(lr_gt_k[i, :, :, 0] + 1j * lr_gt_k[i, :, :, 1])), cmap="gray", interpolation=None)
    plt.title('LR GT:'+str(number), fontdict={'fontsize': 6})
    plt.axis("off")

    plt.subplot(sample_num * 2, 10, index + 14)
    error_map = (np.log(1+np.abs(lr_pd_k[i, :, :, 0] + 1j * lr_pd_k[i, :, :, 1])) - np.log(1+np.abs(lr_gt_k[i, :, :, 0] + 1j * lr_gt_k[i, :, :, 1])))  # [h, w]
    bound = min(abs(error_map.max().item()), abs(error_map.min().item()))
    error_map = np.clip(error_map, -bound, bound)
    plt.imshow(error_map, cmap='bwr', interpolation=None)
    plt.title('LR K Error Map', fontdict={'fontsize': 6})
    plt.axis("off")

    plt.subplot(sample_num * 2, 10, index + 15)
    number = int(np.sum(np.abs(up_k[i])))
    plt.imshow(np.log(1 + np.abs(up_k[i, :, :, 0] + 1j * up_k[i, :, :, 1])), cmap="gray", interpolation=None)
    plt.title('UpSampled Result:' + str(number), fontdict={'fontsize': 6})
    plt.axis("off")

    plt.subplot(sample_num * 2, 10, index + 16)
    number = int(np.sum(np.abs(hr_unConv_pd_k[i])))
    plt.imshow(np.log(1+np.abs(hr_unConv_pd_k[i, :, :, 0] + 1j * hr_unConv_pd_k[i, :, :, 1])), cmap="gray", interpolation=None)
    plt.title('HR unConv Predicted:' + str(number), fontdict={'fontsize': 6})
    plt.axis("off")

    plt.subplot(sample_num * 2, 10, index + 17)
    number = int(np.sum(np.abs(hr_Conv_pd_k[i])))
    plt.imshow(np.log(1 + np.abs(hr_Conv_pd_k[i, :, :, 0] + 1j * hr_Conv_pd_k[i, :, :, 1])), cmap="gray",
               interpolation=None)
    plt.title('HR Conv Predicted:' + str(number), fontdict={'fontsize': 6})
    plt.axis("off")

    plt.subplot(sample_num * 2, 10, index + 18)
    number = int(np.sum(np.abs(hr_gt_k[i])))
    plt.imshow(np.log(1+np.abs(hr_gt_k[i, :, :, 0] + 1j * hr_gt_k[i, :, :, 1])), cmap="gray", interpolation=None)
    plt.title('HR GT:' + str(number), fontdict={'fontsize': 6})
    plt.axis("off")

    plt.subplot(sample_num * 2, 10, index + 19)
    error_map = (np.log(1+np.abs(hr_unConv_pd_k[i, :, :, 0] + 1j * hr_unConv_pd_k[i, :, :, 1])) - np.log(1+np.abs(hr_gt_k[i, :, :, 0] + 1j * hr_gt_k[i, :, :, 1]))) # [h, w]
    bound = min(abs(error_map.max().item()), abs(error_map.min().item()))
    error_map = np.clip(error_map, -bound, bound)
    plt.imshow(error_map, cmap='bwr', interpolation=None)
    plt.title('HR unConv K Error Map', fontdict={'fontsize': 6})
    plt.axis("off")

    plt.subplot(sample_num * 2, 10, index + 20)
    error_map = (np.log(1 + np.abs(hr_Conv_pd_k[i, :, :, 0] + 1j * hr_Conv_pd_k[i, :, :, 1])) - np.log(
      1 + np.abs(hr_gt_k[i, :, :, 0] + 1j * hr_gt_k[i, :, :, 1]))) # [h, w]
    bound = min(abs(error_map.max().item()), abs(error_map.min().item()))
    error_map = np.clip(error_map, -bound, bound)
    plt.imshow(error_map, cmap='bwr', interpolation=None)
    plt.title('HR Conv K Error Map', fontdict={'fontsize': 6})
    plt.axis("off")

    index += 20

  plt.suptitle('UP-PSNR:%.4f  UP-SSIM:%.4f  UnConv-PSNR:%.4f  Conv-PSNR:%.4f  UnConv-SSIM:%.4f  Conv-SSIM:%.4f' % (up_psnr, up_ssim, unConv_psnr, Conv_psnr, unConv_ssim, Conv_ssim), fontdict={'fontsize': 10})
  plt.tight_layout()
  plt.savefig(path)
  plt.close()

# metric calculator

def cal_psnr(pred_i, i_gt):
  # 计算一个batch中所有sample的psnr并求和
  np_pred = pred_i
  np_gt = i_gt
  p = 0.0
  for i in range(np_gt.shape[0]):
    abs_pred = np.abs(np_pred[i, :, :, 0] + 1j * np_pred[i, :, :, 1])
    abs_gt = np.abs(np_gt[i, :, :, 0] + 1j * np_gt[i, :, :, 1])
    peak_signal = (abs_gt.max() - abs_gt.min())
    mse = np.power((abs_pred - abs_gt), 2).mean()
    p += 10 * log10(peak_signal ** 2 / mse)
  return p

def cal_ssim(pred_i, i_gt):
  """

  Args:
    pred_i: [b, h, w, 2]
    i_gt: [b, h, w, 2]

  Returns:
    一个batch中所有sample的ssim之和

  """
  ssim = 0.0
  for i in range(i_gt.shape[0]):
    abs_pred = np.abs(pred_i[i, :, :, 0] + 1j * pred_i[i, :, :, 1])
    abs_gt = np.abs(i_gt[i, :, :, 0] + 1j * i_gt[i, :, :, 1])
    ssim += SSIM(abs_gt, abs_pred, data_range=(abs_gt.max() - abs_gt.min()))
  return ssim
