import numpy as np
from fftc import ifft2c_new, fft2c_new
import torch
import os
import h5py
import matplotlib.pylab as plt

def save_image(np_data, path, type='np'):
  if type == 'torch':
    data = np_data.numpy()
  else:
    data = np_data
  if data.ndim == 3:
    plt.imshow(np.abs(data[:, :, 0] + 1j * data[:, :, 1]), cmap="gray")  # [h, w, c]
  elif data.ndim == 2:
    plt.imshow(np.abs(data), cmap="gray")
  plt.axis("off")
  plt.savefig(path)
  plt.close()

def save_k(np_data, path, type='np'):
  if type == 'torch':
    data = np_data.numpy()
  else:
    data = np_data
  plt.figure(dpi=300)
  plt.imshow(np.log(1 + np.abs(data[:, :, 0] + 1j * data[:, :, 1])), cmap="gray")
  plt.axis("off")
  plt.savefig(path)
  plt.close()

def normal_in_i(image):
  """
  normalized to N(0, 1)

  Args:
    data: image, array [h, w, 2]
  """
  mean = np.mean(image)
  std = np.std(image)
  tmp = (image - mean)/std
  return tmp

def concat_all_h5py(crop_size, root_path, save_path):
  """
  From h5 to npy
  
  Args:
      crop_size (int): cropped size of each slice, e.g. 320
      root_path (str): root path to h5 files
      save_path (str): save path to npy file
  """
  g = os.walk(root_path)
  data_list = []    # [num_volumes, 26, 640, 372, 2]
  count = 0
  for _, _, file_list in g:
    for file_name in file_list:
      # open file
      if file_name[-2:] != 'h5':
        continue
      id = file_name[-10:-3]
      file_path = os.path.join(root_path, file_name)
      print(id)
      try:
        volume = h5py.File(file_path)['kspace'][()]   # [26, 640, 372]
      except:
        continue
      # crop the size if needed
      width = volume.shape[2]
      height = volume.shape[1]
      if width < crop_size or height < crop_size:
        continue
      else:
        count += 1
        print(count)
        w_from = (width-crop_size)//2
        w_to = w_from + crop_size
        h_from = (height - crop_size) // 2
        h_to = h_from + crop_size
      # complex to double channel
      volume = np.stack([volume.real, volume.imag], axis=-1)    # [26, 640, 372, 2]
      save_k(volume[15], root_path + '/images320/' + id + 'BeforeCropNorm(K).png')  # check
      # convert to image
      imag_volume = ifft2c_new(torch.from_numpy(volume)).numpy()
      save_image(imag_volume[15], root_path+'/images320/'+id+'BeforeCropNorm.png')  # check
      # crop
      imag_volume = imag_volume[5:, h_from:h_to, w_from:w_to, :]    # [21, 320, 320, 2]
      # normalize each sample
      for i in range(imag_volume.shape[0]):
        imag_volume[i, :, :, :] = normal_in_i(imag_volume[i, :, :, :])
      save_image(imag_volume[10], root_path + '/images320/' + id + 'AfterCropNorm.png')  # check
      # back to k space
      k_volume = fft2c_new(torch.from_numpy(imag_volume)).numpy()
      save_k(k_volume[10], root_path + '/images320/' + id + 'AfterCropNorm(K).png')  # check

      data_list.append(k_volume)

  data_concat = np.concatenate(data_list, axis=0)   # [num_volumes*volume_len, h, w, 2]
  np.save(save_path, data_concat)

def down_sample_I(np_I_data, scale=2):
  """
  Downsample image
  
  Args:
    np_I_data: image [N, h, w, c]
    scale: down sample ratio

  Returns: down sampled image [N, h/2, w/2, c]
  """
  m = nn.AvgPool2d(scale, stride=scale)
  i = torch.from_numpy(np_I_data)  # [n, 256, 256, 2]
  save_image(i[20], 'image_before_ds.png', 'torch') # check

  i = i.permute(0, 3, 1, 2)
  ds_i = m(i)

  ds_i = ds_i.permute(0, 2, 3, 1)
  save_image(ds_i[20], 'image_after_ds.png', 'torch') # check

  return ds_i.numpy()