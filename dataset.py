from torch.utils.data import Dataset
from fftc import *
from utils import *
import random

def build_grid(resolution):
  ranges = [np.linspace(0., 1., num=res) for res in resolution]
  grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
  grid = np.stack(grid, axis=-1)
  grid = np.reshape(grid, [resolution[0], resolution[1], -1])
  grid = grid.astype(np.float32)
  return torch.tensor(grid)

class TrainDataSet(Dataset):
  def __init__(self, dataPath, LRdataPath, maskPath):
    self.k_gt, self.i_gt, self.mask, \
    self.sampled_num, self.unsampled_num, \
    self.LR_image, self.LR_k, self.LR_pos, self.LR_pos_norm = self.read_data(dataPath, LRdataPath, maskPath)

    self.image_num = self.k_gt.shape[0]
    self.channel = self.k_gt.shape[3]
    self.h = self.k_gt.shape[1]
    self.w = self.k_gt.shape[2]

  def __len__(self):
    return self.image_num

  def __getitem__(self, index):

    return {'sampled_k': torch.from_numpy(self.sampled_k[index]).type(torch.FloatTensor),
            'sampled_pos': torch.from_numpy(self.sampled_pos[index]).type(torch.IntTensor),
            'sampled_pos_norm': self.sampled_pos_norm[index].type(torch.FloatTensor),

            'unsampled_pos_norm': self.unsampled_pos_norm[index].type(torch.FloatTensor),
            'unsampled_pos': torch.from_numpy(self.unsampled_pos[index]).type(torch.IntTensor),

            'k_us': self.k_us[index, :, :, :].type(torch.FloatTensor),
            'i_gt': self.i_gt[index, :, :, :].type(torch.FloatTensor),
            'k_gt': self.k_gt[index, :, :, :].type(torch.FloatTensor),
            'selected_mask': self.selected_mask[index, :, :, :].type(torch.FloatTensor),

            'LR_i_gt': self.LR_image[index].type(torch.FloatTensor),
            'LR_k_gt': self.LR_k[index].type(torch.FloatTensor),
            'LR_pos': torch.from_numpy(self.LR_pos).type(torch.IntTensor),
            'LR_pos_norm': self.LR_pos_norm.type(torch.FloatTensor)
            }

  def reassign_mask(self):
    """
    A warp-up function to undersample each training sample randomly before every N epoch
    """
    self.sampled_k, self.sampled_pos, self.sampled_pos_norm, \
    self.unsampled_pos, self.unsampled_pos_norm, \
    self.k_us, self.selected_mask = self.generate_data()

  def read_data(self, dataPath, LRdataPath, maskPath):
    """
    Load data, mask and derive Low-Resolution position
    Args:
      dataPath: HR k-space data path
      LRdataPath: LR k-space data path
      maskPath: mask data path

    Returns:
      k_data, i_data, LR_i_data, LR_k_data, mask: tensor [num of samples, h/lh, w/lw, c]
      LR_pos_seq_norm : tensor [num of unsampled points, 2]    normalized coordinate sequence
      LR_pos_seq : array [num of unsampled points, 2]   unnormalized
      sampled_num, unsampled_num : scalar
    """
    k_data = np.load(dataPath)  # [num_of_sample, H, W, 2]
    LR_k_data = np.load(LRdataPath)

    self.HR_grid = build_grid([k_data.shape[1], k_data.shape[2]])  # h, w
    self.LR_grid = build_grid([LR_k_data.shape[1], LR_k_data.shape[2]])  # lh, lw

    k_data = torch.from_numpy(k_data)
    i_data = ifft2c_new(k_data, needShift=True)

    LR_k_data = torch.from_numpy(LR_k_data)
    LR_i_data = ifft2c_new(LR_k_data, needShift=True)

    LR_pos_seq = [[], []]
    for i in range(LR_i_data.shape[1]):
      for j in range(LR_i_data.shape[2]):
        LR_pos_seq[0].append(i)
        LR_pos_seq[1].append(j)
    LR_pos_seq_norm = self.LR_grid[LR_pos_seq]  # [num of unsampled points, 2]
    LR_pos_seq = np.array(LR_pos_seq).transpose(1, 0)  # [num of unsampled points, 2]

    mask = np.load(maskPath)  # [num_of_mask, H, W]
    sampled_num = mask[0, :, :].sum()
    unsampled_num = mask.shape[1] * mask.shape[2] - sampled_num
    mask = np.concatenate((mask[:, :, :, np.newaxis], mask[:, :, :, np.newaxis]), axis=-1)  # [num_of_mask, H, W, 2]

    return k_data, i_data, mask, sampled_num, unsampled_num, LR_i_data, LR_k_data, LR_pos_seq, LR_pos_seq_norm

  def select_mask(self):
    """
    randomly select a pre-generated mask
    """
    indexes = np.random.randint(low=0, high=self.mask.shape[0])  # [1]
    mask = (self.mask[indexes])  # [H, W, c]

    return mask

  def generate_data(self):
    """
    Convert 2D data to 1D input/query via randomly selected mask

    Returns:
       sampled_k / sampled_pos: list [image_num, array(input_len, 2)]
       sampled_pos_norm : list [image_num, tensor(input_len, 2)]
       unsampled_pos : list [image_num, array(query_len, 2)]
       unsampled_pos_norm : list [image_num, tensor(query_len, 2)]]
       k_us : tensor (image_num, h, w, c)
       selected_mask : tensor (image_num, h, w, c)

    """
    selected_mask = []
    sampled_k = []
    sampled_pos = []
    sampled_pos_norm = []
    unsampled_pos_norm = []
    unsampled_pos = []

    k_us = copy.deepcopy(self.k_gt)
    for i in range(self.k_gt.shape[0]):
      mask = self.select_mask()  # [H, W, c]
      k_us[i][~mask] = 0
      selected_mask.append(~mask)

      kgt = self.k_gt[i, :, :, :]  # [H, W, C]
      sampled_indexes = np.nonzero(mask[:, :, 0])   # 1 for sample, 0 for unsample [2, num of sampled points]
      sampled_real = kgt[:, :, 0][sampled_indexes]  # [num of sampled points]
      sampled_imag = kgt[:, :, 1][sampled_indexes]  # [num of sampled points]
      sampled_complex = np.concatenate((sampled_real[:, np.newaxis], sampled_imag[:, np.newaxis]),
                                       axis=-1)     # [num of sampled points, 2]
      sampled_indexes_norm = self.HR_grid[sampled_indexes]  # tensor [num of sampled points, 2]
      sampled_indexes = np.array(sampled_indexes).transpose(1, 0)  # [num of sampled points, 2]

      unsampled_indexes = np.nonzero(~mask[:, :, 0])  # [2, num of unsampled points]
      unsampled_indexes_norm = self.HR_grid[unsampled_indexes]  # tensor [num of unsampled points, 2]
      unsampled_indexes = np.array(unsampled_indexes).transpose(1, 0) # [num of unsampled points, 2]

      sampled_k.append(sampled_complex)
      sampled_pos.append(sampled_indexes)
      sampled_pos_norm.append(sampled_indexes_norm)
      unsampled_pos_norm.append(unsampled_indexes_norm)
      unsampled_pos.append(unsampled_indexes)

    selected_mask = torch.from_numpy(np.stack(selected_mask, axis=0))

    return sampled_k, sampled_pos, sampled_pos_norm, \
           unsampled_pos, unsampled_pos_norm, \
           k_us, selected_mask

class TestDataSet(Dataset):
  def __init__(self, dataPath, LRdataPath, maskPath, test_time_train=True):
    self.k_gt, self.i_gt, self.mask, \
    self.sampled_num, self.unsampled_num, \
    self.LR_image, self.LR_k, self.LR_pos, self.LR_pos_norm = self.read_data(dataPath, LRdataPath, maskPath)  # [num_of_sample, h, w, 2]

    self.image_num = self.k_gt.shape[0]
    self.channel = self.k_gt.shape[3]
    self.h = self.k_gt.shape[1]
    self.w = self.k_gt.shape[2]

    self.sampled_k, self.sampled_pos, self.sampled_pos_norm, \
    self.unsampled_pos, self.unsampled_pos_norm, \
    self.k_us, self.selected_mask = self.generate_data()

  def __len__(self):

    return self.image_num

  def __getitem__(self, index):

    return {'sampled_k': torch.from_numpy(self.sampled_k[index]).type(torch.FloatTensor),
            'sampled_pos': torch.from_numpy(self.sampled_pos[index]).type(torch.IntTensor),
            'sampled_pos_norm': self.sampled_pos_norm[index].type(torch.FloatTensor),

            'unsampled_pos_norm': self.unsampled_pos_norm[index].type(torch.FloatTensor),
            'unsampled_pos': torch.from_numpy(self.unsampled_pos[index]).type(torch.IntTensor),

            'k_us': self.k_us[index, :, :, :].type(torch.FloatTensor),
            'i_gt': self.i_gt[index, :, :, :].type(torch.FloatTensor),
            'k_gt': self.k_gt[index, :, :, :].type(torch.FloatTensor),
            'selected_mask': self.selected_mask[index, :, :, :].type(torch.FloatTensor),

            'LR_i_gt': self.LR_image[index].type(torch.FloatTensor),
            'LR_k_gt': self.LR_k[index].type(torch.FloatTensor),
            'LR_pos': torch.from_numpy(self.LR_pos).type(torch.IntTensor),
            'LR_pos_norm': self.LR_pos_norm.type(torch.FloatTensor)
            }

  def read_data(self, dataPath, LRdataPath, maskPath):
    k_data = np.load(dataPath)  # [num_of_sample, H, W, 2]
    LR_k_data = np.load(LRdataPath)

    self.HR_grid = build_grid([k_data.shape[1], k_data.shape[2]])  # h, w
    self.LR_grid = build_grid([LR_k_data.shape[1], LR_k_data.shape[2]])  # lh, lw

    k_data = torch.from_numpy(k_data)
    i_data = ifft2c_new(k_data, needShift=True)

    LR_k_data = torch.from_numpy(LR_k_data)
    LR_i_data = ifft2c_new(LR_k_data, needShift=True)

    LR_pos_seq = [[], []]
    for i in range(LR_i_data.shape[1]):
      for j in range(LR_i_data.shape[2]):
        LR_pos_seq[0].append(i)
        LR_pos_seq[1].append(j)
    LR_pos_seq_norm = self.LR_grid[LR_pos_seq]  # [num of unsampled points, 2]
    LR_pos_seq = np.array(LR_pos_seq).transpose(1, 0)  # [num of unsampled points, 2]

    mask = np.load(maskPath)  # [num_of_mask, H, W]
    sampled_num = mask[0, :, :].sum()
    unsampled_num = mask.shape[1] * mask.shape[2] - sampled_num
    mask = np.concatenate((mask[:, :, :, np.newaxis], mask[:, :, :, np.newaxis]), axis=-1)  # [num_of_mask, H, W, 2]

    return k_data, i_data, mask, sampled_num, unsampled_num, LR_i_data, LR_k_data, LR_pos_seq, LR_pos_seq_norm

  def select_mask(self):
    indexes = np.random.randint(low=0, high=self.mask.shape[0])  # [1]
    mask = (self.mask[indexes])  # [H, W, c]
    return mask

  def generate_data(self):
    selected_mask = []
    sampled_k = []  # list : [image num, array(input_len, 2)]
    sampled_pos = []  # list : [image num, array(input_len, 2)]
    sampled_pos_norm = []  # list : [image num, tensor(input_len, 2)]
    unsampled_pos_norm = []  # list : [image num, tensor(query_len, 2))]
    unsampled_pos = []  # list : [image num, array(query_len, 2)]

    k_us = copy.deepcopy(self.k_gt)
    for i in range(self.k_gt.shape[0]):
      mask = self.select_mask()  # [H, W, c]
      k_us[i][~mask] = 0
      selected_mask.append(~mask)

      kgt = self.k_gt[i, :, :, :]  # [H, W, C]
      sampled_indexes = np.nonzero(mask[:, :, 0])  # [2, num of sampled points]
      sampled_real = kgt[:, :, 0][sampled_indexes]  # [num of sampled points]
      sampled_imag = kgt[:, :, 1][sampled_indexes]  # [num of sampled points]
      sampled_complex = np.concatenate((sampled_real[:, np.newaxis], sampled_imag[:, np.newaxis]),
                                       axis=-1)  # [num of sampled points, 2]
      sampled_indexes_norm = self.HR_grid[sampled_indexes]  # tensor [num of sampled points, 2]
      sampled_indexes = np.array(sampled_indexes).transpose(1, 0)  # [num of sampled points, 2]

      unsampled_indexes = np.nonzero(~mask[:, :, 0])  # [2, num of unsampled points]
      unsampled_indexes_norm = self.HR_grid[unsampled_indexes]  # tensor [num of unsampled points, 2]
      unsampled_indexes = np.array(unsampled_indexes).transpose(1, 0) # [num of unsampled points, 2]

      sampled_k.append(sampled_complex)
      sampled_pos.append(sampled_indexes)
      sampled_pos_norm.append(sampled_indexes_norm)
      unsampled_pos_norm.append(unsampled_indexes_norm)
      unsampled_pos.append(unsampled_indexes)

    selected_mask = torch.from_numpy(np.stack(selected_mask, axis=0))

    return sampled_k, sampled_pos, sampled_pos_norm, \
           unsampled_pos, unsampled_pos_norm, \
           k_us, selected_mask