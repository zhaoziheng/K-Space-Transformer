from layers import *
import math
from fftc import *
from utils import *
import torch.nn.functional as F

def build_grid(resolution):
  ranges = [np.linspace(0., 1., num=res) for res in resolution]
  grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
  grid = np.stack(grid, axis=-1)
  grid = np.reshape(grid, [resolution[0], resolution[1], -1])
  grid = grid.astype(np.float32)
  return torch.tensor(grid)

class Transformer(nn.Module):

  def __init__(self, lr_size, channel=2, d_model=512, nhead=8, num_encoder_layers=6,
               num_LRdecoder_layers=6, num_HRdecoder_layers=6, dim_feedforward=2048,
               HR_conv_channel=64, HR_conv_num=3, HR_kernel_size=5,
               dropout=0.1, activation="relu"):
    super().__init__()

    self.num_HRdecoder_layers = num_HRdecoder_layers

    self.encoder_embed_layer = nn.Sequential(
      nn.Linear(channel, d_model),
      nn.ReLU(inplace=True),
      nn.Linear(d_model, d_model)
    )

    self.pe_layer = PositionalEncoding(d_model, magnify=250.0)

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
    self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

    decoder_layerLR = TransformerDecoderLayerLR(d_model, nhead, dim_feedforward, dropout, activation)
    self.decoderLR = TransformerDecoderLR(d_model=d_model,
                                          lr_size=lr_size,
                                          channel=2,
                                          decoder_layer=decoder_layerLR,
                                          num_layers=num_LRdecoder_layers)

    decoder_layerHR = TransformerDecoderLayerHR(d_model, nhead, dim_feedforward, dropout, activation)
    self.decoderHR = TransformerDecoderHR(d_model=d_model,
                                          channel=2,
                                          decoder_layer=decoder_layerHR,
                                          num_layers=num_HRdecoder_layers,
                                          conv_channel=HR_conv_channel,
                                          conv_num=HR_conv_num,
                                          kernel_size=HR_kernel_size)

    self._reset_parameters()

    self.d_model = d_model
    self.nhead = nhead

  def _reset_parameters(self):
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

  def forward(self, src, lr_pos, src_pos, hr_pos, k_us, unsampled_pos, up_scale, mask, conv_weight, stage):
    """

    Args:
      src: [bs, src_len, c] intensity of sampled points
      lr_pos: [bs, lh*lw, 2] normalized coordinates of LR query points
      src_pos: [bs, src_len, 2] normalized coordinates of sampled points
      hr_pos: [bs, query_len, 2] normalized coordinates of unsampled points
      k_us: [bs, h, w, c] zero-filled specturm
      mask: [bs, h, w, c] undersampling mask, 1 means unsampled and 0 means sampled
      unsampled_pos: [bs, query_len, 2] coordinates of unsampled points(unnormalized)
      up_scale: LR upsample ratio to HR

    Returns:

    """
    unsampled_pos = unsampled_pos.permute(0, 2, 1).cpu().numpy()  # pos [bs, 2, quey_len]

    # encode
    src_embed = self.encoder_embed_layer(src)
    src_pe = self.pe_layer(src_pos)      # [bs, src_len, d]
    Encoder_memory = self.encoder(src_embed, pos=src_pe) # [bs, src_len, d]

    # lr decode
    lr_pe = self.pe_layer(lr_pos)        # [bs, lh*lw, d]
    LR_Trans_outputs, LR_memory = self.decoderLR(Encoder_memory, lr_pe)   # [num_of_layers, img:[bs, lh, lw, c]], [bs, lh*lw, d]

    # upsample in image domain
    LR_i = LR_Trans_outputs[-1].permute(0, 3, 1, 2).contiguous()   # [bs, c, lh, lw]
    Up_LR_i = F.interpolate(LR_i, scale_factor=up_scale, mode='bicubic').permute(0, 2, 3, 1).contiguous() # [bs, h, w, c]
    # back to k space
    Up_LR_k = fft2c_new(Up_LR_i)    # [bs, h, w, c]
    if stage == 'LR':
      HR_Trans_outputs = HR_Conv_outputs = [torch.zeros_like(Up_LR_i)] * self.num_HRdecoder_layers
      return LR_Trans_outputs, Up_LR_i, Up_LR_k, HR_Trans_outputs, HR_Conv_outputs
    else:
      # select unsampled points' predicted values
      unsampled_value = []
      for i in range(unsampled_pos.shape[0]):
        unsampled_value.append(Up_LR_k[i, :, :, :][unsampled_pos[i, :, :]])  # k [query_len, c]
      unsampled_value = torch.stack(unsampled_value, dim=0)        # k [bs, query_len, c]

      # hr decode
      hr_pe = self.pe_layer(hr_pos)  # [bs, query_len, d]
      HR_Trans_outputs, HR_Conv_outputs = self.decoderHR(lr_memory=LR_memory,
                                                         query_pe=hr_pe,
                                                         query_value=unsampled_value,
                                                         unsampled_pos=unsampled_pos,
                                                         k_us=k_us,
                                                         mask=mask,
                                                         conv_weight=conv_weight,
                                                         stage=stage)    # [num_of_layers, [bs, h, w, c]]

      return LR_Trans_outputs, Up_LR_i, Up_LR_k, HR_Trans_outputs, HR_Conv_outputs

class TransformerEncoder(nn.Module):

  def __init__(self, encoder_layer, num_layers):
    super().__init__()
    self.layers = _get_clones(encoder_layer, num_layers)
    self.num_layers = num_layers

  def with_pos_embed(self, tensor, pos):
    return tensor if pos is None else tensor + pos

  def forward(self, src, pos):
    output = self.with_pos_embed(src, pos)

    for layer in self.layers:
      output = layer(output)

    return output

class TransformerDecoderLR(nn.Module):

  def __init__(self, d_model, lr_size, channel, decoder_layer, num_layers):
    super().__init__()
    self.layers = _get_clones(decoder_layer, num_layers)
    self.num_layers = num_layers

    self.channel = channel
    self.lr_size = lr_size

    self.LR_predict_layers = []
    self.LR_norm_layers = []
    self.LR_embed_layers = []
    self.ConvBlocks = []

    for i in range(num_layers):
      self.LR_predict_layers.append(nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.ReLU(inplace=True),
        nn.Linear(d_model, channel),
      ))
      self.LR_norm_layers.append(nn.LayerNorm(d_model, eps=1e-6))
    self.LR_norm_layers = nn.ModuleList(self.LR_norm_layers)
    self.LR_predict_layers = nn.ModuleList(self.LR_predict_layers)

  def forward(self, encoder_memory, lr_pe):

    Transformer_interpredict = []

    input = lr_pe
    for i in range(len(self.layers)):
      output_memory = self.layers[i](input, encoder_memory)   # [bs, lh*lw, d]

      output = self.LR_predict_layers[i](self.LR_norm_layers[i](output_memory))  # k [b, lh*lw, c]
      output = torch.reshape(output, (output.shape[0], self.lr_size, self.lr_size, self.channel))   # k [b, lh, lw, c]
      output = ifft2c_new(output, needShift=True)   # img [b, lh, lw, c]
      Transformer_interpredict.append(output)

      input = output_memory

    return Transformer_interpredict, output_memory

class TransformerDecoderHR(nn.Module):

  def __init__(self, d_model, channel, decoder_layer, num_layers, conv_channel=64, conv_num=3, kernel_size=5):
    super().__init__()
    self.layers = _get_clones(decoder_layer, num_layers)
    self.num_layers = num_layers

    self.HR_predict_layers = []
    self.HR_norm_layers = []
    self.HR_embed_layers = []
    self.ConvBlocks = []

    for i in range(num_layers):
      self.HR_predict_layers.append(nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.ReLU(inplace=True),
        nn.Linear(d_model, channel),
      ))
      self.HR_norm_layers.append(nn.LayerNorm(d_model, eps=1e-6))
      self.ConvBlocks.append(CNN_Block(in_channels=channel, mid_channels=conv_channel, num_convs=conv_num, kernel_size=kernel_size))
    self.HR_norm_layers = nn.ModuleList(self.HR_norm_layers)
    self.HR_predict_layers = nn.ModuleList(self.HR_predict_layers)
    self.HR_embed_layer = nn.Sequential(
      nn.Linear(channel, d_model),
      nn.ReLU(inplace=True),
      nn.Linear(d_model, d_model)
    )
    self.ConvBlocks = nn.ModuleList(self.ConvBlocks)

  def forward(self, lr_memory, query_pe, query_value, unsampled_pos, k_us, mask, conv_weight, stage):
    input = query_pe + self.HR_embed_layer(query_value)    # value + pe

    Transformer_interpredict = []
    CNN_interpredict = []      # [[b, h, w, d]... ...]

    for i in range(len(self.layers)):
      output_memory = self.layers[i](input, lr_memory)   # [b, query_len, d]

      output = self.HR_predict_layers[i](self.HR_norm_layers[i](output_memory))  # k [b, query_len, c]
      output = fill_in_k(k_us, unsampled_pos, output)  # k [bs, h, w, c]
      output = ifft2c_new(output)                      # img [bs, h, w, c]
      Transformer_interpredict.append(output)

      if stage == 'RM':
        conv_output = self.ConvBlocks[i](k_us, output, mask)         # img [bs, h, w, c]
        CNN_interpredict.append(conv_output)
      else:
        CNN_interpredict.append(torch.zeros_like(output))

      if i+1 == len(self.layers):
        return Transformer_interpredict, CNN_interpredict
      else:
        if stage == 'RM':
          conv_output = fft2c_new(conv_output)   # k [bs, h, w, c]
          # select unsampled points' predicted values
          unsampled_value = []
          for j in range(conv_output.shape[0]):
            unsampled_value.append(conv_output[j, :, :, :][unsampled_pos[j, :, :]])  # k [query_len, c]
          unsampled_value = torch.stack(unsampled_value, dim=0)            # k [bs, query_len, c]
          input = conv_weight * self.HR_embed_layer(unsampled_value) + output_memory   # [b, query_len, d]
        else:
          input = output_memory

class CNN_Block(nn.Module):
  def __init__(self, in_channels=2, mid_channels=48, num_convs=4, kernel_size=3):
    super(CNN_Block, self).__init__()
    self.convs = []

    # first layers
    self.convs.append(nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size//2))
    self.convs.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
    # N * middle layers
    for i in range(num_convs):
      self.convs.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size//2))
      self.convs.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
    # final layers
    self.convs.append(nn.Conv2d(mid_channels, in_channels, kernel_size=1))

    self.convs = nn.ModuleList(self.convs)

  def DataConsistency(self, k_rec, k_in, mask):
    # mask中1表示没采样，0表示采样
    k_rec_masked = k_rec * mask
    k_out = k_rec_masked + k_in
    return k_out

  def forward(self, k_sampled, i_in, mask):
    output = i_in.permute(0, 3, 1, 2).contiguous()   # [bs, c, h, w]

    for layer in self.convs:
      output = layer(output)

    output = output.permute(0, 2, 3, 1).contiguous() # [bs, h, w, c]
    output = output + i_in

    k_rec = fft2c_new(output)
    k_rec = self.DataConsistency(k_rec, k_sampled, mask)
    output = ifft2c_new(k_rec)

    return output

class PositionalEncoding(nn.Module):

  def __init__(self, pe_dim=128, magnify=100.0):
    super(PositionalEncoding, self).__init__()
    # Compute the division term
    # note that the positional dim for x and y is equal to dim_of_pe/2
    self.dim = pe_dim
    self.div_term = nn.Parameter(torch.exp(torch.arange(0, self.dim/2, 2) * -(2 * math.log(10000.0) / self.dim)), requires_grad=False)      # [32]
    self.magnify = magnify

  def forward(self, p_norm):
    """
    given position:[bs, h*w*0.2, 2]

    return pe
    """

    p = p_norm * self.magnify  # normalized 到 [0, magnify] 之间

    no_batch = False
    if p.dim() == 2:    # no batch size
      no_batch = True
      p = p.unsqueeze(0)

    p_x = p[:, :, 0].unsqueeze(2)                            # [bs, h*w*0.2, 1]
    p_y = p[:, :, 1].unsqueeze(2)
    # assert p_x.shape[1] == p_y.shape[1]
    pe_x = torch.zeros(p_x.shape[0], p_x.shape[1], self.dim // 2).to(torch.device('cuda'))     # [bs, h*w*0.2, 64]
    pe_x[:, :, 0::2] = torch.sin(p_x * self.div_term)                       # [bs, h*w*0.2, 32]
    pe_x[:, :, 1::2] = torch.cos(p_x * self.div_term)


    pe_y = torch.zeros(p_x.shape[0], p_x.shape[1], self.dim // 2).to(torch.device('cuda'))     # [bs, h*w*0.2, 64]
    pe_y[:, :, 0::2] = torch.sin(p_y * self.div_term)
    pe_y[:, :, 1::2] = torch.cos(p_y * self.div_term)

    pe = torch.cat([pe_x, pe_y], dim=2)                      # [bs, h*w*0.2, 128]

    if no_batch:
      pe = pe.squeeze(0)

    # [len, dim]
    return pe

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

