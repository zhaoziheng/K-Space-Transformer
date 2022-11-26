from multi_head_attn import *

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = MultiHeadAttention(nhead, d_model, d_k=d_model//nhead, d_v=d_model//nhead, dropout=dropout)    # 自带dropout & residual & norm

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)

    def forward(self, src):

        q = k = src
        src = self.self_attn(q, k, src)[0]

        src2 = self.norm(src)

        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src

class TransformerDecoderLayerLR(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.self_attn = MultiHeadAttention(nhead, d_model, d_k=d_model//nhead, d_v=d_model//nhead, dropout=dropout)

        self.multihead_attn = MultiHeadAttention(nhead, d_model, d_k=d_model//nhead, d_v=d_model//nhead, dropout=dropout)

        self.ffn1 = nn.Sequential(
          nn.Linear(d_model, dim_feedforward),
          _get_activation_md(activation),
          nn.Dropout(dropout),
          nn.Linear(dim_feedforward, d_model),
          nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory):

        tgt = self.multihead_attn(tgt, memory, memory)[0]

        tgt = self.self_attn(tgt, tgt, tgt)[0]

        tgt2 = self.norm(tgt)

        tgt2 = self.ffn1(tgt2)
        tgt = tgt + tgt2

        return tgt

class TransformerDecoderLayerHR(nn.Module):

  def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
    super().__init__()
    self.multihead_attn = MultiHeadAttention(nhead, d_model, d_k=d_model//nhead, d_v=d_model//nhead, dropout=dropout)

    self.ffn1 = nn.Sequential(
      nn.Linear(d_model, dim_feedforward),
      _get_activation_md(activation),
      nn.Dropout(dropout),
      nn.Linear(dim_feedforward, d_model),
      nn.Dropout(dropout)
    )
    self.norm = nn.LayerNorm(d_model)

  def forward(self, tgt, memory):
    # tgt: [bs, h*w, d] hr-deocder query
    tgt = self.multihead_attn(tgt, memory, memory)[0]

    tgt2 = self.norm(tgt)

    tgt2 = self.ffn1(tgt2)
    tgt = tgt + tgt2

    return tgt

def _get_activation_fn(activation):
  """Return an activation function given a string"""
  if activation == "relu":
    return F.relu
  if activation == "gelu":
    return F.gelu
  if activation == "glu":
    return F.glu
  raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_activation_md(activation):
  """Return an activation function given a string"""
  if activation == "relu":
    return nn.ReLU()
  if activation == "gelu":
    return nn.GELU()
  if activation == "glu":
    return nn.GLU()
  raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
