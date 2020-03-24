
import torch
import torch.nn as nn
import torch.nn.functional as F
from sparse_attn import SparseAttention

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        #self.dropout = nn.Dropout(attn_dropout)
        self.use_sparse = True

        self.sa = SparseAttention()

    def forward(self, q, k, v, mask=None):

        # bs x pos x key .. bs x key x pos

        # bs x pos x pos .. bs x pos x key

        attn = torch.matmul(q / self.temperature, k.permute(0,2,1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        if self.use_sparse:
            mb, ins, outs = attn.shape[0], attn.shape[1], attn.shape[2]
            sparse_attn = attn.reshape((mb*ins, outs))
            sparse_attn = self.sa(sparse_attn)
            sparse_attn = sparse_attn.reshape((mb,ins,outs))
            attn = sparse_attn*1.0

        #attn = self.dropout(F.softmax(attn, dim=-1))
        
        attn = F.softmax(attn, dim=-1)

        output = torch.matmul(attn, v)

        return output, attn

