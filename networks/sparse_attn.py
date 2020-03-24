
import torch
import torch.nn as nn
import numpy

class SparseAttention(nn.Module):
    def __init__(self, top_k = 3):
        super(SparseAttention,self).__init__()
        top_k += 1
        self.top_k = top_k

    def forward(self, attn_s):

        attn_plot = []
        eps = 10e-8
        time_step = attn_s.size()[1]
        if time_step <= self.top_k:
            return attn_s
        else:
            delta = torch.topk(attn_s, self.top_k, dim= 1)[0][:,-1] + eps
            delta = delta.reshape((delta.shape[0],1))


        attn_w = attn_s - delta.repeat(1, time_step)
        attn_w = torch.clamp(attn_w, min = 0)
        attn_w_sum = torch.sum(attn_w, dim = 1, keepdim=True)
        attn_w_sum = attn_w_sum + eps
        attn_w_normalize = attn_w / attn_w_sum.repeat(1, time_step)

        return attn_w_normalize

