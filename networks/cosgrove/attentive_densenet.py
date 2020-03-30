'''
Module which takes new (spatial) layers and performs attention to weight over all previous spatial layers.  

Needs to know beforehand number of channels in each layer.  

-For key and query, do mean-pooling over spatial dimensions and then do an FC-layer.  

-For value, do a conv-layer.  When using on the current time-step, resize values to the current size, then reshape.  

(new_layer) --> (attentive_input).  Keeps and updates a key_lst and value_lst internally.  

-Should attention be same or different object?  Probably easiest to make it the same object.  

-Need to figure out how to resize the value layers

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.cosgrove.attention import ScaledDotProductAttention
import numpy as np

class AttentiveDensenet(nn.Module):
    def __init__(self, layer_channels, key_size, val_size, n_heads):
        super(AttentiveDensenet, self).__init__()

        self.layer_channels = layer_channels
        self.key_size = key_size
        self.val_size = val_size
        self.n_heads = n_heads

        self.key_layers = []
        self.query_layers = []
        self.val_layers = []

        self.out_layers = []

        self.gammas = []

        for ch in layer_channels:
            self.key_layers.append(nn.Conv2d(ch, key_size*n_heads,1))
            self.query_layers.append(nn.Conv2d(ch, key_size*n_heads,1))
            self.val_layers.append(nn.Conv2d(ch, val_size*n_heads, 1))
            out_layer = nn.Sequential(nn.Conv2d(ch + val_size*n_heads,ch,3,stride=1,padding=1), nn.BatchNorm2d(ch), nn.ReLU(), nn.Conv2d(ch, ch, 3,stride=1,padding=1))
            self.out_layers.append(out_layer)
            self.gammas.append(nn.Parameter(torch.tensor(0.0)))
 
            for convs in [self.key_layers[-1], self.query_layers[-1], self.val_layers[-1], out_layer[0], out_layer[3]]:
                nn.init.xavier_uniform(convs.weight.data, 1.)

        self.query_layers = nn.ModuleList(self.query_layers)
        self.key_layers = nn.ModuleList(self.key_layers)
        self.val_layers = nn.ModuleList(self.val_layers)
        self.out_layers = nn.ModuleList(self.out_layers)
        self.gammas = nn.ParameterList(self.gammas)

        self.layer_index = None

        self.attn = ScaledDotProductAttention(np.power(key_size, 0.5))

    def reset(self):
        self.key_lst = []
        self.val_lst = []
        self.layer_index = 0

    def forward(self, x, read, write):
    
        sz_b, ch, h, w = x.shape

        if write:
            h_key = self.key_layers[self.layer_index](x) #bs x n_heads*key_size x h x w
            val = self.val_layers[self.layer_index](x)
            self.key_lst.append(h_key)
            self.val_lst.append(val)
        else:
            self.key_lst.append(None)
            self.val_lst.append(None)

        if not read:
            self.layer_index += 1
            return x

        h_query = self.query_layers[self.layer_index](x)
        query = h_query.reshape((sz_b,self.n_heads,self.key_size,h,w)).permute(0,3,4,1,2).reshape((sz_b*h*w*self.n_heads, 1, self.key_size)) #sz_b, h, w, n_heads key_size

        #print('key shape', key.shape)
        #print('query shape', query.shape)
        
        vals_reshaped = []
        keys_reshaped = []

        for ind in range(len(self.val_lst)):

            val = self.val_lst[ind]
            key = self.key_lst[ind]
            
            if val is None:
                assert key is None
                continue

            val_resized = F.interpolate(val, (h,w), mode='nearest')
            key_resized = F.interpolate(key, (h,w), mode='nearest')

            #if h > val.shape[2]:
            #    #upsample
            #    val_resized = F.interpolate(val, (h,w), mode='nearest')
            #    key_resized = F.interpolate(key, (h,w), mode='nearest')
            #else:
                #downsample
            #    maxpool = nn.AvgPool2d((val.shape[2]//h, val.shape[2]//h))
            #    val_resized = maxpool(val)
            #    key_resized = maxpool(key)

            val_resized = val_resized.reshape((sz_b, self.n_heads,self.val_size,h,w)).permute(0,3,4,1,2).reshape((sz_b*h*w*self.n_heads, 1, self.val_size))#.repeat(1,self.n_heads,1,1)
            vals_reshaped.append(val_resized)
            key_old = key_resized.reshape((sz_b,self.n_heads,self.key_size,h,w)).permute(0,3,4,1,2).reshape((sz_b*h*w*self.n_heads, 1, self.key_size))
            keys_reshaped.append(key_old)

        vals_tensor = torch.cat(vals_reshaped, dim = 1)
        keys_tensor = torch.cat(keys_reshaped, dim = 1)

        #print('query shape', query.shape)
        #print('vals tensor shape', vals_tensor.shape)
        #print('keys tensor shape', keys_tensor.shape)

        att_out,iatt = self.attn(query, keys_tensor, vals_tensor)

        att_out = att_out.reshape((sz_b, h, w, self.n_heads, self.val_size)).permute(0,3,4,1,2).reshape((sz_b, self.n_heads*self.val_size, h, w))

        att_out = torch.cat([att_out, x], dim=1)

        #print('att out shape', att_out.shape)
        att_out = self.out_layers[self.layer_index](att_out)

        att_out = x + att_out * self.gammas[self.layer_index]

        self.layer_index += 1

        return att_out

if __name__ == "__main__":
    channels = [3,64,64,128,256]

    ad = AttentiveDensenet(channels, 32, 32, 4)

    ad.reset()

    for ch in channels: 

        if ch == 3:
            h = 32
        elif ch == 64:
            h = 16
        else:
            h = 4

        layerx = torch.randn(64, ch, h, h)

        inp_out = ad(layerx)









