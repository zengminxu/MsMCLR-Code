import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math 
from einops import rearrange

# import sys 
# sys.path.append('./ops/')
# from sota import TEA, TAM, TDN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def self_attention(q, k, v):
    """query, key, value: (B, T, E): batch, temporal, embedding
    """
    dk = q.size(-1)  
    scores = torch.bmm(q, torch.transpose(k, 1, 2)) / math.sqrt(dk)  
    scores = F.softmax(scores, -1)  
    output = torch.bmm(scores, v)  

    return output, scores

# def compute_shift(x, type, amount=1):
#     # x = (batch, T, ...) x[32,8,32,13,25]
#     pad = torch.zeros_like(x).to(x.device)[:, :amount] 
#     if type == 'left':
#         xt2 = torch.cat([x.clone()[:, amount:], pad], 1) 
#     elif type == 'right':
#         xt2 = torch.cat([pad, x.clone()[:, :-amount]], 1)
#     xt2.to(x.device) 
#     return xt2

def compute_shift(x, type, amount=1):
    # x = (batch, T, ...) 
    pad = torch.zeros_like(x).to(x.device)[:, :,:,:amount] 
    if type == 'left':
        xt2 = torch.cat([x.clone()[:,:,:, amount:], pad], 3)
    elif type == 'right':
        xt2 = torch.cat([pad, x.clone()[:, :-amount]], 1)
    xt2.to(x.device) 
    return xt2

# def compute_shift(x, type, amount=1):
#     # x = (batch, T, ...) x[32,8,32,13,25]
#     pad1 = torch.zeros_like(x).to(x.device)[:, :,:,:amount]
#     pad2 = torch.zeros_like(x).to(x.device)[:, :,:,:,:amount]
#     if type == 'left':
#         xt2 = torch.cat([x.clone()[:,:,:, amount:], pad1], 3)  
#         xv3 = torch.cat([x.clone()[:,:,:,:, amount:], pad2], 4)
#     elif type == 'right':
#         xt2 = torch.cat([pad1, x.clone()[:,:,:, :-amount]], 1)
#         xv3 = torch.cat([pad2, x.clone()[:, :,:,:,:-amount]], 4)
#     xt2.to(x.device)  
#     xv3.to(x.device)
#     return xt2, xv3

class M2A(nn.Module):
    def __init__(self, in_channels, n_segment, n_div, blocks):  
        super().__init__()
        self.n_segment = n_segment
        self.rsize = in_channels // n_div

        def block2mod(block):
            name, attn_shape = block  # block=('motion+attn', 'b t (c h w)')  name = motion+attn  attn_shape='b t (c h w)'
            if name == 'motion': 
                m = Motion(n_segment, '(b t) c h w', 'b t c h w')
            # elif name == 'tdn':
            #     m = TDN(self.rsize, n_segment=n_segment)
            # elif name == 'tam':
            #     m = TAM(self.rsize, n_segment=n_segment)
            # elif name == 'tea':
            #     m = TEA(self.rsize, reduction=1, n_segment=n_segment)
            # elif name == 'attn': 
            #     m = Attention(n_segment, attn_shape)
            # elif name == 'patch_mattn': 
            #     m = SpatialPatchMotionAttention(n_segment, attn_shape, args.patch_size)
            # elif '+' in name: # combination of motion & attention modules (motion+attn)
            elif '+' in name:  # combination of motion & attention modules (motion+attn)
                mods = name.split('+')  # name='motion+attn'
                motion_m, attn_m = mods[0], mods[1]  # motion_m='motion' ,attn_m='attn'
                
                # get the motion module
                motion_mod = block2mod((motion_m, ''))  

                # get the attention module
                if attn_m == 'attn':
                    m = CustomMotionAttention(n_segment, attn_shape, motion_mod, self.rsize)
                # elif attn_m == 'tam':
                #     m = CustomMotionTAM(n_segment, attn_shape, self.rsize, motion_mod)
                else: 
                    raise NotImplementedError
            else: 
                raise NotImplementedError
            return m 
        
        self.modules_list = nn.ModuleList([])
        for block in blocks:  # blocks=[('motion+attn', 'b t (c h w)')]
            m = block2mod(block)
            self.modules_list.append(m)

        self.conv_down = nn.Conv2d(in_channels, self.rsize, 1)
        self.conv_up = nn.Conv2d(self.rsize, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        shortcut = x  
        x = self.conv_down(x)  
        
        for m in self.modules_list:
            x = m(x)

        x = self.conv_up(x) 
        x = self.sigmoid(x)  

        x = x * shortcut + shortcut   
        return x
    
class CustomMotionAttention(nn.Module):
    def __init__(self, n_segment, einops_to, motion_mod, in_channels):
        super().__init__()
        self.name = 'attn'
        self.n_segment = n_segment
        self.einops_from = '(b t) c h w'
        self.einops_to = einops_to
        self.motion_mod = motion_mod
        self.attn_activations = None
        self.in_channels = in_channels

    def forward(self, x):
        bt, c, h, w = x.size()

        # layer norm  # x[256,32,13,25] einops_from=bt, c, h, w  einops_to=b,t,chw
        xn = rearrange(x, f'{self.einops_from} -> {self.einops_to}', t=self.n_segment, c=c, h=h, w=w)  
        xn = F.layer_norm(xn, [xn.size(-1)])  
        xn = rearrange(xn, f'{self.einops_to} -> {self.einops_from}', t=self.n_segment, c=c, h=h, w=w) 

        # motion 
        xn = self.motion_mod(xn)  

        # attention -- don't use Attention class bc of layernorm
        xn = rearrange(xn, f'{self.einops_from} -> {self.einops_to}', t=self.n_segment, c=c, h=h, w=w)  
        x_attn, _ = self_attention(xn, xn, xn) 
        x_attn = rearrange(x_attn, f'{self.einops_to} -> {self.einops_from}', t=self.n_segment, c=c, h=h, w=w)
       
        self.attn_activations = x_attn

        # skip connection
        x = x_attn + x 

        return x 
    
class Motion(nn.Module):
    def __init__(self, n_segment, einops_from, einops_to):
        super().__init__()
        self.n_segment = n_segment
        self.einops_from, self.einops_to = einops_from, einops_to  #　einops_from　'(b t) c h w'　　einops_to　'b t c h w'
        self.name = 'motion'
        #  self.conv_down = nn.Conv2d(x.size[1], (x.size[1])/2, 1)
    def forward(self, x):  
        x = rearrange(x, f'{self.einops_from} -> {self.einops_to}', t=self.n_segment) 
        x = compute_shift(x, type='left') - x 
        x = rearrange(x, f'{self.einops_to} -> {self.einops_from}')  
        
        return x 
    
    # def forward(self, x):  # x[256,32,13,25] N*M, C, T, V
    #     x = rearrange(x, f'{self.einops_from} -> {self.einops_to}', t=self.n_segment) 
    #     xt2, xv3 =compute_shift(x, type='left')
    #     x2 = xt2 - x  
    #     x3 = xv3 - x
    #     # x = torch.stack((x2,x3), 0) 
    #     # x = torch.squeeze(x) 
    #     # x = x2 + x3 
    #     x = torch.cat((x2,x3), 1) 
    #     x = conv_down(x)
    #     x = rearrange(x, f'{self.einops_to} -> {self.einops_from}')
    #     return x 
    
if __name__ == '__main__':
    import numpy as np

    x = np.random.random([256,256,13,25])
    x = torch.tensor(x,dtype=torch.float32)
    temporal_module = 'motion+attn'
    network_blocks = [
            (temporal_module, 'b t (c h w)'),
        ]
    in_channels = 256
    this_segment = 8
    n_div = 8
    mod = M2A(in_channels, this_segment, n_div, network_blocks) 
    
    output = mod(x)
    print(output)
