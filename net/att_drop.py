import torch
from torch import nn

# # Spatial Drop
# class DropBlock_Ske(nn.Module):
#     def __init__(self, num_point=25, keep_prob=0.9):
#         super(DropBlock_Ske, self).__init__()
#         self.keep_prob = keep_prob  # keep_margin --control the importance margin to drop.
#         self.num_point = num_point

#     def forward(self, input, mask):  # n,c,t,v
#         n, c, t, v = input.size()
#         mask[mask >= self.keep_prob] = 2.0  
#         mask[mask < self.keep_prob] = 1.0
#         mask[mask == 2.0] = 0.0  
#         
#         mask = mask.view(n, 1, 1, self.num_point)  # Ms 
#         
#         return input * mask * mask.numel() / mask.sum()  # X = X*Ms * count(Ms) / count_ones(Ms)

# # Temporal Drop
# class DropBlockT_1d(nn.Module):
#     def __init__(self, keep_prob=0.9):
#         super(DropBlockT_1d, self).__init__()
#         self.keep_prob = keep_prob

#     def forward(self, input, mask):  # mask(NM, C * V, T) 
#         n, c, t, v = input.size()
#         input1 = input.permute(0, 1, 3, 2).contiguous().view(n, c * v, t)  # （n,c,v,t）--> (n,c*v,t)
#         mask[mask >= self.keep_prob] = 2.0
#         mask[mask < self.keep_prob] = 1.0
#         mask[mask == 2.0] = 0.0

#         return (input1 * mask * mask.numel() / mask.sum()).view(n, c, v, t).permute(0, 1, 3, 2)  # (n,c*v,t)-->(n, c, t, v)
#         # X = X*Mt * count(Mt) / count_ones(Mt)
#         


# class Simam_Drop(nn.Module):
#     def __init__(self, num_point=25, e_lambda=1e-4, keep_prob=0.9):
#         super(Simam_Drop, self).__init__()
#         self.activaton = nn.Sigmoid()
#         self.e_lambda = e_lambda  

#         self.dropSke = DropBlock_Ske(num_point=num_point, keep_prob=keep_prob)
#         self.dropT_skip = DropBlockT_1d(keep_prob=keep_prob)

#     def forward(self, x):
#         NM, C, T, V = x.size()
#         num = V * T - 1  
#         x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)  
#         y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / num + self.e_lambda)) + 0.5  
#         att_map = self.activaton(y)  
#         att_map_s = att_map.mean(dim=[1, 2]) 
#         att_map_t = att_map.permute(0, 1, 3, 2).contiguous().view(NM, C * V, T)  
#         output = self.dropT_skip(self.dropSke(x, att_map_s), att_map_t)
#         return output

class Simam_Drop(nn.Module):
    def __init__(self, channel, reduct_ratio=2, bias=True, **kwargs):
        super(Simam_Drop, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.Hardswish(),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)  
        x_v = x.mean(2, keepdims=True).transpose(2, 3)  
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2)) 
        x_t, x_v = torch.split(x_att, [T, V], dim=2)
        x_t_att = self.conv_t(x_t).sigmoid()
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_att = x_t_att * x_v_att
        return x_att


if __name__ == '__main__':
    NM, C, T, V = 256, 16, 13, 25
    x = torch.randn((NM, C, T, V))
    # drop_sk = Simam_Drop(num_point=25, keep_prob=0.9)
    drop_sk = Simam_Drop(16,2,True)
    w = drop_sk(x)
    print(w.shape)
