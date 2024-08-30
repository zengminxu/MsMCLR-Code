import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph
from net.att_drop import Simam_Drop 
from net.m2a import M2A


class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks."""

    def __init__(self, in_channels, hidden_channels, hidden_dim, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, kernel_size, 1, **kwargs), 
        ))
        self.fc = nn.Linear(hidden_dim, num_class)
     
        self.dropout = Simam_Drop(hidden_dim) 
        temporal_module = 'motion+attn'
        this_segment = 8
        n_div = 8
        network_blocks = [(temporal_module, 'b t (c h w)'),]
        self.m2a = M2A(hidden_dim,this_segment, n_div, network_blocks)

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        

    def forward(self, x, drop=False):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
       
        if drop:
            # y = self.dropout(x)
            x1 = self.m2a(x)

            yjp = divide_s(x, part_group) 
            _, Cp, Tp, Vp = yjp.shape
            assert Vp == 10
            yjb = divide_s(x, body_group)  
            _, Cb, Tb, Vb = yjb.shape
            assert Vb == 5

            yjp_att = self.m2a(yjp) 
            yjb_att = self.m2a(yjb) 
          

            
            # global pooling
            x1 = F.avg_pool2d(x1, x1.size()[2:]) 
            x1 = x1.view(N, M, -1).mean(dim=1)

            # prediction
            x1 = self.fc(x1) 
            x1 = x1.view(x1.size(0), -1) 
            # global pooling
            yjp_att = F.avg_pool2d(yjp_att, yjp_att.size()[2:]) 
            yjb_att = F.avg_pool2d(yjb_att, yjb_att.size()[2:])
            # y = torch.div(yjp_att+yjb_att,2)
            yjp_att = yjp_att.view(N, M, -1).mean(dim=1) 
            yjb_att = yjb_att.view(N, M, -1).mean(dim=1)
            # prediction
            yjp_att = self.fc(yjp_att) 
            yjb_att = self.fc(yjb_att)
            yjp_att = yjp_att.view(yjp_att.size(0), -1) 
            yjb_att = yjb_att.view(yjb_att.size(0), -1)
            return x1, yjp_att,yjb_att  # q_extreme,q_extreme_drop
        else:
            # global pooling
            # x = self.m2a(x)
            x = F.avg_pool2d(x, x.size()[2:])  
            x = x.view(N, M, -1).mean(dim=1)

            # prediction
            x = self.fc(x) 
            x = x.view(x.size(0), -1) 

            return x

def divide_s(x, parts):
    B, C, T, V = x.size() 
    part_skeleton = [torch.Tensor(part).long() for part in parts] 
    
    x_sum = None
    for pa in part_skeleton:
        xx = None
        for p in pa:
            if xx is None:
                xx = x[:, :, :, p].unsqueeze(-1)
                #print(xx.size())
            else:
                xx = torch.cat((xx, x[:, :, :, p].unsqueeze(-1)), dim=-1)
        xx = xx.mean(-1) # B, C, T, M -> B, C, T   [M:the number of joint in this part]
        #print(xx.size())
        # pa_len = len(pa)
        # max_len = len(part_skeleton[0])
        # if pa_len < max_len:
        #     for _ in range(pa_len, max_len):
        #         xx = torch.cat((xx, torch.zeros_like(x[:, :, :, 0].unsqueeze(-1))), dim=-1)
        if x_sum is None:
            x_sum = xx.unsqueeze(-1)
            #print(x_sum.size())
        else:
            x_sum = torch.cat((x_sum, xx.unsqueeze(-1)), dim=-1)
    # x_sum (N, C, T, P)  [P:the number of part(5)]
    # assert x_sum.size() == (B, C, T, 5), "part_spatial divide error"
    return x_sum

body_group = [
    np.array([5, 6, 7, 8, 22, 23]) - 1,     # left_arm
    np.array([9, 10, 11, 12, 24, 25]) - 1,  # right_arm
    np.array([13, 14, 15, 16]) - 1,         # left_leg
    np.array([17, 18, 19, 20]) - 1,         # right_leg
    np.array([1, 2, 3, 4, 21]) - 1          # torso
]

part_group = [
    np.array([1, 2, 21]) - 1, #spine
    np.array([3, 4]) - 1, #head 
    np.array([5, 6, 7]) - 1, #left arm
    np.array([8, 22, 23]) - 1, #left hand
    np.array([9, 10, 11]) - 1, #right arm
    np.array([12, 24, 25]) - 1, #right hand
    np.array([13, 14]) - 1, #left leg
    np.array([15, 16]) - 1, #left foot
    np.array([17, 18]) - 1, #right leg
    np.array([19, 20]) - 1  #right foot
]

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A