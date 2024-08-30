import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class
from net.m2a import M2A
import numpy as np


class AimCLR(nn.Module):
    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain

        # temporal_module = 'motion+attn'
        # this_segment = 8
        # n_div = 8
        # network_blocks = [(temporal_module, 'b t (c h w)'),]
        # self.m2a = M2A(hidden_dim,this_segment, n_div, network_blocks)

        if not self.pretrain:
            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature

            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_k = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)

            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_k.fc)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

            # create the queue
            self.register_buffer("queue", torch.randn(feature_dim, queue_size))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        gpu_index = keys.device.index
        self.queue[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0  # for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K

    def forward(self, im_q_extreme, im_q, im_k=None, nnm=False, topk=1):
        """
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """

        if nnm:
            return self.nearest_neighbors_mining(im_q, im_k, im_q_extreme, topk)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # q = self.m2a(q)
        # _, q = self.encoder_q(im_q, drop=True)  # NxC im_q[128, 3, 50, 25, 2] 

        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        q_extreme, q_extreme_drop, q_extreme_drop_b= self.encoder_q(im_q_extreme, drop=True)  # NxC  
        # q_extreme = self.encoder_q(im_q_extreme)  #, q_extreme_drop

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)
        q_extreme_drop_b = F.normalize(q_extreme_drop_b, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        # q1 = get_group(q)
        # k1 = get_group(k)
        # group_contrastive_loss = compute_group_contrastive_loss(q1,k1)

        # Compute logits of normally augmented query using Einstein sum
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()  

        # Compute logits_e of extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        # apply temperature
        logits_e /= self.T
        logits_e = torch.softmax(logits_e, dim=1)

        # Compute logits_ed of dropped extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        # apply temperature
        logits_ed /= self.T
        logits_ed = torch.softmax(logits_ed, dim=1)
        # b
        # positive logits: Nx1
        l_pos_ed_b = torch.einsum('nc,nc->n', [q_extreme_drop_b, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_ed_b = torch.einsum('nc,ck->nk', [q_extreme_drop_b, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_ed_b = torch.cat([l_pos_ed_b, l_neg_ed_b], dim=1)
        # apply temperature
        logits_ed_b /= self.T
        logits_ed_b = torch.softmax(logits_ed_b, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()  
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()
                
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, logits_e, logits_ed,logits_ed_b, labels_ddm #,group_contrastive_loss

    def nearest_neighbors_mining(self, im_q, im_k, im_q_extreme, topk=1):

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # q = self.m2a(q)
        # _, q = self.encoder_q(im_q, drop=True)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        q_extreme, q_extreme_drop,q_extreme_drop_b = self.encoder_q(im_q_extreme, drop=True)  # NxC
        # q_extreme = self.encoder_q(im_q_extreme)  #, q_extreme_drop

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)
        q_extreme_drop_b = F.normalize(q_extreme_drop_b, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        # q1 = get_group(q)
        # k1 = get_group(k)
        # group_contrastive_loss = compute_group_contrastive_loss(q1,k1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])

        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])

        l_pos_ed_b = torch.einsum('nc,nc->n', [q_extreme_drop_b, k]).unsqueeze(-1)
        l_neg_ed_b = torch.einsum('nc,ck->nk', [q_extreme_drop_b, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        logits_ed_b = torch.cat([l_pos_ed_b, l_neg_ed_b], dim=1)

        logits /= self.T
        logits_e /= self.T
        logits_ed /= self.T
        logits_ed_b /= self.T

        logits_e = torch.softmax(logits_e, dim=1)
        logits_ed = torch.softmax(logits_ed, dim=1)
        logits_ed_b = torch.softmax(logits_ed_b, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()

        # nearest neighbors mining to expand the positive set
        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_ed, topk, dim=1)
        _, topkdix_ed_b = torch.topk(l_neg_ed_b, topk, dim=1)

        topk_onehot = torch.zeros_like(l_neg)

        topk_onehot.scatter_(1, topkdix, 1)
        topk_onehot.scatter_(1, topkdix_e, 1)
        topk_onehot.scatter_(1, topkdix_ed, 1)
        topk_onehot.scatter_(1, topkdix_ed_b, 1)

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)

        self._dequeue_and_enqueue(k)

        return logits, pos_mask, logits_e, logits_ed,logits_ed_b, labels_ddm #,group_contrastive_loss



# def get_group(output):
#     logits = torch.softmax(output, dim=-1)
#     _ , target = torch.max(logits, dim=-1)
#     groups ={}
#     for x,y in zip(target, logits):
#         # x = x.cpu().numpy()#.astype()
#         group = groups.get(x.item(),[])
#         group.append(y)
#         groups[x.item()]= group
#     return groups

# def compute_group_contrastive_loss(grp_dict_un,grp_dict_lab):
#     loss = []
#     l_fast =[]
#     l_slow =[]
#     for key in grp_dict_un.keys():
#         if key in grp_dict_lab:
#             l_fast.append(torch.stack(grp_dict_un[key]).mean(dim=0))
#             l_slow.append(torch.stack(grp_dict_lab[key]).mean(dim=0))
#     if len(l_fast) > 0:
#         l_fast = torch.stack(l_fast)
#         l_slow = torch.stack(l_slow)
#         loss = simclr_loss(l_fast,l_slow)
#         loss = max(torch.tensor(0.000).cuda(),loss)
#     else:
#         loss= torch.tensor(0.0).cuda()
#     return loss

# Temperature_group = 0.5

# def simclr_loss(output_fast,output_slow,normalize=True):
#     out = torch.cat((output_fast, output_slow), dim=0)
#     sim_mat = torch.mm(out, torch.transpose(out,0,1))
#     if normalize:
#         sim_mat_denom = torch.mm(torch.norm(out, dim=1).unsqueeze(1), torch.norm(out, dim=1).unsqueeze(1).t())
#         sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)
#     sim_mat = torch.exp(sim_mat / Temperature_group)
#     if normalize:
#         sim_mat_denom = torch.norm(output_fast, dim=1) * torch.norm(output_slow, dim=1)
#         sim_match = torch.exp(torch.sum(output_fast * output_slow, dim=-1) / sim_mat_denom / Temperature_group)
#     else:
#         sim_match = torch.exp(torch.sum(output_fast * output_slow, dim=-1) / Temperature_group)
#     sim_match = torch.cat((sim_match, sim_match), dim=0)
#     norm_sum = torch.exp(torch.ones(out.size(0)) / Temperature_group )
#     norm_sum = norm_sum.cuda()
#     loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))
#     return loss