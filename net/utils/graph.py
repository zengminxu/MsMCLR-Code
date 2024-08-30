import numpy as np

class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop # 1
        self.dilation = dilation  # 1

        self.get_edge(layout)  # layout ='ntu-rgb+d'
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)  #(25,25) 0,1,inf
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]  # [(0,0),(1,1),...,(24,24)]自连矩阵
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]  # 依照骨架图的物理连接矩阵
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]  # 让物理连接矩阵从0开始 [(0,1),(1,20),...(24,11)]
            self.edge = self_link + neighbor_link # 自连矩阵和物理连接矩阵合并 [(0,0),(1,1),...,(24,24),(0,1),(1,20),...(24,11)]
            self.center = 21 - 1 # 中心关节为21（从0开始的话要减1）
        elif layout == 'nw-ucla':
            self.num_node = 20
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 3), (4, 3), (5, 3), (6, 5), (7, 6),
                              (8, 7), (9, 3), (10, 9), (11, 10), (12, 11), (13, 1),
                              (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                              (20, 19)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 3 - 1
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation) # range(0,2)
        adjacency = np.zeros((self.num_node, self.num_node)) # (25,25) 全0
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)  # 初始化邻接矩阵

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))  # A (1,25,25)
            A[0] = normalize_adjacency  # A (1,25,25)
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1  # 邻接矩阵

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf # 一个25*25的 全是 inf 的矩阵
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)] # np.linalg.matrix_power(A, d)快速求图中距离为d的节点  d=0 1
    # transfer_mat len=2 里面是两个25*25的矩阵，一个是d=0的 一个是d=1的
    arrive_mat = (np.stack(transfer_mat) > 0)  # （2,25,25） 根据>0 把其转换为 true false 矩阵
    for d in range(max_hop, -1, -1):  # 倒序。 range有三个参数，从左到右依次是: 计数从 start 开始（默认是0）、计数到 stop 结束（不包括 stop）、步长（默认为1）。
        hop_dis[arrive_mat[d]] = d
    return hop_dis  # 变成0 1 inf矩阵 (25,25)


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


if __name__ == '__main__':
    # import tools
    g = Graph(layout='ntu-rgb+d').A  # g(1,25,25)
    print(g)

# from audioop import reverse
# import sys
# import numpy as np

# sys.path.extend(['../'])
# # from graph import tools


# def get_hierarchical_graph(num_node, edges):
#     A = []
#     for edge in edges:  # edge为edges位置0/1/.../5处的部分
#         A.append(get_graph(num_node, edge))
#     # A1 = np.stack(A) # 6个 (3, 25, 25)堆叠在一起  成为最终的邻接矩阵
#     # A = np.array(A[0])
#     A = A[0].tolist()
#     return A   # (6,3,25,25)  # 有

# def get_graph(num_node, edges):

#     I = edge2mat(edges[0], num_node) # 自连矩阵 主对角线为1 的25*25 矩阵

#     Forward = normalize_digraph(edge2mat(edges[1], num_node))  # 正向 edges[1]长度为150  取A[j, i]这150个对应的位置为1 得到 (25,25)再进行初始化
#     Reverse = normalize_digraph(edge2mat(edges[2], num_node))  # 反向
#     A = np.stack((I, Forward, Reverse))  # 三个矩阵堆叠在一起 
#     return A # (3, 25, 25)  # 有

# def edge2mat(link, num_node):  # link=[(0, 0), (1, 1), (12, 12), (16, 16)]  num_node=25
#     A = np.zeros((num_node, num_node))
#     for i, j in link:
#         A[j, i] = 1
#     return A # 有 

# def normalize_digraph(A):
#     Dl = np.sum(A, 0)  # Dl(25,)
#     h, w = A.shape # 25 25
#     Dn = np.zeros((w, w))  # Dn(25,25)
#     for i in range(w):
#         if Dl[i] > 0:
#             Dn[i, i] = Dl[i] ** (-1)  # Dn 对角矩阵
#     AD = np.dot(A, Dn) # 归一化
#     return AD # 有

# def get_edgeset(dataset='NTU', CoM=21):
#     groups = get_groups(dataset=dataset, CoM=CoM)
    
#     for i, group in enumerate(groups):
#         group = [i - 1 for i in group]
#         groups[i] = group # 让节点从0开始 groups[[0],[1, 12, 16],[13, 17, 20]，[2, 4, 8, 14, 18]，[3, 5, 9, 15, 19]，[6, 10]，[7, 11, 21, 22, 23, 24]]

#     identity = []  # (i,i)
#     forward_hierarchy = []  # (i,j)
#     reverse_hierarchy = []  # (j,i)

#     for i in range(len(groups) - 1):  # len(groups)=2 2-1=1 range(len(groups) - 1)=0 1
#         self_link = groups[i] + groups[i + 1] # 如i=0时， self_link=[0,1, 12, 16]
#         self_link = [(i, i) for i in self_link]  # self_link=[(0,0),(1,1), (12,12), (16,16)]
#         identity.append(self_link)  # identity的位置0处为[(0,0),(1,1), (12,12), (16,16)] 位置1处为[(1,1), (12,12), (16,16),(13,13),(17,17),(20,20)] 
#         forward_g = []
#         for j in groups[i]:
#             for k in groups[i + 1]:  #  如i=0时 j=0  k=1 12 16
#                 forward_g.append((j, k))  # forward_g=[(0,1),(0,12),(0,16)]
#         # print(forward_g)
#         PC=[(0,1),(0,16),(1,20),(20,2),(20,8),(2,3),(8,9),(9,10),(10,11),(11,24),(24,23),(16,17),(17,18),(18,19)]
#         for h in PC:
#             forward_g.append(h)
#         # 不行 要用循环
#         forward_hierarchy.append(forward_g)  # forward_hierarchy的位置0处为[(0,1),(0,12),(0,16)]
        
#         reverse_g = []
#         for j in groups[-1 - i]:  # 如i=0时groups[-1 - i]= [7, 11, 21, 22, 23, 24]  j= 7 11 21 ... 24
#             for k in groups[-2 - i]:  # [6, 10] k=6 10
#                 reverse_g.append((j, k))  # reverse_g=[(7,6),(7,10),(11,6),(11,10),...,(24,6),(24,10)]
#         PC_reverse=[(4,5),(5,6),(6,7),(7,22),(22,21),(12,13),(13,14),(14,15)]
#         for g in PC_reverse:
#             reverse_g.append(g)   
#         reverse_hierarchy.append(reverse_g)  # reverse_hierarchy的位置0处为[(7,6),(7,10),(11,6),(11,10),...,(24,6),(24,10)]

#     edges = []
#     for i in range(len(groups) - 1): # range(len(groups) - 1)=0 1    i=0时，edges的位置0处为[[(0,0),(1,1), (12,12), (16,16)],[(0,1),(0,12),(0,16)],[(1,0),(12,0),(16,0)]]
#         edges.append([identity[i], forward_hierarchy[i], reverse_hierarchy[-1 - i]])
#         # edges 长度为1 里面identity 25  forward_hierarchy 150  和reverse_hierarchy 150
#     return edges  # 长度为6 每个位置下有identity forward_hierarchy 和reverse_hierarchy 三个  # 有

# def get_groups(dataset='NTU', CoM=21):
#     groups  =[]
    
#     if dataset == 'NTU':
#         if CoM == 2:
#             groups.append([2])  # HK层次节点集
#             groups.append([1, 21])
#             groups.append([13, 17, 3, 5, 9])
#             groups.append([14, 18, 4, 6, 10])
#             groups.append([15, 19, 7, 11])
#             groups.append([16, 20, 8, 12])
#             groups.append([22, 23, 24, 25])

#         ## Center of mass : 21
#         elif CoM == 21:
#             # groups.append([21])
#             # groups.append([2, 3, 5, 9])
#             # groups.append([4, 6, 10, 1])
#             # groups.append([7, 11, 13, 17])
#             # groups.append([8, 12, 14, 18])
#             # groups.append([22, 23, 24, 25, 15, 19])
#             # groups.append([16, 20])
#             groups.append([1,2,21,3,4,9,10,11,12,24,25,17,18,19,20])
#             groups.append([5,6,7,8,22,23,13,14,15,16])

#         ## Center of Mass : 1
#         elif CoM == 1:
#             groups.append([1])
#             groups.append([2, 13, 17])
#             groups.append([14, 18, 21])
#             groups.append([3, 5, 9, 15, 19])
#             groups.append([4, 6, 10, 16, 20])
#             groups.append([7, 11])
#             groups.append([8, 12, 22, 23, 24, 25])
#             # groups[[1],[2, 13, 17],[14, 18, 21]，[3, 5, 9, 15, 19]，[4, 6, 10, 16, 20]，[7, 11]，[8, 12, 22, 23, 24, 25]]
#         else:
#             raise ValueError()
        
#     return groups  # 有





# num_node = 25

# class Graph:
#     def __init__(self, CoM=21, labeling_mode='spatial'):
#         self.num_node = num_node  # 25
#         self.CoM = CoM  # 21
#         self.A = self.get_adjacency_matrix(labeling_mode)
        

#     def get_adjacency_matrix(self, labeling_mode=None):
#         if labeling_mode is None:
#             return self.A
#         if labeling_mode == 'spatial':
#             A = get_hierarchical_graph(num_node, get_edgeset(dataset='NTU', CoM=self.CoM)) # L, 3, 25, 25
#         else:
#             raise ValueError()
#         return A  # , self.CoM  # A[6,3,25,25]

# if __name__ == '__main__':
#     # import tools
#     g = Graph(CoM=21).A
#     print(g)