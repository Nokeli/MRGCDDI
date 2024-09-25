import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,RGCNConv,RGATConv,HGTConv
from torch_geometric.nn import global_mean_pool
import numpy as np
import csv
import os
import random


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(1, deterministic=True)
def reset_parameters(w):
    stdv = 1. / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(32, 32, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(in_channels, 2 * out_channels)
        self.linear2 = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x


class MRCGNN_sub(nn.Module):
    def __init__(self, feature, hidden1, hidden2, decoder1, dropout,zhongzi,train_data):
        super(MRCGNN_sub, self).__init__()

        self.lin = nn.Linear(feature, feature)
        # self.encoder_o1 = RGCNConv(feature, hidden1,num_relations=65)
        # self.encoder_o2 = RGCNConv(hidden1 , hidden2,num_relations=65)
        self.encoder_o1 = HGTConv(feature,hidden1,train_data.metadata(),heads=2)
        self.encoder_o2 = HGTConv(hidden1, hidden2, train_data.metadata(), heads=2)
        # self.encoder_o1 = RGATConv(feature, hidden1, num_relations=65)
        # self.encoder_o2 = RGATConv(hidden1, hidden2, num_relations=65)
        self.dropout = dropout
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
#         self.mlp = nn.ModuleList([nn.Linear(448, 256),
#                                   nn.ELU(),
#                                   nn.Dropout(p=0.1),
#                                   nn.Linear(256, 128),
#                                   nn.ELU(),
#                                   nn.Dropout(p=0.1),
#                                   nn.Linear(128, 65)
#                                   ])

#     def MLP(self, vectors, layer):
#             for i in range(layer):
#                 vectors = self.mlp[i](vectors)

#         return vectors
    def forward(self, data_o):


        #RGCN for DDI event graph and two corrupted graph
        # x_o, adj ,e_type= data_o.x, data_o.edge_index,data_o.edge_type
        # e_type=torch.tensor(e_type,dtype=torch.int64)
        # e_type=torch.tensor(e_type,dtype=torch.int64).cuda()
        # print(data_o.x_dict)
        x_dict = {
            node_type: self.lin(x).relu_()
            for node_type, x in data_o.x_dict.items()
        }
        x1_o = self.encoder_o1(x_dict, data_o.edge_index_dict)
        # x1_o = F.dropout(x1_o, self.dropout, training=self.training)
        x1_os = x1_o
        x2_o = self.encoder_o2(x1_os, data_o.edge_index_dict)
        x2_os = x2_o['drug']


#         x1_o_a = F.relu(self.encoder_o1(x_a, adj,e_type))
#         x1_o_a = F.dropout(x1_o_a, self.dropout, training=self.training)
#         x1_os_a = x1_o_a
#         x2_o_a = self.encoder_o2(x1_os_a, adj,e_type)
#         x2_os_a = x2_o_a

#         x1_o_a_a = F.relu(self.encoder_o1(x_o, adj, e_type1))
#         x1_o_a_a = F.dropout(x1_o_a_a, self.dropout, training=self.training)
#         x1_os_a_a= x1_o_a_a
#         x2_o_a_a = self.encoder_o2(x1_os_a_a, adj, e_type1)
#         x2_os_a_a = x2_o_a_a

        # readout
        h_os = self.read(x2_os)
        #h_os = x2_os
        #h_os = self.sigm(h_os)



#         # contrastive learning#
#         ret_os = self.disc(h_os, x2_os, x2_os_a)
#         ret_os_a =self.disc(h_os, x2_os, x2_os_a_a) #self.disc(h_os, x2_os, x2_os_a_a)

#         a = [int(i) for i in list(idx[0])]
#         b = [int(i) for i in list(idx[1])]

#         aa = torch.tensor(a, dtype=torch.long)
#         bb = torch.tensor(b, dtype=torch.long)
#         #layer attnetion
#         final = torch.cat((self.attt[0] * x1_o, self.attt[1] * x2_o), dim=1)

#         entity1 = final[aa]
#         entity2 = final[bb]

#         #skip connection
#         entity1_res = self.features1[aa].to('cuda')
#         entity2_res = self.features1[bb].to('cuda')
#         entity1 = torch.cat((entity1, entity1_res), dim=1)
#         entity2 = torch.cat((entity2, entity2_res), dim=1)

#         concatenate = torch.cat((entity1, entity2), dim=1)
#         feature = self.MLP(concatenate, 7)
#         log = feature

        # return log, ret_os, ret_os_a, x2_os
        return h_os,x1_o['drug'],x2_o['drug']

class MRCGNN(nn.Module):
    def __init__(self, feature, hidden1, hidden2, decoder1, dropout,zhongzi,train_data1):
        super(MRCGNN, self).__init__()
        self.gnn = MRCGNN_sub(feature, hidden1, hidden2, decoder1, dropout,zhongzi,train_data1)
        self.mlp = nn.ModuleList([nn.Linear(448, 256),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(256, 128),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(128, 65)
                                  ])
        
        self.attt = torch.zeros(2)
        self.attt[0] = 0.5
        self.attt[1] = 0.5
        self.attt = nn.Parameter(self.attt)
        self.disc = Discriminator(hidden2 * 2)
        self.projection_head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(inplace=True), nn.Linear(32, 32))
        drug_list = []
        with open('data/drug_listxiao.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                drug_list.append(row[0])
        features = np.load('trimnet/drug_emb_trimnet' + str(zhongzi) + '.npy')

        ids = np.load('trimnet/drug_idsxiao.npy')
        ids = ids.tolist()
        features1 = []
        for i in range(len(drug_list)):
            features1.append(features[ids.index(drug_list[i])])
        features1 = np.array(features1)

        self.features1 = torch.from_numpy(features1).cuda() #
    def MLP(self, vectors, layer):
        for i in range(layer):
            vectors = self.mlp[i](vectors)
        return vectors

    def forward_cl(self, data_o):
        x,x1_o,x2_o = self.gnn(data_o)
        # x = self.pool(x, batch)
        x = self.projection_head(x)
        #         a = [int(i) for i in list(idx[0])]
        return x,x1_o,x2_o
    def forward_classifiation(self,x1_o,x2_o,idx):
        a = [int(i) for i in list(idx[0])]
        b = [int(i) for i in list(idx[1])]

        aa = torch.tensor(a, dtype=torch.long)
        bb = torch.tensor(b, dtype=torch.long)
        # layer attnetion
        final = torch.cat((self.attt[0] * x1_o, self.attt[1] * x2_o), dim=1)

        entity1 = final[aa]
        entity2 = final[bb]

        # skip connection
        entity1_res = self.features1[aa].to('cuda')
        entity2_res = self.features1[bb].to('cuda')
        entity1 = torch.cat((entity1, entity1_res), dim=1)
        entity2 = torch.cat((entity2, entity2_res), dim=1)

        concatenate = torch.cat((entity1, entity2), dim=1)
        feature = self.MLP(concatenate, 7)
        log = feature
        return log

    def loss_cl(self, x1, x1_o_2,x2):
#         T = 0.1
#         batch_size,_= x1.size()
#         x1_abs = x1.norm(dim=1)
#         x2_abs = x2.norm(dim=1)
        
#         sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
#         sim_matrix = torch.exp(sim_matrix / T)
#         pos_sim = sim_matrix[range(batch_size), range(batch_size)]
#         loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
#         loss = - torch.log(loss).mean()
        ret_os = self.disc(x1, x1_o_2, x2)

        return ret_os