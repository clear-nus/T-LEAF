import sys
sys.path.insert(1, '../../')
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Sequential,Linear,ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter
from torch_geometric.nn import GATConv, AGNNConv, GraphConv, ARMAConv, SGConv, SAGEConv, ChebConv
# from Util import cat_mean
import random



def get_neighbor_list(x, edge_index, bi_direction = False):
    ret = [[] for i in range(len(x))]
    for i in range(edge_index.shape[1]):
        ret[int(edge_index[0][i])].append(int(edge_index[1][i]))
        if bi_direction and (int(edge_index[1][i]) != int(edge_index[0][i])):
            ret[int(edge_index[1][i])].append(int(edge_index[0][i]))
    return ret

def sample_path(neighbor_list, num_path = 10, max_length = 4, from_start_node = False):
    ret = []
    for i in range(num_path):
        if from_start_node:
            cur = 1
        else:
            cur = random.choice(range(len(neighbor_list)))
        ret.append([cur])
        for j in range(max_length-1):
            if neighbor_list[cur] == []:
                break
            else:
                cur = random.choice(neighbor_list[cur])
                ret[-1].append(cur)
    return ret

class Node_only_random_agg(torch.nn.Module): # set the learning rate to 0.001
    def __init__(self, dropout = 0,INPUT_DIMENSION=100, HIDDEN_DIMENSION=512, OUTPUT_DIMENSION=200):
        super(Node_only_random_agg,self).__init__()
        # self.conv1 = GCNConv(INPUT_DIMENSION, HIDDEN_DIMENSION)
        # self.conv2 = GCNConv(HIDDEN_DIMENSION, HIDDEN_DIMENSION//2)
        # self.conv3 = GCNConv(HIDDEN_DIMENSION//2, OUTPUT_DIMENSION)

        self.conv1 = GCNConv(INPUT_DIMENSION, HIDDEN_DIMENSION*8)
        self.conv2 = GCNConv(HIDDEN_DIMENSION*8, HIDDEN_DIMENSION*4)
        self.conv3 = GCNConv(HIDDEN_DIMENSION*4, HIDDEN_DIMENSION*2)
        self.conv4 = GCNConv(HIDDEN_DIMENSION*2, HIDDEN_DIMENSION)
        self.conv5 = GCNConv(HIDDEN_DIMENSION, HIDDEN_DIMENSION//2)
        self.conv6 = GCNConv(HIDDEN_DIMENSION//2, OUTPUT_DIMENSION)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index,edge_attr = data.x, data.edge_index,data.edge_attr
        neighbor_list = get_neighbor_list(x, edge_index)

        out = F.dropout(F.relu(self.conv1(x,edge_index)),p=self.dropout,training=self.training) # conv
        out = F.dropout(F.relu(self.conv2(out,edge_index)),p=self.dropout,training=self.training) # conv
        out = F.dropout(F.relu(self.conv3(out,edge_index)),p=self.dropout,training=self.training) # conv
        out = F.dropout(F.relu(self.conv4(out,edge_index)),p=self.dropout,training=self.training) # conv
        out = F.dropout(F.relu(self.conv5(out,edge_index)),p=self.dropout,training=self.training) # conv
        out = F.dropout(F.relu(self.conv6(out,edge_index)),p=self.dropout,training=self.training) # conv
        # out = torch.mean(out,dim=0,keepdim=True)

        paths = sample_path(neighbor_list, num_path=10, max_length=5, from_start_node = True)
        out = torch.mean(torch.stack([out[paths[i][j]] for i in range(len(paths)) for j in range(len(paths[i]))]), dim=0, keepdim = True)

        # if torch.norm(out) < 0.01:
        #     print("norm ", torch.norm(out))
        # out=out/torch.max(torch.norm(out), min_clip)
        out = out/torch.norm(out)
        return out

class Node_only_random_agg_min_clip(torch.nn.Module): # set the learning rate to 0.001
    def __init__(self, device, dropout = 0,INPUT_DIMENSION=100, HIDDEN_DIMENSION=512, OUTPUT_DIMENSION=200):
        super(Node_only_random_agg_min_clip,self).__init__()
        # self.conv1 = GCNConv(INPUT_DIMENSION, HIDDEN_DIMENSION)
        # self.conv2 = GCNConv(HIDDEN_DIMENSION, HIDDEN_DIMENSION//2)
        # self.conv3 = GCNConv(HIDDEN_DIMENSION//2, OUTPUT_DIMENSION)

        self.conv1 = GCNConv(INPUT_DIMENSION, HIDDEN_DIMENSION*4)
        self.conv2 = GCNConv(HIDDEN_DIMENSION*4, HIDDEN_DIMENSION)
        # self.conv3 = GCNConv(HIDDEN_DIMENSION*4, HIDDEN_DIMENSION*2)
        # self.conv4 = GCNConv(HIDDEN_DIMENSION*2, HIDDEN_DIMENSION)
        self.conv5 = GCNConv(HIDDEN_DIMENSION, HIDDEN_DIMENSION//2)
        self.conv6 = GCNConv(HIDDEN_DIMENSION//2, OUTPUT_DIMENSION)
        self.dropout = dropout

        self.min_clip = torch.tensor(1e-3, dtype=torch.float32).to(device)
        self.device = device

    def forward(self, data):
        x, edge_index,edge_attr = data.x, data.edge_index,data.edge_attr
        neighbor_list = get_neighbor_list(x, edge_index)

        out = F.dropout(F.relu(self.conv1(x,edge_index)),p=self.dropout,training=self.training) # conv
        out = F.dropout(F.relu(self.conv2(out,edge_index)),p=self.dropout,training=self.training) # conv
        # out = F.dropout(F.relu(self.conv3(out,edge_index)),p=self.dropout,training=self.training) # conv
        # out = F.dropout(F.relu(self.conv4(out,edge_index)),p=self.dropout,training=self.training) # conv
        out = F.dropout(F.relu(self.conv5(out,edge_index)),p=self.dropout,training=self.training) # conv
        out = F.dropout(F.relu(self.conv6(out,edge_index)),p=self.dropout,training=self.training) # conv
        # out = torch.mean(out,dim=0,keepdim=True)

        paths = sample_path(neighbor_list, num_path=10, max_length=5, from_start_node = True)
        out = torch.mean(torch.stack([out[paths[i][j]] for i in range(len(paths)) for j in range(len(paths[i]))]), dim=0, keepdim = True)

        # if torch.norm(out) < 0.01:
        #     print("norm ", torch.norm(out))
        out=out/torch.max(torch.norm(out), self.min_clip)
        # out = out/torch.norm(out)
        return out


class Net(torch.nn.Module):
    def __init__(self, dropout=0,INPUT_DIMENSION=50, HIDDEN_DIMENSION=50, OUTPUT_DIMENSION=100):
        super(Net,self).__init__()
        self.conv1 = GCNConv(INPUT_DIMENSION, HIDDEN_DIMENSION)
        self.conv2 = GCNConv(HIDDEN_DIMENSION, HIDDEN_DIMENSION)
        self.conv3 = GCNConv(HIDDEN_DIMENSION, OUTPUT_DIMENSION)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index,edge_attr = data.x, data.edge_index,data.edge_attr

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training,p=self.dropout)
        x = self.conv3(x, edge_index)

        return torch.sum(x,0,True)/x.shape[0]

class EdgeNet_MLP(torch.nn.Module):
    def __init__(self, dropout = 0,INPUT_DIMENSION=50, HIDDEN_DIMENSION=50, OUTPUT_DIMENSION=100):
        super(EdgeNet_MLP,self).__init__()
        # MLP (perf is worse than conv, but reasonable okay)
        # self.conv1 = Sequential(Linear(INPUT_DIMENSION*2,HIDDEN_DIMENSION),ReLU(),Linear(HIDDEN_DIMENSION,HIDDEN_DIMENSION))
        # self.conv2 = Sequential(Linear(HIDDEN_DIMENSION+INPUT_DIMENSION, OUTPUT_DIMENSION),ReLU(),Linear(OUTPUT_DIMENSION,OUTPUT_DIMENSION))
        self.conv1 = Sequential(Linear(INPUT_DIMENSION*2,1024),ReLU(),Linear(1024,512),ReLU(),Linear(512,128),ReLU(),Linear(128,INPUT_DIMENSION))
        self.conv2 = Sequential(Linear(INPUT_DIMENSION+INPUT_DIMENSION, 1024), ReLU(),Linear(1024,512),ReLU(),Linear(512,256), ReLU(),
                                Linear(256,OUTPUT_DIMENSION))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index,edge_attr = data.x, data.edge_index,data.edge_attr

        row, col = edge_index

        out = torch.cat([x[row], edge_attr], dim=1)

        out = F.dropout(self.conv1(out), p=self.dropout, training=self.training) # MLP

        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        out = F.dropout(self.conv2(out), p=self.dropout, training=self.training) # MLP
        out = torch.mean(out,dim=0,keepdim=True)
        return out

class EdgeNet_Ori(torch.nn.Module):
    def __init__(self, device, dropout = 0,INPUT_DIMENSION=50, HIDDEN_DIMENSION=50, OUTPUT_DIMENSION=100):
        super(EdgeNet_Ori,self).__init__()
        # MLP (perf is worse than conv, but reasonable okay)
        # self.conv1 = Sequential(Linear(INPUT_DIMENSION*2,HIDDEN_DIMENSION),ReLU(),Linear(HIDDEN_DIMENSION,HIDDEN_DIMENSION))
        # self.conv2 = Sequential(Linear(HIDDEN_DIMENSION+INPUT_DIMENSION, OUTPUT_DIMENSION),ReLU(),Linear(OUTPUT_DIMENSION,OUTPUT_DIMENSION))

        self.conv1 = GCNConv(INPUT_DIMENSION*2, HIDDEN_DIMENSION)
        self.conv2 = GCNConv(HIDDEN_DIMENSION+INPUT_DIMENSION, OUTPUT_DIMENSION)
        self.dropout = dropout
        self.device = device

    def forward(self, data):
        x, edge_index,edge_attr = data.x, data.edge_index,data.edge_attr

        row, col = edge_index

        if x.shape[0]-1 == edge_attr.shape[0]:
            linear = True
        else:
            linear = False

        if linear:
            temp = torch.rand((1,50)).to(self.device)
            out = torch.cat([x,torch.cat([edge_attr, temp],dim=0)],dim=1)
        else:
            out = torch.cat([x[row], edge_attr], dim=1)

        out = F.dropout(F.relu(self.conv1(out,edge_index)),p=self.dropout,training=self.training) # conv
        # out = F.dropout(self.conv1(out), p=self.dropout, training=self.training) # MLP

        if linear:
            out = scatter_mean(out, torch.arange(out.shape[0]).to(self.device), dim=0, dim_size=x.size(0))
        else:
            out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        out = F.dropout(F.relu(self.conv2(out,edge_index)),p=self.dropout,training=self.training) # conv
        # out = F.dropout(self.conv2(out), p=self.dropout, training=self.training) # MLP
        out = torch.mean(out,dim=0,keepdim=True)

        return out

class Node_only(torch.nn.Module):
    def __init__(self, dropout = 0,INPUT_DIMENSION=100, HIDDEN_DIMENSION=256, OUTPUT_DIMENSION=200):
        super(Node_only,self).__init__()
        # self.conv1 = GCNConv(INPUT_DIMENSION, HIDDEN_DIMENSION)
        # self.conv2 = GCNConv(HIDDEN_DIMENSION, HIDDEN_DIMENSION//2)
        # self.conv3 = GCNConv(HIDDEN_DIMENSION//2, OUTPUT_DIMENSION)

        self.conv1 = GCNConv(INPUT_DIMENSION, HIDDEN_DIMENSION*2)
        self.conv2 = GCNConv(HIDDEN_DIMENSION*2, HIDDEN_DIMENSION)
        self.conv3 = GCNConv(HIDDEN_DIMENSION, HIDDEN_DIMENSION//2)
        self.conv4 = GCNConv(HIDDEN_DIMENSION//2, OUTPUT_DIMENSION)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index,edge_attr = data.x, data.edge_index,data.edge_attr

        out = F.dropout(F.relu(self.conv1(x,edge_index)),p=self.dropout,training=self.training) # conv
        out = F.dropout(F.relu(self.conv2(out,edge_index)),p=self.dropout,training=self.training) # conv
        out = F.dropout(F.relu(self.conv3(out,edge_index)),p=self.dropout,training=self.training) # conv
        out = F.dropout(F.relu(self.conv4(out,edge_index)),p=self.dropout,training=self.training) # conv
        out = torch.mean(out,dim=0,keepdim=True)
        return out



class Node_only_Type(torch.nn.Module):
    def __init__(self,device, dropout = 0,INPUT_DIMENSION=100, HIDDEN_DIMENSION=256, OUTPUT_DIMENSION=200):
        super(Node_only_Type,self).__init__()
        # self.conv1 = GCNConv(INPUT_DIMENSION, HIDDEN_DIMENSION)
        # self.conv2 = GCNConv(HIDDEN_DIMENSION, HIDDEN_DIMENSION//2)
        # self.conv3 = GCNConv(HIDDEN_DIMENSION//2, OUTPUT_DIMENSION)

        self.conv1 = GCNConv(INPUT_DIMENSION, HIDDEN_DIMENSION*2)
        self.conv2 = GCNConv(HIDDEN_DIMENSION*2, HIDDEN_DIMENSION)
        self.conv3 = GCNConv(HIDDEN_DIMENSION, HIDDEN_DIMENSION//2)
        self.conv4 = GCNConv(HIDDEN_DIMENSION//2, OUTPUT_DIMENSION)
        self.dropout = dropout
        self.device = device

    def forward(self, data):
        x, edge_index,edge_attr = data.x, data.edge_index,data.edge_attr

        out = F.dropout(F.relu(self.conv1(x,edge_index)),p=self.dropout,training=self.training) # conv
        out = F.dropout(F.relu(self.conv2(out,edge_index)),p=self.dropout,training=self.training) # conv
        out = F.dropout(F.relu(self.conv3(out,edge_index)),p=self.dropout,training=self.training) # conv
        out = F.dropout(F.relu(self.conv4(out,edge_index)),p=self.dropout,training=self.training) # conv
        out = self.cat_mean(x,out)
        return out

    def cat_mean(self,x, out):
        init, common, accept, edge = [], [], [], []
        node_feautre = torch.tensor(np.load("../../dataset/Synthetic/node.npy", allow_pickle=True), dtype=torch.float32).to(self.device)
        values=[]
        zeros=torch.zeros(out[0].shape,dtype=torch.float32).unsqueeze(0).to(self.device)
        for index in range(x.shape[0]):
            if torch.all(x[index].eq(node_feautre[0])):
                init.append(out[index].unsqueeze(0))
            elif torch.all(x[index].eq(node_feautre[1])):
                common.append(out[index].unsqueeze(0))
            elif torch.all(x[index].eq(node_feautre[2])):
                accept.append(out[index].unsqueeze(0))
            else:
                edge.append(out[index].unsqueeze(0))
        if len(init)!=0:
            init = torch.mean(torch.cat(init,dim=0), dim=0, keepdim=True)
            values.append(init)
        if len(accept)!=0:
            accept = torch.mean(torch.cat(accept, dim=0), dim=0, keepdim=True)
            values.append(accept)
        if len(edge) !=0:
            edge = torch.mean(torch.cat(edge, dim=0), dim=0, keepdim=True)
            values.append(edge)
        if len(common)!=0:
            common = torch.mean(torch.cat(common, dim=0), dim=0, keepdim=True)
            values.append(common)
        for i in range(len(values),4):
            values.append(zeros)

        output = torch.cat(values, dim=0).flatten().unsqueeze(0)
        return output

class Node_only_syntax(torch.nn.Module):
    def __init__(self, dropout = 0,INPUT_DIMENSION=50, HIDDEN_DIMENSION=256, OUTPUT_DIMENSION=200):
        super(Node_only_syntax,self).__init__()
        # self.conv1 = GCNConv(INPUT_DIMENSION, HIDDEN_DIMENSION)
        # self.conv2 = GCNConv(HIDDEN_DIMENSION, HIDDEN_DIMENSION//2)
        # self.conv3 = GCNConv(HIDDEN_DIMENSION//2, OUTPUT_DIMENSION)

        self.conv1 = GCNConv(INPUT_DIMENSION, HIDDEN_DIMENSION*2)
        self.conv2 = GCNConv(HIDDEN_DIMENSION*2, HIDDEN_DIMENSION)
        self.conv3 = GCNConv(HIDDEN_DIMENSION, HIDDEN_DIMENSION//2)
        self.conv4 = GCNConv(HIDDEN_DIMENSION//2, OUTPUT_DIMENSION)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index,edge_attr = data.x, data.edge_index,data.edge_attr

        out = F.dropout(F.relu(self.conv1(x,edge_index)),p=self.dropout,training=self.training) # conv
        out = F.dropout(F.relu(self.conv2(out,edge_index)),p=self.dropout,training=self.training) # conv
        out = F.dropout(F.relu(self.conv3(out,edge_index)),p=self.dropout,training=self.training) # conv
        out = F.dropout(F.relu(self.conv4(out,edge_index)),p=self.dropout,training=self.training) # conv
        out = torch.mean(out,dim=0,keepdim=True)
        return out

class Node_only_global(torch.nn.Module):
    def __init__(self, dropout = 0,INPUT_DIMENSION=100, HIDDEN_DIMENSION=256, OUTPUT_DIMENSION=200):
        super(Node_only_global,self).__init__()
        # self.conv1 = GCNConv(INPUT_DIMENSION, HIDDEN_DIMENSION)
        # self.conv2 = GCNConv(HIDDEN_DIMENSION, HIDDEN_DIMENSION//2)
        # self.conv3 = GCNConv(HIDDEN_DIMENSION//2, OUTPUT_DIMENSION)

        self.conv1 = GCNConv(INPUT_DIMENSION, HIDDEN_DIMENSION*2)
        self.conv2 = GCNConv(HIDDEN_DIMENSION*2, HIDDEN_DIMENSION)
        self.conv3 = GCNConv(HIDDEN_DIMENSION, HIDDEN_DIMENSION//2)
        self.conv4 = GCNConv(HIDDEN_DIMENSION//2, OUTPUT_DIMENSION)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index,edge_attr = data.x, data.edge_index,data.edge_attr

        out = F.dropout(F.relu(self.conv1(x,edge_index)),p=self.dropout,training=self.training) # conv
        out = F.dropout(F.relu(self.conv2(out,edge_index)),p=self.dropout,training=self.training) # conv
        out = F.dropout(F.relu(self.conv3(out,edge_index)),p=self.dropout,training=self.training) # conv
        out = F.dropout(F.relu(self.conv4(out,edge_index)),p=self.dropout,training=self.training) # conv
        # out = torch.mean(out,dim=0,keepdim=True)
        return out[-1].unsqueeze(0)

class MLP(torch.nn.Module):
    def __init__(self, ninput=2*200, nhidden=512, nclass=2, dropout=0):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(ninput, nhidden)
        self.fc2 = torch.nn.Linear(nhidden, 128)
        self.fc3 = torch.nn.Linear(128, nclass)
        self.dropout = dropout

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.dropout(F.relu(self.fc1(x)),p=self.dropout))))
class MLP_new(torch.nn.Module):
    def __init__(self, ninput=2*200, nhidden=1500, nclass=2, dropout=0):
        super(MLP_new, self).__init__()
        self.fc1 = torch.nn.Linear(ninput, nhidden)
        self.fc2 = torch.nn.Linear(nhidden, nclass)
        # self.fc3 = torch.nn.Linear(nhidden//2, nhidden//2)
        # self.fc4 = torch.nn.Linear(nhidden//2, nhidden//4)
        # self.fc5 = torch.nn.Linear(nhidden//4, nclass)
        # self.dropout = dropout
        self.relu = torch.nn.ReLU()
        # self.relu = torch.nn.Sigmoid()

    def forward(self, x):
        # out=self.fc3(F.relu(self.fc2(F.dropout(F.relu(self.fc1(x)),p=self.dropout))))
        # return self.fc5(F.dropout(F.relu(self.fc4(out)), p=self.dropout))
        # out = self.fc5(self.relu(self.fc4(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))))))
        out = self.fc2(self.relu(self.fc1(x)))
        return out

class MLP_type(torch.nn.Module):
    def __init__(self, ninput=2*800, nhidden=2048, nclass=2, dropout=0):
        super(MLP_type, self).__init__()
        self.fc1 = torch.nn.Linear(ninput, nhidden)
        self.fc2 = torch.nn.Linear(nhidden, 512)
        self.fc3 = torch.nn.Linear(512, 128)
        self.fc4 = torch.nn.Linear(128, nclass)
        self.dropout = dropout

    def forward(self, x):
        return self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.dropout(F.relu(self.fc1(x)),p=self.dropout))))))

class MLP_MSE(torch.nn.Module):
    def __init__(self, ninput=2*200, nhidden=512, nclass=1, dropout=0):
        super(MLP_MSE, self).__init__()
        self.fc1 = torch.nn.Linear(ninput, nhidden)
        self.fc2 = torch.nn.Linear(nhidden, 128)
        self.fc3 = torch.nn.Linear(128, nclass)
        self.dropout = dropout

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.dropout(F.relu(self.fc1(x)),p=self.dropout))))


class NNConv(MessagePassing):
    def __init__(self,
                 in_channels = 50,
                 out_channels = 100,
                 nn = Sequential(Linear(50,50*100),ReLU(),Linear(50*100,50*100)),
                 aggr='add',
                 root_weight=True,
                 bias=True,
                 **kwargs):
        super(NNConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        out = self.propagate(edge_index, x=x, pseudo=pseudo)
        out = torch.mean(out,dim=0,keepdim=True)
        return out


    def message(self, x_j, pseudo):
        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
        out = torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)
        return out

    def update(self, aggr_out, x):
        if aggr_out.shape[0] != x.shape[0]:
            print ('size mismatch')
            return aggr_out

        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
class Assign_Embedder(torch.nn.Module):
    def __init__(self,input_size,output_size,device):
        super(Assign_Embedder,self).__init__()
        self.hidden_size=output_size
        self.lstm=torch.nn.LSTM(input_size,output_size)
        self.device = device


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h_0,c_0 = self.initHidden_Cell()
        output_h,output_c=self.lstm(edge_attr.unsqueeze(1),(h_0,c_0))
        out=torch.max(output_h,dim=0)[0]
        return out

    def initHidden_Cell(self):
        '''

        :return: h0 c0 (num_layers * num_directions, batch, hidden_size)
        '''
        return torch.randn([1, 1, self.hidden_size]).to(self.device), torch.randn(
                [1, 1, self.hidden_size]).to(self.device)


class GATConv_wrapper(torch.nn.Module):
    def __init__(self, dropout = 0,INPUT_DIMENSION=100, HIDDEN_DIMENSION=256, OUTPUT_DIMENSION=200):
        super(GATConv_wrapper,self).__init__()
        self.f = GATConv(INPUT_DIMENSION,OUTPUT_DIMENSION)

    def forward(self, data):
        out = self.f(data.x,data.edge_index)
        out = torch.mean(out,dim=0,keepdim=True)
        return out

class AGNNConv_wrapper(torch.nn.Module):
    def __init__(self, dropout = 0,INPUT_DIMENSION=100, HIDDEN_DIMENSION=256, OUTPUT_DIMENSION=200):
        super(AGNNConv_wrapper,self).__init__()
        self.f = AGNNConv(True)

    def forward(self, data):
        out = self.f(data.x,data.edge_index)
        out = torch.mean(out,dim=0,keepdim=True)
        return out

class GraphConv_wrapper(torch.nn.Module):
    def __init__(self, dropout = 0,INPUT_DIMENSION=100, HIDDEN_DIMENSION=256, OUTPUT_DIMENSION=200):
        super(GraphConv_wrapper,self).__init__()
        self.f1 = GraphConv(INPUT_DIMENSION,OUTPUT_DIMENSION)
        self.f2 = GraphConv(OUTPUT_DIMENSION, OUTPUT_DIMENSION)
        self.f3 = GraphConv(OUTPUT_DIMENSION, OUTPUT_DIMENSION)

    def forward(self, data):
        out = self.f1(data.x,data.edge_index)
        out = self.f2(out, data.edge_index)
        out = self.f3(out, data.edge_index)
        out = torch.mean(out[:10],dim=0,keepdim=True)
        # out = out[0].unsqueeze(0)
        return out

class ARMAConv_wrapper(torch.nn.Module):
    def __init__(self, dropout = 0,INPUT_DIMENSION=100, HIDDEN_DIMENSION=256, OUTPUT_DIMENSION=200):
        super(ARMAConv_wrapper,self).__init__()
        self.f = ARMAConv(INPUT_DIMENSION,OUTPUT_DIMENSION)

    def forward(self, data):
        out = self.f(data.x,data.edge_index)
        out = torch.mean(out,dim=0,keepdim=True)
        return out

class SGConv_wrapper(torch.nn.Module):
    def __init__(self, dropout = 0,INPUT_DIMENSION=100, HIDDEN_DIMENSION=256, OUTPUT_DIMENSION=200):
        super(SGConv_wrapper,self).__init__()
        self.f = SGConv(INPUT_DIMENSION,OUTPUT_DIMENSION)

    def forward(self, data):
        out = self.f(data.x,data.edge_index)
        out = torch.mean(out,dim=0,keepdim=True)
        return out

class SAGEConv_wrapper(torch.nn.Module):
    def __init__(self, dropout = 0,INPUT_DIMENSION=100, HIDDEN_DIMENSION=256, OUTPUT_DIMENSION=200):
        super(SAGEConv_wrapper,self).__init__()
        self.f = SAGEConv(INPUT_DIMENSION,OUTPUT_DIMENSION)
        # self.nn = torch.nn.Linear(OUTPUT_DIMENSION, 1)

    def forward(self, data):
        out = self.f(data.x,data.edge_index)
        # out = torch.mean(out,dim=0,keepdim=True)
        out = out[0].unsqueeze(0)
        return out

class ChebConv_wrapper(torch.nn.Module):
    def __init__(self, dropout = 0,INPUT_DIMENSION=100, HIDDEN_DIMENSION=256, OUTPUT_DIMENSION=200):
        super(ChebConv_wrapper,self).__init__()
        self.f = ChebConv(INPUT_DIMENSION,OUTPUT_DIMENSION,2)

    def forward(self, data):
        out = self.f(data.x,data.edge_index)
        out = torch.mean(out,dim=0,keepdim=True)
        return out






class Edge_Embedder(torch.nn.Module):
    def __init__(self, dropout = 0, INPUT_DIMENSION = 50, HIDDEN_DIMENSION=200, OUTPUT_DIMENSION=100):
        super(Edge_Embedder,self).__init__()

        self.dropout = dropout
        self.activation = torch.nn.ReLU()

        # conv 100 200 50 origin
        self.conv1 = GCNConv(INPUT_DIMENSION, HIDDEN_DIMENSION)
        self.conv2 = GCNConv(HIDDEN_DIMENSION, OUTPUT_DIMENSION)
        # self.conv3 = GCNConv(HIDDEN_DIMENSION//2, HIDDEN_DIMENSION//4)
        # self.conv4 = GCNConv(HIDDEN_DIMENSION//4, OUTPUT_DIMENSION)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        out = F.dropout(self.activation(self.conv1(x,edge_index)),p=self.dropout,training=self.training) # conv
        out = F.dropout(self.activation(self.conv2(out,edge_index)),p=self.dropout,training=self.training) # conv
        out = torch.mean(out,dim=0,keepdim=True)

        return out

################## customized layer #########################
class my_random_walk(torch.nn.Module):
    def __init__(self, dropout = 0,INPUT_DIMENSION=100, HIDDEN_DIMENSION=512, OUTPUT_DIMENSION=200):
        super(my_random_walk,self).__init__()
        print ('my_random_walk')
        self.fc1 = torch.nn.Linear(INPUT_DIMENSION, HIDDEN_DIMENSION)
        self.fc2 = torch.nn.Linear(HIDDEN_DIMENSION, HIDDEN_DIMENSION)
        self.fc3 = torch.nn.Linear(HIDDEN_DIMENSION, OUTPUT_DIMENSION)

        self.relu = torch.nn.ReLU()
    def sample_neighbors_and_aggregate(self, x, neighbor_list, num_path=5, max_length=4):
        ret = []
        for node_idx in range(len(x)):
            temp = []
            for i in range(num_path):
                cur = node_idx
                temp.append([cur])
                for j in range(max_length - 1):
                    if neighbor_list[cur] == []:
                        break
                    else:
                        cur = random.choice(neighbor_list[cur])
                        temp[-1].append(cur)
            cur_emb = torch.mean(torch.stack([x[temp[i][j]] for i in range(len(temp)) for j in range(len(temp[i]))]),
                             dim=0, keepdim=True)
            ret.append(cur_emb)
        return torch.stack(ret)


    def forward(self, data):
        x, edge_index,edge_attr = data.x, data.edge_index,data.edge_attr
        neighbor_list = get_neighbor_list(x, edge_index)
        neighbor_list_bi = get_neighbor_list(x, edge_index, bi_direction = True)

        out = self.relu(self.fc1(x))
        out = self.sample_neighbors_and_aggregate(out, neighbor_list_bi,num_path=5, max_length=4)
        # out = self.relu(self.fc2(out))
        # out = self.sample_neighbors_and_aggregate(out, neighbor_list,num_path=5, max_length=4)
        out = self.relu(self.fc3(out))

        paths = sample_path(neighbor_list, num_path=10, max_length=4)
        out = torch.mean(torch.stack([out[paths[i][j]] for i in range(len(paths)) for j in range(len(paths[i]))]), dim=0, keepdim = True)

        return out

###############################################################################