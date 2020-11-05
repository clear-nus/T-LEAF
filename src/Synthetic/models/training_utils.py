import torch
import numpy as np
from torch_geometric.data import Data

def add_y_label(device, args,x):
    node_feature = torch.tensor(np.load(args.dataset_root+"node.npy", allow_pickle=True), dtype=torch.float32).to(device)
    values = []
    # zeros = torch.zeros(out[0].shape, dtype=torch.float32).unsqueeze(0)
    for index in range(x.shape[0]):
        if torch.all(x[index].eq(node_feature[0])):
            values.append(0)
        elif torch.all(x[index].eq(node_feature[1])):
            values.append(1)
        elif torch.all(x[index].eq(node_feature[2])):
            values.append(2)
        else:
            values.append(3)
    return torch.LongTensor(values)


def replace_edge_attr(device, args, input, edge_embedder):
    if args.syntax:
        if not args.nodes_only:
            return input
        else:
            return edge2node(device, args, input)
    data=input[0]
    if args.replace:
        edge_attr = []
        with torch.no_grad():
            for edge in range(len(input[1])):
                edge_attr.append(edge_embedder(input[1][edge]))

            if len(edge_attr) > 0:
                data.edge_attr = torch.cat(edge_attr).float().to(device)
            else:
                print("no edge attr",input[0])
                return None
    return edge2node(device, args, data)

def edge2node(device, args, data):
    if not args.nodes_only:
        return data
    nnode=len(data.x)
    if not args.global_node:
        nodes=torch.cat([data.x,data.edge_attr],dim=0)
    else:
        # global_feature=torch.rand(data.x[0].shape).unsqueeze(0)
        if not args.syntax:
            global_npy=np.load(args.dataset_root+"global.npy",allow_pickle=True)
        else:
            global_npy = np.load(args.dataset_root + "global_50.npy", allow_pickle=True)
        global_feature=torch.tensor(global_npy,dtype=torch.float32).to(device)
        nodes = torch.cat([data.x, data.edge_attr,global_feature], dim=0)
    edge_index = []
    for index,edge in enumerate(data.edge_index.t()):
        start,end = int(edge[0]), int(edge[1])
        edge_index.append([start,nnode])
        edge_index.append([nnode,end])
        nnode+=1
    if args.global_node:
        for index in range(nnode):
            edge_index.append([index,nnode])
        nnode+=1
    edge_index.sort()
    edge_index=torch.tensor(edge_index,dtype=torch.long)
    if not args.syntax:
        y=add_y_label(device, args, nodes)
    else:
        y=torch.zeros([nodes.shape[0]])
    ret=Data(edge_index=edge_index.t().contiguous().to(device),x=nodes.to(device),y=y.to(device))
    return ret

def new_dataset(device, args, dataset, edge_embedder):
    dataset_new=[]
    for data_i in range(len(dataset)):
        data_formula=replace_edge_attr(device, args, dataset[data_i][0], edge_embedder)
        if data_formula==None:
            continue
        true=[]
        false=[]
        for index in range(len(dataset[data_i][1])):
            new_true=replace_edge_attr(device, args, dataset[data_i][1][index], edge_embedder)
            if new_true==None:
                continue
            new_false=replace_edge_attr(device, args, dataset[data_i][2][index], edge_embedder)
            if new_false==None:
                continue
            true.append(new_true)
            false.append(new_false)
        dataset_new.append([data_formula,true,false])
    return dataset_new

def newsyntax_dataset(device, args, dataset, edge_embedder):
    dataset_new=[]
    for data_i in range(len(dataset)):
        data_formula=dataset[data_i][0]
        true=[]
        false=[]
        for index in range(len(dataset[data_i][1])):
            true.append(replace_edge_attr(device, args, dataset[data_i][1][index], edge_embedder))
            false.append(replace_edge_attr(device, args, dataset[data_i][2][index], edge_embedder))
        dataset_new.append([data_formula,true,false])
    return dataset_new