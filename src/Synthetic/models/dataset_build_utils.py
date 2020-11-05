import sys
sys.path.insert(1, '../../')
import scipy.io as scio
import numpy as np
from torch_geometric.data import Data

import torch
import os


import random
import copy
import re

def build_edgeembedder_dataset(device,path_synthetic,path):
    def formula2graph(formulas,size):
        def trans(formula):
            graph=[]
            formula=formula.strip()
            ors=formula.split(" | ")
            for ands in ors:
                graph.append(ands.split(" & "))
            return graph
        def graph2numer(graph):
            graph_node_features = []
            graph_edge_index = [[], []]

            counter = 0
            graph_node_features.append(or_feature)
            for i in range(len(graph)):
                counter += 1
                graph_node_features.append(and_feature)
                graph_edge_index[0].append(0)
                graph_edge_index[1].append(counter)
                parent = counter
                for j in range(len(graph[i])):
                    counter += 1
                    graph_node_features.append(feature[graph[i][j]])
                    graph_edge_index[0].append(parent)
                    graph_edge_index[1].append(counter)
            graph_edge_index = torch.Tensor(graph_edge_index).long()
            graph_node_features = torch.Tensor(graph_node_features).float()
            graph_edge_index=graph_edge_index.to(device)
            graph_node_features=graph_node_features.to(device)
            return Data(edge_index=graph_edge_index, x=graph_node_features)

        node_feature = np.load(path_synthetic+"/prop.npy", allow_pickle=True)
        op_feature= np.load(path_synthetic+"/op.npy", allow_pickle=True)
        and_feature=op_feature[0]
        or_feature=op_feature[1]
        feature={}
        for i in range(size):
            feature["p{}".format(i)]=node_feature[i]
            feature["!p{}".format(i)]=1-node_feature[i]
        dataset_train=[]
        dataset_test=[]
        mask=mask = np.random.choice([0, 1],len(formulas), replace = True, p = [0.2, 0.8])
        for index,formula in enumerate(formulas):
            graph=trans(formula[0][0])
            data_graph=graph2numer(graph)
            # print(formula[0], graph, data_graph)
            true_data_graph=[]
            false_data_graph=[]
            for true_formula in formula[1]:
                graph_true=trans(true_formula)
                true_data_graph.append(graph2numer(graph_true))
                # print(true_formula, graph_true, true_data_graph[-1])
            for false_formula in formula[2]:
                graph_false=trans(false_formula)
                false_data_graph.append(graph2numer(graph_false))
                # print(false_formula, graph_false, false_data_graph[-1])
            if mask[index]==0:
                dataset_test.append([data_graph,true_data_graph,false_data_graph])
            else:
                dataset_train.append([data_graph,true_data_graph,false_data_graph])
        return dataset_train, dataset_test

    data=scio.loadmat(path)
    size=data["size"][0][0]
    complement_formula=data["final_prop"]
    dataset_train, dataset_test=formula2graph(complement_formula,size)
    return dataset_train, dataset_test

#######################################################
def gen_edge_prop_data(device, prop_list, prop_features):
    # assume prop_features is dictionary
    # feature on the node, three layers of nodes: propositions, AND, OR
    graph_node_features = []
    graph_edge_index = [[],[]]

    counter = 0
    graph_node_features.append(prop_features['OR'])
    for i in range(len(prop_list)):
        counter += 1
        graph_node_features.append(prop_features['AND'])
        graph_edge_index[0].append(0)
        graph_edge_index[1].append(counter)
        parent = counter
        for j in range(len(prop_list[i])):
            counter += 1
            graph_node_features.append(prop_features[prop_list[i][j]])
            graph_edge_index[0].append(parent)
            graph_edge_index[1].append(counter)
    graph_edge_index = torch.Tensor(graph_edge_index).long()
    graph_node_features = torch.Tensor(graph_node_features).float()

    graph_edge_index,graph_node_features=graph_edge_index.to(device),graph_node_features.to(device)

    # return graph_node_features, graph_edge_features, graph_edge_index
    return Data(edge_index=graph_edge_index, x=graph_node_features)



########################################################

def gen_graph_data(assign, node_feature, edge_feature):
    num_states = int(len(assign)/3)

    node_feature_tensor = torch.tensor(node_feature).float().to(device)
    edge_feature_tensor = torch.tensor(edge_feature).float().to(device)

    graph_node_features = []
    graph_edge_features = []
    graph_edge_index = []

    graph_node_features.append(node_feature_tensor[0].unsqueeze(0))
    for i in range(num_states-1):
        graph_node_features.append(node_feature_tensor[1].unsqueeze(0))
        graph_node_features.append(node_feature_tensor[1].unsqueeze(0))
    graph_node_features.append(node_feature_tensor[2].unsqueeze(0))
    graph_node_features = torch.cat(graph_node_features).float()

    graph_edge_index = [[0]+list(np.arange(num_states)), list(np.arange(num_states+1))]
    graph_edge_index = torch.tensor(graph_edge_index).long().to(device)

    graph_edge_features = torch.rand((1,50)).float().to(device)
    for i in range(num_states):
        temp = assign[0+i]*edge_feature_tensor[0]+(1-assign[0+i])*edge_feature_tensor[3] # p0 or !p0
        temp *= assign[1+i]*edge_feature_tensor[1]+(1-assign[1+i])*edge_feature_tensor[4] # p1 or !p1
        temp *= assign[2+i] * edge_feature_tensor[2] + (1 - assign[2+i]) * edge_feature_tensor[5] # p2 or !p2
        graph_edge_features = torch.cat((graph_edge_features,temp.unsqueeze(0).float()))

    # return graph_node_features, graph_edge_features, graph_edge_index
    return Data(edge_attr=graph_edge_features.to(device), edge_index=graph_edge_index.to(device), x=graph_node_features.to(device))

###########################################################

def writeltl(path,data):
    s=""
    for item in data:
        s+=str(item)+"\n"
    with open(path,"w") as file:
        file.write(s)

def writefile(path,data):
    with open(path) as file:
        file.write(data)

def readltl(dir):
    with open(dir) as file:
        ltls=file.read()
        ltls=ltls.split("\n")[0:-1]
        return ltls

def readData_syntax(device, dir,name):
    edge_attr = torch.load(dir + "/" + str(name) + ".edge_attr").requires_grad_(True)
    edge_index = torch.tensor(torch.load(dir + "/" + str(name) + ".edge_index"), dtype=torch.long)
    nodes = torch.load(dir + "/" + str(name) + ".nodes").requires_grad_(True)
    edge_attr, edge_index, nodes = edge_attr.to(device), edge_index.to(device), nodes.to(device)
    ret = Data(edge_attr=edge_attr, edge_index=edge_index.t().contiguous(), x=nodes)
    return ret


def readData(device, dir,name, edge_feature, video = False):
    edge_attr=torch.load(dir+"/"+str(name)+".edge_attr").requires_grad_(True)
    edge_index=torch.load(dir+"/"+str(name)+".edge_index").long()
    nodes=torch.load(dir+"/"+str(name)+".nodes").requires_grad_(True)
    edge_attr,edge_index,nodes=edge_attr.to(device),edge_index.to(device),nodes.to(device)
    ret = Data(edge_attr=edge_attr,edge_index=edge_index.t().contiguous(), x=nodes)

    if video:
        edge_strings = parse_assignments(dir + "/" + str(name) + ".edge_label")
    else:
        edge_strings = parse_assignments(dir+"/"+str(name)+"edge_string.txt")
    edge_datas = []
    for edge in range(len(edge_strings)):
        edge_datas.append(gen_edge_prop_data(device,edge_strings[edge], edge_feature))
    return [ret, edge_datas]

def build_dataset(device, datapath, edge_feature):
    ltls=readltl(datapath+"/formula.txt")
    np.random.seed(27)
    mask = np.random.choice([0, 1],len(ltls), replace = True, p = [0.2, 0.8])  # 0 for test and 1 for train

    dataset_train = []
    dataset_test = []
    Path_Dataset=datapath+"/ltls"
    for index in range(len(ltls)):
        graph_data=readData(device, Path_Dataset+"/"+str(index+1),"formula", edge_feature)
        trues_data=[]
        false_data=[]
        number=len(os.listdir(Path_Dataset+"/"+str(index+1)+"/true_numeric"))//4
        if number ==0:
            continue
        if index % 100 == 0:
            print(index, number)
        for i in range(number):
            true_i=readData(device, Path_Dataset+"/"+str(index+1)+"/true_numeric",i, edge_feature)
            false_i=readData(device, Path_Dataset+"/"+str(index+1)+"/false_numeric",i, edge_feature)
            trues_data.append(true_i)
            false_data.append(false_i)
        if mask[index] == 1:
            dataset_train.append([graph_data,trues_data,false_data])
        else:
            dataset_test.append([graph_data, trues_data, false_data])
    return dataset_train, dataset_test

def sample_true_assign(filename, num_or_sample = 1, num_complete_sample = 1):
    def get_temp(input_string):
        processed_temp = []
        temp = input_string.replace(' ', '').replace('(', '').replace(')', '').split('&X')
        temp = list(map(lambda x: x.split('|'), temp))
        temp = list(map(lambda x: list(map(lambda y: y.split('&'), x)), temp))
        # sample OR components
        for n in range(num_or_sample):
            processed_temp.append([])
            for i in range(len(temp)):
                processed_temp[-1].append(random.sample(temp[i], 1)[0])
        return processed_temp

    def sample_complete(input):
        ret = []
        # complete the missing propositions
        for i in range(len(input)):
            for n in range(num_complete_sample):
                for j in range(len(input[i])):
                    for prop in range(len(complete_prop_list)):
                        if '1' in input[i][j]:
                            input[i][j].remove('1')
                        if (complete_prop_list[prop][0] not in input[i][j]) and (
                                complete_prop_list[prop][1] not in input[i][j]):
                            input[i][j].append(random.sample(complete_prop_list[prop], 1)[0])
                ret.append(input[i])
        return ret


    complete_prop_list = [['p0','!p0'],['p1','!p1'],['p2','!p2']]

    trueltls = readltl(filename)
    true_all = []
    false_all = []
    for ltl in range(len(trueltls)):
        one_true_assign = trueltls[ltl]
        one_false_assign = copy.deepcopy(one_true_assign)
        rep = {}
        for i in range(len(complete_prop_list)):
            rep[complete_prop_list[i][0]] = complete_prop_list[i][1]
            rep[complete_prop_list[i][1]] = complete_prop_list[i][0]
        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        one_false_assign = pattern.sub(lambda m: rep[re.escape(m.group(0))], one_false_assign)


        true_temp = get_temp(one_true_assign)
        false_temp = get_temp(one_false_assign)

        true_all += sample_complete(true_temp)
        false_all += sample_complete(false_temp)
    return true_all, false_all



def parse_assignments(filename):
    edge_strings = readltl(filename)

    ret = []
    for edge in range(len(edge_strings)):
        out = []
        temp = edge_strings[edge].replace('(','').replace(')','').replace(" ", "").split('|')
        for i in range(len(temp)):
            out.append(temp[i].split("&"))
        ret.append(out)
    return ret

def prop2binary(input, num_prop):
    complete_prop_list = [['p0', '!p0'], ['p1', '!p1'], ['p2', '!p2']]
    ret = []
    for i in range(len(input)):
        for j in range(len(complete_prop_list)):
            if complete_prop_list[j][0] in input[i]:
                ret.append(1)
            elif complete_prop_list[j][1] in input[i]:
                ret.append(0)
    return ret




def build_dataset_onebyone(device, datapath, edge_feature):
    ltls=readltl(datapath+"/formula.txt")
    np.random.seed(27)
    mask = np.random.choice([0, 1], len(ltls), replace=True, p=[0.2, 0.8])  # 0 for test and 1 for train

    dataset_train = []
    dataset_test = []
    Path_Dataset=datapath+"/ltls"
    for index in range(len(ltls)):
        graph_data=readData(device, Path_Dataset+"/"+str(index+1),"formula", edge_feature)
        number=len(os.listdir(Path_Dataset+"/"+str(index+1)+"/true_numeric"))//4
        # print("number of assign in dataset ",number)
        if number ==0:
            continue
        for i in range(number):
            trues_data = []
            false_data = []
            true_i=readData(device, Path_Dataset+"/"+str(index+1)+"/true_numeric",i, edge_feature)
            false_i=readData(device, Path_Dataset+"/"+str(index+1)+"/false_numeric",i, edge_feature)
            trues_data.append(true_i)
            false_data.append(false_i)
            if mask[index] == 1:
                dataset_train.append([graph_data, [true_i], [false_i]])
            else:
                dataset_test.append([graph_data,[true_i],[false_i]])
    return dataset_train, dataset_test


def readsyntaxData(device, dir,name):
    # edge_attr=torch.load(dir+"/"+str(name)+".edge_attr")
    edge_index=torch.load(dir+"/"+str(name)+".edge_index")
    nodes=torch.load(dir+"/"+str(name)+".nodes")
    edge_attr=torch.zeros([edge_index.shape[0],50])
    edge_attr,edge_index,nodes=edge_attr.to(device),edge_index.to(device),nodes.to(device)
    return Data(edge_attr=edge_attr,edge_index=edge_index.t().contiguous(), x=nodes)

def readsyntaxassignData(device, dir,name):
    # edge_attr=torch.load(dir+"/"+str(name)+".edge_attr")
    edge_index=torch.load(dir+"/"+str(name)+"syntax.edge_index")
    # deal with empty edge index case
    if len(edge_index.shape) == 1:
        edge_index = torch.tensor([[0,0]]).long()

    nodes = torch.load(dir+"/"+str(name)+"syntax.nodes")
    edge_attr=torch.zeros([edge_index.shape[0],50])
    edge_attr,edge_index,nodes=edge_attr.to(device),edge_index.to(device),nodes.to(device)
    return Data(edge_attr=edge_attr,edge_index=edge_index.t().contiguous(), x=nodes)


def build_syntaxdataset_onebyone(datapath):
    ltls=readltl(datapath+"/formula.txt")
    np.random.seed(27)
    mask = np.random.choice([0, 1], len(ltls), replace=True, p=[0.2, 0.8])  # 0 for test and 1 for train

    dataset_train = []
    dataset_test = []
    Path_Dataset=datapath+"/ltls"
    for index in range(len(ltls)):
        graph_data=readsyntaxData(device, Path_Dataset+"/"+str(index+1),"syntax")
        trues_data=[]
        false_data=[]
        number=len(os.listdir(Path_Dataset+"/"+str(index+1)+"/true_syntax"))//2
        if number==0:
            continue
        if index%100 == 0:
            print(index, number)
        for i in range(number):
            true_i=readsyntaxassignData(device, Path_Dataset+"/"+str(index+1)+"/true_syntax",i)
            false_i=readsyntaxassignData(device, Path_Dataset+"/"+str(index+1)+"/false_syntax",i)
            trues_data.append(true_i)
            false_data.append(false_i)
            if mask[index] == 1:
                dataset_train.append([graph_data, [true_i], [false_i]])
            else:
                dataset_test.append([graph_data,[true_i],[false_i]])
    return dataset_train, dataset_test

def build_syntaxdataset(datapath):
    ltls=readltl(datapath+"/formula.txt")
    np.random.seed(27)
    mask = np.random.choice([0, 1], len(ltls), replace=True, p=[0.2, 0.8])  # 0 for test and 1 for train

    dataset_train = []
    dataset_test = []
    Path_Dataset=datapath+"/ltls"
    for index in range(len(ltls)):
        graph_data=readsyntaxData(Path_Dataset+"/"+str(index+1),"syntax")
        trues_data=[]
        false_data=[]
        number=len(os.listdir(Path_Dataset+"/"+str(index+1)+"/true_syntax"))//2
        if number==0:
            continue
        if index%100 == 0:
            print(index, number)
        for i in range(number):
            true_i=readsyntaxassignData(device, Path_Dataset+"/"+str(index+1)+"/true_syntax",i)
            false_i=readsyntaxassignData(device, Path_Dataset+"/"+str(index+1)+"/false_syntax",i)
            trues_data.append(true_i)
            false_data.append(false_i)
        if mask[index] == 1:
            dataset_train.append([graph_data, trues_data, false_data])
        else:
            dataset_test.append([graph_data,trues_data,false_data])
    return dataset_train, dataset_test



def build_video_edgeembedder_dataset(device,path_video,name_dataset):
    def trans(formula):
        graph=[]
        formula=formula.replace("(","").replace(")","").strip()
        ors=formula.split(" | ")
        for ands in ors:
            graph.append(ands.split(" & "))
        return graph
    def graph2numer(graph):
        graph_node_features = []
        graph_edge_index = [[], []]

        counter = 0
        graph_node_features.append(or_feature)
        for i in range(len(graph)):
            counter += 1
            graph_node_features.append(and_feature)
            graph_edge_index[0].append(0)
            graph_edge_index[1].append(counter)
            parent = counter
            for j in range(len(graph[i])):
                counter += 1
                graph_node_features.append(feature[graph[i][j]])
                graph_edge_index[0].append(parent)
                graph_edge_index[1].append(counter)
        graph_edge_index = torch.Tensor(graph_edge_index).long().to(device)
        graph_node_features = torch.Tensor(graph_node_features).float().to(device)
        return Data(edge_index=graph_edge_index, x=graph_node_features)
    op_feature= np.load(path_video+"/op.npy", allow_pickle=True)
    and_feature=op_feature[0]
    or_feature=op_feature[1]
    one_feature=torch.ones([200],dtype=torch.float32)
    zero_feature=torch.zeros([200],dtype=torch.float32)
    videos=os.listdir(path_video+name_dataset)
    dataset_train=[]
    dataset_test=[]
    mask = np.random.choice([0, 1], len(videos), replace=True, p=[0.2, 0.8])
    for index,recipe in enumerate(videos):
        if '.pt' in recipe:
            continue
        feature={}
        path_recipe=path_video+name_dataset+recipe
        node_feature=torch.load(path_recipe+"/prop_dict_numerical")
        data = scio.loadmat(path_recipe+"/edge_dataset.mat")
        formulas=data["edge_dataset"]
        for i in range(node_feature.shape[0]):
            feature["p{}".format(i)]=node_feature[i]
            feature["!p{}".format(i)]=1-node_feature[i]
        feature["1"]=one_feature
        feature["0"]=zero_feature
        for formula in formulas:
            graph=trans(formula[0][0])
            data_graph=graph2numer(graph)
            # print(formula[0], graph, data_graph)
            true_data_graph=[]
            false_data_graph=[]
            for true_formula in formula[1]:
                graph_true=trans(true_formula)
                true_data_graph.append(graph2numer(graph_true))
                # print(true_formula, graph_true, true_data_graph[-1])
            for false_formula in formula[2]:
                graph_false=trans(false_formula)
                false_data_graph.append(graph2numer(graph_false))
                # print(false_formula, graph_false, false_data_graph[-1])
            if mask[index]==0:
                dataset_test.append([data_graph,true_data_graph,false_data_graph])
            else:
                dataset_train.append([data_graph,true_data_graph,false_data_graph])
    return dataset_train, dataset_test




def build_video_dataset_onebyone(device, path_video, name_dataset):
    videos=os.listdir(path_video+name_dataset)

    mask = np.random.choice([0, 1], len(videos), replace=True, p=[0.2, 0.8])

    one_feature=torch.ones([200],dtype=torch.float32)
    zero_feature=torch.zeros([200],dtype=torch.float32)
    op_feature = np.load(path_video + "/op.npy", allow_pickle=True)

    dataset_train = []
    dataset_test = []
    counter = 0
    for index, recipe in enumerate(videos):
        counter += 1
        feature={}
        path_recipe=path_video+name_dataset+recipe
        node_feature=torch.load(path_recipe+"/prop_dict_numerical")
        print(recipe)
        for i in range(node_feature.shape[0]):
            feature["p{}".format(i)]=node_feature[i]
            feature["!p{}".format(i)]=1-node_feature[i]
        feature["1"]=one_feature
        feature["0"]=zero_feature
        feature["AND"]=op_feature[0]
        feature["OR"] = op_feature[1]
        graph_data = readData(device, path_recipe, "formula", feature, video=True)
        number=len(os.listdir(path_recipe+"/true_numeric"))//4
        # print("number of assign in dataset ",number)
        if number ==0:
            continue
        for i in range(number):
            trues_data = []
            false_data = []
            true_i=readData(device, path_recipe+"/true_numeric",i, feature, video=True)
            false_i=readData(device, path_recipe+"/false_numeric",i, feature, video=True)
            trues_data.append(true_i)
            false_data.append(false_i)
            if mask[index] == 1:
                dataset_train.append([graph_data, [true_i], [false_i]])
            else:
                dataset_test.append([graph_data,[true_i],[false_i]])


    return dataset_train, dataset_test