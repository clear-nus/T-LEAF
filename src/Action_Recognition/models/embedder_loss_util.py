import torch
import numpy as np
import os
from torch_geometric.data import Data
from src.Synthetic.models.Util import finite2aut, aut2graph_finite, aut2graph
from src.Synthetic.models.Graph import Graph, Node, Edge
import re



WORD_DIMESION = 100

INIT=0
COMMON=1
FINAL=2


def read_strlist(path):
    '''

    :param path: strlist data path
    :return: list of string ["ab","cd]
    '''
    with open(path) as file:
        ltls = file.read()
        ltls = ltls.split("\n")[0:-1]
        return ltls


def gen_edge_prop_data(device, prop_list, prop_features, soft_features = None):
    # if is_assign:
    #     print ('prop list', prop_list)
        # print ('prop features', prop_features)

    # assume prop_features is dictionary
    # feature on the node, three layers of nodes: propositions, AND, OR
    graph_node_features = []
    graph_edge_index = [[], []]


    if soft_features is not None:
        counter = 0
        graph_node_features.append(torch.tensor(prop_features['OR']).to(device).float().requires_grad_())
        counter += 1
        graph_node_features.append(torch.tensor(prop_features['AND']).to(device).float().requires_grad_())
        graph_edge_index[0].append(0)
        graph_edge_index[1].append(counter)
        parent = counter
        for j in range(len(soft_features)):
            counter += 1
            graph_node_features.append(soft_features[j].requires_grad_())
            graph_edge_index[0].append(parent)
            graph_edge_index[1].append(counter)
        graph_node_features = torch.stack(graph_node_features).float()
    else:
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
        graph_node_features = torch.Tensor(graph_node_features).float()

    graph_edge_index = torch.Tensor(graph_edge_index).long()


    # return graph_node_features, graph_edge_features, graph_edge_index
    # if is_assign:
    #     print ('node features',len(graph_node_features))

    return Data(edge_index=graph_edge_index.to(device), x=graph_node_features.requires_grad_().to(device))


def parse_assignments(filename):
    edge_strings = read_strlist(filename)

    ret = []
    for edge in range(len(edge_strings)):
        out = []
        temp = edge_strings[edge].replace('(', '').replace(')', '').replace(" ", "").split('|')
        for i in range(len(temp)):
            out.append(temp[i].split("&"))
        ret.append(out)
    return ret





def get_formula_video(device, pred_actions, pred_objects, path_recipe, path_video, edge_embedder, glove_features, action_cat, ingredient_cat, info_embedder, differentiable):
    all, graph_node_feature, op_feature = info_embedder


    one_feature = torch.ones([200], dtype=torch.float32)
    zero_feature = torch.zeros([200], dtype=torch.float32)


    feature = {}
    try:
        node_feature = torch.load(path_recipe + "/prop_dict_numerical")
    except:
        print ('file not found', path_recipe)
        return None, None

    ltl_next_relation = read_strlist(path_recipe + "ltl_next_relations.txt")
    ltl_feasible_relation = read_strlist(path_recipe + "ltl_feasible_relations.txt")

    prop_dict = read_strlist(path_recipe + "/prop_dict.txt")
    prop_list = []
    for i in prop_dict:
        temp = i.replace(' ', '').split(':')
        prop_list.append((temp[1], temp[0]))
    for i in range(node_feature.shape[0]):
        feature["p{}".format(i)] = node_feature[i]
        feature["!p{}".format(i)] = 1 - node_feature[i]
    feature["1"] = one_feature
    feature["0"] = zero_feature
    feature["AND"] = op_feature[0]
    feature["OR"] = op_feature[1]
    feature[""] = zero_feature



    # pred_actions = [[all['actions_list'].index(prop_list[0][0]),all['actions_list'].index(prop_list[1][0])]]
    # pred_objects = [[all['objects_list'].index(prop_list[0][1]),all['objects_list'].index(prop_list[1][1])]]
    p_indexs, temp, pred_data, grad_test = process_pred(device, pred_actions, pred_objects, prop_list,  ltl_feasible_relation, ltl_next_relation, feature, all, path_video, glove_features, action_cat, ingredient_cat, differentiable)
    if temp is False:
        return None, None, None

    graph_data = build_formula(device, graph_node_feature, feature, p_indexs, ltl_feasible_relation, ltl_next_relation)




    graph_data_new = replace_edge_attr(device, graph_data, edge_embedder, path_video)
    pred_data_new = replace_edge_attr(device, pred_data, edge_embedder, path_video)


    return graph_data_new, pred_data_new, grad_test


def assign2numer(assign, dataset_root):
    '''
    numerical the assign
    :param assigns:
    :return:
    '''
    all_node_feature = np.load(dataset_root + "node.npy", allow_pickle=True)
    WORD_DIMESION = 100
    nodes = []
    edge_index = []
    for index in range(len(assign) + 1):
        if index == 0:
            nodes.append(all_node_feature[0])
        elif index == len(assign):
            nodes.append(all_node_feature[2])
            edge_index.append([index - 1, index])
        else:
            nodes.append(all_node_feature[1])
            edge_index.append([index - 1, index])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    nodes = torch.tensor(nodes, dtype=torch.float32)
    edge_attr = torch.zeros([edge_index.shape[0], WORD_DIMESION * 2], dtype=torch.float32)
    return [edge_index, nodes, edge_attr], assign


def write_strlist(path, data):
    '''

    :param path: file path
    :param data: a list of string
    :return:
    '''
    s = ""
    for item in data:
        s += str(item) + "\n"
    with open(path, "w") as file:
        file.write(s)



def process_pred(device, pred_actions_raw, pred_objects_raw, prop_list, ltl_feasible_relation, ltl_next_relation, edge_feature, all, dataset_root, glove_features, action_cat, ingredient_cat, differentiable):
    # print (pred_actions_raw.shape, pred_objects_raw.shape)

    softmax = torch.nn.Softmax(dim=1)
    temperature = 1

    # pred_actions_raw = softmax(pred_actions_raw/temperature)
    # pred_objects_raw = softmax(pred_objects_raw/temperature)
    if differentiable:
        pred_actions = torch.max(pred_actions_raw,-1)[1].detach()
        pred_objects = torch.max(pred_objects_raw, -1)[1].detach()
    else:
        pred_actions = pred_actions_raw
        pred_objects = pred_objects_raw
    temp = []
    for i in range(len(pred_actions)):
        action_name = all['actions'][pred_actions[i]]
        object_name = all['ingredients'][pred_objects[i]]

        if (action_name, object_name) in prop_list:
            temp.append(prop_list.index((action_name, object_name)))

    if len(temp) == 0:
        return temp, False, [], []



    prop_regen=regenerate_prop(ltl_next_relation,ltl_feasible_relation, temp) # this function is wrong




    assign_str = []
    for i in range(len(temp)):
        assign_str.append('')
        for j in range(len(prop_regen)):
            if prop_regen[j] == temp[i]:
                assign_str[-1] += 'p' + str(prop_regen[j])
            else:
                assign_str[-1] += '!p' + str(prop_regen[j])

            if j < len(prop_regen) - 1:
                assign_str[-1] += ' & '

    # path_pred = path_recipe + "/pred_numeric"
    # if not os.path.exists(path_pred):
    #     os.mkdir(path_pred)

    assign_data, assign_edge_label = assign2numer(assign_str, dataset_root)

    edge_index, nodes, edge_attr = assign_data[0].to(device), assign_data[1].to(device), assign_data[2].to(device)

    data = Data(edge_attr=edge_attr, edge_index=edge_index.t().contiguous(), x=nodes)
    # print (assign_str)


    edge_s = []
    edge_strings = assign_edge_label
    for edge in range(len(edge_strings)):
        out = []
        edge_temp = edge_strings[edge].replace('(', '').replace(')', '').replace(" ", "").split('|')
        for i in range(len(edge_temp)):
            out.append(edge_temp[i].split("&"))
        edge_s.append(out)
    edge_datas = []
    for edge in range(len(edge_s)):
        edge_datas.append(gen_edge_prop_data(device, edge_s[edge], edge_feature))


    # torch.save(assign_data[0], path_pred + "/pred.edge_index")
    # torch.save(assign_data[1], path_pred + "/pred.nodes")
    # torch.save(assign_data[2], path_pred + "/pred.edge_attr")
    # write_strlist(path_pred + "/pred.edge_label", assign_edge_label)



    ############  differential features #####################
    if differentiable:
        soft_prop_features = []
        for t in range(len(pred_actions_raw)):

            soft_prop_features.append(torch.mean(torch.stack([pred_actions_raw[t][i] * pred_objects_raw[t][
                i] * torch.tensor(glove_features[action_cat[i]] + glove_features[ingredient_cat[i]]).to(device) for i in
                                                              range(len(pred_actions_raw[t]))]), 0))

        assign_data, assign_edge_label = assign2numer([''], dataset_root)

        edge_index, nodes, edge_attr = assign_data[0].to(device), assign_data[1].to(device), assign_data[2].to(device)
        data = Data(edge_attr=edge_attr, edge_index=edge_index.t().contiguous(), x=nodes)

        edge_datas = [gen_edge_prop_data(device, edge_s[edge], edge_feature, soft_features=soft_prop_features)]

    return temp, True, [data, edge_datas],edge_datas[0].x.sum() # torch.stack(soft_prop_features).sum(),


def replace_edge_attr(device, input, edge_embedder, dataset_root):
    data = input[0]
    edge_attr = []
    # with torch.no_grad():
    for edge in range(len(input[1])):
        edge_attr.append(edge_embedder(input[1][edge]))
    data.edge_attr = torch.cat(edge_attr).float().to(device)
    return edge2node(device, data, dataset_root)


def edge2node(device, data, dataset_root):
    nnode = len(data.x)
    nodes = torch.cat([data.x, data.edge_attr], dim=0)

    edge_index = []
    for index, edge in enumerate(data.edge_index.t()):
        start, end = int(edge[0]), int(edge[1])
        edge_index.append([start, nnode])
        edge_index.append([nnode, end])
        nnode += 1

    edge_index.sort()
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    y = add_y_label(device, nodes, dataset_root)
    edge_index, nodes, y = edge_index.to(device), nodes.to(device), y.to(device)

    ret = Data(edge_index=edge_index.t().contiguous(), x=nodes, y=y)
    return ret


def add_y_label(device, x, dataset_root):
    node_feautre = torch.tensor(np.load(dataset_root + "/node.npy", allow_pickle=True), dtype=torch.float32).to(device)
    values = []
    # zeros = torch.zeros(out[0].shape, dtype=torch.float32).unsqueeze(0)
    for index in range(x.shape[0]):
        if torch.all(x[index].eq(node_feautre[0])):
            values.append(0)
        elif torch.all(x[index].eq(node_feautre[1])):
            values.append(1)
        elif torch.all(x[index].eq(node_feautre[2])):
            values.append(2)
        else:
            values.append(3)
    return torch.LongTensor(values)


def read_cat(cat_path):

    ingredient_cat = read_strlist(cat_path+"ingredient_cat.txt")
    action_cat = read_strlist(cat_path + "action_cat.txt")
    return ingredient_cat, action_cat


def regenerate_prop(ltl_next_relation,ltl_feasible_relation, prop_indexs):
    all_props = []
    sub_next = []
    for next_relation in ltl_next_relation:
        for prop in prop_indexs:
            if str(prop) in next_relation and next_relation not in sub_next:
                sub_next.append(next_relation)
                all_props+=re.findall(r"\d+", next_relation)
    sub_feasible = []
    for feasible_relation in ltl_feasible_relation:
        for prop in prop_indexs:
            if str(prop) in feasible_relation and feasible_relation not in sub_feasible:
                sub_feasible.append(feasible_relation)
                all_props+=re.findall(r"\d+", feasible_relation)
    all_props=list(set(all_props))
    all_prop_num=[]
    for prop in all_props:
        all_prop_num.append(int(prop))
    return all_prop_num

def graph2numer(graph, node_feature):
    '''
    numerical the graph
    :param graph:
    :return:
    '''
    nodes = []
    edge_index = []
    edge_attr = []
    edge_label = []
    for index, node in enumerate(graph.nodes):
        if node.label == INIT:
            nodes.append(node_feature[0])
        elif node.label == FINAL:
            nodes.append(node_feature[2])
        else:
            nodes.append(node_feature[1])
        for edge in node.edges:
            edge_index.append([edge.src, edge.dst])
            # edge_attr.append(edge_feature)
            edge_label.append(edge.label)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    nodes = torch.tensor(nodes, dtype=torch.float32)
    edge_attr = torch.zeros([edge_index.shape[0], WORD_DIMESION * 2], dtype=torch.float32)
    return [edge_index, nodes, edge_attr], edge_label

def trans2graph(ltl_next_relation, ltl_feasible_relation, constraint_string):
    '''
    combine next and feasible ltl, then transform to automata graph
    :param ltl_next_relation:
    :param ltl_feasible_relation:
    :return:
    '''
    ltl_all = ""
    # print(len(ltl_next_relation),len(ltl_feasible_relation))
    # print(constraint)
    for ltl in ltl_next_relation:
        ltl_all += ltl + " & "
    for ltl in ltl_feasible_relation:
        ltl_all += ltl + " & "
    ltl_all = ltl_all + constraint_string
    # print(ltl_all)
    aut = finite2aut(ltl_all)
    # print(aut.to_str())
    # nodes = aut2graph_finite(aut)
    nodes = aut2graph(aut)
    init_node = int(aut.get_init_state_number())
    graph = Graph(ltl_all, nodes, init_node)
    return graph

def generate_constraints(props):
    '''
    generate the list of one prop true constrains, both list and Global string
    :param size:
    :return:
    '''
    size = len(props)
    constraints = []
    constraint_string = "G ("
    for i in range(size):
        constraint_item = ""
        constraint_string += " ( "
        constraint_item += "{}".format(props[i])
        for j in range(size):
            if j == i:
                continue
            constraint_item += " & !{}".format(props[j])
        constraint_string += constraint_item
        constraint_string += " ) | "
        constraints.append(constraint_item)
    constraint_string = constraint_string[0:-2]
    constraint_string += " )"

    return constraints, constraint_string

def generate_constraints_size(size):
    '''
    generate the list of one prop true constrains, both list and Global string
    :param size:
    :return:
    '''
    constraints = []
    constraint_string = "G ("
    for i in range(size):
        constraint_item = ""
        constraint_string += " ( "
        constraint_item += "p{}".format(i)
        for j in range(size):
            if j == i:
                continue
            constraint_item += " & !p{}".format(j)
        constraint_string += constraint_item
        constraint_string += " ) | "
        constraints.append(constraint_item)
    constraint_string = constraint_string[0:-2]
    constraint_string += " )"

    return constraints, constraint_string

def parse_labels(edge_strings):
    ret = []
    for edge in range(len(edge_strings)):
        out = []
        temp = edge_strings[edge].replace('(', '').replace(')', '').replace(" ", "").split('|')
        for i in range(len(temp)):
            out.append(temp[i].split("&"))
        ret.append(out)
    return ret

def numerical(device, graph, node_feature, edge_feature):

    formula_data, formula_edge_label = graph2numer(graph, node_feature)
    edge_index, nodes, edge_attr = formula_data[0].to(device), formula_data[1].to(device), formula_data[2].to(device)

    ret = Data(edge_attr=edge_attr, edge_index=edge_index.t().contiguous(), x=nodes)

    edge_strings = parse_labels(formula_edge_label)
    edge_datas = []

    for edge in range(len(edge_strings)):
        edge_datas.append(gen_edge_prop_data(device, edge_strings[edge], edge_feature))
    return [ret, edge_datas]

def build_formula(device, node_feature,edge_feature, prop_indexs, ltl_feasible_relation, ltl_next_relation):

    all_props = []
    sub_next = []
    for next_relation in ltl_next_relation:
        for prop in prop_indexs:
            if str(prop) in next_relation and next_relation not in sub_next:
                sub_next.append(next_relation)
            if str(prop) in next_relation and "p{}".format(prop) not in all_props:
                all_props.append("p{}".format(prop))
    sub_feasible = []
    for feasible_relation in ltl_feasible_relation:
        for prop in prop_indexs:
            if str(prop) in feasible_relation and feasible_relation not in sub_feasible:
                sub_feasible.append(feasible_relation)
            if str(prop) in feasible_relation and "p{}".format(prop) not in all_props:
                all_props.append("p{}".format(prop))

    constraints, cons_str = generate_constraints(all_props)
    if len(sub_next) > 2:
        sub_next=np.random.choice(sub_next,2)
    if len(sub_feasible) >5 :
        sub_feasible=np.random.choice(sub_feasible, 5)

    graph_formula = trans2graph(sub_next, sub_feasible, cons_str)
    # print(graph_formula.formula)
    graph_data= numerical(device, graph_formula, node_feature, edge_feature)

    return graph_data

