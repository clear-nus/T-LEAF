import sys
import os
sys.path.append(os.getcwd())
import torch.nn.functional as F
from src.Action_Recognition.models.embedder_loss_util import generate_constraints_size, trans2graph, replace_edge_attr, build_formula, process_pred, gen_edge_prop_data, graph2numer
from torch_geometric.data import Data
import torch
import random

def get_embedder_loss(device, pred_actions, pred_objects, meta_embedder,edge_embedder, info):
    ingreds = None

    graph_data, pred_data = sample_formula(device, ingreds, pred_actions, pred_objects, edge_embedder, info)
    if graph_data is None:
        return 1

    loss = F.pairwise_distance(meta_embedder(graph_data), meta_embedder(pred_data))
    return loss[0]

def sample_formula(device, ingreds, pred_actions, pred_objects, edge_embedder, info):
    all, feature, node_feature, word_embedding, order_rules, affordable_rules, actions_name, ingreds_name, WORD_DIMESION, INGREDS_SAMPLE_SIZE, ORDER_RULE_SAMPLE_SIZE, FEASIBLE_RULE_SAMPLE_SIZE, dataset_root = info

    ings, prop_dict, next_relation, ltl_next_relation, feasible_relation, ltl_feasible_relation = extract_relation_single(ingreds, pred_actions, pred_objects, info)

    size = len(prop_dict)

    constraints, cons_str = generate_constraints_size(size)
    graph_ltl_video = trans2graph(ltl_next_relation, ltl_feasible_relation, cons_str)


    formula_data, formula_edge_label, prop_dict, prop_dict_numerical = \
        numerical_video_dataset_single(word_embedding, graph_ltl_video, size, prop_dict, node_feature)


    ################################################

    prop_list = []
    for i in prop_dict:
        temp = i.replace(' ', '').split(':')
        prop_list.append((temp[1], temp[0]))
    for i in range(prop_dict_numerical.shape[0]):
        feature["p{}".format(i)] = prop_dict_numerical[i]
        feature["!p{}".format(i)] = 1 - prop_dict_numerical[i]

    p_indexs, temp, pred_data, edge_strings = process_pred(device, pred_actions, pred_objects, prop_list,
                                                               ltl_feasible_relation, ltl_next_relation, feature, all,dataset_root, glove_features=None, action_cat=None, ingredient_cat=None, differentiable=False)


    if temp is False:
        return None, None

    graph_data = build_formula(device, node_feature, feature, p_indexs, ltl_feasible_relation, ltl_next_relation)

    graph_data_new = replace_edge_attr(device, graph_data, edge_embedder, dataset_root)
    pred_data_new = replace_edge_attr(device, pred_data, edge_embedder, dataset_root)

    return graph_data_new, pred_data_new


def extract_relation_single(ings=None, pred_actions=None, pred_objects=None, info=None):
    '''

    :param steps:
    :return:
    '''
    all, feature, node_feature, word_embedding, order_rules, affordable_rules, actions_name, ingreds_name, WORD_DIMESION, INGREDS_SAMPLE_SIZE, ORDER_RULE_SAMPLE_SIZE, FEASIBLE_RULE_SAMPLE_SIZE, dataset_root = info
    prop_dict = {}
    prop_count = 0
    if ings is None and pred_actions is None:
        ings = random.sample(range(len(ingreds_name)), INGREDS_SAMPLE_SIZE)
    elif ings is None:
        ings = list(set(pred_objects))

    order_rule_ings = []
    affordable_rule_ings = []
    if pred_actions is None:
        for rule in order_rules:
            if rule[0][1] in ings:
                order_rule_ings.append(rule)
        for rule in affordable_rules:
            if rule[1] in ings:
                affordable_rule_ings.append(rule)
    else:
        for rule in order_rules:
            if rule[0][1] in pred_objects and ((rule[0][0] in pred_actions) or (rule[1][0] in pred_actions)):
                order_rule_ings.append(rule)
        for rule in affordable_rules:
            if rule[1] in pred_objects and (rule[0] in pred_actions):
                affordable_rule_ings.append(rule)
        # print ('order_rule_ings', order_rule_ings)
        # print ('affordable_rule_ings', affordable_rule_ings)

    if len(order_rule_ings) > ORDER_RULE_SAMPLE_SIZE:
        order_rule_sub = random.sample(order_rule_ings, ORDER_RULE_SAMPLE_SIZE)
    else:
        order_rule_sub = order_rule_ings
    if len(affordable_rule_ings) > FEASIBLE_RULE_SAMPLE_SIZE:
        affordable_rule_sub = random.sample(affordable_rule_ings, FEASIBLE_RULE_SAMPLE_SIZE)
    else:
        affordable_rule_sub = affordable_rule_ings
    ltl_next_video = []
    for pair in order_rule_sub:
        # ingredient,action_first,action_second=pair[0],pair[1],pair[2]
        # build the prop dictionary
        prop_key1 = ingreds_name[pair[0][1]] + " : " + actions_name[pair[0][0]]
        prop_key2 = ingreds_name[pair[1][1]] + " : " + actions_name[pair[1][0]]
        if prop_key1 in prop_dict.keys():
            a1 = prop_dict[prop_key1]
        else:
            a1 = prop_count
            prop_dict[prop_key1] = prop_count
            prop_count += 1
        if prop_key2 in prop_dict.keys():
            a2 = prop_dict[prop_key2]
        else:
            a2 = prop_count
            prop_dict[prop_key2] = prop_count
            prop_count += 1
        ltl_next_video.append("((!p{} U p{}) | G!p{})".format(a2, a1, a2))
    ltl_not_feasible_video = []
    for pair in affordable_rule_sub:
        ingredient, action = ingreds_name[pair[1]], actions_name[pair[0]]
        prop_key = ingredient + " : " + action
        if prop_key in prop_dict.keys():
            a = prop_dict[prop_key]
        else:
            a = prop_count
            prop_dict[prop_key] = prop_count
            prop_count += 1
        ltl_not_feasible_video.append("!p{}".format(a))
    return ings, prop_dict, order_rule_sub, ltl_next_video, affordable_rule_sub, ltl_not_feasible_video


def numerical_video_dataset_single(word_embedding, graph_formula, size, prop_dict, node_feature):
    '''
    build the numerical video dataset, include the formula assigns text, and numerical formula and asssigns, prop_dict
    :param prop_dict:
    :param next_relation:
    :param graph_ltl_video:
    :param true_assigns:
    :param false_assigns:
    :return:
    '''
    WORD_DIMESION = 100

    # build prop dictionary numerical dataset
    prop_dict_numerical=torch.zeros([size,2*WORD_DIMESION])
    for prop in prop_dict.keys():
        prop_splits=prop.split(" : ")
        ingredient, action= prop_splits[0], prop_splits[1]
        if ingredient not in word_embedding.keys():
            word_embedding[ingredient]=torch.rand([WORD_DIMESION],dtype=torch.float32)
            print("miss",ingredient)
        if action not in word_embedding.keys():
            word_embedding[action]=torch.rand([WORD_DIMESION],dtype=torch.float32)
            print("miss",action)
        value=torch.cat((word_embedding[ingredient],word_embedding[action]),dim=0)
        prop_dict_numerical[prop_dict[prop]]=value


    #build numerical Dataset
    formula_data, formula_edge_label=graph2numer(graph_formula, node_feature)
    return formula_data, formula_edge_label, prop_dict, prop_dict_numerical