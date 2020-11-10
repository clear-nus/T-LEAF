import sys
import os
sys.path.append(os.getcwd())

import torch
from src.Synthetic.models.Graph import *
from src.Synthetic.models.Util import read_pk_file, finite2aut, aut2graph, write_strlist, read_strlist
import scipy.io as scio
import numpy as np
import os
import random
import argparse
import pickle as pk


def extract_relation_recipe(steps):
    '''

    :param steps:
    :return:
    '''
    prop_dict = {}
    prop_count = 0
    # extract the ingredient and action in list
    actions, ingredients = [], []
    for step in steps:
        for substep in step:
            actions.append(substep[0])
            ingredients += substep[1]

    actions = list(set(actions))
    ingredients = list(set(ingredients))

    ing_act_next_video = []
    for ing in ingredients:
        for action_first in actions:
            for action_second in actions:
                if "{} : {} : {}".format(ing, action_first, action_second) in next_relation_all:
                    ing_act_next_video.append([ing, action_first, action_second])

    ing_act_not_feasible_video = []
    for ing in ingredients:
        for action in actions:
            if "{} : {}".format(ing, action) not in feasible_relation_all:
                ing_act_not_feasible_video.append([ing, action])
    if len(ing_act_next_video) > args.order_sample_size:
        ing_act_next_video = random.sample(ing_act_next_video, args.order_sample_size)
    if len(ing_act_not_feasible_video) > args.affordable_sample_size:
        ing_act_not_feasible_video = random.sample(ing_act_not_feasible_video, args.affordable_sample_size)
    ltl_next_video = []
    for pair in ing_act_next_video:
        ingredient, action_first, action_second = pair[0], pair[1], pair[2]
        # build the prop dictionary
        prop_key1 = ingredient + " : " + action_first
        prop_key2 = ingredient + " : " + action_second
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
    for pair in ing_act_not_feasible_video:
        ingredient, action = pair[0], pair[1]
        prop_key = ingredient + " : " + action
        if prop_key in prop_dict.keys():
            a = prop_dict[prop_key]
        else:
            a = prop_count
            prop_dict[prop_key] = prop_count
            prop_count += 1
        ltl_not_feasible_video.append("!p{}".format(a))
    return prop_dict, ing_act_next_video, ltl_next_video, ing_act_not_feasible_video, ltl_not_feasible_video


def extract_relation():
    '''

    :param steps:
    :return:
    '''
    prop_dict = {}
    prop_count = 0
    ings = random.sample(range(len(ingreds_name)), args.ingreds_sample_size)
    order_rule_ings = []
    for rule in order_rules:
        if rule[0][1] in ings:
            order_rule_ings.append(rule)
    affordable_rule_ings = []
    for rule in affordable_rules:
        if rule[1] in ings:
            affordable_rule_ings.append(rule)
    if len(order_rule_ings) > args.order_sample_size:
        order_rule_sub = random.sample(order_rule_ings, args.order_sample_size)
    else:
        order_rule_sub = order_rule_ings
    if len(affordable_rule_ings) > args.affordable_sample_size:
        affordable_rule_sub = random.sample(affordable_rule_ings, args.affordable_sample_size)
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


def generate_constraints(size):
    '''
    generate the list of one prop true constrains, both list and Global string
    :param size:
    :return:
    '''
    constraints=[]
    constraint_string="G ("
    for i in range(size):
        constraint_item=""
        constraint_string += " ( "
        constraint_item += "p{}".format(i)
        for j in range(size):
            if j==i:
                continue
            constraint_item += " & !p{}".format(j)
        constraint_string += constraint_item
        constraint_string += " ) | "
        constraints.append(constraint_item)
    constraint_string=constraint_string[0:-2]
    constraint_string+= " )"

    return constraints, constraint_string

def trans2graph(ltl_next_relation, ltl_feasible_relation, constraint_string):
    '''
    combine next and feasible ltl, then transform to automata graph
    :param ltl_next_relation:
    :param ltl_feasible_relation:
    :return:
    '''
    ltl_all=""
    # print(len(ltl_next_relation),len(ltl_feasible_relation))
    # print(constraint)
    for ltl in ltl_next_relation:
        ltl_all+=ltl+" & "
    for ltl in ltl_feasible_relation:
        ltl_all += ltl +" & "
    ltl_all=ltl_all + constraint_string
    # print(ltl_all)
    aut=finite2aut(ltl_all)
    # print(aut.to_str())
    nodes=aut2graph(aut)
    init_node=int(aut.get_init_state_number())
    graph=Graph(ltl_all,nodes,init_node)
    return graph

def generateassigns(graph,constraints,size):
    '''
    generate true assigns and corresponding false assigns
    :param graph:
    :param constraints:
    :return:
    '''
    true_assigns= graph.find_assign()
    # print(graph.true_order)
    false_assigns=[]
    for assign in true_assigns:
        steps=len(assign)
        while True:
            false_assign=random.sample(constraints,steps)
            if not graph.check(false_assign,size):
                # print(false_assign)
                false_assigns.append(false_assign)
                break
    return true_assigns, false_assigns


def generate_finite_assign(edge):
    '''
    generate the true assign and false assign for each edge label.
    :param edge:
    :return:
    '''
    p_nots = ["!p{}".format(i) for i in range(size)]
    ps = ["p{}".format(i) for i in range(size)]

    prop_splits = edge.split(" & ")
    indexs = []
    false_indexs = []
    for prop in prop_splits:
        if prop in p_nots:
            indexs.append(-1 * p_nots.index(prop) - 1)
            false_indexs.append((p_nots.index(prop) + 1))
        if prop in ps:
            indexs.append(ps.index(prop) + 1)
            false_indexs.append(-1 * ps.index(prop) - 1)
    num_trues = 2 ** (size - len(indexs))
    if num_trues > 5:
        num_trues = 5
    true_assigns = []
    false_assigns = []
    count = 0
    while count < num_trues:
        true_assign = []
        for i in range(size):
            if -1 * (i + 1) in indexs:
                true_assign.append(-1 * (i + 1))
                continue
            if i + 1 in indexs:
                true_assign.append(i + 1)
                continue
            if random.random() < 0.5:
                true_assign.append(i + 1)
            else:
                true_assign.append(-1 * (i + 1))
        false_item = random.sample(false_indexs, 1)
        false_assign = true_assign.copy()
        false_assign[abs(false_item[0]) - 1] = false_item[0]
        # print(false_assign,true_assign)
        s_false, s_true = "", ""
        for index in range(size):
            if true_assign[index] < 0:
                s_true += "!p{}".format(index)
            else:
                s_true += "p{}".format(index)
            if false_assign[index] < 0:
                s_false += "!p{}".format(index)
            else:
                s_false += "p{}".format(index)
            if index < size - 1:
                s_true += " & "
                s_false += " & "
        true_assigns.append(s_true)
        false_assigns.append(s_false)
        count += 1
    # print("label: ",edge, "true :", true_assigns , "false: ", false_assigns)
    return true_assigns, false_assigns

def build_edgedataset(graph_formula,size):
    '''
    build edge dataset, and formulate the true and false assigns,
    :param graph:
    :return:
    '''

    all_edges=[]

    for index,node in enumerate(graph_formula.nodes):
        for edge in node.edges:
            if edge.label not in all_edges:
                all_edges.append(edge.label)

    edge_dataset=[]
    for edge in all_edges:
        if edge == "1" or edge=="0":
            continue
        edge_true_assigns, edge_false_assigns = generate_finite_assign(edge)
        if len(edge_true_assigns)==0 or len(edge_false_assigns)==0:
            continue
        edge_dataset.append([edge,edge_true_assigns,edge_false_assigns])
    return edge_dataset


def graph2numer(graph):
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


def assign2numer(assign):
    '''
    numerical the assign
    :param assigns:
    :return:
    '''
    nodes = []
    edge_index = []
    for index in range(len(assign) + 1):
        if index == 0:
            nodes.append(node_feature[0])
        elif index == len(assign):
            nodes.append(node_feature[2])
            edge_index.append([index - 1, index])
        else:
            nodes.append(node_feature[1])
            edge_index.append([index - 1, index])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    nodes = torch.tensor(nodes, dtype=torch.float32)
    edge_attr = torch.zeros([edge_index.shape[0], WORD_DIMESION * 2], dtype=torch.float32)
    return [edge_index, nodes, edge_attr], assign

def numerical_video_dataset(name_receipt, word_embedding, graph_formula, true_assigns, false_assigns):
    '''
    build the numerical video dataset, include the formula assigns text, and numerical formula and asssigns, prop_dict
    :param prop_dict:
    :param next_relation:
    :param graph_ltl_video:
    :param true_assigns:
    :param false_assigns:
    :return:
    '''

    # build prop dictionary numerical dataset
    prop_dict_numerical = torch.zeros([size, 2 * WORD_DIMESION])
    for prop in prop_dict.keys():
        prop_splits = prop.split(" : ")
        ingredient, action = prop_splits[0], prop_splits[1]
        if ingredient not in word_embedding.keys():
            word_embedding[ingredient] = torch.rand([WORD_DIMESION], dtype=torch.float32)
            miss_key.append(ingredient)
            print("miss", ingredient)
        if action not in word_embedding.keys():
            word_embedding[action] = torch.rand([WORD_DIMESION], dtype=torch.float32)
            miss_key.append(action)
            print("miss", action)
        value = torch.cat((word_embedding[ingredient], word_embedding[action]), dim=0)
        prop_dict_numerical[prop_dict[prop]] = value

    video_path = dataset_path + "/" + name_receipt
    if not os.path.exists(video_path):
        os.mkdir(video_path)
    write_strlist(video_path + "/prop_dict.txt", prop_dict)
    torch.save(prop_dict_numerical, video_path + "/prop_dict_numerical")

    # wrtire formula, true assigns and false assgins, relations,
    write_strlist(video_path + "/formula.txt", [graph_formula.formula])
    write_strlist(video_path + "/false_assigns.txt", false_assigns)
    write_strlist(video_path + "/true_assigns.txt", true_assigns)

    write_strlist(video_path + "/next_relations.txt", next_relation)
    write_strlist(video_path + "/ltl_next_relations.txt", ltl_next_relation)
    write_strlist(video_path + "/feasible_relations.txt", feasible_relation)
    write_strlist(video_path + "/ltl_feasible_relations.txt", ltl_feasible_relation)

    formula_data, formula_edge_label = graph2numer(graph_formula)
    torch.save(formula_data[0], video_path + "/formula.edge_index")
    torch.save(formula_data[1], video_path + "/formula.nodes")
    torch.save(formula_data[2], video_path + "/formula.edge_attr")
    write_strlist(video_path + "/formula.edge_label", formula_edge_label)

    path_true = video_path + "/true_numeric"
    if not os.path.exists(path_true):
        os.mkdir(path_true)
    for index, assign in enumerate(true_assigns):
        assign_data, assign_edge_label = assign2numer(assign)
        torch.save(assign_data[0], path_true + "/{}.edge_index".format(index))
        torch.save(assign_data[1], path_true + "/{}.nodes".format(index))
        torch.save(assign_data[2], path_true + "/{}.edge_attr".format(index))
        write_strlist(path_true + "/{}.edge_label".format(index), assign_edge_label)

    path_false = video_path + "/false_numeric"
    if not os.path.exists(path_false):
        os.mkdir(path_false)
    for index, assign in enumerate(false_assigns):
        assign_data, assign_edge_label = assign2numer(assign)
        torch.save(assign_data[0], path_false + "/{}.edge_index".format(index))
        torch.save(assign_data[1], path_false + "/{}.nodes".format(index))
        torch.save(assign_data[2], path_false + "/{}.edge_attr".format(index))
        write_strlist(path_false + "/{}.edge_label".format(index), assign_edge_label)

def numerical_edge_dataset(prop_dict, edge_dataset, name_receipt):
    '''
    build edge dataset one by one, include the prop dict, edge formula ,edge false and true assigns.
    :param edge_dataset:
    :return:
    '''
    # build prop dictionary numerical dataset
    prop_dict_numerical = torch.zeros([size, 2 * WORD_DIMESION])
    for prop in prop_dict.keys():
        prop_splits = prop.split(" : ")
        ingredient, action = prop_splits[0], prop_splits[1]
        value = torch.cat((word_embedding[ingredient], word_embedding[action]), dim=0)
        prop_dict_numerical[prop_dict[prop]] = value
    edge_path = edge_dataset_path + "/" + name_receipt
    if not os.path.exists(edge_path):
        os.mkdir(edge_path)
    write_strlist(edge_path + "/prop_dict.txt", prop_dict)
    torch.save(prop_dict_numerical, edge_path + "/prop_dict_numerical")

    # formula text
    data = {}
    # data["prop_dict"]=prop_dict
    data["edge_dataset"] = edge_dataset
    scio.savemat(edge_path + "/edge_dataset.mat", data)

def read_steps(dir,name):
    def extract(s):
        actions=s.split("; ")[0:-1]
        act_split=[]
        for action in actions:
            temp = action.split(" : ")
            verb = temp[0].strip()
            objects= temp[1].replace("[","").replace("]","").replace("'","")
            objects_new=[]
            for object in objects.split(","):
                object_new=object.strip().lower()
                if object_new=="":
                    continue
                objects_new.append(object_new)
            act_split.append([verb,objects_new])
        return act_split
    path=dir+name
    with open(path+"/steps.txt","r") as f:
        file=f.read()
        steps=[]
        for line in file.split("\n"):
            if line.strip() =="" or line =="() : ) ":
                continue
            step=extract(line)
            if len(step)==0:
                continue
            steps.append(step)
        return steps

def readrecipes(path):
    '''
    read all recipe data and process:

    data[recipe] = [recipes[steps[substeps[verb,objects[]]]]]
    :return:
    '''
    recipes=os.listdir(path)
    kitchen=[]
    name=[]
    data={}
    for recipe in recipes:
        recipe_dic=read_steps(path,recipe)
        name.append(recipe.strip())
        kitchen.append(recipe_dic)
    data["name"] = name
    data["recipe"]=kitchen
    return data

def read_allrelation(path):
    feasible_relation=read_strlist(path+"feasible_relation.txt")
    next_relation = read_strlist(path+"next_relation.txt")

    return feasible_relation, next_relation

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", default=False, action="store_true")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="random seed for reprobudicibility")
    parser.add_argument("--rule_path", type=str, default="./src/Imitation_Cooking/data/cooking_rules/")
    parser.add_argument("--num_formula", type=int, default=300)
    parser.add_argument("--ingreds_sample_size", type=int, default=2)
    parser.add_argument("--order_sample_size", type=int, default=2)
    parser.add_argument("--affordable_sample_size", type=int, default=5)
    parser.add_argument("--dataset_root", type=str, default="./datasets/Action_Recognition/")
    parser.add_argument("--dataset_edge_name", type=str, default="/edge_embedder_dataset_test/")
    parser.add_argument("--dataset_meta_name", type=str, default="/meta_embedder_dataset_test/")
    parser.add_argument("--recipe_path", type=str, default="/recipes/")
    parser.add_argument("--video_info_path", type=str, default="/videos/info/")
    args = parser.parse_args()






    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + str(args.device_id) if args.cuda else 'cpu')


    torch.manual_seed(args.seed)
    random.seed(args.seed)

    WORD_DIMESION = 100

    node_feature = np.load(args.dataset_root + "node.npy", allow_pickle=True)

    # action_feature = read_pk_file(args.rule_path + "/actions_features.pk")
    # ingreds_feature = read_pk_file(args.rule_path + "/ingreds_features.pk")
    # affordable_rules = read_pk_file(args.rule_path + "/affordable.pk")
    # order_rules = read_pk_file(args.rule_path + "/ordering.pk")
    # actions_name = read_pk_file(args.rule_path + "/actions_name.pk")
    # ingreds_name = read_pk_file(args.rule_path + "/ingreds_name.pk")

    feasible_relation_all, next_relation_all = read_allrelation(args.dataset_root +args.video_info_path)
    origin_embedding = pk.load(open(args.dataset_root + args.video_info_path + "rel_glove_features_6B100", "rb"))
    word_embedding = {}
    for key in origin_embedding.keys():
        word_embedding[key] = torch.tensor(origin_embedding[key], dtype=torch.float32)

    # word_embedding = {}
    # for key in action_feature.keys():
    #     word_embedding[actions_name[key]] = torch.tensor(action_feature[key], dtype=torch.float32)
    # for key in ingreds_feature.keys():
    #     word_embedding[ingreds_name[key]] = torch.tensor(ingreds_feature[key], dtype=torch.float32)

    dataset_path = args.dataset_root + args.dataset_meta_name
    edge_dataset_path = args.dataset_root + args.dataset_edge_name
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    if not os.path.exists(edge_dataset_path):
        os.mkdir(edge_dataset_path)

    data = readrecipes(args.dataset_root+args.recipe_path)  # modification
    recipes = data["recipe"]
    name = data["name"]






    formula_dataset = []
    miss_key = []
    # index = 0
    # while index < args.num_formula:
    for index, recipe in enumerate(recipes):
        prop_dict, next_relation, ltl_next_relation, feasible_relation, ltl_feasible_relation = extract_relation_recipe(recipe)
        size = len(prop_dict)
        if size == 0:
            print(" dataset ", " no reasonable relation")
            continue
        print(index, prop_dict, next_relation, feasible_relation)
        constraints, cons_str = generate_constraints(size)
        graph_ltl_video = trans2graph(ltl_next_relation, ltl_feasible_relation, cons_str)
        true_assigns, false_assigns = generateassigns(graph_ltl_video, constraints, size)
        edge_dataset_video = build_edgedataset(graph_ltl_video, size)
        formula_dataset.append([graph_ltl_video.formula, true_assigns, false_assigns])



        numerical_video_dataset(name[index], word_embedding, graph_ltl_video, true_assigns, false_assigns)
        numerical_edge_dataset(prop_dict, edge_dataset_video, name[index])
        index += 1

    print(miss_key)

