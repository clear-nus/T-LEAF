import torch
from src.Synthetic.models.Util import finite2aut, aut2graph_finite
from src.Synthetic.models.Graph import Graph, Node, Edge
import re
import numpy as np

def read_strlist(path):
    '''

    :param path: strlist data path
    :return: list of string ["ab","cd]
    '''
    with open(path) as file:
        ltls = file.read()
        ltls = ltls.split("\n")[0:-1]
        return ltls

def read_formula(path):
    with open(path+"formula.txt") as file:
        formula=file.read()
        return formula

def read_cat(cat_path):

    ingredient_cat = read_strlist(cat_path+"ingredient_cat.txt")
    action_cat = read_strlist(cat_path + "action_cat.txt")
    return ingredient_cat, action_cat


def get_checker_loss(pred_actions, pred_objects, path_recipe, path_video, cat_path):
    global dataset_root, all, edge_embedder_path, edge_embedder
    dataset_root = path_video
    all = {}
    ingredients, actions = read_cat(cat_path)
    all["actions"] = actions
    all["ingredients"] = ingredients
    ltl_next_relation = read_strlist(path_recipe + "ltl_next_relations.txt")
    ltl_feasible_relation = read_strlist(path_recipe + "ltl_feasible_relations.txt")


    prop_dict = read_strlist(path_recipe + "/prop_dict.txt")
    prop_list = []
    for i in prop_dict:
        temp = i.replace(' ', '').split(':')
        prop_list.append((temp[1], temp[0]))

    p_indexs, temp, assign_str, props = process_pred(pred_actions, pred_objects, prop_list, path_recipe,  ltl_feasible_relation, ltl_next_relation)

    if temp is False:
        return 1

    # if not formula_regen:
    #     formula_txt = read_formula(path_recipe)
    #     aut = finite2aut(formula_txt)
    #     # print(aut.to_str())
    #     nodes = aut2graph_finite(aut)
    #     init_node = int(aut.get_init_state_number())
    #     graph_formula = Graph(formula_txt, nodes, init_node)
    # else:
    graph_formula = build_formula(p_indexs, ltl_feasible_relation, ltl_next_relation)
    if graph_formula.check_plist(assign_str,props):
        return 0
    else:
        return 1


def regenerate_prop(ltl_next_relation,ltl_feasible_relation, prop_indexs):
    all_props = []
    sub_next = []
    for next_relation in ltl_next_relation:
        for prop in prop_indexs:
            if str(prop) in next_relation and next_relation not in sub_next:
                sub_next.append(next_relation)
                all_props+=re.findall(r"\d+",next_relation)
    sub_feasible = []
    for feasible_relation in ltl_feasible_relation:
        for prop in prop_indexs:
            if str(prop) in feasible_relation and feasible_relation not in sub_feasible:
                sub_feasible.append(feasible_relation)
                all_props+=re.findall(r"\d+",feasible_relation)
    all_props=list(set(all_props))
    all_prop_num=[]
    for prop in all_props:
        all_prop_num.append(int(prop))

    return all_prop_num

def process_pred(pred_actions, pred_objects, prop_list, path_recipe, ltl_feasible_relation, ltl_next_relation):
    temp = []
    for i in range(len(pred_actions)):
        action_name = all['actions'][pred_actions[i]]
        object_name = all['ingredients'][pred_objects[i]]

        if (action_name, object_name) in prop_list:
            temp.append(prop_list.index((action_name, object_name)))

    if len(temp) == 0:
        return temp, False, [],[]


    prop_regen=regenerate_prop(ltl_next_relation,ltl_feasible_relation, temp)
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

    return temp, True, assign_str, prop_regen


def build_formula(prop_indexs, ltl_feasible_relation, ltl_next_relation):
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
        nodes = aut2graph_finite(aut)
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

    return graph_formula