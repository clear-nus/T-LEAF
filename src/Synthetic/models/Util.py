import sys
sys.path.insert(1, '../../')
import random
from src.Synthetic.models.Graph import Node , Graph ,Edge
import spot
import numpy as np
import os
import torch
from torch_geometric.data import Data
import sys
import pickle as pk

sys.path.append("/home/user/Documents/zf/ltlfkit/")
from ltlf2hoa import *

MINI_STATE=2
MINI_RUN=2

INIT=0
COMMON=1
FINAL=2

INPUT_SHAPE=50
OUTPUT_SHAPE=100

MAX_SIZE=50

Node_SIZE=3

LTL_FINITE_NO_ACC=2

def write_strlist(path,data):
    '''

    :param path: file path
    :param data: a list of string
    :return:
    '''
    s=""
    for item in data:
        s+=str(item)+"\n"
    with open(path,"w") as file:
        file.write(s)

def writetext(path,data):
    '''

    :param path: file path
    :param data: text data
    :return:
    '''
    with open(path) as file:
        file.write(data)

def read_strlist(path):
    '''

    :param path: strlist data path
    :return: list of string ["ab","cd]
    '''
    with open(path) as file:
        ltls=file.read()
        ltls=ltls.split("\n")[0:-1]
        return ltls

def readgraphData(dir,name):
    '''

    :param dir: directory
    :param name:
    :return:
    '''
    edge_attr=torch.load(dir+"/"+str(name)+".edge_attr")
    edge_index=torch.load(dir+"/"+str(name)+".edge_index")
    nodes=torch.load(dir+"/"+str(name)+".nodes")
    return Data(edge_attr=edge_attr,edge_index=edge_index.t().contiguous(), x=nodes)

def finite2aut(ltl):
    '''
    use ltlfkit to translate the finite ltl to graph
    :param ltl:
    :return:
    '''

    ltl_string=str(ltl)
    ltl_new=""
    for index in range(len(ltl_string)):
        s=ltl_string[index]
        if s=="p" or s==" ":
            ltl_new+=s
        else:
            ltl_new+=s+" "
    try:
        hoa=ltl2hoa(ltl_new)
        aut=spot.automaton(hoa)
        return aut
    except:
        return "-1"

def infinite2aut(ltl):
    '''
    use spot to translate the finite ltl into buchi automata
    :param ltl:
    :return:
    '''
    ltl_string=str(ltl)
    try:
        aut=spot.translate(ltl_string)
        return aut
    except:
        return -1
def aut2graph_finite(aut):
    '''

    :param aut: ltl finite automata to graph
    :return: return nodes of the graph
    '''
    bdict = aut.get_dict()
    num_states=aut.num_states()
    nodes=[]
    finals=[]
    for s in range(0, num_states):
        edges=[]
        id=s
        label=COMMON
        if id == int(aut.get_init_state_number()):
            label=INIT
        for t in aut.out(s):
            #
            if t.dst== LTL_FINITE_NO_ACC:
                continue
            edges.append(Edge(t.src,t.dst,spot.bdd_format_formula(bdict, t.cond)))
            if not str(t.acc)=="{}":
                finals.append(t.dst)
        nodes.append(Node(id,edges,label))
    for final in finals:
        nodes[final].label=FINAL
    return nodes



def aut2graph(aut):
    '''

    :param aut: automata to graph
    :return: return nodes of the graph
    '''
    bdict = aut.get_dict()
    num_states=aut.num_states()
    nodes=[]
    finals=[]
    for s in range(0, num_states):
        edges=[]
        id=s
        label=COMMON
        if s==0:
            label=INIT
        for t in aut.out(s):
            edges.append(Edge(t.src,t.dst,spot.bdd_format_formula(bdict, t.cond)))
            if not str(t.acc)=="{}":
                finals.append(t.dst)
        nodes.append(Node(id,edges,label))
    for final in finals:
        nodes[final].label=FINAL
    return nodes


def random_assignment(number,graph):
    '''
    used to gennerate the false assginments
    :param number: the number of false assignments
    :param graph: the graph for the formula
    :return:
    '''
    random.seed(number)
    assignment_item = generate_simple_item()
    num_prop=len(assignment_item)
    false_assinments=[]
    count=0
    while count < number:
        assigment=[]
        length=len(graph.true_assign[count])
        count_prop=0
        while count_prop != length:
            index=random.randint(0,num_prop-1)
            assigment.append(assignment_item[index])
            count_prop+=1
        if not graph.check(assigment):
            ltl_assign=assignment2ltl(assigment)
            count+=1
            false_assinments.append(ltl_assign)
    return false_assinments


def assignment2ltl(assignment):
    '''
    tranform a assignment to ltl
    :param assignment:
    :return:
    '''
    ltl_s=""
    i=len(assignment)-1
    while i>=0:
        if i==len(assignment)-1:
            ltl_s=assignment[i]
        else:
            ltl_s="{} & X ({})".format(assignment[i],ltl_s)
        i-=1
    return ltl_s

def ltls2graphs(ltls,Path):
    '''

    :param ltls: the origin dataset of ltls
    :return: graphs for each ltl, and true assgins and false assigns
    '''
    graphs=[]
    true_assigns=[]
    false_assigns=[]
    for index,ltl_string in enumerate(ltls):
        ltl=spot.formula(ltl_string)
        aut=ltl.translate()
        nodes=aut2graph(aut)
        init_node=int(aut.get_init_state_number())
        graph=Graph(ltl_string, nodes, init_node)
        path=Path+str(index+1)+"/"
        true_ltl=read_strlist(path+"trueltl.txt")
        false_ltl=read_strlist(path+"falseltl.txt")
        true_assigns.append(true_ltl)
        false_assigns.append(false_ltl)
        graphs.append(graph)
    return graphs,true_assigns,false_assigns


def numeric_all(dir):
    '''

    :param dir:
    :return:
    '''
    prop_feature = np.random.random([MAX_SIZE,INPUT_SHAPE])
    node_feature = np.random.random([Node_SIZE,INPUT_SHAPE])

    symbols = ["t", "f"]
    singles = ["F", "X", "G", "!"]
    doubles = ["U", "V", "&", "|", "M", "R", "W", "i", "e", "^"]

    syntax_feature = np.random.random([len(singles+doubles),INPUT_SHAPE])

    syntax_feature = np.concatenate((np.ones([1,INPUT_SHAPE]),np.zeros([1,INPUT_SHAPE]),syntax_feature),axis=0)

    np.save(dir+"/syntax", syntax_feature)
    np.save(dir+"/prop", prop_feature)
    np.save(dir+"/node", node_feature)


# numeric_all("../../dataset/Synthetic")

def readtxtfile(path):
    with open(path) as file:
        s=file.read().split("\n")
        return s[0:-1]
def read_pk_file(file_path):
    with open(file_path,"rb") as file:
        ret=pk.load(file, )
    return ret
