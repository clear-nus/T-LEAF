import sys
import os
sys.path.append("/home/user/Documents/zf/ltlfkit/")
sys.path.append(os.getcwd())

import networkx as nx

from src.Synthetic.models.Util import *
from src.Synthetic.models.Graph import  *
from ltlf2hoa import *
from ltlf2nnf import *
import matplotlib.pyplot as plt
# import spot

INIT=0
COMMON=1
FINAL=2
REJECT=3
EDGE=4
REJECT_EDGE=5

edge2node=True

def aut_graph(ltl):
    # ltl="(! p2 U p1) | (G ! p2)"
    aut_fkit=finite2aut(ltl)
    # print(aut_fkit.to_str())
    nodes=aut2graph(aut_fkit)
    init_node = int(aut_fkit.get_init_state_number())
    graph_aut = Graph(ltl, nodes, init_node)
    return graph_aut



def graph_visual(graph_aut):
    nodes=graph_aut.nodes
    dg=nx.DiGraph()
    edge_labels, node_labels={}, {}
    nnode=len(nodes)
    for node in nodes:
        dg.add_node(node.id)
        node_labels[node.id]=node.label
        for edge in node.edges:
            dg.add_node(nnode)
            if edge.dst==LTL_FINITE_NO_ACC:
                node_labels[nnode]=REJECT_EDGE
            else:
                node_labels[nnode]=EDGE
            dg.add_edge(edge.src, nnode)
            dg.add_edge(nnode, edge.dst)
            nnode+=1
            edge_labels[(edge.src, edge.dst)]=edge.label
    return dg, node_labels, edge_labels

def visualize(dg,node_labels, edge_labels):
    plt.plot()
    nodes_accept=[ key for key in node_labels.keys() if node_labels[key]==FINAL]
    nodes_start=[ key for key in node_labels.keys() if node_labels[key]==INIT]
    nodes_edges=[ key for key in node_labels.keys() if node_labels[key]==EDGE]
    nodes_common=[ key for key in node_labels.keys() if node_labels[key]==COMMON]
    nodes_reject=[ key for key in node_labels.keys() if node_labels[key]==REJECT]
    nodes_reject_edge=[ key for key in node_labels.keys() if node_labels[key]==REJECT_EDGE]
    nx.draw_networkx_edges(dg,pos=nx.circular_layout(dg), edgelist=dg.edges)
    # nx.draw_networkx_edge_labels(dg, pos=nx.circular_layout(dg))
    nx.draw_networkx_nodes(dg, with_labels=True, pos=nx.circular_layout(dg),nodelist=nodes_accept, node_color="green", node_size=500)
    nx.draw_networkx_nodes(dg, pos=nx.circular_layout(dg),nodelist=nodes_start, node_color="yellow", node_size=500)
    nx.draw_networkx_nodes(dg, pos=nx.circular_layout(dg),nodelist=nodes_edges, node_color="orange", node_size=100)
    nx.draw_networkx_nodes(dg, pos=nx.circular_layout(dg),nodelist=nodes_common, node_color="blue", node_size=100)
    nx.draw_networkx_nodes(dg, pos=nx.circular_layout(dg),nodelist=nodes_reject, node_color="red", node_size=100)
    nx.draw_networkx_nodes(dg, pos=nx.circular_layout(dg),nodelist=nodes_reject_edge, node_color="black", node_size=100)
    # nx.draw(dg, pos=nx.circular_layout(dg), with_labels=True)
    plt.savefig('./src/Action_Recognition/visualization/figs/dfa_visualization.pdf')


def graph_info(ltl):
    info=[{ "outer_degree":{}, "inner_degree": {}} for i in range(4)]
    aut = finite2aut(ltl)
    num_states = aut.num_states()
    finals = []
    ids=[]
    inner_deg=[0 for i in range(num_states)]
    outer_deg=[0 for i in range(num_states)]
    for s in range(0, num_states):
        id=s
        if id == LTL_FINITE_NO_ACC:
            ids.append(REJECT)
        elif id == int(aut.get_init_state_number()):
            ids.append(INIT)
        else:
            ids.append(COMMON)
        for t in aut.out(s):
            inner_deg[t.dst]+=1
            outer_deg[t.src]+=1
            if not str(t.acc) == "{}":
                if t.dst not in finals:
                    finals.append(t.dst)
    for final in finals:
        ids[final]=FINAL
    for index in range(num_states):
        if inner_deg[index] not in info[ids[index]]["inner_degree"].keys():
            info[ids[index]]["inner_degree"][inner_deg[index]]=1
        else:
            info[ids[index]]["inner_degree"][inner_deg[index]] += 1
        if outer_deg[index] not in info[ids[index]]["outer_degree"].keys():
            info[ids[index]]["outer_degree"][outer_deg[index]]=1
        else:
            info[ids[index]]["outer_degree"][outer_deg[index]]+=1
    return info


def statistic(dir):
    infors=[]
    infor_total=[({},{}) for i in range(4)]
    videos=os.listdir(dir)
    for video in videos:
        path=dir+video+"/formula.txt"
        ltl=read_strlist(path)[0]
        info=graph_info(ltl)
        infors.append(str(video)+"\n"+str(info))
        for i in range(4):
            for key in info[i]["outer_degree"].keys():
                if key not in infor_total[i][0].keys():
                    infor_total[i][0][key]=info[i]["outer_degree"][key]
                else:
                    infor_total[i][0][key] += info[i]["outer_degree"][key]
            for key in info[i]["inner_degree"].keys():
                if key not in infor_total[i][1].keys():
                    infor_total[i][1][key] = info[i]["inner_degree"][key]
                else:
                    infor_total[i][1][key] += info[i]["inner_degree"][key]
        # print(info)
    # print(infor_total)
    # write_strlist(dir+"analysis.txt",infors)
    node_name=["starting","common","accept","reject"]
    # for i in range(4):
    #     print(node_name[i])
    #     print("outer_dgeree: ",infor_total[i][0])
    #     print("inner_degree: ", infor_total[i][1])

        # dg, node_labels, edge_labels=graph_visual(graph_aut)

# statistic("../../../dataset/Tasty_Video/base_rm2/")
ltl="((!p0 U p0) | G!p0) & ((!p1 U p1) | G!p1) & !p2 & !p3 & !p4 & !p5 & !p6 & G ( ( p0 & !p1 & !p2 & !p3 & !p4 & !p5 & !p6 ) |  ( p1 & !p0 & !p2 & !p3 & !p4 & !p5 & !p6 ) |  ( p2 & !p0 & !p1 & !p3 & !p4 & !p5 & !p6 ) |  ( p3 & !p0 & !p1 & !p2 & !p4 & !p5 & !p6 ) |  ( p4 & !p0 & !p1 & !p2 & !p3 & !p5 & !p6 ) |  ( p5 & !p0 & !p1 & !p2 & !p3 & !p4 & !p6 ) |  ( p6 & !p0 & !p1 & !p2 & !p3 & !p4 & !p5 )  )"
graph_aut=aut_graph(ltl)
dg, node_labels, edge_labels=graph_visual(graph_aut)
visualize(dg,node_labels, edge_labels)