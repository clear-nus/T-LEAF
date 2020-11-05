import sys
import os
sys.path.append(os.getcwd())
import spot
import torch
import numpy as np
import os
import random
from src.Synthetic.models.Util import *
from src.Synthetic.models.Graph import *
import time
import sys
import argparse




def generatetextltl(path,number_props,tree_size,priority=""):
    '''
    generatee textltl and the
    :param path: the path store textltl
    :param number_props: the number of props
    :param tree_size: the tree size of ltl automata, means the complexity of the props
    :param priority: frequency detail about the operator in ltl
    :return:
    '''

    def checkfalse(number,graph, true_assigns):
        '''
        used to gennerate the false assginments
        :param number: the number of false assignments
        :param graph: the graph for the formula
        :return:
        '''
        def generateassign(size, index):
            if index == size - 1:
                return ["!p{}".format(index), "p{}".format(index)]
            else:
                now_items = ["!p{} & ".format(index), "p{} & ".format(index)]
                ret = []
                next_items = generateassign(size, index + 1)
                for item in now_items:
                    for nextitem in next_items:
                        ret.append(item + nextitem)
                return ret
        new=True
        if new:
            assignment_item=generateassign(number_props,0)
        num_prop = len(assignment_item)
        count = 0
        times=0
        while count < number:
            assigment = []
            length = len(true_assigns[count])
            count_prop = 0
            while count_prop != length:
                times+=1
                index = random.randint(0, num_prop - 1)
                assigment.append(assignment_item[index])
                count_prop += 1
            if not graph.check(assigment,Size):
                count+=1
                times=0
            if times>40:
                print("no false for",graph.formula)
                return False
        return True

    ltls=[]
    f=spot.randltl(number_props,allow_dups=False,tree_size=tree_size,
                   ltl_priorities=priority)
    count=0

    while count < samples:
        ltl=next(f)
        ltl_string=str(ltl)
        # print(ltl_string)
        if str(ltl) == "1" or str(ltl)=="0":
            continue
        t1=time.time()
        print(str(ltl),count)
        aut = finite2aut(ltl)
        if str(aut) == "-1":
            continue
        nodes=aut2graph(aut)
        init_node=int(aut.get_init_state_number())
        graph=Graph(ltl_string,nodes,init_node)
        if len(graph.nodes)>20:
            print(ltl_string,"nodes too many")
            continue
        graph.find_assign()
        if len(graph.true_assign) >=5 and checkfalse(len(graph.true_assign),graph,graph.true_assign):
            print(ltl_string,time.time()-t1)
            ltls.append(ltl)
            count+=1
        else:
            print("graph true assigns length",len(graph.true_assign))
    write_strlist(path,ltls)

def generate_props(path,size):
    '''

    :param path: the path to store the props
    :param size: the size of ltls
    :return:
    '''

    def combinate(prop):
        if len(prop) == 0:
            return [""]
        else:
            props = []
            p1 = combinate(prop[1:])
            for item in p1:
                props.append(prop[0] + " " + item)
                props.append(item)
                props.append("!" + prop[0] + " " + item)
            return props

    def op_combine(num):
        if num == 0:
            return [""]
        else:
            ops = op_combine(num - 1)
            p = []
            for item in ops:
                p.append("& " + item)
                p.append("| " + item)
            return p

    def generateprop(op, prop):
        aps = prop.split(" ")
        ops = op.strip().split(" ")
        s = aps[0]
        for index in range(len(ops)):
            s += " " + ops[index] + " " + aps[index + 1]
        return s
    single_props=[]
    for i in range(size):
        single_props.append("p{}".format(i))
    props=combinate(single_props)
    ops=[]
    for ops_len in range(1,size):
        ops.append(op_combine(ops_len))
    prop_len=[[] for i in range(0,size)]
    for prop in props:
        prop=prop.strip()
        aps=prop.split(" ")
        if prop == "":
            continue
        prop_len[len(aps)-1].append(prop)
    propositions=[]
    for i in range(0,size):
        if i==0:
            for prop in prop_len[i]:
                propositions.append(prop)
        else:
            for prop in prop_len[i]:
                for op in ops[i-1]:
                    propositions.append(generateprop(op,prop))
    # print(len(propositions),propositions)
    propositions.append("1")
    write_strlist(path,propositions)

def numeric_props(size):
    '''
    generate the props and numeric the props
    :param size: the size of props
    :return:
    '''
    def prop2value(p, prop):
        ps = ["p{}".format(i) for i in range(size)]
        ps += ["!p{}".format(i) for i in range(size)]
        if prop == "1":
            return np.ones(INPUT_SHAPE)
        elif prop == "0":
            return np.zeros(INPUT_SHAPE)
        else:
            aps = prop.split(" ")
            value = []
            for i, ap in enumerate(aps):
                if ap in ps:
                    value.append(p[ps.index(ap)])
                elif ap == "&":
                    value.append("*")
                elif ap == "|":
                    value.append("+")
            value_1 = []
            i = 0
            count = 0
            while i < len(value):
                op = value[i]
                if str(op) == "*":
                    value_1[count - 1] = value_1[count - 1] * value[i + 1]
                    i += 1
                elif str(op) == "+":
                    i += 1
                    continue
                else:
                    value_1.append(value[i])
                    count += 1
                i += 1
            avg = np.mean(value_1,axis=0)
            avg = -np.log(avg)
            avg = avg/np.linalg.norm(avg)
            return avg
    if not os.path.exists(Dataset+"/props.txt"):
        generate_props(Dataset+"/props.txt",size)
    props = readtxtfile(Dataset+"/props.txt")
    prop_feature = np.load(SyntheticData+"/prop.npy", allow_pickle=True)
    prop_feature = prop_feature[0:size]
    nodes_feature = np.load(SyntheticData+"/node.npy", allow_pickle=True)
    ones=np.ones([INPUT_SHAPE])
    p_both=np.concatenate([prop_feature,ones-prop_feature])
    props_assign=[]
    for prop in props:
        value=prop2value(p_both,prop)
        props_assign.append(value)
    np.save(Dataset+"/prop",props_assign)
    np.save(Dataset+"/node",nodes_feature)

def syntaxgraph2nume(graph,nodes_labels,num_label):
    nodes=[]
    edge_index=[]
    for node in graph.nodes:
        if node.label in nodes_labels:
            index=nodes_labels.index(node.label)
        else:
            print("not find the syntax label" + node.label)
            continue
        nodes.append(num_label[index])
        for edge in node.edges:
            edge_index.append([edge.src,edge.dst])
    edge_index=torch.tensor(edge_index,dtype=torch.long)
    nodes=torch.tensor(nodes,dtype=torch.float32)
    return nodes,edge_index



def syntax_generate(ltl,index,edge_feature,proposition_name):

    symbols = []
    num_label = []
    for i in range(Size):
        symbols.append("p{}".format(i))
        num_label.append(edge_feature[proposition_name.index("p{}".format(i))])
    symbols += ["t", "f"]
    singles = ["F", "X", "G", "!"]
    doubles = ["U", "V", "&", "|", "M", "R", "W", "i", "e", "^"]
    nodes_labels = symbols + singles + doubles

    symbol_feature = np.load(SyntheticData+"/syntax.npy", allow_pickle=True)

    num_label = np.vstack((np.array(num_label), symbol_feature))
    def syntaxgraph(ltl):
        s = ltl
        s_formula = spot.formula(s)
        s_lbt = s_formula.to_str("lbt").split(" ")
        count = len(s_lbt) - 1
        nodes = []
        for index, symbol in enumerate(s_lbt):
            node = Node(index, [], symbol)
            nodes.append(node)
        items = []
        while count >= 0:
            if s_lbt[count] in symbols:
                items.append(count)
                count -= 1
                continue
            elif s_lbt[count] in singles:
                edge = Edge(count, items[-1], "")
                items[-1] = count
                nodes[count].edges.append(edge)
                count -= 1
            elif s_lbt[count] in doubles:
                edge1 = Edge(count, items[-1], "")
                edge2 = Edge(count, items[-2], "")
                items = items[0:-2]
                items.append(count)
                nodes[count].edges.append(edge1)
                nodes[count].edges.append(edge2)
                count -= 1
        return nodes

    def ltl_transform(s):
        assigns = s.split(" & X ")
        str_assigns = ""
        for index, assign in enumerate(assigns):
            if index < len(assigns) - 1:
                str_assigns += "X" * (index + 1) + assign + " & "
            else:
                str_assigns += "X" * (index + 1) + assign
        return str_assigns

    nodes = syntaxgraph(ltl)
    graph = Graph(ltl, nodes, 0)
    nodes, edge_index = syntaxgraph2nume(graph, nodes_labels, num_label)
    torch.save(edge_index, Path_Dataset + "/" + str(index + 1) + "/syntax.edge_index")
    torch.save(nodes, Path_Dataset + "/" + str(index + 1) + "/syntax.nodes")
    trues_syntax = Path_Dataset + "/" + str(index + 1) + "/true_syntax"
    false_syntax = Path_Dataset + "/" + str(index + 1) + "/false_syntax"
    if not os.path.exists(trues_syntax):
        os.mkdir(trues_syntax)
        os.mkdir(false_syntax)
    trues_assignment = read_strlist(Path_Dataset + "/" + str(index + 1) + "/trueltl.txt")
    false_assignment = read_strlist(Path_Dataset + "/" + str(index + 1) + "/falseltl.txt")
    for f_index, f_ltl in enumerate(false_assignment):
        if f_ltl == "":
            continue
        f_ltl=ltl_transform(f_ltl)
        f_nodes = syntaxgraph(f_ltl)
        f_graph = Graph(f_ltl, f_nodes, 0)
        f_nodes, f_edge_index = syntaxgraph2nume(f_graph, nodes_labels, num_label)
        torch.save(f_edge_index,
                   Path_Dataset + "/" + str(index + 1) + "/false_syntax/" + str(f_index) + "syntax.edge_index")
        torch.save(f_nodes, Path_Dataset + "/" + str(index + 1) + "/false_syntax/" + str(f_index) + "syntax.nodes")
    for t_index, t_ltl in enumerate(trues_assignment):
        if t_ltl == "":
            continue
        t_ltl = ltl_transform(t_ltl)
        t_nodes = syntaxgraph(t_ltl)
        t_graph = Graph(t_ltl, t_nodes, 0)
        t_nodes, t_edge_index = syntaxgraph2nume(t_graph, nodes_labels, num_label)
        torch.save(t_edge_index,
                   Path_Dataset + "/" + str(index + 1) + "/true_syntax/" + str(t_index) + "syntax.edge_index")
        torch.save(t_nodes, Path_Dataset + "/" + str(index + 1) + "/true_syntax/" + str(t_index) + "syntax.nodes")



def generatenumdataset():
    '''
    write the dataset into the disk
    :return:
    '''
    node_feature = np.load(Dataset+"/node.npy", allow_pickle=True)
    edge_feature = np.load(Dataset+"/prop.npy", allow_pickle=True)
    proposition_name = readtxtfile(Dataset+"/props.txt")
    print(node_feature.shape)

    def get_edge_feature(label):
        if label == "1":
            return np.ones([INPUT_SHAPE])
        if label == "0":
            return np.zeros([INPUT_SHAPE])
        if label in proposition_name:
            return edge_feature[proposition_name.index(label)]
        return label_simplify(label)

    def label_simplify(label):
        value = []
        while "(" in label:
            if "(" in label:
                left = label.index("(")
                right = label.index(")")
                sub_string = label[left + 1:right]
                value.append(proposition_name.index(sub_string))
                if right < len(label) - 1:
                    value.append(label[right + 2])
                    label = label[right + 4:]
                else:
                    label = ""
        value_1 = []
        count = 0
        for i in range(len(value)):
            if value[i] == "&":
                value_1[count - 1] = value_1[count-1] * edge_feature[value[i + 1]]
            elif value[i] == "|":
                value_1.append(np.zeros([INPUT_SHAPE]))
                count += 1
            else:
                value_1.append(edge_feature[value[i]])
                count += 1
        avg = np.mean(value_1, axis=0)
        avg = -np.log(avg)
        avg = avg / np.linalg.norm(avg)
        return avg

    def get_edge_string(dir, graph, true_assign, false_index):
        if args.ltlfkit:
            graph_edge_string =graph
        else:
            aut_spot=infinite2aut(graph.formula)
            nodes=aut2graph(aut_spot)
            init_node = int(aut_spot.get_init_state_number())
            graph_edge_string = Graph(graph.formula, nodes, init_node)
        formular_edge = []
        for index, node in enumerate(graph_edge_string.nodes):
            for edge in node.edges:
                formular_edge.append(edge.label)
        write_strlist(dir + "/formulaedge_string.txt", formular_edge)
        for index, assign in enumerate(true_assign):
            write_strlist(dir + "/true_numeric/" + str(index) + "edge_string.txt", assign)
        for index, indexs in enumerate(false_index):
            string_edges = []
            for x in indexs:
                string_edges.append(proposition_name[x])
            write_strlist(dir + "/false_numeric/" + str(index) + "edge_string.txt", string_edges)

    def assignment2ltl(assignment):
        '''
        tranform a assignment to ltl
        :param assignment:
        :return:
        '''
        ltl_s = ""
        i = len(assignment) - 1
        while i >= 0:
            if i == len(assignment) - 1:
                ltl_s = "({})".format(assignment[i])
            else:
                ltl_s = "({}) & X {}".format(assignment[i], ltl_s)
            i -= 1
        return ltl_s

    def random_assignment(number, graph, true_assigns,new):
        '''
        used to gennerate the false assginments
        :param number: the number of false assignments
        :param graph: the graph for the formula
        :return:
        '''
        if new:
            assignment_item=[]
            assign_indexs=[]
            index=0
            while index < len(proposition_name):
                temp=0
                if "|" in proposition_name[index]:
                    index+=1
                    continue
                for i in range(Size):
                    if "p{}".format(i) not in proposition_name[index]:
                        temp=1
                        break
                if temp==0:
                    assign_indexs.append(index)
                    assignment_item.append(proposition_name[index])
                index+=1
        else:
            assignment_item = proposition_name

        num_prop = len(assignment_item)
        false_assinments = []
        false_index = []
        count = 0
        while count < number:
            assigment = []
            indexs = []
            length = len(true_assigns[count])
            count_prop = 0
            while count_prop != length:
                index = random.randint(0, num_prop - 1)
                if new:
                    indexs.append(assign_indexs[index])
                else:
                    indexs.append(index)
                assigment.append(assignment_item[index])
                count_prop += 1
            if not graph.check(assigment,Size):
                false_index.append(indexs)
                ltl_assign = assignment2ltl(assigment)
                count += 1
                false_assinments.append(ltl_assign)
        return false_assinments, false_index

    def numerical(graph, true_edge_features, false_indexs, index):
        def graph2numer(graph):
            nodes = []
            edge_index = []
            edge_attr = []
            for index, node in enumerate(graph.nodes):
                if node.label == INIT:
                    nodes.append(node_feature[0])
                elif node.label == FINAL:
                    nodes.append(node_feature[2])
                else:
                    nodes.append(node_feature[1])
                for edge in node.edges:
                    edge_index.append([edge.src, edge.dst])
                    edge_attr.append(get_edge_feature(edge.label))
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            nodes = torch.tensor(nodes, dtype=torch.float32)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
            return [edge_index, nodes, edge_attr]

        def falseassign2numer(assgin):
            nodes = []
            edge_index = []
            edge_attr = []
            for index in range(len(assgin) + 1):
                if index == 0:
                    nodes.append(node_feature[0])
                elif index == len(assgin):
                    prop_index = assgin[index - 1]
                    nodes.append(node_feature[2])
                    edge_index.append([index - 1, index])
                    edge_attr.append(edge_feature[prop_index])
                else:
                    prop_index = assgin[index - 1]
                    nodes.append(node_feature[1])
                    edge_index.append([index - 1, index])
                    edge_attr.append(edge_feature[prop_index])
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            nodes = torch.tensor(nodes, dtype=torch.float32)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
            return [edge_index, nodes, edge_attr]

        def trueassign2numer(true_edgefeature):
            nodes = []
            edge_index = []
            for index in range(len(true_edgefeature) + 1):
                if index == 0:
                    nodes.append(node_feature[0])
                elif index == len(true_edgefeature):
                    nodes.append(node_feature[2])
                    edge_index.append([index - 1, index])
                else:
                    nodes.append(node_feature[1])
                    edge_index.append([index - 1, index])
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            nodes = torch.tensor(nodes, dtype=torch.float32)
            edge_attr = torch.tensor(true_edgefeature, dtype=torch.float32)
            return [edge_index, nodes, edge_attr]

        if args.ltlfkit:
            graph_data = graph2numer(graph)
        else:
            aut_spot=infinite2aut(graph.formula)
            nodes=aut2graph(aut_spot)
            init_node = int(aut_spot.get_init_state_number())
            graph_spot = Graph(graph.formula, nodes, init_node)
            graph_data=graph2numer(graph_spot)
        false_data = []
        trues_data = []
        for true_edge_feature in true_edge_features:
            data = trueassign2numer(true_edge_feature)
            trues_data.append(data)
        for false_index in false_indexs:
            data = falseassign2numer(false_index)
            false_data.append(data)
        path_true = Path_Dataset + "/" + str(index + 1) + "/true_numeric"
        if not os.path.exists(path_true):
            os.mkdir(path_true)
        for i in range(len(trues_data)):
            torch.save(trues_data[i][0], path_true + "/" + str(i) + ".edge_index")
            torch.save(trues_data[i][1], path_true + "/" + str(i) + ".nodes")
            torch.save(trues_data[i][2], path_true + "/" + str(i) + ".edge_attr")
        path_false = Path_Dataset + "/" + str(index + 1) + "/false_numeric"
        if not os.path.exists(path_false):
            os.mkdir(path_false)
        for i in range(len(false_data)):
            torch.save(false_data[i][0], path_false + "/" + str(i) + ".edge_index")
            torch.save(false_data[i][1], path_false + "/" + str(i) + ".nodes")
            torch.save(false_data[i][2], path_false + "/" + str(i) + ".edge_attr")
        torch.save(graph_data[0], Path_Dataset + "/" + str(index + 1) + "/formula.edge_index")
        torch.save(graph_data[1], Path_Dataset + "/" + str(index + 1) + "/formula.nodes")
        torch.save(graph_data[2], Path_Dataset + "/" + str(index + 1) + "/formula.edge_attr")

    def true_complement(true_assigns,size):
        def complement(s, size):
            indexs = []
            for i in range(size):
                if "!p{}".format(i) in s:
                    indexs.append(-1 * (i + 1))
                elif "p{}".format(i) in s:
                    indexs.append(i + 1)
                else:
                    x = random.random()
                    if x < 0.5:
                        indexs.append(i + 1)
                    else:
                        indexs.append(-1 * (i + 1))
            s_back = ""
            for index in indexs:
                if index < 0:
                    s_back += "!p{}".format(-index - 1)
                else:
                    s_back += "p{}".format(index - 1)
                if abs(index) < size:
                    s_back += " & "
            return s_back

        def assign_complement(s, size):
            # assigns = s.split("& X ")
            assigns=s
            # print(assigns)
            new_assigns = [[] for i in range(len(assigns))]
            for index, assign in enumerate(assigns):
                subassigns = assign.split("|")
                for subassign in subassigns:
                    c_assign = complement(subassign, size)
                    new_assigns[index].append(c_assign)
            sample_assign = []
            for subassign in new_assigns:
                sample_assign.append(random.sample(subassign, 1)[0])
            # print(sample_assign)
            return sample_assign
        new_assigns=[]
        for i in range(3):
            for true_assign in true_assigns:
                new_assigns.append(assign_complement(true_assign,size))
        return new_assigns

    Path_ltl=Dataset+"/ltls"
    ltls=read_strlist(Dataset+"/formula.txt")
    new=True
    count=0
    for i,ltl_string in enumerate(ltls):
        print(ltl_string)
        ltl=spot.formula(ltl_string)
        # print(count)
        aut=finite2aut(ltl)
        if str(aut)=="-1":
            continue
        nodes=aut2graph(aut)
        init_node=int(aut.get_init_state_number())
        graph=Graph(ltl_string,nodes,init_node)
        graph.find_assign()
        if new:
            true_assign = true_complement(graph.true_assign,Size)
        else:
            true_assign = graph.true_assign
        true_edge_feature=[]
        for assign in true_assign:
            features=[]
            for assign_item in assign:
                if assign_item not in proposition_name:
                    print("assign item", assign_item)
                feature=get_edge_feature(assign_item)
                if np.max(feature)>1:
                    print(">1",assign,np.max(feature),np.argmax(feature))
                features.append(feature)
            if len(features)==0:
                continue
            true_edge_feature.append(features)
        true_assign_ltl=[]
        for tr_as in true_assign:
            if tr_as==[]:
                continue
            true_assign_ltl.append(assignment2ltl(tr_as))
        false_assign_ltl,false_index=random_assignment(len(true_assign_ltl),graph,true_assign,new)
        count+=1
        dir=Path_ltl+"/"+str(count)
        if not os.path.exists(dir):
            os.mkdir(dir)
        write_strlist(dir+"/trueltl.txt",true_assign_ltl)
        write_strlist(dir+"/falseltl.txt",false_assign_ltl)
        write_strlist(dir+"/formula.txt",[ltl_string])

        numerical(graph,true_edge_feature,false_index,i)

        syntax_generate(ltl_string,i,edge_feature,proposition_name)
        get_edge_string(dir, graph, true_assign, false_index)


def generatedataset(dataset_root,name, size, num, treesize,priority):
    global ltltype, Size, samples, Dataset, Path_Dataset, SyntheticData
    ltltype = name
    Size = size
    samples = num
    SyntheticData=dataset_root
    Dataset = dataset_root+"{}".format(ltltype)
    Path_Dataset = dataset_root+"{}/ltls".format(ltltype)
    if not os.path.exists(Dataset):
        os.mkdir(Dataset)
        os.mkdir(Path_Dataset)
    generatetextltl(Dataset+"/formula.txt",Size,tree_size=treesize,priority=priority)
    generate_props(Dataset+"/props.txt",Size)
    numeric_props(Size)
    generatenumdataset()

def syntax_retrans(dataset_root,name, size, num,):
    def syntaxgraph(ltl):
        s = ltl
        s_formula = spot.formula(s)
        s_lbt = s_formula.to_str("lbt").split(" ")
        count = len(s_lbt) - 1
        nodes = []
        for index, symbol in enumerate(s_lbt):
            node = Node(index, [], symbol)
            nodes.append(node)
        items = []
        while count >= 0:
            if s_lbt[count] in symbols:
                items.append(count)
                count -= 1
                continue
            elif s_lbt[count] in singles:
                edge = Edge(count, items[-1], "")
                items[-1] = count
                nodes[count].edges.append(edge)
                count -= 1
            elif s_lbt[count] in doubles:
                edge1 = Edge(count, items[-1], "")
                edge2 = Edge(count, items[-2], "")
                items = items[0:-2]
                items.append(count)
                nodes[count].edges.append(edge1)
                nodes[count].edges.append(edge2)
                count -= 1
        return nodes

    ltltype=name
    Size=size
    samples = num
    Dataset = dataset_root+"{}".format(ltltype)
    Path_Dataset = dataset_root+"{}/ltls".format(ltltype)
    ltls = read_strlist(Dataset + "/formula_nnf.txt")
    node_feature = np.load(Dataset+"/node.npy", allow_pickle=True)
    edge_feature = np.load(Dataset+"/prop.npy", allow_pickle=True)
    proposition_name = readtxtfile(Dataset+"/props.txt")
    symbols = []
    num_label = []
    for i in range(Size):
        symbols.append("p{}".format(i))
        num_label.append(edge_feature[proposition_name.index("p{}".format(i))])
    symbols += ["t", "f"]
    singles = ["F", "X", "G", "!"]
    doubles = ["U", "V", "&", "|", "M", "R", "W", "i", "e", "^"]
    nodes_labels = symbols + singles + doubles

    symbol_feature = np.load(dataset_root+"syntax.npy", allow_pickle=True)
    num_label = np.vstack((np.array(num_label), symbol_feature))
    for index, ltl_string in enumerate(ltls):
        print(ltl_string)
        ltl = spot.formula(ltl_string)
        nodes = syntaxgraph(ltl)
        graph = Graph(ltl, nodes, 0)
        nodes, edge_index = syntaxgraph2nume(graph, nodes_labels, num_label)
        torch.save(edge_index, Path_Dataset + "/" + str(index + 1) + "/syntax.edge_index")
        torch.save(nodes, Path_Dataset + "/" + str(index + 1) + "/syntax.nodes")


def trans2noFG(dataset_root,name):
    Dataset = dataset_root+"{}".format(name)
    ltls = read_strlist(Dataset + "/formula.txt")
    ltl_news=[]
    for index, ltl_string in enumerate(ltls):
        print(ltl_string)
        ltl_news.append(spot.unabbreviate(ltl_string,"GWRF"))
    print(ltl_news)
    write_strlist(Dataset+"/formula_noFG.txt",ltl_news)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=bool, default=False, help="cuda")
    parser.add_argument("--dataset_root", type=str, default="./datasets/Synthetic/")
    parser.add_argument("--prop_size", type=int, default=3)
    parser.add_argument("--num_formula", type=int, default=1000)
    parser.add_argument("--tree_size", type=int, default=10)
    parser.add_argument("--ltlfkit", type=bool, default=False, help="whether to use ltlfkit")
    args = parser.parse_args()

    dataset_name = str(args.prop_size)+'_'+str(args.tree_size)

    generatedataset(args.dataset_root,dataset_name, args.prop_size,args.num_formula, args.tree_size, priority="equiv=0, implies=0, R=0, W=0, M=0, xor=0")
