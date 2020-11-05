import sys
import os
sys.path.append(os.getcwd())
from src.Synthetic.models.Util import *
import random
import copy
import os
import scipy.io as scio
import argparse




def generate_edge_prop(path,size):
    def generate_and_props(path, size):
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
                    # p.append("| " + item)
                return p

        def generateprop(op, prop):
            aps = prop.split(" ")
            ops = op.strip().split(" ")
            s = aps[0]
            for index in range(len(ops)):
                s += " " + ops[index] + " " + aps[index + 1]
            return s

        single_props = []
        for i in range(size):
            single_props.append("p{}".format(i))
        props = combinate(single_props)
        ops = []
        for ops_len in range(1, size):
            ops.append(op_combine(ops_len))
        prop_len = [[] for i in range(0, size)]
        for prop in props:
            prop = prop.strip()
            aps = prop.split(" ")
            if prop == "":
                continue
            prop_len[len(aps) - 1].append(prop)
        propositions = []
        for i in range(0, size):
            if i == 0:
                for prop in prop_len[i]:
                    propositions.append(prop)
            else:
                for prop in prop_len[i]:
                    for op in ops[i - 1]:
                        propositions.append(generateprop(op, prop))
        # write_strlist(path,propositions)
        return propositions
    def or_prop(and_prop,max_or,number):
        def permulation(start, end, num):
            if num == 1:
                return [[i] for i in range(start, end)]
            else:
                out = []
                for i in range(start, end):
                    nexts = permulation(i + 1, end, num - 1)
                    for next in nexts:
                        out.append([i] + next)
                return out

        or_props=copy.deepcopy(and_prop)
        for num_or in range(1,max_or):
            if size <4:
                props_indexs=permulation(0,len(and_prop),num_or+1)
                for props_index in props_indexs:
                    s=""
                    for i in range(num_or+1):
                        s+=and_prop[props_index[i]]
                        if i < num_or:
                            s+=" | "
                    or_props.append(s)
            else:
                for nums in range(number):
                    props=random.sample(and_prop,num_or+1)
                    s=""
                    for i in range(num_or+1):
                        s+=props[i]
                        if i <num_or:
                            s+=" | "
                    or_props.append(s)
        return or_props
    def true_complement(true_assigns,size):
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
        def complement(s, size):
            indexs = []
            for i in range(size):
                if "!p{}".format(i) in s:
                    indexs.append([-1 * (i + 1)])
                elif "p{}".format(i) in s:
                    indexs.append([i + 1])
                else:
                    indexs.append([-1*i-1,i+1])
            s_backs = [""]
            for index_i in indexs:
                s_news = []
                for s_back in s_backs:
                    for index in index_i:
                        if index < 0:
                            s_new =s_back + "!p{}".format(-index - 1)
                        else:
                            s_new =s_back+ "p{}".format(index - 1)
                        if abs(index) < size:
                            s_new += " & "
                        s_news.append(s_new)
                s_backs=s_news
            return s_backs

        def assign_complement(s, size):
            new_assigns = []
            subassigns = s.split(" | ")
            for subassign in subassigns:
                c_assign = complement(subassign, size)
                new_assigns+=c_assign
            set_assign=set(new_assigns)
            return list(set_assign)
        total_assign=set(generateassign(size,0))
        new_assigns=[]
        for true_assign in true_assigns:
            trues=assign_complement(true_assign,size)
            falses=list(total_assign-set(trues))
            if len(falses) < 1 or len(trues) < 1:
                continue
            new_assigns.append([true_assign,random.sample(trues*5,5),random.sample(falses*5,5)])
        return new_assigns
    and_prop=generate_and_props(path,size)
    or_props=or_prop(and_prop,5,1000)
    complement_true=true_complement(or_props,size)

    data={}
    data["size"]=size
    data["and_prop"]=and_prop
    data["final_prop"]=complement_true
    if not os.path.exists(path):
        os.mkdir(path)
    scio.savemat(path+"/"+str(size)+".mat",data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=bool, default=False, help="cuda")
    parser.add_argument("--dataset_root", type=str, default="./datasets/Synthetic/")
    parser.add_argument("--prop_size", type=int, default=3)
    args = parser.parse_args()

    generate_edge_prop(args.dataset_root+"edge_dataset/", args.prop_size)