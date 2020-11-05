import sys
sys.path.insert(1, '../../')
import spot

MINI_STATE=2
MINI_RUN=2

INIT=0
COMMON=1
FINAL=2

LABEL_SYMBOL=["a","b","c","d","e","f"]
LABEL_NUMERIC=[]




class Node():
    def __init__(self,id,edges,label):
        self.id=id
        self.edges=edges
        self.label=label

    def checker(self,s, label, size):
        sub_labels = label.split("|")
        indexs = []
        sub_indexs = [[] for i in range(len(sub_labels))]
        for i in range(size):
            if "!p{}".format(i) in s:
                indexs.append(-1 * (i + 1))
            elif "p{}".format(i) in s:
                indexs.append(i + 1)
            for j, sublabel in enumerate(sub_labels):
                if "!p{}".format(i) in sublabel:
                    sub_indexs[j].append(-1 * (i + 1))
                elif "p{}".format(i) in sublabel:
                    sub_indexs[j].append(i + 1)
        for sub_index in sub_indexs:
            count = 0
            for sub in sub_index:
                if sub not in indexs:
                    break
                count += 1
            if count == len(sub_index):
                return True
        return False

    # def valid_edge(self,edge_label,size):
    #     for edge in self.edges:
    #         if self.checker(edge_label,edge.label,size):
    #             return edge.dst
    #     return -1

class Edge():
    def __init__(self,src,dst,label):
        self.src=src
        self.dst=dst
        self.label=label

class Graph():
    def __init__(self,formula,nodes,init_node):
        self.formula=formula
        self.nodes=nodes
        self.true_assign=[]
        self.true_order=[]
        self.init_node=init_node
        self.visited=[False for i in range(len(nodes))]

    def find_assign(self):
        self.final_node=[]
        for node in self.nodes:
            if node.label==FINAL:
                self.final_node.append(node.id)
        def dfs(order,order_edge):
            if len(order)==0:
                return -1
            current_node = order[-1]
            if len(self.true_assign) >= 10:
                return
            self.visited[current_node] = True
            if self.nodes[order[-1]].label==FINAL:
                self.true_assign.append(order_edge.copy())
                self.true_order.append(order.copy())
            else:
                for edge in self.nodes[current_node].edges:
                    dst=int(edge.dst)
                    if self.visited[dst] == False:
                        order_edge.append(edge.label)
                        order.append(dst)
                        dfs(order,order_edge)
                        order.pop()
                        order_edge.pop()
                        self.visited[dst] = False
        order_edge=[]
        order=[]
        state=self.init_node
        order.append(state)
        dfs(order,order_edge)
        return self.true_assign
    def checker(self,s, label, size):
        if label=="1":
            return True
        if label=="0":
            return False
        sub_labels = label.split("|")
        indexs = []
        sub_indexs = [[] for i in range(len(sub_labels))]
        for i in range(size):
            if "!p{}".format(i) in s:
                indexs.append(-1 * (i + 1))
            elif "p{}".format(i) in s:
                indexs.append(i + 1)
            for j, sublabel in enumerate(sub_labels):
                if "!p{}".format(i) in sublabel:
                    sub_indexs[j].append(-1 * (i + 1))
                elif "p{}".format(i) in sublabel:
                    sub_indexs[j].append(i + 1)
        for sub_index in sub_indexs:
            count = 0
            for sub in sub_index:
                if sub not in indexs:
                    break
                count += 1
            if count == len(sub_index):
                return True
        return False

    def check(self,edges,size):
        index = 0
        now_state = [self.init_node]
        while index < len(edges) and len(now_state) > 0:
            next_state = []
            for state in now_state:
                for edge in self.nodes[state].edges:
                    if self.checker(edges[index], edge.label, size):
                            # and edge.dst != state:
                        # print(edges[index], ";", edge.label, ";", state)
                        next_state.append(edge.dst)
            # print(next_state)
            # for state in next_state:
            #     if self.nodes[state].label == FINAL:
            #         return True
            now_state = next_state
            index += 1
        for state in next_state:
            if self.nodes[state].label == FINAL:
                return True
        return False
    def checker_plist(self,s, label, props):
        sub_labels = label.split("|")
        indexs = []
        sub_indexs = [[] for i in range(len(sub_labels))]
        for i in props:
            if "!p{}".format(i) in s:
                indexs.append(-1 * (i))
            elif "p{}".format(i) in s:
                indexs.append(i)
            for j, sublabel in enumerate(sub_labels):
                if "!p{}".format(i) in sublabel:
                    sub_indexs[j].append(-1 * (i))
                elif "p{}".format(i) in sublabel:
                    sub_indexs[j].append(i)
        for sub_index in sub_indexs:
            count = 0
            for sub in sub_index:
                if sub not in indexs:
                    break
                count += 1
            if count == len(sub_index):
                return True
        return False

    def check_plist(self,edges,props):
        '''
        check the assign in edges in true or not
        :param edges:
        :param size:
        :return:
        '''
        index=0
        now_state = [self.init_node]
        while index < len(edges) and len(now_state) > 0:
            next_state = []
            for state in now_state:
                for edge in self.nodes[state].edges:
                    if self.checker_plist(edges[index], edge.label, props):
                        # print(edges[index], ";", edge.label, ";", state)
                        next_state.append(edge.dst)
            # print(next_state)
            # for state in next_state:
            #     if self.nodes[state].label == FINAL:
            #         return True
            now_state = next_state
            index += 1
        for state in next_state:
            if self.nodes[state].label == FINAL:
                return True
        return False
