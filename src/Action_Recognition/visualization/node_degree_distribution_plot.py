import matplotlib.pyplot as plt
import numpy as np

def dic2list(dic):
    keylist = []
    valuelist = []
    for i in dic.keys():
        keylist.append(i)
        valuelist.append(dic[i])
    return keylist, valuelist

node_degree_dist_data = \
    {'starting':
         {'outer_degree':  {18: 230, 8: 24, 16: 142, 9: 41, 7: 2, 14: 20, 10: 5, 12: 6, 6: 3, 1: 4, 3: 1},
          'inner_degree':  {1: 478}},
    'common':
        {'outer_degree':  {1: 948},
        'inner_degree':  {0: 948}},
    'accept':
         {'outer_degree':  {39: 230, 41: 230, 46: 230, 29: 44, 37: 183, 32: 72, 31: 121, 22: 8, 16: 8, 18: 5, 13: 3, 24: 2, 26: 2, 20: 1, 4: 1, 14: 2, 25: 1, 19: 1},
        'inner_degree':  {8: 490, 11: 230, 10: 142, 7: 200, 9: 61, 5: 7, 4: 5, 6: 8, 3: 1}},
    'reject':
         {'outer_degree':  {1: 474},
        'inner_degree':  {119: 230, 31: 24, 70: 62, 63: 17, 69: 56, 93: 24, 24: 2, 36: 18, 21: 2, 41: 2, 30: 2, 39: 14, 28: 2, 74: 2, 64: 10, 18: 1, 56: 1, 6: 1, 50: 1, 42: 1, 27: 1, 54: 1}}}


width =0.3
for k in node_degree_dist_data.keys():
    plt.bar(dic2list(node_degree_dist_data[k]['outer_degree'])[0], dic2list(node_degree_dist_data[k]['outer_degree'])[1], width=width, label = 'Outer')
    plt.bar(np.array(dic2list(node_degree_dist_data[k]['inner_degree'])[0])+ width, dic2list(node_degree_dist_data[k]['inner_degree'])[1], width=width, label='inner')
    plt.legend()
    plt.title(k)
    plt.savefig('./src/Action_Recognition/visualization/figs/node_degree_distribution_'+str(k)+'.pdf')
