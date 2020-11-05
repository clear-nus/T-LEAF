# import sys
# import os
# sys.path.append(os.getcwd())

import random
from gym import spaces
from copy import deepcopy
import pickle as pk

class Env_Cooking():
    def __init__(self, affordable_rules=None, dependency_rule=None, min_num_ingred = 5, max_num_ingred = 5, no_reward = False, checker = False, max_steps = 100):
        super(Env_Cooking, self).__init__()
        self.no_reward = no_reward

        self.min_num_ingred = min_num_ingred
        self.max_num_ingred = max_num_ingred
        self.affordable_rules = affordable_rules
        self.dependency_rule = dependency_rule
        self.checker_reward = checker
        self.max_steps = max_steps


        ############################### add in properties and merging
        ## mixture is a special ingredient
        # 1. it's added for each task
        # 2. only its' properties can be changed (get all properties of other ingredients combined to him

        ## Conflict properties: setup the dominant property

        ## One action one peroperty currently, extend to multiple later

        ## primary ingredient won't get secondary ingredient's property
        # the secondery ingredient's properties will disappear

        ## actions
        # 1. add-ingred1: append all properties of ingred1 to mixture (resolve conflicts)
        # 2. combine-ingred1-ingred2: add-ingred1, add-ingred2
        # 3. top_with-ingred1-ingred2: keep ingred1 (primary ingred) and remove ingred2 (secondary ingred)



        self.actions_prop = {0:'wash', 1:'cut', 2:'add', 3:'cook', 4:'top_with', 5:'combine', 6:'shape', 7:'cool', 8:'pour', 9:'peel', 10:'blend'}
        self.actions = {0: 'wash', 1: 'cut', 2:'add', 3:'cook', 4:'top_with',5:'combine',6:'shape',7:'cool', 8: 'pour', 9:'peel',\
                        10: 'chope', 11: 'garnish_with', 12: 'season_with', 13: 'drizzle',  14: 'mix', \
                        15: 'fold',  16: 'freeze' , 17: 'brush_with', 18: 'bake',
                        19: 'heat', 20: 'crack', 21: 'seal', 22: 'blend', \
                        23: 'fry', 24: 'boil', 25: 'coat_with', 26: 'press', 27: 'beat', 28: 'melt', 29: 'clean',
                        30: 'rinse', 31: 'steam', 32:'fold', 33:'seal', 34:'leave'}
        self.actions_map = {0: [0], 1: [1], 2:[2], 3:[3], 4:[4],5:[5],6:[6],7:[7], 8: [8],  9:[9],\
                        10: [1], 11: [4], 12: [4], 13: [8],  14: [10], \
                        15: [6],  16: [6] , 17: [4], 18: [3],
                        19: [3], 20: [1], 21: [7], 22: [10], \
                        23: [3], 24: [3], 25: [4], 26: [6], 27: [10], 28: [3], 29: [0],
                        30: [0], 31: [3], 32:[6], 33:[6], 34: [7]}

        self.ingreds_prop = {0:'pad', 1: 'mixture', 2: 'liquid', 3:'powder', 4:'vegetable', 5:'fruit', 6:'seasoning', 7:'egg', 8:'solid', 9:'fat', 10:'seafood',\
                            11:'bread', 12:'grain', 13:'pasta', 14:'cream', 15:'meat'}
        self.ingreds = {0:'pad', 1: 'mixture', 2: 'liquid', 3: 'powder', 4: 'vegetable', 5: 'fruit', 6:'seasoning',  7:'egg', 8:'solid', 9:'fat', 10:'seafood',\
                            11:'bread',12:'grain',13:'pasta',\
                        14: 'beef', 15: 'pork', 16: 'milk', 17: 'flour', 18: 'carrot', 19: 'corn', 20: 'apple', 21: 'sugar',
                        22: 'salt', 23:'pepper', 24:'banana', 25:'oil', 26:'butter',27:'chicken',28:'fish',29:'crab',30:'garlic',\
                        31:'chocolate', 32:'water',33:'onion', 34:'syrup', 35:'cherry',36:'lemon',37:'cream',38:'dough',39:'soup',\
                        40:'batter',41:'salmon',42:'bean',43:'tomato',44:'potato',45:'zucchini',46:'juice',47:'cake_mix', 48:'strawberry',\
                        49:'mushroom',50:'meat'}
        self.ingreds_map = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6],
                       7: [7], 8: [8], 9: [9], 10: [10], \
                       11: [11], 12: [12], 13: [13], \
                       14: [15], 15: [15], 16: [2], 17: [3], 18: [4], 19: [4], 20: [5],
                       21: [6], 22: [6], 23: [6], 24: [5], 25: [2,9], 26: [8,9], 27: [5], 28: [10],
                       29: [10], 30: [6], 31: [8], 32: [2], 33: [6,4], 34: [2,6], 35: [5], 36: [5,6], 37: [14],
                       38: [8], 39: [2], 40: [2], 41: [10], 42: [12], 43: [4], 44: [4], 45: [4], 46: [2],47:[3],48:[5], 49:[4], 50:[15]}

        self.ingred_states = {0:'raw', 1:'preprocessed', 2:'heated', 3:'seasoned', 4:'ready', 5:'postprocessed'}

        # self.actions_prop = {0: 'wash', 1: 'cut', 2: 'add', 3: 'cook', 4: 'top_with', 5: 'combine', 6: 'shape',
        #                      7: 'cool', 8: 'pour', 9: 'peel', 10: 'blend'}
        # self.ingreds_prop = {0: 'pad', 1: 'mixture', 2: 'liquid', 3: 'powder', 4: 'vegetable', 5: 'fruit',
        #                      6: 'seasoning', 7: 'egg', 8: 'solid', 9: 'fat', 10: 'seafood', \
        #                      11: 'bread', 12: 'grain', 13: 'pasta', 14: 'cream', 15: 'meat'}

        # action property prereq and effect
        self.actions_prereq = {0: [0], 1: [0], 2: [0], 3: [1, 3], 4: [1, 2], 5: [0], 6: [4], 7: [4], 8: [0], 9: [0], 10: [0]}
        self.actions_effect = {0: [1], 1: [1], 2: [0], 3: [2, 4], 4: [3, 4], 5: [0], 6: [5], 7: [5], 8: [0], 9: [1], 10: [1]}

        # done requirement for each ingredient property (take the max property)
        self.done_requirement = {0: 0, 1: 5, 2: 0, 3: 0, 4: 4, 5: 1, 6: 0, 7: 2, 8: 0, 9: 0, 10: 4, 11: 0, 12: 4, 13: 4, 14: 0, 15: 4}

        # action properties can/cannot be appplied to ingred properties
        if self.affordable_rules is None:
            # remove padding
            # self.affordable_rules = {0: [None, [0,2,3,6,7.9,11,14]], 1: [None, [0, 2,3,6,12,13,14]], 2: [None, [0, 1]], 3: [None, [0]], \
            #                          4: [None, [[0],[0,1]]], 5: [None, [0,1]], 6: [None, [0,2,3,12]], 7: [None, [0]], 8: [[0,2,14,12], None], \
            #                          9: [[4,5], None], 10: [[1,2,3,7,14], None]}

            self.affordable_rules = {0: [None, [2, 3, 6, 7.9, 11, 14]], 1: [None, [2, 3, 6, 12, 13, 14]],
                                     2: [None, [1]], 3: [None, []], \
                                     4: [None, [[], [1]]], 5: [None, [1]], 6: [None, [2, 3, 12]],
                                     7: [None, []], 8: [[2, 14, 12], None], \
                                     9: [[4, 5], None], 10: [[1, 2, 3, 7, 14], None]}
        if self.dependency_rule is None:
            self.dependency_rule = {}
            for i in self.actions_prop:
                for j in self.ingreds_prop:
                    self.dependency_rule[(i,j)] = self.actions_prereq[i]

        self.action_space = spaces.MultiDiscrete([len(self.actions), (max_num_ingred + 1), (max_num_ingred + 1)])
        self.observation_space = spaces.MultiDiscrete(
            [len(self.ingreds), len(self.ingred_states)] * (max_num_ingred + 1))
        self.spec = [(max_num_ingred + 1) * 2, 3]



    def _check_affordance(self, action, ingred, ingred2):
        # print (ingred, ingred2)
        if ingred == 1:
            ingreds_prop = self.mixture_prop
        else:
            ingreds_prop = self.ingreds_map[ingred]
        if ingred2 == 1:
            ingreds_prop2 = self.mixture_prop
        else:
            ingreds_prop2 = self.ingreds_map[ingred2]


        for act_prop in self.actions_map[action]:
            # combine
            if act_prop == 5:
                if ingred == ingred2:
                    return False
                if self.affordable_rules[act_prop][1] is None:
                    for ingred_prop in ingreds_prop:
                        if ingred_prop not in self.affordable_rules[act_prop][0]:
                            return False
                    for ingred_prop in ingreds_prop2:
                        if ingred_prop not in self.affordable_rules[act_prop][0]:
                            return False
                elif self.affordable_rules[act_prop][0] is None:
                    for ingred_prop in ingreds_prop:
                        if ingred_prop in self.affordable_rules[act_prop][1]:
                            return False
                    for ingred_prop in ingreds_prop2:
                        if ingred_prop in self.affordable_rules[act_prop][1]:
                            return False
            # top_with
            elif act_prop == 4:
                if ingred == ingred2:
                    return False
                if self.affordable_rules[act_prop][1] is None:
                    for ingred_prop in ingreds_prop:
                        if ingred_prop not in self.affordable_rules[act_prop][0][0]:
                            return False
                    for ingred_prop in ingreds_prop2:
                        if ingred_prop not in self.affordable_rules[act_prop][0][1]:
                            return False
                elif self.affordable_rules[act_prop][0] is None:
                    for ingred_prop in ingreds_prop:
                        if ingred_prop in self.affordable_rules[act_prop][1][0]:
                            return False
                    for ingred_prop in ingreds_prop2:
                        if ingred_prop in self.affordable_rules[act_prop][1][1]:
                            return False
            else:
                if self.affordable_rules[act_prop][1] is None:
                    for ingred_prop in ingreds_prop:
                        if ingred_prop not in self.affordable_rules[act_prop][0]:
                            return False
                elif self.affordable_rules[act_prop][0] is None:
                    for ingred_prop in ingreds_prop:
                        if ingred_prop in self.affordable_rules[act_prop][1]:
                            return False
        return True

    def _check_dependency(self, action, ingred, ingred_idx):
        if ingred == 1:
            ingreds_prop = self.mixture_prop
        else:
            ingreds_prop = self.ingreds_map[ingred]

        for act_prop in self.actions_map[action]:
            for ingred_prop in ingreds_prop:
                if self.state[ingred_idx*2+1] < min(self.dependency_rule[(act_prop, ingred_prop)]):
                    return False
        return True

    def _check_done(self):
        # hard coded rules for each ingredient category
        for i in range(int(len(self.state)/2)):
            for ingred_prop in self.ingreds_map[self.state[i*2]]:
                if self.state[i * 2+1] < self.done_requirement[ingred_prop]:
                    return False
        return True



    def step(self, a):
        self.counter += 1
        if not isinstance(a,tuple):
            if len(a.shape) > 1:
                a = a.squeeze(1)

        reward = 0
        self.history['states'].append(self.state)

        action, ingred_idx, ingred_idx2 = a
        ingred = self.state[ingred_idx*2]
        ingred2 = self.state[ingred_idx2 * 2]



        if self._check_affordance(action, ingred, ingred2) and self._check_dependency(action, ingred, ingred_idx):
            max_effect_state = -1
            for idx in range(len(self.actions_effect[self.actions_map[action][0]])):
                if (self.actions_prereq[self.actions_map[action][0]][idx] <= self.state[ingred_idx*2+1]) and (max_effect_state <= self.actions_effect[self.actions_map[action][0]][idx]):
                    max_effect_state = self.actions_effect[self.actions_map[action][0]][idx]
                    apply_idx = idx
            if self.state[ingred_idx*2+1] < self.actions_effect[self.actions_map[action][0]][apply_idx]:
                self.state[ingred_idx*2+1] = self.actions_effect[self.actions_map[action][0]][apply_idx]

            # deal with special merging actions
            # add
            if (2 in self.actions_map[action]) or (8 in self.actions_map[action]):
                self.mixture_prop += deepcopy(self.ingreds_map[ingred])
                self.state[1] = max(self.state[1], self.state[ingred_idx * 2 + 1])
                self.state[ingred_idx * 2] = 0
                self.state[ingred_idx * 2 + 1] = 0

            # combine
            elif (5 in self.actions_map[action]):
                self.mixture_prop += deepcopy(self.ingreds_map[ingred])
                self.mixture_prop += deepcopy(self.ingreds_map[ingred2])
                self.state[1] = max(self.state[1], self.state[ingred_idx * 2 + 1], self.state[ingred_idx2 * 2 + 1])
                self.state[ingred_idx * 2] = 0
                self.state[ingred_idx * 2 + 1] = 0
                self.state[ingred_idx2 * 2] = 0
                self.state[ingred_idx2 * 2 + 1] = 0

            # top_with
            elif (4 in self.actions_map[action]):
                self.state[ingred_idx2 * 2] = 0
                self.state[ingred_idx2 * 2 + 1] = 0
            checker_reward = 0

        else:
            checker_reward = -1
            reward -= 1
        reward -= 1
        done = self._check_done()

        self.history['actions'].append(a)
        self.history['rewards'].append([reward])
        self.history['lengths'] += 1


        if done:
            reward += 20

        self.acc_r += reward
        info = {'history': self.history, 'acc_r': self.acc_r}


        if self.no_reward:
            reward = 0
        elif self.checker_reward:
            info['checker_reward'] = checker_reward

        # if done:
        #     print (self.accz_r,self.history['lengths'])
        state_old = deepcopy(self.state)

        # done if reach max steps, but no extra reward given
        if self.counter >= self.max_steps:
            done = True
        if done:
            self.reset()


        return state_old, reward, done, info

    def reset(self):
        # random a set of ingredients 5 or 6
        # ingredients state: raw, washed, preprocessed, cooked, garnished
        self.counter = 0
        num_ingred = random.randint(self.min_num_ingred,self.max_num_ingred)
        # random draw categories
        self.ingreds_current = [1] + random.sample(list(range(1, len(self.ingreds))),  num_ingred)

        self.mixture_prop = deepcopy(self.ingreds_map[0])

        # init all ingredients state to be raw
        self.state = []
        for i in self.ingreds_current:
            self.state.append(i)
            self.state.append(0) # 0 state stands for raw

        if len(self.state) != (self.max_num_ingred+1)*2:
            print (num_ingred, self.ingreds_cat_current, self.ingreds_current,  self.state)

        # assert len(self.state) == self.max_num_ingred+1
        self.history = {'states':[],'actions':[],'rewards':[], 'lengths':0}
        self.acc_r = 0
        return self.state

    def get(self):
        return self

    def extract_rules(self, save_path, use_original_glove = True):
        if use_original_glove:
            glove_dic_all = {}
            glove_dic_actions = {}
            glove_dic_ingreds = {}
            filename = './datasets/GLoVE/glove.6B.100d.txt'
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip().split(' ')
                    glove_dic_all[line[0]] = [float(i) for i in line[1:]]

            for act in self.actions:
                glove_dic_actions[act] = glove_dic_all[self.actions[act].split('_')[0]]
            for ing in self.ingreds:
                glove_dic_ingreds[ing] = glove_dic_all[self.ingreds[ing].split('_')[0]]

            pk.dump(glove_dic_actions, open(save_path + "/actions_features.pk", "wb"))
            pk.dump(glove_dic_ingreds, open(save_path + "/ingreds_features.pk", "wb"))


        affordable = []
        ordering = []

        for act in self.actions:
            for ing in self.ingreds:
                if not self._check_affordance(act, ing, ing):
                    affordable.append((act, ing))
        for act in self.actions:
            for act2 in self.actions:
                for ing in self.ingreds:
                    if self._check_affordance(act, ing, ing)  and self._check_affordance(act2, ing, ing) and min(self.actions_prereq[self.actions_map[act2][0]])  >= max(self.actions_effect[self.actions_map[act][0]]):
                        ordering.append(((act,ing), (act2,ing)))

        pk.dump(affordable, open(save_path+"/affordable.pk", "wb"))
        pk.dump(ordering, open(save_path+"/ordering.pk", "wb"))

        pk.dump(self.actions, open(save_path+"/actions_name.pk", "wb"))
        pk.dump(self.ingreds, open(save_path+"/ingreds_name.pk", "wb"))












