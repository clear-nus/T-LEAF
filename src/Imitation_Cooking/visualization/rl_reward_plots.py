import matplotlib.pyplot as plt
import pickle as pk
import numpy as np
import os
from scipy import stats
from matplotlib import rc
import matplotlib
from matplotlib import font_manager as fm, rcParams
import seaborn as sns


def smooth(list, steps):
    ret = []
    for i in range(len(list)):
        start_idx = i
        end_idx = i+steps
        if end_idx > len(list):
            end_idx = len(list)
            start_idx = end_idx-steps
        ret.append(np.mean(list[start_idx:end_idx]))
    return ret

def single_seed(seed):
    path_root = './saved_models/Imitation_Cooking/imitation/logs/'
    smooth_steps = 10
    plot_key = 'mean'

    no_logic_reward = pk.load(open(path_root+'/reward_save_gail_'+str(seed),'rb'))
    checker_reward = pk.load(open(path_root+'/reward_save_gail_checker_'+str(seed),'rb'))
    logic_reward = pk.load(open(path_root+'/reward_save_gail_logic_'+str(seed),'rb'))
    plt.plot(smooth(no_logic_reward[plot_key],smooth_steps),label='baseline')
    plt.plot(smooth(checker_reward[plot_key],smooth_steps),label='checker')
    plt.plot(smooth(logic_reward[plot_key],smooth_steps),label = 'logic')

    plt.legend()
    plt.savefig(path_root+'/rl_reward_compare_'+str(seed))
    plt.close()

def agg_data(save_filename):
    directory1 = './rl_logs/data/'
    directory2 = './rl_logs/data1/'
    plot_key = 'mean'

    all = [[],[],[]]
    labels = ['GAIL', 'GAIL_with_checker', 'GAIL_with_logic']

    data_save = {}

    def get_files(all, directory):
        for filename in os.listdir(directory):
            data = pk.load(open(os.path.join(directory, filename), 'rb'))[plot_key]
            if '_checker_' in filename:
                all[1].append(data)
            elif '_logic_' in filename:
                all[2].append(data)
            else:
                all[0].append(data)
        return all


    all = get_files(all, directory1)
    all = get_files(all, directory2)

    all = np.array(all)

    for i in range(len(labels)):
        data_save[labels[i]] = all[i]

    pk.dump(data_save, open(save_filename, 'wb'))

def plot(load_ename, save_name):
    all = pk.load(open(load_ename, 'rb'))

    labels = ['GAIL', 'GAIL_with_checker', 'GAIL_with_logic']
    labels_name = ['GAIL', 'GAIL with checker', 'GAIL with T-LEAF']
    colors = ['#003d7c', '#46a247', '#f07d08']
    linestyles = [':', '-.', '-']

    smooth_steps = 1
    plot_to = 100

    matplotlib.font_manager._rebuild()
    # plt.style.use('ggplot')
    plt.style.use('seaborn')
    print (plt.style.available)


    sns.set_style("darkgrid", {"axes.facecolor": "0.95"})
    # plt.style.use('presentation')
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = 'Arial'
    # rc('font',**{'family':'sans-serif','serif':['Comic Sans']})
    rcParams['lines.linewidth'] = 2.5
    # params = {'legend.fontsize': 40,
    #           'legend.handlelength': 2}
    # rcParams.update(params)


    fpath = os.path.join("./ArialUnicodeMS.ttf")
    prop = fm.FontProperties(fname=fpath)

    rc('text', usetex=False)
    plt.rcParams['font.size'] = 40

    fig, ax = plt.subplots()

    for i in range(len(all)):
        # print (all[labels[i]].shape)
        avg = np.mean(all[labels[i]], axis=0)
        std = np.std(all[labels[i]], axis=0)
        sem = stats.sem(all[labels[i]], axis = 0)
        fill = sem
        fill = smooth(fill, smooth_steps)

        ax.plot(smooth(avg,smooth_steps)[:plot_to], label=labels_name[i], color=colors[i], linestyle = linestyles[i])
        ax.fill_between(range(len(avg))[:plot_to], (avg - fill)[:plot_to], (avg + fill)[:plot_to],color=colors[i], alpha=.2)
    ax.set_xlabel('Iterations', fontproperties=prop, size=22)
    ax.set_ylabel('Reward',fontproperties=prop, size=22)
    ax.set_title('Creative Cooking Performance',fontproperties=prop, size=22)
    L = plt.legend(loc=4, prop={'size': 19},fontsize=12)
    # plt.setp(L.texts, fontproperties=prop, size=80)
    # plt.gcf().subplots_adjust(bottom=0.15, left=0.2)
    # plt.tight_layout()
    plt.savefig(save_name)
    plt.close()






if __name__ == "__main__":
    # load_name = './rl_logs/cooking_data.pk'
    # save_name = './rl_logs/cooking_perf'
    #
    # agg_data(load_name)

    load_name = 'cooking_data.pk'
    save_name = 'cooking_perf'
    plot(load_name, save_name)
