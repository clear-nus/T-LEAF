import matplotlib.pyplot as plt
import pickle as pk
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib import rc
from matplotlib import font_manager as fm, rcParams

# print (matplotlib.__file__)


def visualize(Y, name_idx, is_syntax):
    colors = ['#f07d08','#0f88eb','#dd4d78', '#003d7c',]

    idx = int((len(Y)-1)/2)
    plt.figure()
    # plt.style.use('seaborn')
    sns.set_style("darkgrid", {"axes.facecolor": "0.95"})
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = 'Arial'
    rc('text', usetex=False)
    plt.rcParams['font.size'] = 22


    ax = plt.gca()
    ax.scatter(Y[0, 0], Y[0,1], c=colors[0],marker='*', s = 1000)
    ax.scatter(Y[1:idx+1,0], Y[1:idx+1,1], c=colors[1], marker = '+', s=100)
    ax.scatter(Y[idx+1:,0], Y[idx+1:,1], c=colors[2], marker = 'x', s=100)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.legend(labels=['Formula', 'Satisfying Assign', "Unstisfying Assign"], loc=4, frameon=True, fancybox=True, framealpha=0.77)
    prefix = './src/Synthetic/visualization/data_and_figs/'
    if is_syntax:
        figname = 'synatx_emb'+str(name_idx)
        plt.title("Syntax Tree Embedding Space")
    else:
        figname = 'dfa_emb'+str(name_idx)
        plt.title("DFA Embedding Space")

    plt.tight_layout()
    plt.savefig(prefix+figname+'.pdf')
    plt.close()


def generate_tsne():
    data_syntax = pk.load(open('./src/Synthetic/visualization/data_and_figs/embedding_syntax.pk','rb'))
    data_dfa = pk.load(open('./src/Synthetic/visualization/data_and_figs/embedding_dfa.pk','rb'))

    for key in [31]:
        num_syntax = int((len(data_syntax[key])-1)/2)
        num_dfa = int((len(data_dfa[key]) - 1) / 2)
        min_idx = min(num_dfa, num_syntax)
        # print (key, min_idx, num_syntax, num_dfa)
        syntax_emb = np.concatenate((data_syntax[key][:1 + min_idx], data_syntax[key][1+num_syntax:1+num_syntax+min_idx]))
        dfa_emb = np.concatenate((data_dfa[key][:1 + min_idx], data_dfa[key][1 + num_dfa:1 + num_dfa + min_idx]))

        tsne = TSNE(n_components=2, perplexity=15)
        Y_syntax = tsne.fit_transform(syntax_emb)
        Y_dfa = tsne.fit_transform(dfa_emb)

        tsne_data = {'syntax': Y_syntax, 'dfa': Y_dfa}
        pk.dump(tsne_data, open("./src/Synthetic/visualization/data_and_figs/tsne_data.pk","wb"))

        visualize(syntax_emb, name_idx='', is_syntax=True)
        visualize(dfa_emb, name_idx='', is_syntax=False)

if __name__ == "__main__":
    # generate_tsne()

    tsne_data = pk.load(open("./src/Synthetic/visualization/data_and_figs/tsne_data.pk","rb"))
    visualize(tsne_data['syntax'], name_idx='', is_syntax=True)
    visualize(tsne_data['dfa'], name_idx='', is_syntax=False)


