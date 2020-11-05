import sys
import os
sys.path.append(os.getcwd())
import torch
from torch_geometric.data import Data

from src.Synthetic.models.dataset_build_utils import build_dataset,build_syntaxdataset,build_dataset_onebyone, build_syntaxdataset_onebyone, readltl
import numpy as np
import itertools
import argparse


from src.Synthetic.models.models import *
from src.Synthetic.models.training_utils import new_dataset, newsyntax_dataset

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle as pk
import os





def load_models():
    try:
        print ('Modes loaded')
        edge_embedder.load_state_dict(torch.load(args.model_save_path + 'edge_embedder_save'))
        model.load_state_dict(torch.load(args.model_save_path+'embedder_save'))
        mlp.load_state_dict(torch.load(args.model_save_path + 'mlp_save'))
    except:
        print ('Models not loaded')


def save_models(epoch):
    print ('Models saved')
    sub_model_path = args.model_save_path + str(args.dataset_name)
    if args.syntax:
        sub_model_path+="_syntax"
    sub_model_path+="/"
    if not os.path.exists(sub_model_path):
        os.mkdir(sub_model_path)
    torch.save(edge_embedder.state_dict(),sub_model_path+'edge_embedder_latest')
    torch.save(model.state_dict(),sub_model_path+'meta_embedder_latest')
    torch.save(mlp.state_dict(),sub_model_path+'mlp_latest')


def visualize(data_point, name_idx):
    tsne = TSNE(n_components=2, perplexity=10)
    Y = tsne.fit_transform(data_point)

    idx = int((len(data_point)-1)/2)
    plt.figure()
    ax = plt.gca()
    ax.scatter(Y[0, 0], Y[0,1], c="b",)
    ax.scatter(Y[1:idx+1,0], Y[1:idx+1,1], c="g")
    ax.scatter(Y[idx+1:,0], Y[idx+1:,1], c="r")

    plt.legend(labels=['formula', 'true', "false"], loc='best')
    if args.syntax:
        figname = './src/Synthetic/visualization/data_and_figs/synatx_plot_'+str(name_idx)
        plt.title("syntax embedding")
    else:
        figname = './src/Synthetic/visualization/data_and_figs/dfa_plot_'+str(name_idx)
        plt.title("DFA embedding")
    plt.savefig(figname)
    plt.close()


def generate_embedding():
    # for data_i in range(len(dataset_train)):
    data_save = {}
    for data_i in [0]:
        with torch.no_grad():
            out_formula = model(dataset_train[data_i][0])
            trues, falses=[],[]
            for index in range(len(dataset_train[data_i][1])):
                out_true=model(dataset_train[data_i][1][0])
                out_false=model(dataset_train[data_i][2][0])
                trues.append(out_true)
                falses.append(out_false)
            embeddings=torch.cat([out_formula]+trues+falses).cpu().numpy()
        data_save[data_i] = embeddings
        print (data_i, len(dataset_train[data_i][1]))
        visualize(embeddings, data_i)
    if args.syntax:
        save_name = './src/Synthetic/visualization/data_and_figs/embedding_syntax.pk'
    else:
        save_name = './src/Synthetic/visualization/data_and_figs/embedding_dfa.pk'
    pk.dump(data_save,open(save_name,'wb'))

def get_dataset(dataset_save_path):
    print('Building dataset ...')
    if args.syntax:
        if args.notonebyone:
            dataset_train, dataset_test = build_syntaxdataset(device, args.dataset_root + args.dataset_name)
        else:
            dataset_train, dataset_test = build_syntaxdataset_onebyone(device,
                                                                       args.dataset_root + args.dataset_name)

        dataset_train = newsyntax_dataset(device, args, dataset_train, edge_embedder)
        dataset_test = newsyntax_dataset(device, args, dataset_test, edge_embedder)

    else:
        if args.notonebyone:
            dataset_train, dataset_test = build_dataset(device, args.dataset_root + args.dataset_name,
                                                        edge_feature_dic)
        else:
            dataset_train, dataset_test = build_dataset_onebyone(device, args.dataset_root + args.dataset_name,
                                                                 edge_feature_dic)


        random.seed(27)
        random.shuffle(dataset_train)
        print('Built')

        # dataset_train = dataset_train[:20]
        # dataset_test = dataset_test[:20]

        print('Converting to new dataset ...')
        dataset_train = new_dataset(device, args, dataset_train, edge_embedder)
        dataset_test = new_dataset(device, args, dataset_test, edge_embedder)
        print('Converted')

    print('Saving dataset ..')
    torch.save((dataset_train, dataset_test), dataset_save_path)
    print('Saved')
    return dataset_train, dataset_test
def load_saved_dataset(dataset_save_path):
    print('Laoding saved dataset')

    dataset_train, dataset_test = torch.load(dataset_save_path, map_location=device)
    print('Loaded')
    return dataset_train, dataset_test

if __name__ == '__main__':
    print ('Default: set --replace --nodes_only --random_path_agg')
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", default=False, action="store_true")
    parser.add_argument("--syntax", default=False, action="store_true")
    parser.add_argument("--replace", default=False, action="store_true")
    parser.add_argument("--dataset_root", type=str, default="./datasets/Synthetic/")
    parser.add_argument("--dataset_name", type=str, default='3_10')
    parser.add_argument("--margin", type=float, default=1)
    parser.add_argument("--model_save_path", type=str, default="./saved_models/Synthetic/meta_embedder/")
    parser.add_argument("--prop_size", type=int, default=3)
    parser.add_argument("--edge_embedder_rootpath", type=str, default="./saved_models/Synthetic/edge_embedder/")
    parser.add_argument("--edge_embedder_name", type=str,
                        default="edge_embedder_3_latest.pt")
    parser.add_argument("--is_assign", default=False, action="store_true")
    parser.add_argument("--nodes_only", default=False, action="store_true")
    parser.add_argument("--rebuild_dataset", default=False, action="store_true")
    parser.add_argument("--global_node", default=False, action="store_true")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--type_pooling", default=False, action="store_true")
    parser.add_argument("--skip_connected", default=False, action="store_true")
    parser.add_argument("--undirected", default=False, action="store_true")
    parser.add_argument("--type_node", default=False, action="store_true")
    parser.add_argument("--random_path_agg", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=1111, help="random seed for reprobudicibility")
    parser.add_argument("--notonebyone", default=False, action="store_true", help = "dont build dataset one by one, for saving the embedding of the same formula")
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device('cuda:' + str(args.device_id) if args.cuda else 'cpu')

    onebyone_save_name = '_not1b1' if args.notonebyone else ''
    dataset_save_path = args.dataset_root + args.dataset_name + '/saved_dataset'
    if args.syntax:
        dataset_save_path += "_syntax_noFG" + onebyone_save_name + ".pt"
    else:
        if args.global_node:
            dataset_save_path += "_global"
        if args.skip_connected:
            dataset_save_path += "_skip_connected"
        if args.undirected:
            dataset_save_path += "_undirected"
        dataset_save_path += onebyone_save_name + ".pt"



    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)






    ######### extract edge features_
    edge_feature_dic = {}
    node_feature = np.load(args.dataset_root + "prop.npy", allow_pickle=True)
    op_feature = np.load(args.dataset_root + "op.npy", allow_pickle=True)


    for i in range(args.prop_size):
        edge_feature_dic["p{}".format(i)] = node_feature[i]
        edge_feature_dic["!p{}".format(i)] = 1 - node_feature[i]
    edge_feature_dic['AND'] = op_feature[0]
    edge_feature_dic['OR'] = op_feature[1]
    # if dataset_name == 'SimpleLTL':
    #     prop_list = ['p0','p1','p2','!p0','!p1','!p2']
    # else:
    #     prop_list = ['p0', 'p1', 'p2','p3','p4','p5', '!p0', '!p1', '!p2','!p3','!p4','!p5']
    edge_feature_dic['1'] = np.ones((50))
    edge_feature_dic['0'] = np.zeros((50))
    #########################################################

    edge_embedder = torch.load(args.edge_embedder_rootpath+args.edge_embedder_name, map_location=device)


    ######## build dataset
    if args.rebuild_dataset:
        dataset_train, dataset_test = get_dataset(dataset_save_path)
    else:
        try:
            dataset_train, dataset_test = load_saved_dataset(dataset_save_path)
        except:
            print('\n No saved dataset\n')
            dataset_train, dataset_test = get_dataset(dataset_save_path)


    assign_embedder = Node_only()

    # model = EdgeNet_Ori(device)
    if not args.nodes_only:
        model = EdgeNet_MLP()
    elif args.syntax:
        model = Node_only_syntax()
    else:
        if args.global_node:
            model = Node_only_global()
        elif args.type_pooling:
            model = Node_only_Type(device)
        elif args.random_path_agg:
            model = Node_only_random_agg_min_clip(device)
        else:
            model = Node_only()

    syntax_name = '_syntax' if args.syntax else ''
    meta_embedder_path = args.model_save_path + str(args.dataset_name) +syntax_name + '/' + 'meta_embedder_latest'

    model.load_state_dict(torch.load(meta_embedder_path, map_location=device))
    model.to(device)

    generate_embedding()

