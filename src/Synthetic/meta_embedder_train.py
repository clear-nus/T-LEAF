import sys
import os
sys.path.append(os.getcwd())
import torch


from src.Synthetic.models.dataset_build_utils import build_dataset,build_syntaxdataset,build_dataset_onebyone, build_syntaxdataset_onebyone, readltl
import numpy as np
import itertools
import argparse

import random
from src.Synthetic.models.models import *
from src.Synthetic.models.training_utils import replace_edge_attr, new_dataset, newsyntax_dataset


import os






def train():
    model.train()
    mlp.train()
    # assign_embedder.train()
    acc_list = []
    loss_list_cross_entropy = []
    loss_list_triplet = []
    count=0
    loss_train_all = 0
    loss_mlp_all = 0
    out_false = []
    out_true = []
    prediction = []
    for data_i in range(len(dataset_train)):
        data = (dataset_train[data_i][0])
        out_formula = model(data)
        for index in range(len(dataset_train[data_i][1])):
            data = (dataset_train[data_i][1][index])
            out_true.append(model(data))
            data = (dataset_train[data_i][2][index])
            out_false.append(model(data))
            # loss_train =triplet_loss(out_formula,out_true[-1],out_false[-1])
            # loss_train_all += loss_train

            #mlp train
            mlp.train()
            input = torch.cat((torch.cat((out_formula, out_true[-1])).flatten().unsqueeze(0), \
                                     torch.cat((out_formula, out_false[-1])).flatten().unsqueeze(0))).to(device)

            # print ('training sample ', data_i)
            # print ('true', input[0][100:106])
            # print ('false',input[1][100:106])
            pred = mlp(input)
            prediction.append(pred)
            target = torch.LongTensor([1, 0]).to(device)
            loss_mlp = cross_entropy(pred, target)
            loss_mlp_all += loss_mlp
            count+=1

        if data_i % 256 == 0 and data_i !=0 or data_i == len(dataset_train)-1:
            # print (data_i,'/',len(dataset_train), ' triplet loss: ', float(loss_train_all/count),)
            print (data_i,'/',len(dataset_train)," mlp loss", float(loss_mlp_all/count))

            optimizer.zero_grad()
            # loss_train_all /= count
            loss_mlp_all /= count
            (loss_mlp_all).backward()
            optimizer.step()

            loss_list_cross_entropy.append(float(loss_mlp_all.cpu()))
            # loss_list_triplet.append(float(loss_train_all.cpu()))
            # calculate accuracy
            total_true=0
            for pred in prediction:
                _, predicted = torch.max(pred.data, 1)
                total_true += (predicted == target).sum().item()
            acc = total_true / (len(prediction)*2)
            acc_list.append(float(acc))


            loss_train_all = 0
            loss_mlp_all = 0
            out_false = []
            out_true = []
            prediction = []
            count = 0

    return np.mean(acc_list)

def train_edge_embedder():
    edge_embedder.train()
    acc_list = []
    loss_list_cross_entropy = []
    loss_list_triplet = []
    for data_i in range(len(dataset_train)):
        data = replace_edge_attr(device, args, dataset_train[data_i][0], edge_embedder)
        out_formula = model(data)
        loss_train_all=0
        loss_mlp_all=0
        out_false=[]
        out_true=[]
        prediction=[]
        for index in range(len(dataset_train[data_i][1])):
            data = replace_edge_attr(device, args, dataset_train[data_i][1][index], edge_embedder)
            out_true.append(model(data))
            data = replace_edge_attr(device, args, dataset_train[data_i][2][index], edge_embedder)
            out_false.append(model(data))
            loss_train =triplet_loss(out_formula,out_true[-1],out_false[-1])
            loss_train_all += loss_train


        if data_i % 500 == 0:
            print (data_i,'/',len(dataset_train), ' train loss: ', loss_train)
        # print("loss_mlp {}".format(loss_mlp))

        # optimizer_embedder.zero_grad()
        optimizer_edge_embedder.zero_grad()
        loss_train_all /= len(dataset_train[data_i][1])
        (loss_train_all).backward()
        # optimizer_embedder.step()
        optimizer_edge_embedder.step()


def train_embedder():
    model.train()
    if args.is_assign:
        assign_embedder.eval()
    count=0
    loss_train_all = 0
    out_false = []
    out_true = []
    for data_i in range(len(dataset_train)):
        # start=time.time()
        with torch.no_grad():
            data = dataset_train[data_i][0]
        # print("time load formula ", time.time()-start)

        out_formula = model(data)
        # print("time run formula ", time.time()-start)
        for index in range(len(dataset_train[data_i][1])):
            data = dataset_train[data_i][1][index]
            if not args.is_assign:
                out_true.append(model(data))
            else:
                out_true.append(assign_embedder(data))
            data = (dataset_train[data_i][2][index])
            if not args.is_assign:
                out_false.append(model(data))
            else:
                out_false.append(assign_embedder(data))
                # print("time for run assign ",time.time()-start)

            loss_train =triplet_loss(out_formula,out_true[-1],out_false[-1])
            loss_train_all += loss_train
            count +=1


        if data_i % 256 == 0  and data_i !=0 or data_i == len(dataset_train)-1:
            print (data_i,'/',len(dataset_train), ' triplet embedder loss: ', float(loss_train_all/count))
        # print("loss_mlp {}".format(loss_mlp))
            optimizer_embedder.zero_grad()
            loss_train_all /= count
            # print("loss train in embedder",loss_train_all)
            (loss_train_all).backward()
            optimizer_embedder.step()

            count = 0
            loss_train_all = 0
            out_false = []
            out_true = []


def train_assign_embedder():
    model.eval()
    assign_embedder.train()
    count=0
    loss_train_all = 0
    out_false = []
    out_true = []
    for data_i in range(len(dataset_train)):
        # start=time.time()
        with torch.no_grad():
            data = dataset_train[data_i][0]
        # print("time load formula ", time.time()-start)

        out_formula = model(data)
        # print("time run formula ", time.time()-start)
        for index in range(len(dataset_train[data_i][1])):
            data = dataset_train[data_i][1][index]
            if not args.is_assign:
                out_true.append(model(data))
            else:
                out_true.append(assign_embedder(data))
            data = (dataset_train[data_i][2][index])
            if not args.is_assign:
                out_false.append(model(data))
            else:
                out_false.append(assign_embedder(data))
                # print("time for run assign ",time.time()-start)

            loss_train =triplet_loss(out_formula,out_true[-1],out_false[-1])
            loss_train_all += loss_train
            count +=1


        if data_i % 256 == 0  and data_i !=0 or data_i == len(dataset_train)-1:
            print (data_i,'/',len(dataset_train), ' triplet assign embedder loss: ', float(loss_train_all/count))
        # print("loss_mlp {}".format(loss_mlp))
            optimizer_assign_embedder.zero_grad()
            loss_train_all /= count
            # print("loss train in embedder",loss_train_all)
            (loss_train_all).backward()
            optimizer_assign_embedder.step()

            count = 0
            loss_train_all = 0
            out_false = []
            out_true = []


def train_mlp():
    model.eval()
    if args.is_assign:
        assign_embedder.eval()
    acc_list = []
    loss_list_cross_entropy = []
    loss_list_triplet = []
    count=0
    loss_train_all = 0
    loss_mlp_all = 0
    out_false = []
    out_true = []
    prediction = []
    for data_i in range(len(dataset_train)):
        with torch.no_grad():
            data = (dataset_train[data_i][0])
            out_formula = model(data)
        for index in range(len(dataset_train[data_i][1])):
            with torch.no_grad():
                data_true = (dataset_train[data_i][1][index])
                data_false = (dataset_train[data_i][2][index])
                if not args.is_assign:
                    out_true.append(model(data_true))
                    out_false.append(model(data_false))
                else:
                    out_true.append(assign_embedder(data_true))
                    out_false.append(assign_embedder(data_false))

                loss_train = triplet_loss(out_formula, out_true[-1], out_false[-1])
                loss_train_all += loss_train

            #mlp train
            mlp.train()
            input = torch.cat((torch.cat((out_formula, out_true[-1])).flatten().unsqueeze(0), \
                                     torch.cat((out_formula, out_false[-1])).flatten().unsqueeze(0))).to(device)

            pred = mlp(input)
            prediction.append(pred)
            target = torch.LongTensor([1, 0]).to(device)

            loss_mlp = cross_entropy(pred, target)
            loss_mlp_all += loss_mlp
            count+=1

        if data_i % 256 == 0  and data_i !=0 or data_i == len(dataset_train)-1:
            print(data_i, '/', len(dataset_train), ' mlp loss: ',
                  float(loss_mlp_all / count))
            optimizer_mlp.zero_grad()
            loss_train_all /= count
            loss_mlp_all /= count
            (loss_mlp_all).backward()
            optimizer_mlp.step()


            loss_list_cross_entropy.append(float(loss_mlp_all.cpu()))
            loss_list_triplet.append(float(loss_train_all.cpu()))
            # calculate accuracy
            total_true=0
            for pred in prediction:
                _, predicted = torch.max(pred.data, 1)
                total_true += (predicted == target).sum().item()
            acc = total_true / (len(prediction)*2)
            acc_list.append(float(acc))
            count = 0
            loss_train_all = 0
            loss_mlp_all = 0
            out_false = []
            out_true = []
            prediction = []

    # print(acc_list)
    return np.mean(acc_list)


def eval():
    acc_list = []
    loss_list_cross_entropy = []
    loss_list_triplet = []
    model.eval()
    if args.is_assign:
        assign_embedder.eval()
    for data_i in range(len(dataset_test)):
        data = (dataset_test[data_i][0])
        out_formula = model(data)
        loss_train=0
        loss_mlp=0
        out_false=[]
        out_true=[]
        prediction=[]
        for index in range(len(dataset_test[data_i][1])):
            data_true = (dataset_test[data_i][1][index])
            data_false = (dataset_test[data_i][2][index])
            if not args.is_assign:
                out_true.append(model(data_true))
                out_false.append(model(data_false))
            else:
                out_true.append(assign_embedder(data_true))
                out_false.append(assign_embedder(data_false))

            loss_train += triplet_loss(out_formula,out_true[-1],out_false[-1])

            mlp.eval()
            input = torch.cat((torch.cat((out_formula, out_true[-1])).flatten().unsqueeze(0), \
                                     torch.cat((out_formula, out_false[-1])).flatten().unsqueeze(0))).to(device)

            pred = mlp(input)
            prediction.append(pred)
            target = torch.LongTensor([1, 0]).to(device)

            loss_mlp += cross_entropy(pred, target)
            # loss_by_iter.append(float(mlp_loss.cpu()))
        loss_train /= len(dataset_test[data_i][1])
        loss_mlp /= len(dataset_test[data_i][1])
        # print("loss_mlp {}".format(loss_mlp))
        loss_list_cross_entropy.append(float(loss_mlp.cpu()))
        loss_list_triplet.append(float(loss_train.cpu()))

        # calculate accuracy
        total_true=0
        for pred in prediction:
            _, predicted = torch.max(pred.data, 1)
            total_true += (predicted == target).sum().item()
        acc = total_true / (len(prediction)*2)
        acc_list.append(float(acc))
        # print("eval set acc {}".format(acc))
    print("eval at epoch ",np.mean(acc_list))



def load_models():
    try:
        print ('Modes loaded')
        sub_model_path = args.model_save_path + str(args.dataset_name)
        if args.syntax:
            sub_model_path += "_syntax"
        sub_model_path += "/"

        edge_embedder.load_state_dict(torch.load(sub_model_path+'edge_embedder_latest'))
        model.load_state_dict(torch.load(sub_model_path+'meta_embedder_latest'))
        mlp.load_state_dict(torch.load(sub_model_path+'mlp_latest'))
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

def get_dataset(dataset_save_path):
    print('Building dataset ...')
    if args.syntax:
        dataset_train, dataset_test = build_syntaxdataset_onebyone(device, args.dataset_root + args.dataset_name)
        dataset_train = newsyntax_dataset(device, args, dataset_train, edge_embedder)
        dataset_test = newsyntax_dataset(device, args, dataset_test, newsyntax_dataset)

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
    print('Loading saved dataset')

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
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + str(args.device_id) if args.cuda else 'cpu')



    dataset_save_path = args.dataset_root + args.dataset_name + '/saved_dataset'
    if args.syntax:
        dataset_save_path += "_syntax_noFG.pt"
    else:
        if args.global_node:
            dataset_save_path += "_global"
        if args.skip_connected:
            dataset_save_path += "_skip_connected"
        if args.undirected:
            dataset_save_path += "_undirected"
        dataset_save_path += ".pt"





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
            print ('\n No saved dataset\n')
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


    if args.type_pooling:
        mlp = MLP_type()
    else:
        mlp = MLP()

    optimizer_edge_embedder = torch.optim.Adam(edge_embedder.parameters(), lr=0.001, weight_decay=5e-4)
    optimizer_embedder = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=0.001, weight_decay=5e-4)
    optimizer_assign_embedder = torch.optim.Adam(assign_embedder.parameters(), lr=0.001, weight_decay=5e-4)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), mlp.parameters()), lr=0.001, weight_decay=5e-4)

    triplet_loss = torch.nn.TripletMarginLoss(margin=args.margin, p=2)
    cross_entropy = torch.nn.CrossEntropyLoss()

    edge_embedder.to(device)
    model.to(device)
    assign_embedder.to(device)
    mlp.to(device)
    triplet_loss = triplet_loss.to(device)
    cross_entropy = cross_entropy.to(device)

    model.train()
    assign_embedder.train()

    acc_list = []
    loss_list_cross_entropy = []
    loss_list_triplet = []





    for epoch in range(1000):
        save_models(epoch)

        print('\nEpoch ', epoch)

        # # load_models()
        for i in range(0):
            if i == 0:
                print ('\nEdge embedder training')
            print ("Edge embedder training; Epoch: ", i)
            train_edge_embedder()
        for i in range(5):
            if i == 0:
                print ('\nMain embedder trianing:')
            print ('Main embedder training; Epoch: ', i)
            train_embedder()
        eval()
        if args.is_assign:
            for i in range(10):
                if i == 0:
                    print("\n Main assign embedder training: ")
                print("Assign embedder training Epoch ",i)
                train_assign_embedder()
            eval()
        for i in range(10):
            if i == 0:
                print ('\nPredictor training:')
            acc = train_mlp()
            print("Predictor traininig at epoch {} acc is {}".format(i, acc))
            eval()