import sys
import os
sys.path.append(os.getcwd())
import torch
import numpy as np

import argparse
from src.Synthetic.models.models import MLP, Edge_Embedder
from src.Synthetic.models.dataset_build_utils import build_edgeembedder_dataset



def train_embedder():
    edge_embedder.train()
    loss_list_triplet = []
    count=0
    loss_train_all = 0
    out_false = []
    out_true = []
    for data_i in range(len(dataset_train)):
        data = dataset_train[data_i][0]
        out_formula = edge_embedder(data)
        for index in range(len(dataset_train[data_i][1])):
            data = dataset_train[data_i][1][index]
            out_true.append(edge_embedder(data))
            data =dataset_train[data_i][2][index]
            out_false.append(edge_embedder(data))
            loss_train =triplet_loss(out_formula,out_true[-1],out_false[-1])
            loss_train_all += loss_train
            count +=1



        if data_i % 128 == 0 and data_i != 0:
            print (data_i,'/',len(dataset_train), ' embedder loss: ', float(loss_train_all)/count)

            optimizer_embedder.zero_grad()
            loss_train_all /= count
            loss_train_all.backward()
            optimizer_embedder.step()

            loss_list_triplet.append(float(loss_train_all.cpu()))

            loss_train_all = 0
            out_false = []
            out_true = []
            count=0
    return np.mean(loss_list_triplet)

def train_mlp():
    edge_embedder.eval()
    loss_list_cross_entropy = []
    count=0
    out_false = []
    out_true = []
    prediction=[]
    loss_mlp_all=0
    for data_i in range(len(dataset_train)):
        data = dataset_train[data_i][0]
        out_formula = edge_embedder(data)
        for index in range(len(dataset_train[data_i][1])):
            with torch.no_grad():
                data = dataset_train[data_i][1][index]
                out_true.append(edge_embedder(data))
                data =dataset_train[data_i][2][index]
                out_false.append(edge_embedder(data))

                count +=1

            # mlp train
            mlp.to(device)
            input = torch.cat((torch.cat((out_formula, out_true[-1])).flatten().unsqueeze(0), \
                                     torch.cat((out_formula, out_false[-1])).flatten().unsqueeze(0))).to(device)
            pred = mlp(input)
            prediction.append(pred)
            target = torch.LongTensor([1, 0]).to(device)

            loss_mlp = cross_entropy(pred, target)
            loss_mlp_all += loss_mlp

        if data_i % 128 == 0 and data_i != 0:
            print (data_i,'/',len(dataset_train), ' mlp loss: ', float(loss_mlp_all)/count)

            optimizer_mlp.zero_grad()
            loss_mlp_all /= count
            loss_mlp_all.backward()
            optimizer_mlp.step()

            loss_list_cross_entropy.append(float(loss_mlp_all.cpu()))

            loss_mlp_all = 0
            prediction = []
            out_false = []
            out_true = []
            count=0
    return np.mean(loss_list_cross_entropy)


def eval():
    acc_list = []
    loss_list_cross_entropy = []
    loss_list_triplet = []
    edge_embedder.eval()
    for data_i in range(len(dataset_test)):
        data = (dataset_test[data_i][0])
        out_formula = edge_embedder(data)
        loss_train=0
        loss_mlp=0
        out_false=[]
        out_true=[]
        prediction=[]
        for index in range(len(dataset_test[data_i][1])):
            data_true = (dataset_test[data_i][1][index])
            data_false = (dataset_test[data_i][2][index])
            out_true.append(edge_embedder(data_true))
            out_false.append(edge_embedder(data_false))
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
    print("eval at epoch ", np.mean(acc_list))

def load_saved_dataset(dataset_save_path):
    print('Loading saved dataset')

    dataset_train, dataset_test = torch.load(dataset_save_path)
    print('Loaded')
    return dataset_train, dataset_test
def get_dataset(dataset_save_path):
    print ('Building dataset ...')
    dataset_train, dataset_test = build_edgeembedder_dataset(device, args.dataset_root,
                                                             args.dataset_root + args.dataset_name + str(args.prop_size))

    print('Saving dataset ..')
    torch.save((dataset_train, dataset_test), dataset_save_path)
    print('Saved')
    return dataset_train, dataset_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cuda", default=False, action="store_true")
    parser.add_argument("--syntax", default=False, action="store_true")
    parser.add_argument("--replace", default=False, action="store_true")
    parser.add_argument("--dataset_root", type=str, default="./datasets/Synthetic/")
    parser.add_argument("--dataset_name", type=str, default='/edge_dataset/')
    parser.add_argument("--margin", type=float, default=1)
    parser.add_argument("--model_save_path", type=str, default="./saved_models/Synthetic/edge_embedder/")
    parser.add_argument("--prop_size", type=int, default=3)
    parser.add_argument("--rebuild_dataset", default=False, action="store_true")
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    dataset_save_path = args.dataset_root + args.dataset_name + args.prop_size + "_saved_dataset.pt"
    device = torch.device('cuda:' + str(args.device_id) if args.cuda else 'cpu')

    edge_embedder = Edge_Embedder()


    mlp = MLP()

    optimizer_embedder = torch.optim.Adam(edge_embedder.parameters(), lr=0.001, weight_decay=5e-4)
    optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=0.001, weight_decay=5e-4)

    triplet_loss = torch.nn.TripletMarginLoss(margin=args.margin, p=1)
    cross_entropy = torch.nn.CrossEntropyLoss()

    edge_embedder.to(device)
    mlp.to(device)
    triplet_loss = triplet_loss.to(device)
    cross_entropy = cross_entropy.to(device)

    edge_embedder.train()
    acc_list = []
    loss_list_cross_entropy = []
    loss_list_triplet = []

    if args.rebuild_dataset:
        dataset_train, dataset_test = get_dataset(dataset_save_path)
    else:
        try:
            dataset_train, dataset_test = load_saved_dataset(dataset_save_path)
        except:
            e = sys.exc_info()[0]
            print("<p>Error: %s</p>" % e)
            print ('\nNo saved dataset\n')
            dataset_train, dataset_test = get_dataset(dataset_save_path)


    for epoch in range(200):
        torch.save(edge_embedder, args.model_save_path +'edge_embedder_'+ str(args.prop_size) + "_latest.pt")
        for i in range(20):
            loss=train_embedder()
            print("train embedder at epoch {} loss is {}".format(i, loss))
        eval()
        for i in range(10):
            loss=train_mlp()
            print("train mlp at epoch {} loss is {}".format(i, loss))
            eval()
