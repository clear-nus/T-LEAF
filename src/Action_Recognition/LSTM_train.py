import sys
import os
sys.path.append(os.getcwd())
from src.Action_Recognition.models.arguments_LSTM import get_args
from src.Action_Recognition.models.checker_loss import get_checker_loss
from src.Action_Recognition.models.embedder_loss_util import get_formula_video, read_cat
from src.Action_Recognition.models.preprocess_util import build_dataset_action_recognition, shuffledata, accuracy_cal, recall_precision_f1
from src.Synthetic.models.Util import read_strlist

import torch
import torch.nn.functional as F
import pickle as pk
from src.Synthetic.models.models import Edge_Embedder, Node_only_random_agg_min_clip
from src.Action_Recognition.models.models_predictor import Step_Model_Lstm, Step_Model_Lstm_classification_lstm
import numpy as np




def embedder_loss(pred_actions, pred_objects, video_name, glove_features, action_cat, ingredient_cat):
    graph_data, pred_data, grad_test = get_formula_video(device, pred_actions, pred_objects, args.logic_dataset_path+args.logic_dataset_name+video_name+"/", args.logic_dataset_path, edge_embedder, glove_features, action_cat, ingredient_cat, info_embedder, differentiable=True)
    if graph_data is None:
        return zero, grad_test
    else:
        graph_embedding, pred_embedding = meta_embedder(graph_data), meta_embedder(pred_data)
        # norm_graph_embedding, norm_pred_embedding = graph_embedding/torch.norm(graph_embedding), pred_embedding/torch.norm(pred_embedding)
        loss = F.pairwise_distance(graph_embedding, pred_embedding)
        # return torch.max(loss[0]-half, zero)
        return loss[0], grad_test

def checker_loss(pred_actions, pred_objects, video_name):
    loss = get_checker_loss(pred_actions, pred_objects, args.logic_dataset_path+args.logic_dataset_name+video_name+"/", args.logic_dataset_path, args.video_dataset_info)
    return loss



def train(optimizer,loss_model,train_video, train_action_label, train_ingre_label, train_video_name, glove_features, action_cat, ingredient_cat):
    step_model.train()
    if args.meta_train:
        meta_embedder.train()
    count = 0
    batch_size=args.batch_size
    loss_batch_action, loss_batch_ingre=0,0

    acc_action_batch, acc_ingre_batch=0,0
    predict_action_batch, predict_ingre_batch=0,0
    loss_embedder_batch =0
    loss_checker_batch =0
    loss_diff_batch = 0
    for index, video in enumerate(train_video):
        if args.memory_predictor:
            out_feature_list=[]
            for index_step, frame_step in enumerate(video):
                out_feature= step_model(torch.tensor(frame_step,dtype=torch.float32).to(device))
                out_feature_list.append(out_feature)
            out_feature_tensor = torch.stack(out_feature_list)
            output_action, output_ingre=step_model.classification_forward(out_feature_tensor)
        else:
            output_action_batch = []
            output_ingre_batch = []
            for index_step, frame_step in enumerate(video):
                # print("frame output", frame_step)
                out_action, out_ingre = step_model(torch.tensor(frame_step,dtype=torch.float32).to(device))
                # print("action output",out_action)
                output_action_batch.append(out_action.unsqueeze(0))
                output_ingre_batch.append(out_ingre.unsqueeze(0))
                output_action = torch.cat(output_action_batch).squeeze()
                output_ingre = torch.cat(output_ingre_batch).squeeze()



        count += 1
        target_action = train_action_label[index].to(device)
        target_ingre = train_ingre_label[index].to(device)

        loss_video_action = loss_model(output_action, target_action)
        loss_video_ingre = loss_model(output_ingre, target_ingre)
        if torch.any(torch.isnan(loss_video_ingre)):
            loss_video_ingre[torch.isnan(loss_video_ingre)]=0
            print("loss_vidoe_ingredient", loss_video_ingre)
            exit(0)
        if torch.any(torch.isnan(loss_video_action)):
            loss_video_action[torch.isnan(loss_video_action)]=0
            print("loss video action", loss_video_action)
            exit(0)
        predicted_action = torch.max(output_action,-1)[1]==target_action
        predicted_ingre = torch.max(output_ingre, -1)[1] == target_ingre
        if args.embedder_loss:
            # print ('debug',torch.max(output_action,-1)[1])
            # loss_embedder = embedder_loss(torch.max(output_action,-1)[1], torch.max(output_ingre, -1)[1], train_video_name[index])
            loss_embedder, grad_test = embedder_loss(output_action, output_ingre, train_video_name[index], glove_features, action_cat, ingredient_cat)


            # print("loss embedder ",loss_embedder)
            if torch.any(torch.isnan(loss_embedder)):
                print("loss_embedder",loss_embedder)
                exit(0)
            loss_embedder_batch += loss_embedder
        if args.checker_loss:
            loss_cheker = checker_loss(torch.max(output_action, -1)[1], torch.max(output_ingre, -1)[1],
                                          train_video_name[index])
            # print("loss_checker", loss_cheker)
            loss_checker_batch += loss_cheker


        acc_action = predicted_action.sum().item() * 1.0 / predicted_action.shape[0]
        acc_ingre = predicted_ingre.sum().item() * 1.0 / predicted_ingre.shape[0]

        loss_batch_action += loss_video_action
        loss_batch_ingre += loss_video_ingre
        acc_action_batch += acc_action
        acc_ingre_batch += acc_ingre
        predict_action_batch += predicted_action.sum().item()
        predict_ingre_batch += predicted_ingre.sum().item()

        if count!=0 and count%batch_size==0 :
            print("\ntrain action loss {} acc {} predict {} at batch {}".format(loss_batch_action/batch_size, acc_action_batch/batch_size,
                                                                              predict_action_batch/batch_size, (count)//batch_size))
            print("train ingredient loss {} acc {} predict {} at batch {}".format(loss_batch_ingre/batch_size, acc_ingre_batch/batch_size,
                                                                                  predict_ingre_batch/batch_size, (count)//batch_size))

            loss=(loss_batch_action+loss_batch_ingre)
            print_grad = False
            if print_grad:
                loss = 0

            if args.embedder_loss:
                loss+= loss_embedder_batch*args.lamda
                print("loss compare ", loss_batch_action, loss_batch_ingre, loss_embedder_batch)
            elif args.checker_loss:
                loss +=loss_checker_batch

            loss/=batch_size

            if print_grad:
                # loss = grad_test
                print_indicator = True

                if grad_test is None:
                    loss = (loss_batch_action+loss_batch_ingre)
                    print_indicator = False
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            # torch.nn.utils.clip_grad_norm_(step_model.parameters(), clip)
            # torch.nn.utils.clip_grad_norm(action_modeel.parameters(), clip)
            # for group in optimizer.param_groups:
            #     for param in group["params"]:
            #         param.grad[torch.isnan(param.grad)]=0

            # print grad
            if print_grad and print_indicator:
            # if print_grad and not print_indicator:
                for p in step_model.parameters():
                    if p.grad is not None:
                        print('loss', loss,'step_model_grad', p.name, p.grad)


            loss_batch_action, loss_batch_ingre = 0, 0
            acc_action_batch, acc_ingre_batch = 0, 0
            predict_action_batch, predict_ingre_batch = 0, 0

            loss_embedder_batch=0
            loss_checker_batch=0
            loss_diff_batch=0

def eval(loss_model,eval_video,eval_action_label, eval_ingre_label, eval_video_name, glove_features, action_cat, ingredient_cat):
    step_model.eval()
    loss_action, loss_ingre=0,0
    accs_action, accs_ingre=0,0
    predict_action, predict_ingre=0,0
    top5_action, top5_ingre=0, 0
    f1_action, f1_ingre=0, 0
    recall_action, recall_ingre=0, 0
    precision_action, precision_ingre=0, 0
    if len(eval_video) == 0:
        return 0.,0.
    for index,video in enumerate(eval_video):

        if args.memory_predictor:
            out_feature_list=[]
            for index_step, frame_step in enumerate(video):
                out_feature= step_model(torch.tensor(frame_step,dtype=torch.float32).to(device))
                out_feature_list.append(out_feature)
            out_feature_tensor = torch.stack(out_feature_list)
            output_action, output_ingre=step_model.classification_forward(out_feature_tensor)
        else:
            output_action_batch = []
            output_ingre_batch = []
            for index_step, frame_step in enumerate(video):
                # print("frame output", frame_step)
                out_action, out_ingre = step_model(torch.tensor(frame_step,dtype=torch.float32).to(device))
                # print("action output",out_action)
                output_action_batch.append(out_action.unsqueeze(0))
                output_ingre_batch.append(out_ingre.unsqueeze(0))
                output_action = torch.cat(output_action_batch).squeeze()
                output_ingre = torch.cat(output_ingre_batch).squeeze()

        target_action = eval_action_label[index].to(device)
        target_ingre = eval_ingre_label[index].to(device)


        loss_video_action = loss_model(output_action, target_action)
        loss_video_ingre = loss_model(output_ingre, target_ingre)

        predicted_action = torch.max(output_action,-1)[1]==target_action
        predicted_ingre = torch.max(output_ingre, -1)[1] == target_ingre
        acc_action = predicted_action.sum().item() * 1.0 / predicted_action.shape[0]
        acc_ingre = predicted_ingre.sum().item() * 1.0 / predicted_ingre.shape[0]
        if args.embedder_loss:
            # loss_embedder, grad_test = embedder_loss(torch.max(output_action,-1)[1], torch.max(output_ingre, -1)[1], eval_video_name[index], glove_features, action_cat, ingredient_cat)
            loss_embedder, grad_test = embedder_loss(output_action, output_ingre,
                                                     eval_video_name[index], glove_features, action_cat, ingredient_cat)
            # print("loss_embedder ", loss_embedder)
        if args.checker_loss:
            loss_checker = checker_loss(torch.max(output_action,-1)[1], torch.max(output_ingre, -1)[1], eval_video_name[index])
            print("loss checker ", loss_checker)

        top5_ingre+=accuracy_cal(output_ingre,target_ingre, args.top_k_acc)
        top5_action+=accuracy_cal(output_action, target_action, args.top_k_acc)

        pre_ing, recall_ing, f1_ing=recall_precision_f1(args.macro, output_ingre, target_ingre)
        pre_act, recall_act, f1_act=recall_precision_f1(args.macro, output_action, target_action)

        precision_action+=pre_act
        precision_ingre+=pre_ing
        recall_action+=recall_act
        recall_ingre+=recall_ing
        f1_ingre+=f1_ing
        f1_action+=f1_act


        loss_action += loss_video_action
        loss_ingre += loss_video_ingre
        accs_action += acc_action
        accs_ingre += acc_ingre
        predict_action += predicted_action.sum().item()
        predict_ingre += predicted_ingre.sum().item()


    num_video=len(eval_video)
    print("\neval action loss {} predicted {} acc {} top5 {} f1 {} precision {} recall {}".format((loss_action)/num_video, predict_action/num_video,
                                                            accs_action/num_video, top5_action/num_video, f1_action/num_video,
                                                                    precision_action/num_video, recall_action/num_video))
    print("eval ingredient loss {} predicted {} acc {} top5 {} f1 {} precision {} recall {}".format((loss_ingre)/num_video, predict_ingre/num_video,
                                                                accs_ingre/num_video, top5_ingre/num_video,f1_ingre/num_video,
                                                                                                    precision_ingre/num_video, recall_ingre/num_video))
    # return loss/len(eval_video), accs/len(eval_video)
    return accs_action/len(eval_video), top5_action/len(eval_video), accs_ingre/len(eval_video), top5_ingre/len(eval_video)

def save_model(epoch):
    save_path = args.model_save_path+"seed_"+str(args.seed)+"/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if args.embedder_loss:
        temp = 'embedder_'
    elif args.checker_loss:
        temp = 'checker_'
    else:
        temp = 'baseline_'

    torch.save(step_model.state_dict(), save_path + temp + str(epoch) + '.pt')
    torch.save(step_model.state_dict(),args.model_save_path+temp+"latest.pt")





if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda:' + str(args.device_id) if args.cuda else 'cpu')

    zero = torch.tensor(0.).to(device)
    half = torch.tensor(0.5).to(device)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    if args.memory_predictor:
        step_model = Step_Model_Lstm_classification_lstm(args.input_dim, args.hidden_dim, args.output_dim,
                                                         args.num_action, args.num_ingredient, device).to(device)
    else:
        step_model = Step_Model_Lstm(args.input_dim, args.hidden_dim, args.output_dim, args.num_action,
                                     args.num_ingredient, args.multi_label, device).to(device)
    if args.eval:
        step_model.load_state_dict(torch.load(args.model_save_path + "embedder_latest.pt", map_location=device))

    meta_embedder = Node_only_random_agg_min_clip(device).to(device)
    meta_embedder.load_state_dict(torch.load(args.meta_embedder_path, map_location=device))
    edge_embedder = torch.load(args.edge_embedder_path, map_location=device)

    all = {}
    ingredients, actions = read_cat(args.video_dataset_info)
    all["actions"] = actions
    all["ingredients"] = ingredients
    op_feature = np.load(args.logic_dataset_path + "/op.npy", allow_pickle=True)
    graph_node_feature = np.load(args.logic_dataset_path + "/node.npy", allow_pickle=True)
    info_embedder = (all, graph_node_feature, op_feature)

    ##########################

    glove_features = pk.load(open(args.video_dataset_info + "rel_glove_features_6B100", "rb"))
    ingredient_cat = read_strlist(args.video_dataset_info + "ingredient_cat.txt")
    action_cat = read_strlist(args.video_dataset_info + "action_cat.txt")

    train_video_name, train_video, train_action_label, train_ingre_label, eval_video_name, eval_video, eval_action_label, eval_ingre_label = \
        build_dataset_action_recognition(args.eval, args.video_dataset_path, args.interval, args.video_dataset_info,
                                         args.num_data, args.step_avg, device)

    # model=RNN(input_dim,hidden_dim,2)
    optimizer = torch.optim.Adam(step_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_model = torch.nn.CrossEntropyLoss().to(device)

    epoch_num = 200
    lr_emb = args.lr
    if args.eval:
        epoch_num = 1
    for epoch in range(epoch_num):
        if not args.eval:
            save_model(epoch)
            train_video, train_action_label, train_ingre_label = shuffledata(train_video, train_action_label,
                                                                             train_ingre_label)
            print("\n ***************** Train at epoch {}".format(epoch))
            train(optimizer, loss_model, train_video, train_action_label, train_ingre_label, train_video_name,
                  glove_features, action_cat, ingredient_cat)
            # if epoch % 40 == 0 and epoch != 0:
            #     lr_emb /= 2
            #     for param in optimizer.param_groups:
            #         param["lr"] = lr_emb
        acc_action, top5_action, acc_ing, top5_ing = eval(loss_model, eval_video, eval_action_label, eval_ingre_label,
                                                          eval_video_name, glove_features, action_cat, ingredient_cat)


