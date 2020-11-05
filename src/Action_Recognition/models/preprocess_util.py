import os
import numpy as np
import torch
from random import shuffle
from sklearn.metrics import precision_score, f1_score, recall_score


def read_strlist(path):
    '''

    :param path: strlist data path
    :return: list of string ["ab","cd]
    '''
    with open(path) as file:
        textfile=file.read()
        lines=textfile.split("\n")[0:-1]
        return lines


def write_strlist(path,data):
    '''

    :param path: file path
    :param data: a list of string
    :return:
    '''
    s=""
    for item in data:
        s+=str(item)+"\n"
    with open(path,"w") as file:
        file.write(s)

def read_alignment(path):
    '''

    :param path: alignment path
    :return: alignment
    '''
    with open(path,"r") as file:
        aligns_str=file.read()
        aligns=aligns_str.split("\n")[0:-2]
    frames_align=[]
    discard=[]
    for index, align in enumerate(aligns):
        start, end = int(align.split(",")[0]), int(align.split(",")[1])
        if start < 0 or end <0 or start == end:
            discard.append(index)
        frames_align.append([start,end])
    return frames_align, discard

def read_frame_step(data, aligns):
    '''
    return corresponding frame data to each steps
    :param data: frame data
    :param aligns: aligns data
    :return:
    '''
    step_frame=[]
    for align in aligns:
        step_frame.append(data[align[0]:align[1]])
    return step_frame

# def read_steps(path):
#     '''
#
#     :param path: path of recipe step text
#     :return: list[steps[step[action,list[ingredients]]]]
#     '''
#     def extract(s):
#         s=s.replace("(","")
#         actions=s.split(")")[0:-1]
#         act_split=[]
#         for action in actions:
#             if len(action.split(":"))<2:
#                 continue
#             verb,objects=action.split(":")[0].strip().lower(), action.split(":")[1].strip()
#             objects_new=[]
#             for object in objects.split(","):
#                 object_new=object.strip().lower()
#                 if object_new=="":
#                     continue
#                 objects_new.append(object_new)
#             act_split.append([verb,objects_new])
#         return act_split
#     with open(path+"/action_ingredient.txt","r") as f:
#         file=f.read()
#         steps=[]
#         for line in file.split("\n"):
#             if line.strip() =="" or line =="() : ) ":
#                 continue
#             step=extract(line)
#             if len(step)==0:
#                 continue
#             steps.append(step)
#         return steps

def read_steps(path):
    def extract(s):
        actions=s.split("; ")[0:-1]
        act_split=[]
        for action in actions:
            temp = action.split(" : ")
            verb = temp[0].strip()
            objects= temp[1].replace("[","").replace("]","").replace("'","")
            objects_new=[]
            for object in objects.split(","):
                object_new=object.strip().lower()
                if object_new=="":
                    continue
                objects_new.append(object_new)
            act_split.append([verb,objects_new])
        return act_split
    with open(path+"/steps.txt","r") as f:
        file=f.read()
        steps=[]
        for line in file.split("\n"):
            if line.strip() =="" or line =="() : ) ":
                continue
            step=extract(line)
            if len(step)==0:
                continue
            steps.append(step)
        return steps

def read_cat(cat_path):

    ingredient_cat = read_strlist(cat_path+"ingredient_cat.txt")
    action_cat = read_strlist(cat_path + "action_cat.txt")
    return ingredient_cat, action_cat

def step_tag(ingredients, actions, steps, device):
    action_labels = []
    ingre_labels = []
    for step in steps:
        tag_step=0
        for action_ingredient in step:
            action, ingredient_action=action_ingredient[0], action_ingredient[1]
            if len(ingredient_action) > 0 :
                tag=0
                for ingre in ingredient_action:
                    if ingre in ingredients:
                        ingredients_label=ingredients.index(ingre)
                        tag=1
                        break
                if tag==0:
                    ingredients_label=len(ingredients)
                if action in actions:
                    actions_label = actions.index(action)
                else:
                    actions_label = len(actions)
                tag_step=1
            if tag_step == 1:
                break
        if tag_step ==0:
            action, ingredient_action = step[0][0], step[0][1]
            if action in actions:
                actions_label = actions.index(action)
            else:
                actions_label = len(actions)
            ingredients_label = len(ingredients)
        action_labels.append(actions_label)
        ingre_labels.append(ingredients_label)

    return [torch.LongTensor(action_labels).to(device), torch.LongTensor(ingre_labels).to(device)]


def read_video_data(dataset_root,interval, kitchen_dataset, size, avg, device):
    '''
    read all frame data in a list
    :param dataset_root: root of tasty video dataset
    :param interval: the interval between interval
    :return: [recipe[[video_frame[steps]],[step_label[steps,[action, ingredient]]]]]
    '''
    videos=os.listdir(dataset_root)
    dataset=[]
    ingredients, actions= read_cat(kitchen_dataset)
    if size > len(videos):
        size = len(videos)
    count =0
    index = 0
    while count < size and index < len(videos):
        video = videos[index]
        index+=1
        video_path=os.path.join(dataset_root,video)
        # print(video)
        frame_data=np.load(video_path+"/resnet50.npy")
        frame_data_interval=frame_data[::interval]
        frame_align, discard=read_alignment(video_path+"/csvalignment.dat")
        raw_steps_text=read_steps(video_path)
        if len(raw_steps_text) < len(frame_align):
            frame_align=frame_align[0:len(raw_steps_text)]
        valid_frame_align, valid_step_text=[],[]
        for i in range(len(frame_align)):
            if i in discard:
                continue
            valid_frame_align.append(frame_align[i])
            valid_step_text.append(raw_steps_text[i])


        steps_text=valid_step_text
        step_frame=read_frame_step(frame_data_interval,valid_frame_align)

        # print(video,len(steps_text),len(step_frame))
        if len(step_frame) <3:
            continue
        step_label=step_tag(ingredients, actions, steps_text, device)

        if avg:
            mean_frame = []
            for frame in step_frame:
                mean_frame.append(torch.mean(torch.tensor(frame,dtype=torch.float32),dim=0).unsqueeze(0))
            if torch.isnan(torch.cat(mean_frame,dim=0)).sum()>0:
                continue
            dataset.append([video, torch.cat(mean_frame,dim=0).to(device),step_label])
            # print(video,step_label[0], step_label[1])
        else:
            empty = 0
            for frame in step_frame:
                if len(frame) ==0:
                    empty=1
                    break
            if empty==1:
                print("empty frame",video,valid_frame_align,steps_text)
                continue
            dataset.append([video, step_frame, step_label])
            # print(video)
        count +=1
    return dataset
# read_video_data("../../dataset/Tasty", 5, "../../dataset/Kitchen/",4000, False ,True)

###########################################################
def build_dataset_action_recognition(eval, video_dataset_path, interval, video_dataset_info, num_data, step_avg, device):
    '''
    build the dataset of training data.
    :return:
    '''

    dataset= read_video_data(video_dataset_path, interval, video_dataset_info, num_data, step_avg, device)
    dataset_size=len(dataset)
    # mask = np.random.choice([0, 1], dataset_size, replace=True, p=[1-args.ratio,args.ratio])
    mask = np.zeros([dataset_size])
    mask[0:int(0.8*dataset_size)]=1
    train_video_name, train_video, train_action_label, train_ingre_label = [], [], [], []
    eval_video_name, eval_video, eval_action_label, eval_ingre_label = [], [], [], []
    start=0
    if eval:
        start=int(0.8*dataset_size)
    for index, data in enumerate(dataset[start:]):
        if mask[start+index] == 1:
            train_video_name.append(data[0])
            train_video.append(data[1])
            train_action_label.append(data[2][0])
            train_ingre_label.append(data[2][1])
        else:
            eval_video_name.append(data[0])
            eval_video.append(data[1])
            eval_action_label.append(data[2][0])
            eval_ingre_label.append(data[2][1])
    return train_video_name, train_video, train_action_label, train_ingre_label, eval_video_name, eval_video, eval_action_label, eval_ingre_label

def shuffledata(train_video,train_action_label,train_ingre_label):
    '''
    shfffle the dataset
    :param train_data:
    :param train_label:
    :return:
    '''
    index=[i for i in range(len(train_video))]
    shuffle(index)
    train_video_new, train_action_label_new, train_ingre_label_new = [], [], []
    for i in index:
        train_video_new.append(train_video[i])
        train_action_label_new.append(train_action_label[i])
        train_ingre_label_new.append(train_ingre_label[i])
    return train_video_new, train_action_label_new, train_ingre_label_new

def accuracy_cal(outputs, targets, k):
    '''

    :param outputs:
    :param targets:
    :return:
    '''
    top_output=torch.topk(outputs,k,dim=-1)[1]
    index_target= torch.cat([targets.unsqueeze(-1)]*k, dim=-1).reshape(top_output.shape)
    predict=top_output==index_target
    return predict.sum()*1.0/targets.shape[0]

def recall_precision_f1(macro, outputs, targets):

    outputs=outputs.cpu()
    targets=targets.cpu()
    output_index = outputs.argmax(dim=1)
    if macro:
        precision=precision_score(targets, output_index, average="macro")
        recall=recall_score(targets, output_index, average="macro")
        f1=f1_score(targets, output_index, average="macro")
    else:
        precision=precision_score(targets, output_index, average="micro")
        recall=recall_score(targets, output_index, average="micro")
        f1=f1_score(targets, output_index, average="micro")

    return precision, recall, f1