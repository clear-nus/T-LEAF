import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import random

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim*2),nn.ReLU(),nn.Linear(hidden_dim*2, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2,output_dim))

    def forward(self, data):
        out=self.mlp(data)
        return out


class Single_MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Single_MLP, self).__init__()
        self.dropout=0.5

        self.fc1= nn.Linear(input_dim, hidden_dim*8)
        self.fc2= nn.Linear(hidden_dim*8, hidden_dim*2)
        self.fc3= nn.Linear(hidden_dim*2, hidden_dim//2)
        self.fc4= nn.Linear(hidden_dim//2, output_dim)

        # self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim*2),nn.ReLU(),nn.Linear(hidden_dim*2, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2,output_dim))

    def forward(self, data):
        out = F.dropout(F.relu(self.fc1(data)),p=self.dropout,training=self.training)
        out = F.dropout(F.relu(self.fc2(out)),p=self.dropout,training=self.training)
        out = F.dropout(F.relu(self.fc3(out)),p=self.dropout,training=self.training)
        out = F.dropout(F.relu(self.fc4(out)),p=self.dropout,training=self.training)
        return out

class Lstm_MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Lstm_MLP, self).__init__()

        self.dropout=0.5

        self.fc1= nn.Linear(input_dim, hidden_dim*2)
        # self.fc2= nn.Linear(hidden_dim*8, hidden_dim*2)
        self.fc3= nn.Linear(hidden_dim*2, hidden_dim//2)
        self.fc4= nn.Linear(hidden_dim//2, output_dim)

        # self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim*2),nn.ReLU(),nn.Linear(hidden_dim*2, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2,output_dim))

    def forward(self, data):
        out = F.dropout(F.relu(self.fc1(data)),p=self.dropout,training=self.training)
        # out = F.dropout(F.relu(self.fc2(out)),p=self.dropout,training=self.training)
        out = F.dropout(F.relu(self.fc3(out)),p=self.dropout,training=self.training)
        out = F.dropout(F.relu(self.fc4(out)),p=self.dropout,training=self.training)
        # out = torch.softmax(out, dim=1)

        return out

class Step_Model_Lstm(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim, num_actions, num_ingredients, multi_label, device):
        super(Step_Model_Lstm,self).__init__()
        self.device = device
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.num_ingredients = num_ingredients
        self.bidirectional = True
        self.num_layers = 1
        self.layer=1
        if self.bidirectional:
            self.layer*=2

        self.lstm=torch.nn.LSTM(input_dim,hidden_dim ,bidirectional=self.bidirectional)
        if multi_label:
            self.action_mlp = MLP(hidden_dim*self.layer, output_dim, num_actions).to(device)
            self.ingredient_mlp = MLP(hidden_dim*self.layer, output_dim, num_ingredients).to(device)
        else:
            self.action_mlp = Single_MLP(hidden_dim*self.layer, output_dim, num_actions).to(device)
            self.ingredient_mlp = Single_MLP(hidden_dim*self.layer, output_dim, num_ingredients).to(device)

    def forward(self, data):
        h_0, c_0 = self.initHidden_Cell()
        output_h, output_c = self.lstm(data.unsqueeze(1),(h_0,c_0))

        out=torch.max(output_h,dim=0)[0]

        out_action = self.action_mlp(out)
        out_ingre = self.ingredient_mlp(out)

        return out_action, out_ingre

    def initHidden_Cell(self):
        '''
        :return: h0 c0 (num_layers * num_directions, batch, hidden_size)
        '''
        num=self.num_layers
        if self.bidirectional:
            num *= 2
        return torch.randn([num, 1, self.hidden_dim]).to(self.device), torch.randn(
            [num, 1, self.hidden_dim]).to(self.device)


class Step_Model_Lstm_classification_lstm(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim, num_actions, num_ingredients, device):
        super(Step_Model_Lstm_classification_lstm,self).__init__()
        self.device = device
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.num_ingredients = num_ingredients
        self.bidirectional = True
        self.num_layers = 1
        self.layer=1
        if self.bidirectional:
            self.layer*=2

        self.lstm=torch.nn.LSTM(input_dim,hidden_dim ,bidirectional=self.bidirectional)


        self.action_lstm = torch.nn.LSTM(hidden_dim*self.layer, hidden_dim, bidirectional=self.bidirectional)
        self.ingredient_lstm = torch.nn.LSTM(hidden_dim*self.layer, hidden_dim, bidirectional=self.bidirectional)

        self.action_mlp = Lstm_MLP(hidden_dim*self.layer, output_dim, num_actions).to(device)
        self.ingredient_mlp = Lstm_MLP(hidden_dim*self.layer, output_dim, num_ingredients).to(device)

    def forward(self, data):
        h, c=self.initHidden_Cell(self.layer, self.hidden_dim)
        output, (h_n,c_n) = self.lstm(data.unsqueeze(1),(h,c))
        out=torch.max(output,dim=0)[0]
        return out

    def classification_forward(self, data):
        action_h, action_c = self.initHidden_Cell(self.layer, self.hidden_dim)
        ing_h, ing_c = self.initHidden_Cell(self.layer, self.hidden_dim)
        feature_out_action = self.action_lstm(data,(action_h,action_c))[0]
        feature_out_ingre = self.ingredient_lstm(data, (ing_h, ing_c))[0]

        out_action = self.action_mlp(feature_out_action).squeeze()
        out_ingre = self.ingredient_mlp(feature_out_ingre).squeeze()

        return out_action, out_ingre

    def initHidden_Cell(self, num, hidden_dim):
        '''
        :return: h0 c0 (num_layers * num_directions, batch, hidden_size)
        '''

        return torch.randn([num, 1, hidden_dim]).to(self.device), torch.randn(
            [num, 1, hidden_dim]).to(self.device)


