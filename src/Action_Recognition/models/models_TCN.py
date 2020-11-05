import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from src.Action_Recognition.models.models_predictor import Single_MLP, Lstm_MLP
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Step_Model_TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, num_actions, num_ingredients):
        super(Step_Model_TCN, self).__init__()

        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.layer = 1

        hidden_dim = num_channels[-1]



        self.action_mlp = Single_MLP(hidden_dim * self.layer, output_size, num_actions)
        self.ingredient_mlp = Single_MLP(hidden_dim * self.layer, output_size, num_ingredients)

    def forward(self, data):
        output_h = self.tcn(data.unsqueeze(-1))
        out = torch.max(output_h, dim=0)[0].squeeze()

        out_action = self.action_mlp(out)
        out_ingre = self.ingredient_mlp(out)
        return out_action, out_ingre



class Tcn_Lstm_MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Tcn_Lstm_MLP, self).__init__()

        self.dropout=0.5

        self.fc1= nn.Linear(input_dim, hidden_dim*4)
        self.fc2= nn.Linear(hidden_dim*4, hidden_dim*2)
        self.fc3= nn.Linear(hidden_dim*2, hidden_dim//2)
        self.fc4= nn.Linear(hidden_dim//2, output_dim)

        # self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim*2),nn.ReLU(),nn.Linear(hidden_dim*2, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2,output_dim))

    def forward(self, data):
        out = F.dropout(F.relu(self.fc1(data)),p=self.dropout,training=self.training)
        out = F.dropout(F.relu(self.fc2(out)),p=self.dropout,training=self.training)
        out = F.dropout(F.relu(self.fc3(out)),p=self.dropout,training=self.training)
        out = F.dropout(F.relu(self.fc4(out)),p=self.dropout,training=self.training)
        # out = torch.softmax(out, dim=1)
        return out

class Step_Model_TCN_classification_lstm(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, num_actions, num_ingredients, device):
        super(Step_Model_TCN_classification_lstm, self).__init__()

        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.layer = 1
        self.init_layer=1
        self.device=device

        hidden_dim = num_channels[-1]
        self.hidden_dim=hidden_dim

        self.lstm_hidden_dim=1024

        self.bidirectional=True
        if self.bidirectional:
            self.init_layer*=2


        self.action_lstm = torch.nn.LSTM(hidden_dim*self.layer, self.lstm_hidden_dim, bidirectional=self.bidirectional)
        self.ingredient_lstm = torch.nn.LSTM(hidden_dim*self.layer, self.lstm_hidden_dim, bidirectional=self.bidirectional)

        self.action_mlp = Lstm_MLP(self.lstm_hidden_dim * self.init_layer, output_size, num_actions)
        self.ingredient_mlp = Lstm_MLP(self.lstm_hidden_dim* self.init_layer, output_size, num_ingredients)

    def forward(self, data):
        output_h = self.tcn(data.unsqueeze(-1))
        out = torch.max(output_h, dim=0)[0].transpose(0,1)
        return out


    def classification_forward(self, data):
        action_h, action_c = self.initHidden_Cell(self.init_layer, self.lstm_hidden_dim)
        ing_h, ing_c = self.initHidden_Cell(self.init_layer, self.lstm_hidden_dim)
        feature_out_action = self.action_lstm(data,(action_h,action_c))[0]
        feature_out_ingre = self.ingredient_lstm(data, (ing_h, ing_c))[0]

        out_action = self.action_mlp(feature_out_action).squeeze()
        out_ingre = self.ingredient_mlp(feature_out_ingre).squeeze()

        return out_action.to(self.device), out_ingre.to(self.device)


    def initHidden_Cell(self, num, hidden_dim):
        '''
        :return: h0 c0 (num_layers * num_directions, batch, hidden_size)
        '''
        return torch.randn([num, 1, hidden_dim]).to(self.device), torch.randn(
            [num, 1, hidden_dim]).to(self.device)