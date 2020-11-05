import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from a2c_ppo_acktr.utils import AddBias, init

cuda = True
device = torch.device("cuda:1" if cuda else "cpu")
"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)

class MultiCategorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(MultiCategorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear_act = init_(nn.Linear(num_inputs, num_outputs[0]))
        self.linear_ingred = init_(nn.Linear(num_inputs, num_outputs[1]))
        self.linear_ingred2 = init_(nn.Linear(num_inputs, num_outputs[2]))

    def forward(self, x):
        x_act = self.linear_act(x)
        x_ingred = self.linear_ingred(x)
        x_ingred2 = self.linear_ingred(x)
        return FixedCategorical(logits=x_act), FixedCategorical(logits=x_ingred), FixedCategorical(logits=x_ingred2)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, std_parametrization = 'exp', min_std=1e-6):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        # self.logstd = AddBias(torch.zeros(num_outputs))
        self.fc_logstd = init_(nn.Linear(num_inputs, num_outputs))

        if std_parametrization == 'exp':
            self.min_std_param = np.log(min_std)
        elif std_parametrization == 'softplus':
            self.min_std_param = np.log(np.exp(min_std) - 1)
        else:
            raise NotImplementedError
        self.std_parametrization = std_parametrization

    def forward(self, x):
        action_mean = self.fc_mean(x)
        std_param_var = self.fc_logstd(x)

        if self.min_std_param is not None:
            std_param_var = torch.max(std_param_var, torch.ones(std_param_var.shape).to(device)*self.min_std_param)
            std_param_var = torch.min(std_param_var, torch.log(torch.ones(std_param_var.shape).to(device) * np.exp(2)))

        if self.std_parametrization == 'exp':
            log_std_var = std_param_var
        elif self.std_parametrization == 'softplus':
            log_std_var = torch.log(torch.log(1. + torch.exp(std_param_var)))
        else:
            raise NotImplementedError

        '''
        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        '''
        return FixedNormal(action_mean, log_std_var.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)