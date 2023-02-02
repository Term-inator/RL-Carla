from enum import Enum

import torch
from torch import nn
import torch.nn.functional as F


class ICMType(Enum):
    LINEAR = 0
    LSTM = 1
    DNN = 2


class LSTMPrediction(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMPrediction, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class DNNPrediction(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DNNPrediction, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class ICM(nn.Module):
    def __init__(self, icm_type, state_dim, action_dim, hidden_dim=256):
        super(ICM, self).__init__()
        if icm_type is ICMType.LINEAR:
            self.predict_module = nn.Linear(state_dim + action_dim, state_dim)
            self.inv_predict_module = nn.Linear(state_dim + state_dim, action_dim)
        elif icm_type is ICMType.LSTM:
            self.predict_module = LSTMPrediction(state_dim + action_dim, hidden_dim, state_dim)
            self.inv_predict_module = LSTMPrediction(state_dim + state_dim, hidden_dim, action_dim)
        elif icm_type is ICMType.DNN:
            self.predict_module = DNNPrediction(state_dim + action_dim, hidden_dim, state_dim)
            self.inv_predict_module = DNNPrediction(state_dim + state_dim, hidden_dim, action_dim)

    def forward(self, state, next_state, action):
        next_state_predict = self.predict_module(torch.cat((state, action), 1))
        action_predict = self.inv_predict_module(torch.cat((state, next_state), 1))
        return next_state_predict, action_predict
