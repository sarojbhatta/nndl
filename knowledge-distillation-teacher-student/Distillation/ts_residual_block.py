import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.extend(['../../'])
from models.base import Embedding_layers, base_NN

class ResidualBlock(torch.nn.Module):
    def __init__(self, hidden_dim: int, act=nn.ReLU(), p: float=0.3, dtype=torch.float64):
        super(ResidualBlock,self).__init__()
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        
        self.act = act
        self.dropout = nn.Dropout(p=p)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True, dtype=self.dtype),
            nn.BatchNorm1d(hidden_dim, dtype=self.dtype),  # Batch Normalization
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True, dtype=self.dtype),
            nn.BatchNorm1d(hidden_dim, dtype=self.dtype),  # Batch Normalization
        )

    def forward(self, x):
        y = self.dropout(self.act(self.fc1(x)))
        y = self.fc2(y)
        return self.dropout(self.act(x+y))


def get_layer(input_dim: int, layer_list: list, p: float, act=nn.ReLU(), dtype=torch.float64) -> nn.Sequential:
    current_layer = nn.Sequential()
    lst_dim = input_dim
    for current_dim in layer_list:
        current_layer.append(nn.Linear(lst_dim, current_dim, bias=True, dtype=dtype))
        current_layer.append(nn.BatchNorm1d(current_dim, dtype=dtype))
        current_layer.append(act)
        current_layer.append(nn.Dropout(p))

        current_layer.append(ResidualBlock(current_dim, act, p, dtype))
        lst_dim = current_dim
    return current_layer

class TS_ResidualNN(base_NN):
    def __init__(self, cat_length: list, cot_length: int, stack_layers: list, 
                 act: str='ReLu', p: float=0.3):
        super(TS_ResidualNN,self).__init__()
        self.act = nn.ReLU()
        if act == 'ReLu':
            self.act = nn.ReLU()
        elif act == 'Tanh':
            self.act = nn.Tanh()
        elif act == 'ELU':
            self.act = nn.ELU()
        
        self.cat_length = cat_length
        self.cot_length = cot_length
        self.stack_layers = stack_layers

        self.embedding_layers = Embedding_layers(cat_length, cot_length)
        self.input_dim = self.embedding_layers.input_dim

        self.deep_layers = get_layer(self.input_dim, stack_layers, p, act=self.act)
        self.fc = nn.Sequential(
            nn.Linear(stack_layers[-1], 1, dtype=torch.float64),
            nn.Sigmoid()
        )
        
    def forward(self, input_cat, input_cot):
        embedding = self.embedding_layers(input_cat, input_cot)
        embedding = self.deep_layers(embedding)

        return self.fc(embedding)
    
    def logits(self, input_cat, input_cot):
        # bypass the sigmoid
        return self.fc(self.deep_layers(self.embedding_layers(input_cat, input_cot)))