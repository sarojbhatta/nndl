import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
sys.path.extend(['../../'])
from models.base import base_NN, Embedding_layers

class SEBlock(nn.Module):
    def __init__(self, input_dim, dtype=torch.float64, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // reduction, dtype=dtype)
        self.fc2 = nn.Linear(input_dim // reduction, input_dim, dtype=dtype)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        # z = torch.mean(x, dim=1, keepdim=True)  # Global average pooling
        z = x
        z = self.fc1(z)
        z = F.relu(z)
        z = self.fc2(z)
        z = self.activation(z)
        return x * z

class SE_Residual_Block(nn.Module):
    def __init__(self, input_dim, dtype=torch.float64):
        super(SE_Residual_Block, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim, dtype=dtype)
        self.fc2 = nn.Linear(input_dim, input_dim, dtype=dtype)
        self.se = SEBlock(input_dim, dtype)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out = self.se(out)  # Apply SE Block
        return out + residual

def get_layer(input_dim: int, layer_list: list, p: float, act=nn.ReLU(), dtype=torch.float64) -> nn.Sequential:
    current_layer = nn.Sequential()
    lst_dim = input_dim
    for current_dim in layer_list:
        current_layer.append(nn.Linear(lst_dim, current_dim, bias=True, dtype=dtype))
        current_layer.append(nn.BatchNorm1d(current_dim, dtype=dtype))
        current_layer.append(act)
        current_layer.append(nn.Dropout(p))

        current_layer.append(SE_Residual_Block(current_dim, dtype))
        lst_dim = current_dim
    return current_layer


class TS_SE_ResidualNN(base_NN):
    def __init__(self, cat_length: list, cot_length: int, stack_layers: list, 
                 act: str='ReLu', p: float=0.3):
        super(TS_SE_ResidualNN,self).__init__()
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