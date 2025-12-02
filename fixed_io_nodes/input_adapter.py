import torch
from torch import nn
from typing import List

class LinearInputAdapter(nn.Module):

    def __init__(
        self,
        input_dim:int,
        output_dim:int,
        hidden_dims:List[int]=[],
        dropout:float=0.1,
        normalization_layer = 'layer_norm',        
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):

            linear = nn.Linear(prev_dim, hidden_dim)
            layers.append(linear)

            #norm before activation (unlike in prev code)
            if normalization_layer == 'layer_norm':
                layers.append(nn.LayerNorm(hidden_dim))
            else:
                raise NotImplementedError(f"Normalization layer {normalization_layer} not implemented")

            layers.append(nn.ReLU())

            #dropout (not on last hidden layer) (why?, it was done in prev code)
            if dropout > 0 and i<len(hidden_dims)-1:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim


        output_linear = nn.Linear(prev_dim, output_dim)

        layers.append(output_linear)
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

   
    def forward(self, x):
        return self.model(x)
