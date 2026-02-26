from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import torch
from torch import nn 
import numpy as np

try:
    from .focal_loss import FocalLoss
except:
    from focal_loss import FocalLoss

class MLP(nn.Module):
    """
    Multilayer Perceptron with [Linear -> BatchNorm1d -> SiLU -> Dropout] blocks.
    Args:
        in_dim (int): Input feature dimension.
        hidden_dims (list[int]): Sizes of hidden layers. e.g., [256, 256, 128]
        out_dim (int): Output feature dimension.
        dropout (float): Dropout probability applied after activation in each hidden layer.
        bias (bool): Use bias in Linear layers.
        final_activation (bool): If True, applies BatchNorm + SiLU after final Linear as well.
        final_bn (bool): If True and final_activation=False, you can still put BN before output layer.
                         (Most common is both False.)
    """
    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dims : list[int] =  [1024,1024],
        out_dim: int = 1024,
        dropout: float = 0.0,
        bias: bool = True,
        final_activation: bool = False,
        final_bn: bool = False,
    ):
        super().__init__()
        dims = [in_dim] + hidden_dims
        layers = []

        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1], bias=bias))
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.SiLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        # Output layer
        self.out = nn.Linear(dims[-1], out_dim, bias=bias)

        # Optional BN/activation around output
        self.final_bn = nn.BatchNorm1d(out_dim) if final_activation or final_bn else None
        self.final_act = nn.SiLU() if final_activation else None

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (B, in_dim)
        """
        x = self.net(x)
        x = self.out(x)
        if self.final_bn is not None:
            x = self.final_bn(x)
        if self.final_act is not None:
            x = self.final_act(x)
        return x
    
class MRLNet(nn.Module):

    def __init__(
        self,        
        num_classes=13,
        input_dim = 1024,
        hidden_dims = [1024, 1024], 
        output_dim = 1024,
        nesting_list: list = [16, 32, 64, 128, 256, 512, 1024],
        relative_importance = [1., 1., 1., 1., 1., 1., 1.],
        **kwargs,
    ):

        super().__init__()

        self.nesting_list = nesting_list
        self.num_classes = num_classes
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.nesting_list = nesting_list
        
        self.relative_importance = relative_importance

        pos_wt = torch.tensor([67.2693, 11.2597,  7.4966, 34.3695, 38.6754, 55.9148, 20.8658, 10.5219,
        82.4634, 11.4098, 97.1219, 45.9571, 22.6204])
        
        # self.loss_fn = FocalLoss(
        #     gamma=2, 
        #     alpha=0.5, 
        #     reduction='sum', 
        #     task_type='multi-label',
        #     num_classes=num_classes)

        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = pos_wt)
        

        self.get_mrl_embedding = MLP(input_dim, hidden_dims, output_dim)
        self.project_to_logit = MRL_Linear_Layer(nesting_list, num_classes)


    def forward(self, embedding):        
        new_embedding = self.get_mrl_embedding(embedding)              
        return self.project_to_logit(new_embedding)
        
    def compute_loss(self, logits, gt):
        loss = 0
        for i in range(len(logits)):
            loss += self.relative_importance[i] * self.loss_fn(logits[i], gt)
            return loss

class MRL_Linear_Layer(nn.Module):
    
    def __init__(self, nesting_list: list = [16, 32, 64, 128, 256, 512, 1024], num_classes=13, **kwargs):
        
        super(MRL_Linear_Layer, self).__init__()
        
        self.nesting_list=nesting_list # set of m in M (Eq. 1)
        self.num_classes=num_classes

        #Instantiating one nn.Linear layer for MRL-E
        setattr(self, "nesting_classifier_0",
                nn.Linear(self.nesting_list[-1], 
                          self.num_classes, **kwargs)) 
                                                  
    def forward(self, x):
            nesting_logits = []
            for i, num_feat in enumerate(self.nesting_list):                   
                efficient_logit = torch.matmul(x[:, :num_feat],
                    (self.nesting_classifier_0.weight[:, :
                num_feat]).t())                
                nesting_logits.append(efficient_logit)
                
            return nesting_logits

if __name__ == "__main__":

    device = "cuda:0"

    net = MRLNet(
        in_dim = 1024,
        hidden_dims = [1024,1024,1024],
        out_dim = 1024,
        dropout = 0.0,
        bias = True,
        final_activation = False,
        final_bn = False).to(device)

    emb = torch.randn(2, 1024).to(device)

    logit = net(emb)
    gt = torch.ones_like(logit[0])

    loss = net.compute_loss(logit, gt)