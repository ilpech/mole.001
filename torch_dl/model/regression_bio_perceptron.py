import torch
from torch import nn

from varname.helpers import debug

class RegressionBioPerceptron(nn.Module):
    def __init__(
        self,
        hidden_size, #80
        annotation_dropout, #0.25
        hidden_dropout
    ):
        super().__init__()
        
        self.input_features_hidden_size = int(hidden_size / 2)
        self.hidden_size = hidden_size
        self.annotation_dropout = annotation_dropout
        self.hidden_dropout = hidden_dropout

        self.annotations_dropout = nn.Dropout(p=self.annotation_dropout) #check value from paper
        self.annotations_linear = nn.Linear(135967, self.input_features_hidden_size)
        
        self.type_ids_linear = nn.Linear(46, self.input_features_hidden_size)
        
        self.linear_layer = nn.Linear(
            self.hidden_size, 
            self.input_features_hidden_size
        )
        self.dropout = nn.Dropout(self.hidden_dropout)
        
        self.activation = nn.ReLU()
        
        self.add_weight_linear = nn.Linear(self.input_features_hidden_size, 1)
        self.mul_weight_linear = nn.Linear(self.input_features_hidden_size, 1)
    
    def forward(
        self, 
        annotations_tensor,
        type_ids_tensor,
        rna_tensor
    ):
        annotations_tensor = self.annotations_dropout(annotations_tensor)
        annotations_tensor = self.annotations_linear(annotations_tensor)
        
        type_ids_tensor = self.type_ids_linear(type_ids_tensor) #ne uvidel 
        
        all_inputs = torch.cat(
            (annotations_tensor, type_ids_tensor), 1 #in original paper they also use rp_in
        )
        x = self.linear_layer(all_inputs)
        x = self.dropout(x)
        x = self.activation(x)
        
        add_weight = self.add_weight_linear(x)
        mul_weight = self.mul_weight_linear(x)
        debug(rna_tensor.shape)
        debug(mul_weight.shape)
        debug((rna_tensor*mul_weight).shape)
        result = (rna_tensor*mul_weight)+add_weight
        debug(result.shape)
        return result
        
        