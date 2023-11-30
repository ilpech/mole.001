import torch
from torch import nn

from varname.helpers import debug

class RegressionBioPerceptron(nn.Module):
    def __init__(
        self,
        input_annotations_len,
        input_type_len,
        input_gene_seq_bow_len,
        input_features_hidden_size,
        hidden_size, #80
        annotation_dropout, #0.25
        hidden_dropout
    ):
        super().__init__()
        self.input_annotations_len = input_annotations_len
        self.input_type_len = input_type_len
        self.input_features_hidden_size = input_features_hidden_size
        self.hidden_size = hidden_size
        self.annotation_dropout = annotation_dropout
        self.hidden_dropout = hidden_dropout
        self.input_gene_seq_bow_len = input_gene_seq_bow_len

        self.annotations_dropout = nn.Dropout(p=self.annotation_dropout) #check value from paper
        self.annotations_linear = nn.Linear(self.input_annotations_len, self.input_features_hidden_size)
        
        self.type_ids_linear = nn.Linear(self.input_type_len, self.input_features_hidden_size)
        
        self.gene_seq_linear = nn.Linear(
            self.input_gene_seq_bow_len,
            self.input_features_hidden_size
        )
        self.gene_seq_bn = nn.BatchNorm1d(self.input_features_hidden_size)
        
        self.linear_layer = nn.Linear(
            self.input_features_hidden_size * 3, 
            self.hidden_size
        )
        self.dropout = nn.Dropout(self.hidden_dropout)
        
        self.activation = nn.ReLU()
        
        self.add_weight_linear = nn.Linear(self.hidden_size, 1)
        self.mul_weight_linear = nn.Linear(self.hidden_size, 1)
    
    def forward(
        self, 
        annotations_tensor,
        type_ids_tensor,
        rna_tensor,
        gene_seq_bow_tensor
    ):
        annotations_tensor = self.annotations_dropout(annotations_tensor)
        annotations_tensor = self.annotations_linear(annotations_tensor)
        
        type_ids_tensor = self.type_ids_linear(type_ids_tensor) #ne uvidel 

        gene_seq_bow_tensor = self.gene_seq_linear(gene_seq_bow_tensor)
        if gene_seq_bow_tensor.size()[0] != 1: # bn need more than 1 value per channel
            gene_seq_bow_tensor = self.gene_seq_bn(self.activation(gene_seq_bow_tensor))
        
        all_inputs = torch.cat(
            (annotations_tensor, type_ids_tensor, gene_seq_bow_tensor), 1 #in original paper they also use rp_in
        )
        x = self.linear_layer(all_inputs)
        x = self.dropout(x)
        x = self.activation(x)
        
        add_weight = self.add_weight_linear(x)
        mul_weight = self.mul_weight_linear(x)
        # debug(rna_tensor.shape)
        # debug(mul_weight.shape)
        # debug((rna_tensor*mul_weight).shape)
        result = (rna_tensor*mul_weight)+add_weight
        # debug(result.shape)
        return result
        
        