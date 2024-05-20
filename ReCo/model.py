# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch


class UniXcoderModel(nn.Module):
    def __init__(self, encoder):
        super(UniXcoderModel, self).__init__()
        self.encoder = encoder
      
    def forward(self, code_inputs=None, nl_inputs=None): 
        if code_inputs is not None:
            outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[0]
            outputs = (outputs*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        else:
            outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[0]
            outputs = (outputs*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)


class CodeBERTModel(nn.Module):
    def __init__(self, encoder):
        super(CodeBERTModel, self).__init__()
        self.encoder = encoder

    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            outputs = self.encoder(code_inputs, attention_mask=code_inputs.ne(1))[1]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        else:
            outputs = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[1]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)


class CodeT5pModel(nn.Module):
    def __init__(self, encoder):
        super(CodeT5pModel, self).__init__()
        self.encoder = encoder

    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            outputs = self.encoder(code_inputs, attention_mask=code_inputs.ne(1))
            return outputs
        else:
            outputs = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))
            return outputs
