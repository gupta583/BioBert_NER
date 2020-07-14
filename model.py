import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel


class ner_bio(nn.Module):
    def __init__(self,vocab_len,config,state_dict):
        super().__init__()
        self.bert = BertModel(config)
        if state_dict is not None:
            self.bert.load_state_dict(state_dict)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size,vocab_len)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,input_ids,attention_mask):
        encoded_layer,_ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        enc = encoded_layer[-1]
        output = self.drop(enc)
        output = self.out(output)

        return output, output.argmax(-1)
