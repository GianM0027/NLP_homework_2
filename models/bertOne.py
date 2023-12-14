import os

import torch
import transformers

from drTorch.modules import TrainableModule
from torch.nn.functional import max_pool2d

bert_model = transformers.BertModel | transformers.RobertaModel


class BertOne(TrainableModule):
    # todo documentation

    def __init__(self,
                 dropout_prob: float = 0.3,
                 hidden_size: int = 768,
                 bert_version: os.path = './bert_models/bert-base-uncased',
                 bert_constructor: bert_model = transformers.BertModel):
        super(BertOne, self).__init__()

        self.bert = bert_constructor.from_pretrained(bert_version)
        self.drop_out = torch.nn.Dropout(dropout_prob)

        self.clf_opc = torch.nn.Linear(hidden_size, 2)
        self.clf_se = torch.nn.Linear(hidden_size, 2)
        self.clf_c = torch.nn.Linear(hidden_size, 2)
        self.clf_st = torch.nn.Linear(hidden_size, 2)

    def forward(self, kwards: dict) -> torch.Tensor:
        output_loss, output_logits = self.bert(**kwards['Conclusion'], return_dict=False)
        output = self.drop_out(output_logits)

        output_clf_opc = self.clf_opc(output)
        output_clf_se = self.clf_se(output)
        output_clf_c = self.clf_c(output)
        output_clf_st = self.clf_st(output)

        output = torch.stack(tensors=(output_clf_opc, output_clf_se, output_clf_c, output_clf_st), dim=1)

        return output
