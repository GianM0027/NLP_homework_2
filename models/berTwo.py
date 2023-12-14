import os

import torch
import transformers

from drTorch.modules import TrainableModule
from torch.nn.functional import max_pool2d


class BerTwo(TrainableModule):

    def __init__(self, dropout_prob: float = 0.3, hidden_size: int = 768, bert_version: os.path = './bert_models/bert-base-uncased'):
        """

        :param dropout_prob:
        :param hidden_size:
        """
        super(BerTwo, self).__init__()

        self.bert_1 = transformers.BertModel.from_pretrained(bert_version)
        self.bert_2 = transformers.BertModel.from_pretrained(bert_version)

        self.drop_out = torch.nn.Dropout(dropout_prob)

        self.clf_opc = torch.nn.Linear(hidden_size * 2, 2)
        self.clf_se = torch.nn.Linear(hidden_size * 2, 2)
        self.clf_c = torch.nn.Linear(hidden_size * 2, 2)
        self.clf_st = torch.nn.Linear(hidden_size * 2, 2)

    def forward(self, kwards: dict) -> torch.Tensor:
        output_loss_conclusion, output_logits_conclusion = self.bert1(**kwards['Conclusion'], return_dict=False)
        output_loss_premise, output_logits_premise = self.bert2(**kwards['Premise'], return_dict=False)

        output_conclusion = self.drop_out(output_logits_conclusion)
        output_premise = self.drop_out(output_logits_premise)

        concatenated_output = torch.cat(output_conclusion, output_premise)

        output_clf_opc = self.clf_opc(concatenated_output)
        output_clf_se = self.clf_se(concatenated_output)
        output_clf_c = self.clf_c(concatenated_output)
        output_clf_st = self.clf_st(concatenated_output)

        output = torch.stack(tensors=(output_clf_opc, output_clf_se, output_clf_c, output_clf_st), dim=1)

        return output
