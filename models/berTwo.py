import os

import torch
import transformers

from drTorch.modules import TrainableModule


from typing import Union
bert_model = transformers.BertModel | transformers.RobertaModel


class BerTwo(TrainableModule):
    # todo documentation

    def __init__(self,
                 dropout_prob:  float = 0.3,
                 hidden_size: int = 768,
                 pretrained_model_name_or_path: Union[str, os.path] = 'bert-base-uncased',
                 bert_constructor: bert_model = transformers.BertModel):
        """

        :param dropout_prob:
        :param hidden_size:
        """
        super(BerTwo, self).__init__()

        self.bert_1 = bert_constructor.from_pretrained(pretrained_model_name_or_path)
        self.bert_2 = bert_constructor.from_pretrained(pretrained_model_name_or_path)

        self.drop_out = torch.nn.Dropout(dropout_prob)

        self.clf_opc = torch.nn.Linear(hidden_size * 2, 2)
        self.clf_se = torch.nn.Linear(hidden_size * 2, 2)
        self.clf_c = torch.nn.Linear(hidden_size * 2, 2)
        self.clf_st = torch.nn.Linear(hidden_size * 2, 2)

    def forward(self, kwards: dict) -> torch.Tensor:
        output_loss_conclusion, output_logits_conclusion = self.bert_1(**kwards['Conclusion'], return_dict=False)
        output_loss_premise, output_logits_premise = self.bert_2(**kwards['Premise'], return_dict=False)

        output_conclusion = self.drop_out(output_logits_conclusion)
        output_premise = self.drop_out(output_logits_premise)

        concatenated_output = torch.cat((output_conclusion, output_premise), -1)

        output_clf_opc = self.clf_opc(concatenated_output)
        output_clf_se = self.clf_se(concatenated_output)
        output_clf_c = self.clf_c(concatenated_output)
        output_clf_st = self.clf_st(concatenated_output)

        output = torch.stack(tensors=(output_clf_opc, output_clf_se, output_clf_c, output_clf_st), dim=1)

        return output
