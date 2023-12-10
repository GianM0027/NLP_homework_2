import torch
import transformers

from drTorch.modules import TrainableModule
from torch.nn.functional import sigmoid


class BertOne(TrainableModule):

    def __init__(self, dropout_prob=0.3, hidden_size=768):
        """

        :param dropout_prob:
        :param hidden_size:
        """
        super(BertOne, self).__init__()

        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(dropout_prob)

        self.clf_opc = torch.nn.Linear(hidden_size, 2)
        self.clf_se = torch.nn.Linear(hidden_size, 2)
        self.clf_c = torch.nn.Linear(hidden_size, 2)
        self.clf_st = torch.nn.Linear(hidden_size, 2)

    def forward(self, **kwards):
        _, output_1 = self.l1(**kwards, return_dict=False)
        output_2 = self.l2(output_1)
        output_clf_opc = self.clf_opc(output_2)
        output_clf_se = self.clf_se(output_2)
        output_clf_c = self.clf_c(output_2)
        output_clf_st = self.clf_st(output_2)

        output = torch.stack(tensors=(output_clf_opc, output_clf_se, output_clf_c, output_clf_st), dim=1)

        return output
