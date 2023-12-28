git import os

import torch
import transformers

from drTorch.modules import TrainableModule


from typing import Union
bert_model = transformers.BertModel | transformers.RobertaModel


class BerThree(TrainableModule):
    """
    A custom PyTorch module implementing a BERT-based model for a specific task with three input components.

    Parameters:
    - dropout_prob (float): The dropout probability for regularization.
    - hidden_size (int): The size of the hidden layers in the BERT model.
    - pretrained_model_name_or_path (str or os.path): The name or path of the pretrained BERT model.
    - bert_constructor (BertModel): The constructor for the BERT model.

    Attributes:
    - bert (BertModel): The BERT model for feature extraction.
    - drop_out (torch.nn.Dropout): Dropout layer for regularization.
    - clf_opc (torch.nn.Linear): Linear layer for the 'Conclusion' classification.
    - clf_se (torch.nn.Linear): Linear layer for the 'Stance Evidence' classification.
    - clf_c (torch.nn.Linear): Linear layer for the 'Claim' classification.
    - clf_st (torch.nn.Linear): Linear layer for the 'Stance' classification.

    Methods:
    - forward(kwards: dict) -> torch.Tensor: Performs forward pass and returns the model's output.

    Note:
    The model expects input data in the form of a dictionary with keys 'Conclusion', 'Premise', and 'Stance',
    each containing sub-dictionaries with keys 'input_ids', 'token_type_ids', and 'attention_mask'.
    """
    def __init__(self,
                 dropout_prob: float = 0.3,
                 hidden_size: int = 768,
                 pretrained_model_name_or_path: Union[str,os.path] = 'bert-base-uncased',
                 bert_constructor: bert_model = transformers.BertModel):
        """
        Initializes the BerThree model with specified parameters.

        :param dropout_prob: The dropout probability for regularization.
        :param hidden_size: The size of the hidden layers in the BERT model.
        :param pretrained_model_name_or_path: The name or path of the pretrained BERT model.
        :param bert_constructor: The constructor for the BERT model.
        """
        super(BerThree, self).__init__()

        self.bert = bert_constructor.from_pretrained(pretrained_model_name_or_path)

        self.drop_out = torch.nn.Dropout(dropout_prob)

        self.clf_opc = torch.nn.Linear(hidden_size * 2 + 1, 2)
        self.clf_se = torch.nn.Linear(hidden_size * 2 + 1, 2)
        self.clf_c = torch.nn.Linear(hidden_size * 2 + 1, 2)
        self.clf_st = torch.nn.Linear(hidden_size * 2 + 1, 2)

    def forward(self, kwards: dict) -> torch.Tensor:
        """
        Performs a forward pass through the BerThree model.

        :param kwards: A dictionary containing 'Conclusion', 'Premise', and 'Stance' data for the model.
        :return: A torch.Tensor representing the model's output.
        """
        output_loss_conclusion, output_logits_conclusion = self.bert(**kwards['Conclusion'], return_dict=False)
        output_loss_premise, output_logits_premise = self.bert(**kwards['Premise'], return_dict=False)
        stance = kwards['Stance'].unsqueeze(1)

        output_conclusion = self.drop_out(output_logits_conclusion)
        output_premise = self.drop_out(output_logits_premise)

        concatenated_output = torch.cat((output_conclusion, output_premise, stance), -1)

        output_clf_opc = self.clf_opc(concatenated_output)
        output_clf_se = self.clf_se(concatenated_output)
        output_clf_c = self.clf_c(concatenated_output)
        output_clf_st = self.clf_st(concatenated_output)

        output = torch.stack(tensors=(output_clf_opc, output_clf_se, output_clf_c, output_clf_st), dim=1)

        return output
