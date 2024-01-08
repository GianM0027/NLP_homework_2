import os

import torch
import transformers

from drTorch.modules import TrainableModule

from typing import Union
bert_model = transformers.BertModel | transformers.RobertaModel


class BertOne(TrainableModule):
    """
    A custom PyTorch module implementing a simplified BERT-based model for a specific task.

    Parameters:
    - dropout_prob (float): The dropout probability for regularization.
    - hidden_size (int): The size of the hidden layers in the BERT model.
    - pretrained_model_name_or_path (str or os.path): The name or path of the pretrained BERT model.
    - bert_constructor (BertModel): The constructor for the BERT model.

    Attributes:
    - bert (BertModel): The BERT model for feature extraction.
    - drop_out (torch.nn.Dropout): Dropout layer for regularization.
    - clf_opc (torch.nn.Linear): Linear layer for the 'Opens to change' classification.
    - clf_se (torch.nn.Linear): Linear layer for the 'Self enhancement' classification.
    - clf_c (torch.nn.Linear): Linear layer for the 'Conversation' classification.
    - clf_st (torch.nn.Linear): Linear layer for the 'Self transcendence' classification.

    Methods:
    - forward(kwards: dict) -> torch.Tensor: Performs forward pass and returns the model's output.

    Note:
    The model expects input data in the form of a dictionary with the key 'Conclusion',
    containing sub-dictionaries with keys 'input_ids', 'token_type_ids', and 'attention_mask'.
    """
    def __init__(self,
                 dropout_prob: float = 0.3,
                 hidden_size: int = 768,
                 pretrained_model_name_or_path: Union[str,os.path] = 'bert-base-uncased',
                 bert_constructor: bert_model = transformers.BertModel):
        """
        Initializes the BertOne model with specified parameters.

        :param dropout_prob: The dropout probability for regularization.
        :param hidden_size: The size of the hidden layers in the BERT model.
        :param pretrained_model_name_or_path: The name or path of the pretrained BERT model.
        :param bert_constructor: The constructor for the BERT model.
        """
        super(BertOne, self).__init__()

        self.bert = bert_constructor.from_pretrained(pretrained_model_name_or_path)
        self.drop_out = torch.nn.Dropout(dropout_prob)
        self.classification_head = torch.nn.Linear(hidden_size, 4)

    def forward(self, kwards: dict) -> torch.Tensor:
        """
        Performs a forward pass through the BertOne model.

        :param kwards: A dictionary containing 'Conclusion' data for the model.
        :return: A torch.Tensor representing the model's output.
        """
        output_loss, output_logits = self.bert(**kwards['Conclusion'], return_dict=False)
        output = self.drop_out(output_logits)
        output = self.classification_head(output)
        output = torch.nn.functional.sigmoid(output)

        return output
