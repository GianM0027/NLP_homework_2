import os

import numpy as np
import pandas as pd
import torch
import transformers

from drTorch.metrics import F1_Score


def create_dfs(directory):
    """
    Given a directory, takes the files .tsv in that path and convert them to a pd.DataFrame

    :param directory: directory with the three files already split and represented in .tsv format
    :return: three dataframes: training, validation and test set (in this order)
    """
    dfs = {}
    for file_path in os.listdir(directory):
        file_path = os.path.join(directory, file_path)

        with open(file_path, 'r') as file:
            file_name = file_path.split('-')[1].split('.tsv')[0]
            dfs[f'{file_name}'] = pd.read_csv(file_path, sep='\t')
            dfs[f'{file_name}'].index = dfs[f'{file_name}'].loc[:, "Argument ID"]
            dfs[f'{file_name}'] = dfs[f'{file_name}'].drop("Argument ID", axis=1)

    return dfs["training"], dfs["validation"], dfs["test"]


def define_mapping():
    """
    Definition of mapping from class 2 categories to class 3 categories

    :return: A python dictionary in the format:
                map = {"Openess_to_change": [category_2nd_level_1, category_2nd_level_2, category_2nd_level_3, ...],
                        "Self_enhancement": [category_2nd_level_1, category_2nd_level_2, category_2nd_level_3, ...],
                        "Conservation": [category_2nd_level_1, category_2nd_level_2, category_2nd_level_3, ...],
                        "Self_transcendence": [category_2nd_level_1, category_2nd_level_2, category_2nd_level_3, ...]}
    """

    openess_to_change = ["Self-direction: thought", "Self-direction: action", "Stimulation", "Hedonism"]
    self_enhancement = ["Hedonism", "Achievement", "Power: dominance", "Power: resources", "Face"]
    conservation = ["Face", "Security: personal", "Security: societal", "Tradition", "Conformity: rules",
                    "Conformity: interpersonal", "Humility"]
    self_transcendence = ["Humility", "Benevolence: caring", "Benevolence: dependability", "Concern",
                          "Universalism: nature", "Universalism: tolerance", "Objectivity"]

    map = {"Openess_to_change": openess_to_change,
           "Self_enhancement": self_enhancement,
           "Conservation": conservation,
           "Self_transcendence": self_transcendence}

    return map


def map_to_level_3(mapping, *label_sets):
    """
    Function which takes a mapping dictionary and a list of dataframes (of level-2 category labels). It returns the updated
    dataframes mapped to level 3

    :param mapping: the mapping dictionary in the format:
                map = {"Openess_to_change": [category_2nd_level_1, category_2nd_level_2, category_2nd_level_3, ...],
                        "Self_enhancement": [category_2nd_level_1, category_2nd_level_2, category_2nd_level_3, ...],
                        "Conservation": [category_2nd_level_1, category_2nd_level_2, category_2nd_level_3, ...],
                        "Self_transcendence": [category_2nd_level_1, category_2nd_level_2, category_2nd_level_3, ...]}
    :param label_sets: dataframes of labels to which apply the mapping.
    :return: the updated dataframes mapped to level 3 categories (from 20 classes to 4 classes)
    """
    dfs = [pd.DataFrame(0, index=labels.index, columns=mapping.keys()) for labels in label_sets]

    for df, labels in zip(dfs, label_sets):
        for index, row in labels.iterrows():
            indexes = [column for column, value in row.items() if value == 1]
            new_columns = {key: 0 for key in mapping.keys()}

            for i in range(len(indexes)):
                for key in mapping.keys():
                    if indexes[i] in mapping[key] and new_columns[key] == 0:
                        new_columns[key] += 1

            df.loc[index] = new_columns

    return tuple(dfs)


def encode(df: pd.DataFrame,
           tokenizer: transformers.models.bert.tokenization_bert.BertTokenizer,
           max_length: int,
           columns: list[str],
           add_special_tokens: bool = True,
           return_token_type_ids: bool = False,
           return_attention_mask: bool = True):
    """
    Tokenizes the columns of a dataframe df specified in the parameter column. The dataframe must contain the columns
    [Conclusion, Stance, Premise]. It encodes the Conclusion and Premise column according to the tokenizer policy. While the
    Stance is encoded in a numerical format (either 0 or 1).

    :param df: dataframe to tokenize
    :param tokenizer: tokenizer object
    :param max_length: length (in words) of the longest sentence in the columns to tokenize (used to determine the padding)
    :param columns: columns to take into account for the tokenization
    :param add_special_tokens: parameter for the tokenizer, decides whether to add or not special tokens like <star_of_-sentence> or <end_of_-sentence>

    :return: the dataframe tokenized
    """

    for column in columns:
        for index, row in df.iterrows():
            input = tokenizer.encode_plus(row[column],
                                          add_special_tokens=add_special_tokens,
                                          max_length=max_length,
                                          return_token_type_ids=return_token_type_ids,
                                          padding="max_length",
                                          return_attention_mask=return_attention_mask,
                                          return_tensors='pt')

            df.at[index, column] = {key: input[key][0] for key in input.keys()}

    df["Stance"] = np.where(df["Stance"] == "against", 0, 1)

    return df


def calculate_max_length(df, columns, tokenizer):
    """
    Calculates maximum length of tokens in a dataframes of non-tokenized sentences, in the columns specified by the parameter "columns".
    This function is used to have a parameter which helps to set a padding during the actual tokenization.

    :param df: the dataframes of sentences
    :param columns: the columns to take into account
    :param tokenizer: the tokenizer that will be used for the following tokenization

    :return: the number of tokens in the longest sentence of the dataframe
    """

    max_token_count = 0
    for column in columns:
        # Calculate the maximum token count in the specified columns
        token_count = df[column].apply(lambda x: len(tokenizer.tokenize(x))).max()
        if token_count > max_token_count:
            max_token_count = token_count

    return max_token_count


def f1_labels_score(real_y, pred_y, n_labels):
    """
    Uses the F1 score (drTorch version) to compute the f1 on the single columns of a dataframe separately

    :param real_y: the real labels
    :param pred_y: the predicted labels
    :param n_labels: the number of classes (number of columns of the real_y dataframe)

    :return: the f1 scores for each label
    """
    scores = []

    for i in range(n_labels):
        test_labels_flat = torch.tensor(real_y.iloc[:, i].to_list())
        y_pred_flat = torch.tensor(pred_y[:, i])
        f1_metric = F1_Score(name="F1_Score", num_classes=2, mode="macro", classes_to_exclude=None)
        scores.append(f1_metric(y_pred_flat, test_labels_flat))

    return scores
