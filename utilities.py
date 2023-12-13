import os

import torch
import transformers

import pandas as pd


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, labels):
        self.data_dict = data_dict
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        pass


class CustomDataset_C(CustomDataset):
    def __init__(self, data_dict, labels):
        super().__init__(data_dict, labels)

    def __getitem__(self, index):
        sample = {
            'Conclusion': {
                'input_ids': self.data_dict['Conclusion']['input_ids'][index],
                'token_type_ids': self.data_dict['Conclusion']['token_type_ids'][index],
                'attention_mask': self.data_dict['Conclusion']['attention_mask'][index]
            }
        }
        label = self.labels[index]
        return sample, label


class CustomDataset_CP(CustomDataset):
    def __init__(self, data_dict, labels):
        super().__init__(data_dict, labels)

    def __getitem__(self, index):
        sample = {
            'Conclusion': {
                'input_ids': self.data_dict['Conclusion']['input_ids'][index],
                'token_type_ids': self.data_dict['Conclusion']['token_type_ids'][index],
                'attention_mask': self.data_dict['Conclusion']['attention_mask'][index]
            },
            'Premise': {
                'input_ids': self.data_dict['Premise']['input_ids'][index],
                'token_type_ids': self.data_dict['Premise']['token_type_ids'][index],
                'attention_mask': self.data_dict['Premise']['attention_mask'][index]
            }
        }
        label = self.labels[index]
        return sample, label


class CustomDataset_CPS(CustomDataset):
    def __init__(self, data_dict, labels):
        super().__init__(data_dict, labels)  # Modifica questa linea

    def __getitem__(self, index):
        sample = {
            'Conclusion': {
                'input_ids': self.data_dict['Conclusion']['input_ids'][index],
                'token_type_ids': self.data_dict['Conclusion']['token_type_ids'][index],
                'attention_mask': self.data_dict['Conclusion']['attention_mask'][index]
            },
            'Premise': {
                'input_ids': self.data_dict['Premise']['input_ids'][index],
                'token_type_ids': self.data_dict['Premise']['token_type_ids'][index],
                'attention_mask': self.data_dict['Premise']['attention_mask'][index]
            },
            'Stance': self.data_dict['Stance'][index]
        }
        label = self.labels[index]
        return sample, label


def download_bert_models(directory: str, versions: list[str]) -> list[str]:
    # todo documente
    """

    Args:
        directory:
        versions:

    Returns:

    """
    versions_paths = []

    for bert_v in versions:
        version_directory = os.path.join(directory, bert_v)

        current_bert = transformers.BertModel.from_pretrained(bert_v)
        current_tokenizers = transformers.BertTokenizer.from_pretrained(bert_v)

        current_bert.save_pretrained(version_directory)
        current_tokenizers.save_pretrained(version_directory)

        versions_paths.append(version_directory)

    return versions_paths


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


def build_dataloaders(train_df: pd.DataFrame,
                      val_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      train_labels_df: pd.DataFrame,
                      val_labels_df: pd.DataFrame,
                      test_labels_df: pd.DataFrame,
                      one_hot_mapping: dict[str, int],
                      bert_version: os.path,
                      model_input: list[str],
                      custom_dataset_builder: callable,
                      batch_size: int,
                      shuffle: bool):
    """

    Build DataLoader instances for training, validation, and testing datasets.


    Build DataLoader instances for training, validation, and testing datasets.


    :params train_df: DataFrame containing training data.
    :params val_df: DataFrame containing validation data.
    :params test_df: DataFrame containing testing data.
    :params train_labels_df: DataFrame containing labels for training data.
    :params val_labels_df: DataFrame containing labels for validation data.
    :params test_labels_df: DataFrame containing labels for testing data.
    :params one_hot_mapping: Mapping dictionary for one-hot encoding.
    :params bert_version: BERT model version for tokenization.
    :params model_input: List of model input features.
    :params batch_size: Batch size for DataLoader.
    :params shuffle: Whether to shuffle the data in DataLoader.

    :returns tuple: DataLoader instances for training, validation, and testing datasets.

    """

    my_tokenizer = transformers.BertTokenizer.from_pretrained(bert_version)

    tmp_train_dict = {}
    tmp_val_dict = {}
    tmp_test_dict = {}

    for input in model_input:
        if input != "Stance":
            tmp_train_dict[input] = my_tokenizer.batch_encode_plus(train_df[input],
                                                                   return_tensors="pt",
                                                                   padding=True)

            tmp_val_dict[input] = my_tokenizer.batch_encode_plus(val_df[input],
                                                                 return_tensors="pt",
                                                                 padding=True)

            tmp_test_dict[input] = my_tokenizer.batch_encode_plus(test_df[input],
                                                                  return_tensors="pt",
                                                                  padding=True)
        else:
            tmp_train_dict[input] = torch.tensor((train_df[input].to_numpy() == 'in favor of').astype(int))
            tmp_val_dict[input] = torch.tensor((val_df[input].to_numpy() == 'in favor of').astype(int))
            tmp_test_dict[input] = torch.tensor((test_df[input].to_numpy() == 'in favor of').astype(int))

    train_labels_tensor = torch.tensor(
        [[one_hot_mapping[element] for element in row] for row in train_labels_df.values])
    val_labels_tensor = torch.tensor([[one_hot_mapping[element] for element in row] for row in val_labels_df.values])
    test_labels_tensor = torch.tensor([[one_hot_mapping[element] for element in row] for row in test_labels_df.values])

    custom_train_dataset = custom_dataset_builder(tmp_train_dict, train_labels_tensor)
    custom_val_dataset = custom_dataset_builder(tmp_val_dict, val_labels_tensor)
    custom_test_dataset = custom_dataset_builder(tmp_test_dict, test_labels_tensor)

    dataloader_train = torch.utils.data.DataLoader(custom_train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle)

    dataloader_val = torch.utils.data.DataLoader(custom_val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False)

    dataloader_test = torch.utils.data.DataLoader(custom_test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)

    return dataloader_train, dataloader_val, dataloader_test
