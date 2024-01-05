import os
from typing import Optional

import torch
import transformers

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

bert_model = transformers.BertModel | transformers.RobertaModel
bert_tokenizer = transformers.BertTokenizer | transformers.RobertaTokenizer


class CustomDataset(torch.utils.data.Dataset):

    """
    A custom PyTorch Dataset class for handling generic data with corresponding labels.

    Parameters:
    - data_dict (dict): A dictionary containing different data components.
    - labels (torch.Tensor): Tensor containing the labels for each data sample.

    Methods:
    - __len__: Returns the number of samples in the dataset.
    - __getitem__: Returns a specific sample and its corresponding label given an index.
    """

    def __init__(self, data_dict, labels):
        self.data_dict = data_dict
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        """
        Placeholder method. Must be implemented in subclasses to define how data is retrieved for a given index.
        """
        pass


class CustomDataset_C(CustomDataset):

    """
    A subclass of CustomDataset for handling data with a 'Conclusion' component.

    Methods:
    - __getitem__: Returns a sample containing the 'Conclusion' data and its corresponding label given an index.
    """

    def __init__(self, data_dict, labels):
        super().__init__(data_dict, labels)

    def __getitem__(self, index):
        """
        Retrieves a sample with 'Conclusion' data and its corresponding label for a given index.
        """
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

    """
    A subclass of CustomDataset for handling data with 'Conclusion' and 'Premise' components.

    Methods:
    - __getitem__: Returns a sample containing 'Conclusion' and 'Premise' data and its corresponding label given an index.
    """

    def __init__(self, data_dict, labels):
        super().__init__(data_dict, labels)

    def __getitem__(self, index):
        """
        Retrieves a sample with 'Conclusion' and 'Premise' data and its corresponding label for a given index.
        """

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
    """
    A subclass of CustomDataset for handling data with 'Conclusion', 'Premise', and 'Stance' components.

    Methods:
    - __getitem__: Returns a sample containing 'Conclusion', 'Premise', 'Stance' data, and its corresponding label given an index.
    """
    def __init__(self, data_dict, labels):
        super().__init__(data_dict, labels)

    def __getitem__(self, index):
        """
        Retrieves a sample with 'Conclusion', 'Premise', 'Stance' data, and its corresponding label for a given index.
        """
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


def download_bert_models(directory: str,
                         versions: list[str],
                         bert_constructors: dict[str, bert_model],
                         bert_tokenizer_constructors: dict[str, bert_tokenizer]) -> list[str]:
    """
    Download and save BERT models and tokenizers to the specified directory for multiple versions.

    This function takes a directory path, a list of BERT model versions, dictionaries of BERT model constructors,
    and BERT tokenizer constructors. It then downloads the corresponding BERT models and tokenizers for each version,
    saving them to the specified directory.

    :param directory: The directory path where BERT models and tokenizers will be saved.
    :param versions: A list of BERT model versions to download.
    :param bert_constructors: A dictionary mapping BERT model versions to their corresponding model constructors.
    :param bert_tokenizer_constructors: A dictionary mapping BERT model versions to their corresponding tokenizer constructors.
    :return: A list of paths to the downloaded and saved BERT models and tokenizers for each version.
    """
    versions_paths = []

    for bert_v in versions:
        version_directory = os.path.join(directory, bert_v)

        # Instantiate current BERT model and tokenizer
        current_bert = bert_constructors[bert_v].from_pretrained(bert_v)
        current_tokenizers = bert_tokenizer_constructors[bert_v].from_pretrained(bert_v)

        # Save current BERT model and tokenizer to the specified directory
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

    my_map = {"Openess_to_change": openess_to_change,
              "Self_enhancement": self_enhancement,
              "Conservation": conservation,
              "Self_transcendence": self_transcendence}

    return my_map


def map_to_level_3(mapping: dict, *label_sets):
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
    dfs = [pd.DataFrame(0, index=labels.index, columns=list(mapping.keys())) for labels in label_sets]

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


def plot_comparison_across_sets(train_df: pd.DataFrame,
                                test_df: pd.DataFrame,
                                val_df: pd.DataFrame,
                                title: str = 'Comparison across Train, Test, and Validation sets') -> None:
    """
    Plot a comparison of class distributions across Train, Test, and Validation sets.

    :param train_df: Pandas DataFrame containing training set data.
    :param test_df: Pandas DataFrame containing test set data.
    :param val_df: Pandas DataFrame containing validation set data.
    :param title: Optional title for the plot.

    :return: None
    """

    classes = train_df.columns

    plt.figure(figsize=(5 * len(classes), 4))

    for i, class_name in enumerate(classes):
        plt.subplot(1, len(classes), i + 1)
        sns.countplot(x=class_name, hue='dataset', data=pd.concat([train_df.assign(dataset='train'),
                                                                   test_df.assign(dataset='test'),
                                                                   val_df.assign(dataset='val')]))
        plt.title(class_name)

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()


def calculate_class_weights(dataframe):
    """
    Calculate class weights based on the formula: n_samples / (n_classes * np.bincount(y))

    :param dataframe: A pandas DataFrame containing binary labels (0 or 1) in each column.
                     The function assumes that each column represents a different class.
    :return: A PyTorch tensor containing class weights for each class in each column of the DataFrame.
             The shape of the tensor is (num_columns, num_classes), where num_columns is the number of columns
             in the DataFrame and num_classes is the number of unique classes.
    :raises ValueError: If the DataFrame has no columns.
    """
    if len(dataframe.columns) == 0:
        raise ValueError("Invalid input: The Dataframe must contain at last one column")

    class_weights_list = []

    n_samples = len(dataframe)

    for column in dataframe.columns:
        column_values = torch.tensor(dataframe[column].values, dtype=torch.float)
        n_classes = dataframe[column].nunique()
        class_counts = torch.bincount(column_values.long(), minlength=2)
        class_weights = n_samples / (n_classes * class_counts.float())
        class_weights_list.append(class_weights)

    class_label_weights = torch.stack(class_weights_list, dim=0)

    return class_label_weights


def build_dataloader(data: pd.DataFrame,
                     labels: pd.DataFrame,
                     one_hot_mapping: dict[str, int],
                     pretrained_model_name_or_path: os.path,
                     tokenizer_constructor: bert_tokenizer,
                     model_input: list[str],
                     custom_dataset_builder: callable,
                     batch_size: int,
                     shuffle: bool) -> torch.utils.data.DataLoader:
    """
    Build a PyTorch DataLoader for a custom dataset, given input data and labels.

    :param data: Input data stored in a pandas DataFrame.
    :param labels: Labels associated with the input data in a pandas DataFrame.
    :param one_hot_mapping: A dictionary mapping label names to corresponding one-hot encoded integer values.
    :param pretrained_model_name_or_path: The name or path of a pretrained BERT model.
    :param tokenizer_constructor: A constructor for the BERT tokenizer.
    :param model_input: A list of input features to be tokenized by the BERT tokenizer.
    :param custom_dataset_builder: A callable that constructs a custom PyTorch Dataset from tokenized inputs and labels.
    :param batch_size: The size of each batch in the DataLoader.
    :param shuffle: Whether to shuffle the data in each epoch.

    :return: A PyTorch DataLoader for the custom dataset.
    """

    my_tokenizer = tokenizer_constructor.from_pretrained(pretrained_model_name_or_path)

    tmp_dict = {}

    for input_feature in model_input:
        if input_feature != "Stance":
            tmp_dict[input_feature] = my_tokenizer.batch_encode_plus(data[input_feature],
                                                                     return_tensors="pt",
                                                                     padding=True,
                                                                     return_token_type_ids=True)
        else:
            tmp_dict[input_feature] = torch.tensor((data[input_feature].to_numpy() == 'in favor of').astype(int))

    labels_tensor = torch.tensor([[one_hot_mapping[element] for element in row] for row in labels.values])

    custom_dataset = custom_dataset_builder(tmp_dict, labels_tensor)

    dataloader = torch.utils.data.DataLoader(custom_dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle)

    return dataloader


def unpacking_dataloader_builder_parameters_strategy(kwargs: Optional[dict]):
    """
     Strategy for unpacking and configuring DataLoader builder parameters.

     This function takes a dictionary of keyword arguments (`kwargs`) and performs specific actions
     to configure parameters for a DataLoader builder. It is designed to handle variations in parameter
     configurations based on different model versions.

     :param kwargs: A dictionary of keyword arguments for configuring DataLoader builder parameters.
                    It typically includes parameters such as 'pretrained_model_name_or_path' that specify
                    the model version to be used.
     :return: The modified dictionary of keyword arguments with updated or added parameters based on the
              specified strategy.
     """

    variable_parameters = {'bert-base-uncased': transformers.BertTokenizer,
                           'roberta-base': transformers.RobertaTokenizer}

    model_version = os.path.basename(kwargs['pretrained_model_name_or_path'])
    kwargs['tokenizer_constructor'] = variable_parameters[model_version]
    return kwargs
