# Human Value Detection

### Assignment 2 - Natural Language Processing

This repository contains the code and documentation for the second assignment of the Natural Language Processing (NLP) course at the University of Bologna, conducted by Professor [Paolo Torroni](https://www.unibo.it/sitoweb/p.torroni) at the University of Bologna.

## Overview
The assignment focuses on the [Human Value Detection challenge](https://aclanthology.org/2022.acl-long.306/). The primary objective was to address a multi-label classification task by identifying human values within arguments and linking them to their respective labels.


## More Information
For detailed project specifications, refer to the Assignment2.ipynb document.

## Requirements
Ensure you have the necessary dependencies by checking the `requirements.txt` file.

```bash
pip install -r requirements.txt
```
## Main Notebook
The main notebook (`main_notebook.ipynb`) serves as the central hub for this project. By executing this notebook, you can perform the following tasks:

- **Data Preparation:** Preprocess the Data, splitting it into training, validation, and test sets.
- **Model Creation:** Implement, train, and evaluate models.
- **Error Analysis:** Conduct error analysis on the best-performing model, comparing errors between validation and test sets.

Feel free to explore and customize the main notebook to experiment with different configurations and settings.

## Data Preparation

Before running the main notebook (`main.ipynb`), you'll need to download and prepare the Penn TreeBank corpus. Follow these steps:

1. Download the [Penn TreeBank corpus](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/dependency_treebank.zip).
2. Extract the contents of the downloaded zip file.
3. Create a folder named `dependency_treebank` in the root directory of this project.
4. Place the extracted files inside the `dependency_treebank` folder.


## drTorch Framework

The drTorch folder contains a framework developed  for creating neural network models using PyTorch.

## Models Implementation

The `models` folder contains the implementation of neural models, including:
* **Random Uniform Classifier:** An individual classifier per category.
* **Majority Classifier:** An individual classifier per category.

**BERT-based Models:**
1. **BERT w/ C:** A BERT-based classifier that takes an argument conclusion as input (using bert-base-uncased or roberta-base).
2. **BERT w/ CP:** BERT-based classifier with an additional input - argument premise (using bert-base-uncased or roberta-base).
3. **BERT w/ CPS:** BERT-based classifier with inputs of both argument premise and conclusion stance (using bert-base-uncased or roberta-base).

**NOTE:**
Although the files in this repository allow for a complete execution of the code, there are important considerations in such regard:
- We found the best hyperparameters for the models by applying a very time consuming grid search, which we do not recommend to execute (the flags for its execution are already set to False in the main notebook).
- The training of our models during the grid-search was conducted using a version of Torch that leverages NVIDIA's CUDA processing. This means that the weights that we store in the folder "results" are collected and retrieved in a CUDA-compatible environment and a version of Torch that supports this configuration. Without these, the execution will not yield the intended results.

## Authors:
For any questions or assistance, feel free to contact:
- [Mauro Dore](mauro.dore@studio.unibo.it)
- [Giacomo Gaiani](giacomo.gaiani@studio.unibo.it)
- [Gian Mario Marongiu](gianmario.marongiu@studio.unibo.it)
- [Riccardo Murgia ](riccardo.murgia2@studio.unibo.it)


