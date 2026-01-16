import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
import numpy as np
import random
import argparse
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Training Attack script")
    parser.add_argument('--task', type=str, choices=['imdb', 'qnli'], required=True,
                        help="Choose between 'imdb' and 'qnli' datasets.")
    args = parser.parse_args()
    return args


args = parse_args()
task = args.task

# Set seed and device for reproducibility.
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the IMDB dataset (using train and test splits only).
if task == 'imdb':
    dataset = load_dataset('imdb')
    train_data = dataset['train']  # training data
    test_data = dataset['test']    # test data will serve as validation
elif task == 'qnli':
    dataset = load_dataset('glue', 'qnli')
    train_data = dataset['train']  # training data
    test_data = dataset['validation']

# 2. Tokenize the texts using BertTokenizer.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 256


def tokenize_function(examples):
    if task == 'imdb':
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)
    elif task == 'qnli':
        return tokenizer(examples['question'], examples['sentence'], padding='max_length', truncation=True, max_length=max_length)


train_data = train_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)

# Set format for PyTorch.
columns = ['input_ids', 'attention_mask', 'label']
train_data.set_format(type='torch', columns=columns)
test_data.set_format(type='torch', columns=columns)

# Create DataLoaders.
if task == 'imdb':
    batch_size = 64  # adjust based on your hardware
else:
    batch_size = 16  # adjust based on your hardware

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

train_data.save_to_disk(f"./data/{task}/train_data")
test_data.save_to_disk(f"./data/{task}/test_data")
