import signal
import argparse
import warnings
import time
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import torch
import sys
sys.path.insert(0, './TextAttack')
from textattack import Attacker
from textattack.attack_recipes import TextFoolerJin2019, TextBuggerLi2018, BERTAttackLi2020
from textattack.models.wrappers import ModelWrapper
import textattack
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")


# -------------------- Argument Parsing --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Training Attack script")
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--portion', '-k', type=float, default=0, required=True)
    parser.add_argument('--attack', type=str, default='fooler', required=True,
                        help="Choose between ['bertattack', 'fooler', 'bugger'].")
    parser.add_argument('--task', type=str, default='imdb', required=True,
                        help="Choose between 'imdb' and 'qnli' datasets.")
    args = parser.parse_args()
    return args


args = parse_args()
sample = args.sample
att = args.attack
task = args.task
k = args.portion


attack_test_dataset_path = f"./data/attack/{task}_attack_1000.csv"


# -------------------- Timeout Handling --------------------
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException()


signal.signal(signal.SIGALRM, timeout_handler)


def safe_attack_example(attack_fn, text_input, label, timeout=10):
    try:
        signal.alarm(timeout)
        result = attack_fn.attack(text_input, label)
        signal.alarm(0)
        return result
    except TimeoutException:
        print("Timeout: Skipping example that took too long.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        signal.alarm(0)


# -------------------- Model Loading --------------------
def loading():
    loaded_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if task == 'imdb':
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=2)
        state_dict = torch.load(f"./model/imdb/{k}_best_model.pt", map_location="cpu")
        model.load_state_dict(state_dict, strict=False)  # <---- ignore extra keys

    elif task == 'qnli':
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=2)
        state_dict = torch.load(f"./model/QNLI/{k}_best_model.pt", map_location="cpu")
        model.load_state_dict(state_dict, strict=False)  # <---- ignore extra keys

    return model, loaded_tokenizer



# -------------------- Wrapper --------------------
class CustomTorchModelWrapper(ModelWrapper):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # Prefer MPS on Apple Silicon
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def _preprocess_text(self, text_input_list):
        x_input = self.tokenizer(
            text_input_list,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        x_input = {key: value.to(self.device) for key, value in x_input.items()}
        return x_input

    def _get_probabilities(self, x_input):
        model = self.model.to(self.device)
        model.eval()
        with torch.no_grad():
            outputs = model(**x_input)
            logits = outputs.logits
        result = torch.argmax(logits, dim=1).cpu().detach().numpy()
        return result

    def __call__(self, text_input_list):
        if isinstance(text_input_list, str):
            text_input_list = [text_input_list]
        x_input = self._preprocess_text(text_input_list)
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**x_input)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        return probabilities.cpu().numpy()


# -------------------- Dataset Loader --------------------
if task == 'imdb':
    def dataframe_to_list(df):
        return list(zip(df['review'], df['sentiment']))
elif task == 'qnli':
    def dataframe_to_list(df):
        examples = []
        for idx, row in df.iterrows():
            text = f"Question: {row['question']} Sentence: {row['sentence']}"
            examples.append(([text], row['label']))
        return examples


# -------------------- Main --------------------
model, loaded_tokenizer = loading()
model_wrapper = CustomTorchModelWrapper(model, loaded_tokenizer)

dataset = pd.read_csv(attack_test_dataset_path)
data = dataframe_to_list(dataset)
dataset = textattack.datasets.Dataset(data)

if sample != 0:
    print(f'Sample = {sample}')
    attack_args = textattack.AttackArgs(num_examples=sample)
else:
    print(f'Sample = ALL')
    attack_args = textattack.AttackArgs(num_examples=len(data))

if att == 'bertattack':
    print('bertattack')
    attack = BERTAttackLi2020.build(model_wrapper)
elif att == 'fooler':
    print('fooler')
    attack = TextFoolerJin2019.build(model_wrapper)
elif att == 'bugger':
    print('bugger')
    attack = TextBuggerLi2018.build(model_wrapper)
else:
    raise ValueError("Invalid attack type. Choose from ['bertattack', 'fooler', 'bugger'].")


start = time.time()
print(f'\n-------------------- attack_test_dataset ({sample} Random samples) --------------------\n\n')

attacker = Attacker(attack, dataset, attack_args)

if att == 'bertattack':
    for i, example in enumerate(dataset):
        print(f"Running example {i}...")
        text_input = example[0]["text"]
        label = example[1]
        result = safe_attack_example(attack, text_input, label, timeout=180)
        if result:
            print(result)
else:
    aa = attacker.attack_dataset(num_workers_per_device=1)
