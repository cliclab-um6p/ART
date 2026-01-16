import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from datasets import load_from_disk
from tqdm import tqdm
import numpy as np
import random
import argparse
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Training BERT-ART script")
    parser.add_argument('--k', type=float, required=True)
    args = parser.parse_args()
    return args


args = parse_args()
k = args.k

# Set seed and device for reproducibility.
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the IMDB dataset (using train and test splits only).

train_data = load_from_disk("./data/imdb/train_data")
test_data = load_from_disk("./data/imdb/test_data")

# Create DataLoaders.
batch_size = 64  # adjust based on your hardware
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 3. Build the model.
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2)
model = model.to(device)

# Global variables to store hook data (they get overwritten each forward pass).
saved_attn_input = None
saved_attn_output = None


def attn_hook(module, input, output):
    global saved_attn_input, saved_attn_output
    saved_attn_input = input[0]
    saved_attn_output = output


# Register the hook on the self-attention module of the first encoder layer.
hook_handle = model.bert.encoder.layer[0].attention.self.register_forward_hook(
    attn_hook)

# 4. Set up the optimizer and TensorBoard writer.
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# lambda_reg = 0  # regularization coefficient
lambda_reg = k  # regularization coefficient

num_epochs = 7
global_step = 0
best_eval_loss = float("inf")  # To track the best evaluation loss
use_reg = True

for epoch in range(num_epochs):
    # --- Training Phase ---
    model.train()
    train_loss_total = 0.0
    train_loss_reg = 0.0
    train_batches = 0
    train_correct = 0
    train_total = 0
    train_progress = tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)")

    for batch in train_progress:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Forward pass (the hook saves the attention inputs/outputs)
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask, labels=labels)
        loss = outputs.loss  # standard classification loss

        # Compute the gradient penalty regularization term.
        if saved_attn_input is not None and saved_attn_output is not None and use_reg:
            attn_out = saved_attn_output[0] if isinstance(
                saved_attn_output, tuple) else saved_attn_output
            grad = torch.autograd.grad(
                attn_out.sum(), saved_attn_input, create_graph=True)[0]
            reg_loss = lambda_reg * grad.abs().sum()
            loss = (1 - lambda_reg) * loss + reg_loss
        else:
            reg_loss = torch.tensor(0.0).to(device)

        loss.backward()
        optimizer.step()

        train_loss_total += loss.item()
        train_loss_reg += reg_loss.item()
        train_batches += 1

        # Accuracy calculation:
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

        global_step += 1
        train_progress.set_postfix(loss=loss.item())

        # Reset hook variables after each batch.
        saved_attn_input = None
        saved_attn_output = None

    avg_train_loss = train_loss_total / train_batches
    avg_train_reg = train_loss_reg / train_batches
    train_accuracy = train_correct / train_total

    # --- Evaluation Phase ---
    # We compute the gradient penalty in evaluation as well to have consistent loss.
    model.eval()
    eval_loss_total = 0.0
    eval_loss_reg = 0.0
    eval_batches = 0
    eval_correct = 0
    eval_total = 0
    eval_progress = tqdm(
        test_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Evaluation)")

    for batch in eval_progress:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        if saved_attn_input is not None and saved_attn_output is not None and use_reg:
            attn_out = saved_attn_output[0] if isinstance(
                saved_attn_output, tuple) else saved_attn_output
            grad = torch.autograd.grad(
                attn_out.sum(), saved_attn_input, create_graph=False)[0]
            reg_loss = lambda_reg * grad.abs().sum()
            loss = (1 - lambda_reg) * loss + reg_loss
        else:
            reg_loss = torch.tensor(0.0).to(device)

        eval_loss_total += loss.item()
        eval_loss_reg += reg_loss.item()
        eval_batches += 1

        # Accuracy calculation for evaluation:
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        eval_correct += (predictions == labels).sum().item()
        eval_total += labels.size(0)

        eval_progress.set_postfix(loss=loss.item())

        # Reset hook variables after each batch.
        saved_attn_input = None
        saved_attn_output = None

    avg_eval_loss = eval_loss_total / eval_batches
    avg_eval_reg = eval_loss_reg / eval_batches
    eval_accuracy = eval_correct / eval_total

    print(f"Epoch {epoch+1}:")
    print(
        f"    Train Loss = {avg_train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}")
    print(
        f"    Eval Loss  = {avg_eval_loss:.4f}, Eval Accuracy  = {eval_accuracy:.4f}, Eval Reg Loss = {avg_eval_reg:.4f}")

    # --- Checkpoint the Best Model ---
    if avg_eval_loss < best_eval_loss:
        best_eval_loss = avg_eval_loss
        torch.save(model.state_dict(),
                   f"./model/imdb/{lambda_reg}_best_model.pt")
        print("New best model saved based on evaluation loss!")

# Remove the hook when done.
hook_handle.remove()
