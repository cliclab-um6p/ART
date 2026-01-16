# 🧠 ART: Attention-Regularized Transformers for Multi-Modal Robustness

This project provides an Implementation of ART framework.

---

## 📁 Project Structure

```
.
├── data/
│   ├── attack/                   # Precomputed attack CSVs for IMDB & QNLI
│   ├── imdb/                     # Preprocessed IMDB dataset (tokenized/saved)
│   └── qnli/                     # Preprocessed QNLI dataset (tokenized/saved)
├── model/                        # Directory for saving trained models
│   ├── imdb/
│   └── QNLI/
├── logs/                         # Logs and experiment outputs
├── TextAttack/                   # TextAttack modules
├── attack.py                     # Launch and analyze adversarial attacks
├── attention.py                  # Attention training for IMDB
├── attention-QNLI.py             # Attention training for QNLI
├── data_generator.py             # Preprocessing and DataLoader creation
├── requirements.txt              # Required Python dependencies
└── README.md                     # You're reading it!
```

---

## 🚀 Quickstart

### 1. Install Dependencies

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Preprocessed Datasets

```bash
python data_generator.py --task imdb
python data_generator.py --task qnli
```

### 3. Run BERT Training

```bash
python BERT-ART-IMDB.py --k 0.24          # For IMDB
python BERT-ART-QNLI.py --k 0.8         # For QNLI
```

### 4. Evaluate Adversarial Robustness

```bash
python attack.py --task imdb --k 0.24 --attack bertattack
python attack.py --task qnli --k 0.8 --attack bugger
```
