# 🧠 ART: Attention-Regularized Transformers for Multi-Modal Robustness

This project provides an Implementation of ART framework.


## Dependencies

- Run `pip install -r requirement.txt` to install all requrements.


## Directories

- `auto_LiRPA`: Contains the logger and `MultiAverageMeter`.
- `model_for_cifar_ART`: ART models for CIFAR-10 and CIFAR-100 experiments.
- `model_for_imagenet_ART`: ART models for Imagenette experiments.
- `parser`: Python scripts for retrieving input parameters from the command line.
  - `parser_cifar.py`: Parser for CIFAR experiments.
  - `parser_imagenette.py`: Parser for Imagenette experiments.
- `robust_evaluate`: Python scripts for evaluating robustness.
  - `fgsm.py`: Evaluates FGSM Attack.
  - `pgd.py`: Evaluates PGD Attack.
- `train`: Python scripts for training models.
  - `train_cifar.py`: Training script for CIFAR experiments.
  - `train_imagenette.py`: Training script for Imagenette experiments.
  - `utils.py`: Contains the data loading code.

## Data

- **CIFAR-10 and CIFAR-100**: These datasets will be automatically downloaded when running `train_cifar` using `datasets.CIFAR10(args.data_dir, train=True, transform=train_transform, download=True)`.
- **Imagenette-v1**: The Imagenette-v1 dataset can be downloaded from [Imagenette-v1](https://s3.amazonaws.com/fast-ai-imageclas/imagenette.tgz).


## Running

### CIFAR-10/100
```python
CUDA_VISIBLE_DEVICES=0 python3 -m train.train_cifar --model "vit_small_patch16_224_ART" --dataset cifar10 --out-dir ./log/ --method 'CLEAN'  --seed 0 --epochs 40 --data-dir ./data/cifar --weight-decay 1e-05 --ART 6e-06

CUDA_VISIBLE_DEVICES=0 python3 -m train.train_cifar --model "deit_tiny_patch16_224_ART" --dataset cifar100 --out-dir ./log/ --method 'CLEAN'  --seed 0 --epochs 40 --data-dir ./data/cifar --weight-decay 1e-05 --ART 6e-05
```

You can switch to other ViT variants using the `--model` option, change the dataset with the `--dataset` option, and select a different method with the `--method` option.


### ImageNette
```python
CUDA_VISIBLE_DEVICES=0 python3 -m train.train_imagenette --model "convit_tiny_ART" --dataset imagenette --out-dir ./log/ --method 'CLEAN'  --seed 0 --epochs 40 --data-dir ./data/imagenette/ --weight-decay 1e-05 --resize 224 --crop 224 --patch 16 --ART 5e-06
```
