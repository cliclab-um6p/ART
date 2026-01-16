# 🧠 ART: Attention-Regularized Transformers for Multi-Modal Robustness

Official implementation of **ART: Attention-Regularized Transformers for Multi-Modal Robustness**, accepted at **Findings of EACL**.

This repository provides a unified **multi-modal framework** for improving the **adversarial robustness of Transformer-based models** via **attention regularization**, covering both **text** and **image** modalities.



## ✨ Overview

ART improves robustness by regularizing Transformer attention maps to be more stable under adversarial perturbations.

**Key highlights**

    ✅ Unified framework for NLP and Vision

    ✅ Plug-and-play attention regularization

    ✅ Compatible with standard Transformer architectures

    ✅ Improves adversarial robustness with minimal clean accuracy degradation

## 📁 Repository Structure
```
ART/
├── text/                # Text-based ART (NLP tasks)
│   ├── data/
│   ├── logs/
│   ├── model/
│   ├── TextAttack/
│   ├── attack.py
│   ├── BERT-ART-IMDB.py
│   ├── BERT-ART-QNLI.py
│   ├── data_generator.py
│   ├── README.md
│   └── requirements.txt
│
├── image/               # Image-based ART (Vision tasks)
│   ├── auto_LiRPA/
│   ├── data/
│   ├── log/
│   ├── model_for_cifar_ART/
│   ├── parser/
│   ├── robust_evaluate/
│   ├── train/
│   ├── README.md
│   └── requirements.txt
│
├── requirements.txt     # (Optional) global requirements
└── README.md            # This file
```
## 📝 Text Modality (NLP)

**Supported Tasks**

- IMDB (Sentiment Classification)
- QNLI (Natural Language Inference)

**Models**

- BERT with ART regularization

**Adversarial Attacks**

- BERT-Attack
- TextBugger
- Other attacks via TextAttack

## 🖼️ Image Modality (Vision)

**Datasets**

- CIFAR-10

- CIFAR-100

- Imagenette

**Models**

- Vision Transformer (ViT)
- DeiT
- ConViT (ART variants)

**Adversarial Attacks**

- FGSM
- PGD

## 📌 Notes

- Each modality has separate dependencies
- Install requirements inside the corresponding folder
- GPU is recommended for training and evaluation

## 📖 Citation

If you use this code or find our work helpful in your research, please cite our paper:

```bibtex
Soon
```

## 🤝 Contributions

We welcome contributions! If you'd like to help improve this project, feel free to open issues for bugs, questions, or enhancement ideas.

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## 📬 Contact

If you have any questions, suggestions, or feedback, feel free to contact us at: [mohammed.bouri@um6p.ma](mailto:mohammed.bouri@um6p.ma)

## 🤝 Acknowledgements

- [Specformer](https://github.com/microsoft/robustlearn/tree/main/specformer)
- [TextAttack](https://github.com/QData/TextAttack)
