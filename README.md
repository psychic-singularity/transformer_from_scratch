# 🧠 Transformer from Scratch in PyTorch

A clean, modular, and educational implementation of the original Transformer architecture proposed in the seminal paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. (2017). 

This project trains an encoder-decoder sequence-to-sequence model on the OPUS Books dataset for English-to-Italian machine translation.

---

## ✨ Key Features

- **Full Architecture from Scratch**:
  - Multi-Head Attention with scaled dot-product attention
  - Positional Encoding (sinusoidal)
  - Feed-Forward Networks with residual skip connections and Layer Normalization
  - Encoder & Decoder stacks with causal / look-ahead masking
- **Interactive Notebooks**:
  - : Beam search decoding implementation
  - : Attention weight maps and visualization
  -  & : Ready-to-run training workflows
- **Dataset & Tokenization**:
  - Automatic tokenization and dataset caching using HuggingFace `datasets` and `tokenizers`
  - Subword tokenization (WordLevel with standard special tokens: `[UNK]`, `[PAD]`, `[SOS]`, `[EOS]`)
- **Experiment Tracking**:
  - Integrated TensorBoard and Weights & Biases (WandB) support

---

## 📁 Repository Structure

```
├── config.py                # Hyperparameters, model configuration, and file paths
├── dataset.py               # BilingualDataset loader and mask generation
├── model.py                 # Core Transformer architecture (PyTorch)
├── train.py                 # Main training script with TensorBoard logging
├── train_wb.py              # Training script with Weights & Biases tracking
├── translate.py             # Inference / translation CLI script
├── Beam_Search.ipynb        # Beam search decoding exploration
├── attention_visual.ipynb   # Attention matrix heatmaps
├── Colab_Train.ipynb        # Google Colab training notebook
├── Local_Train.ipynb        # Local Jupyter training notebook
├── requirements.txt         # Python dependencies
└── conda.txt                # Conda environment specifications
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9 recommended
- PyTorch 2.0+ with CUDA (optional for GPU training)

### 2. Installation

Clone your repository and install dependencies:

```bash
git clone https://github.com/psychic-singularity/transformer_from_scratch.git
cd transformer_from_scratch
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

To train the Transformer on the English-to-Italian OPUS Books dataset:

```bash
python train.py
```

Or to train with Weights & Biases tracking:

```bash
python train_wb.py
```

Model weights and tokenizer definitions (`tokenizer_en.json`, `tokenizer_it.json`) will automatically be saved into the `opus_books_weights/` directory.

---

## 🔮 Inference & Translation

Once you have trained weights (or placed checkpoints in the weights directory), translate sentences from English to Italian:

```bash
python translate.py "Where are you going?"
```

You can also test translation by providing an index from the dataset:

```bash
python translate.py 42
```

---

## 📊 Hyperparameters (`config.py`)

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `batch_size` | `8` | Batch size per training step |
| `num_epochs` | `20` | Total training epochs |
| `lr` | `1e-4` | Learning rate |
| `seq_len` | `350` | Maximum sequence length |
| `d_model` | `512` | Embedding / latent dimensionality |
| `datasource` | `'opus_books'` | Dataset name from HuggingFace |
| `lang_src` / `lang_tgt` | `en` / `it` | Source and target languages |

---

## 📚 References

- Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (NeurIPS 2017)
- [OPUS Books Dataset](https://huggingface.co/datasets/opus_books) via HuggingFace Datasets
