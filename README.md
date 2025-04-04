# Transformer Implementation from Scratch 🚀

This repository contains my PyTorch from-scratch implementation of the Transformer model and a Transformer-based neural machine translation model for English to French translation. The model is based on the architecture described in the paper "Attention Is All You Need" (Vaswani et al., 2017).

Original paper link: [Attention is all you need](https://arxiv.org/html/1706.03762v7)

## Project Overview

This project provides a clear and educational implementation of the Transformer architecture by breaking down its components into separate modules. Each component is initially implemented using NumPy to enhance understanding of the underlying mathematics, and then integrated using PyTorch for the complete model.

## Repository Structure

```
│transformer-from-scratch/
│
├── data/
│   ├── English.txt
│   ├── French.txt
│   ├── shakespeare.txt
│
├── en-fr-transformer/
│   ├── en-fr-train.ipynb                  # Training notebook for EN-FR transformer
│   ├── model.py                           # Complete Transformer model
│   ├── en-fr-toy-transformer-experiment.ipynb
│
├── simple-transformer/                    # Building blocks
│   ├── decoder.py
│   ├── encoder.py
│   ├── normalisation.ipynb
│   ├── positional_encoding.ipynb
│   ├── self_attention.ipynb
│
└── README.md                              # Documentation
``` 


## Model Architecture

The `model.py` file contains the main Transformer model with:

* 3 encoder layers
* 3 decoder layers
* 8 attention heads
* Embedding dimension of 512
* Feed-forward dimension of 2048
* Dropout rate of 0.1
* Maximum sequence length of 100 characters

### Key Components

* **MultiHeadAttention:** Implements the multi-head self-attention mechanism.
* **PositionWiseFeedForward:** The feed-forward network applied to each position.
* **PositionalEncoding:** Adds positional information to the embeddings.
* **EncoderLayer:** Full encoder layer with self-attention and feed-forward networks.
* **DecoderLayer:** Full decoder layer with self-attention, cross-attention, and feed-forward networks.
* **Transformer:** The complete model integrating all components.

### Model Parameters

* Source vocabulary size: 108 (English characters + special tokens)
* Target vocabulary size: 143 (French characters including accented characters + special tokens)
* Total trainable parameters: 22,254,724 (22M)

## Dataset

The model is trained on a dataset of English-French sentence pairs. The dataset is preprocessed to:

* Filter sentences that exceed the maximum sequence length.
* Ensure all characters in the sentences are within the defined vocabularies.
* Add special tokens (`START`, `END`, `PADDING`, `UNKNOWN`).

## Training Details

* Hardware: Trained on MPS (Metal Performance Shaders) for Apple Silicon
* Training time: 100 minutes for 2 epochs
* Batch size: 32
* Optimizer: AdamW with learning rate 1e-4
* Loss function: Cross-entropy loss (ignoring padding tokens)
* Gradient clipping: Applied at 1.0

## Training Progress

**Epoch 1**

* Initial loss: \~1.8342
* Final train loss: 1.3595
* Validation loss: 0.9313

**Epoch 2**

* Initial loss: \~0.8570
* Final train loss: 0.8604
* Validation loss: 0.6674

## Sample Translations

| English Input                        | French Translation                      |
| :----------------------------------- | :-------------------------------------- |
| what should we do when the day starts? | `<SOS>`Ce que devrais-nous faire le dire ? |
| what do you think about the book?      | `<SOS>`Ce que vous pensez-vous ?         |
| where are you going?                 | `<SOS>`Où vous avez en train de matin ?  |
| I want a new book.                   | `<SOS>`Je veux un livre un livre.       |

## Notes

* The current model is trained at the character level rather than using subword tokens (like BPE).
* While the model shows promising results for simple sentences, it still has limitations for more complex translations.
* The sample translations indicate that the model has begun to learn patterns but still produces some inaccuracies.


Thank you for exploring this Transformer implementation! I hope it is helpful for understanding this powerful architecture. Contributions and feedback are welcome!
