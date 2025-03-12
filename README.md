# Transformer implementation from scratch 🚀 
This repository contains my PyTorch from-scratch implementation of the transformer model, built step-by-step following the architecture introduced in the "Attention Is All You Need" paper.

original paper link: https://arxiv.org/html/1706.03762v7

## Project Overview
This project aims to provide a clear and educational implementation of the transformer architecture by breaking down its components into separate modules. Each component is first implemented using NumPy for better understanding of the underlying mathematics, and then integrated using PyTorch for the complete model.

## Repository Structure

**positional_encoding.ipynb**: Implementation of tokenization and positional encoding to incorporate sequence order information

**normalisation.ipynb**: Implementation of layer normalization 

**self_attention.ipynb**: Implementation of the Self-Attention mechanism, the core of transformer architecture

**encoder.py**: Complete encoder implementation that combines normalization, positional encoding, and self-attention

**toy-transformer-experiment.ipynb**: Demonstration of a simple English to French translation task

