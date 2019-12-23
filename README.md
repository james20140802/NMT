# NMT
Neural Machine Translation by SangHyup Kim

### Table of Contents
1. [Introduction](#Introduction)
2. [Model](#Model)

### Introduction
This repository contains the neural machine translation by SangHyup Kim.

### Model
This model is based on the transformer.
#### 1. Embedding
Embedding layer shares its weights with pre-softmax linear transformation. Unlike the usual embedding 
which is used like a look-up table, this embedding layer is like a encoder in auto-encoder. It means 
that the pre-softmax linear transformation function as a decoder.

#### 2. Attention
Attention layer is multi-headed attention. In this layer I use convolution layer instead of splited linear.

#### 3. Feed forward network
Feed forward network consists of two convolution layers.
