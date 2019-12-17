# NMT
Neural Machine Translation by SangHyup Kim

### Table of Contents
1. [Introduction](#Introduction)
2. [Model](#Model)

### Introduction
This repository contains the neural machine translation by SangHyup Kim.

### Model
####1. Embedding
Embedding layer shares its weights with pre-softmax linear transformation. Unlike the usual embedding 
which is used like a look-up table, this embedding layer is like a encoder in auto-encoder. It means 
that the pre-softmax linear transformation function as a decoder.
####2. Convolution Block
Convolution block is used to get the latent vector in both encoder and decoder. The block uses residual
connection for better optimization. 
