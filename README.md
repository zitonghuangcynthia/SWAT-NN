# SWAT-NN: Simultaneous Weights and Architecture Training for Neural Networks in a Latent Space

This is the official repository for our paper:
**"SWAT-NN: Simultaneous Weights and Architecture Training for Neural Networks in a Latent Space"**  
[View on arXiv](https://arxiv.org/abs/2506.08270)

## Overview

This repository contains all code and scripts used in our experiments on simultaneous weight and architecture optimization of neural networks using a GPT2-based autoencoder framework. The method encodes MLPs into a universal latent space and enables fine-grained architecture discovery (e.g., neuron-level activation, layer width) and finds sparse, compact models with strong performance.

The workflow is divided into several stages:

1. **Train the AutoEncoder** on synthetic datasets of MLPs.
2. **Train sparse MLPs** on real-world or benchmark datasets (e.g., CORNN), using trained embedding space to optimize for performant and compact MLPs.
3. (Optional) **Compression of networks and variable input/output size** using the latent-space embedding to compress large networks.


## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```


## Train autoencoder

Navigate to the autoencoder directory and train the model:
```
cd train_autoencoder
python main.py --cuda 0
```
We provide a pretrained AutoEncoder checkpoint to help reproduce our results.
The checkpoint file is not hosted in this repository, but can be downloaded from the following Google Drive link:
[Download pretrain_ae.pth](https://tinyurl.com/SWAT-NN)

After downloading, place the file under the checkpoints/ directory:
```
checkpoints/pretrain_ae.pth
```

You do not need to modify any code — all scripts (main.py, train_mlp.py, etc.) automatically load the checkpoint if it exists.
If you’d prefer to retrain from scratch, simply remove the checkpoint or comment out the loading lines.

## Train MLPs on Benchmark Dataset

Navigate to the autoencoder directory and train the model:
```
cd train_mlp
python train_mlp.py --cuda 0
```

### Dataset

This training uses the [CORNN dataset](https://github.com/CWCleghornAI/CORNN.git) as the benchmark for functional approximation tasks.

We **do not host the dataset in this repository**. Please follow the instructions in the official [CORNN GitHub repository](https://github.com/CWCleghornAI/CORNN.git) to download and prepare the dataset.

In our code, we assume that the CORNN dataset is available as a Python module (e.g., `lib/CORNN.py`) in the codebase, or properly installed and imported.


## (Discussion) Compression of networks and variable input/output size
This section explores how SWAT-NN can compress large MLPs into smaller subnetworks using the embedding-based optimization framework. We demonstrate this by decomposing a large pretrained MLP into two smaller subnetworks and then compressing them individually using a 4 hidden layer to 2 hidden layer AutoEncoder variant.

Navigate into compress_large_NN folder:

1. Generate a Large MLP
```
python generate_dataset.py
```
Generates a large MLP model and dataset.

2. Split into Two Subnetworks
```
python split_small_MLP.py
```
Splits the large model into two smaller subnetworks.
**Optional:** These two steps can be skipped — we provide their checkpoints in compress_large_NN/checkpoints/.

3. Compress the Subnetworks
```
python main_compress.py --cuda 0
```
Uses SWAT-NN to compress the two subnetworks individually.
Note: This compression uses a different AutoEncoder from the one used in earlier sections.

AutoEncoder Checkpoint (4-to-2 variant):
Please download the pretrained 4-to-2 AutoEncoder checkpoint from the following [pretrained_4_to_2_ae.pth](https://drive.google.com/file/d/1dZ9Cv47JgaFYkxJVd2aZLkqFDaYPXPmT/view?usp=drive_link) and place it in the compress_large_NN/checkpoints/ folder.

4. Visualize Output Consistency
```
python scatter_plot.py
```
Generates scatter plots comparing predicted vs. ground truth outputs.
