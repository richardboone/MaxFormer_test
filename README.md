# C-Grad: Conservative Gradients for Spiking Neural Networks

This repository contains the official implementation of **C-Grad** (Conservative Custom Gradient), a method for stabilizing backward propagation in Spiking Neural Networks.

## Overview

We introduce a "Conservative" gradient estimation strategy that interpolates between the standard surrogate gradient and a corrected gradient direction only when high-confidence misalignment is detected. This method effectively mitigates inconsistent gradient issues in deep SNNs while maintaining training stability.

## Requirements

* python >= 3.8
* pytorch >= 1.12
* spikingjelly
* timm
* wandb

Install dependencies:
```bash
pip install -r requirements.txt




#### Requirement:

```bash
  pip install timm==0.6.12 spikingjelly==0.0.0.0.12 opencv-python==4.8.1.78 wandb einops PyYAML Pillow six torch

  ### OPTIONAL 1: apex
  git clone https://github.com/NVIDIA/apex
  cd apex
  # if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
  # otherwise
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./

  ### OPTIONAL 2: cupy
  
#### Running the code

Please check the bash file in each folder (cifar10-100, event). It can be run directly through the provided `.sh` file. You will need to specify the data path in the bash file.





#### Acknowledgement

This code is built on the Maxformer framework (https://github.com/bic-L/MaxFormer)
