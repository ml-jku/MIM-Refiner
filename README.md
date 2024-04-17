# MIM-Refiner

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mim-refiner-a-contrastive-learning-boost-from/self-supervised-image-classification-on)](https://paperswithcode.com/sota/self-supervised-image-classification-on?p=mim-refiner-a-contrastive-learning-boost-from) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mim-refiner-a-contrastive-learning-boost-from/image-clustering-on-imagenet)](https://paperswithcode.com/sota/image-clustering-on-imagenet?p=mim-refiner-a-contrastive-learning-boost-from)

[[`Project Page`](https://ml-jku.github.io/MIM-Refiner)] [[`Paper`](https://arxiv.org/abs/2402.10093)] [[`Models`](https://github.com/ml-jku/MIM-Refiner#pre-trained-models)] [[`Codebase
Demo
Video`](https://youtu.be/80kc3hscTTg)] [[`BibTeX`](https://github.com/ml-jku/MIM-Refiner#citation)]

Pytorch implementation and pre-trained models of MIM-Refiner.



<p align="center">
<img width="90%" alt="mimrefiner_schematic" src="https://raw.githubusercontent.com/ml-jku/MIM-Refiner/main/docs/imgs/schematic.svg">
</p>

# Pre-trained Models

Pre-trained models can be found [here](https://ml.jku.at/research/mimrefiner/download/)

They can also be loaded via torchhub:

```
import torch

model = torch.hub.load("ml-jku/MIM-Refiner", "mae_refined_l16")
model = torch.hub.load("ml-jku/MIM-Refiner", "mae_refined_h14")
model = torch.hub.load("ml-jku/MIM-Refiner", "mae_refined_twob14")
model = torch.hub.load("ml-jku/MIM-Refiner", "d2v2_refined_l16")
model = torch.hub.load("ml-jku/MIM-Refiner", "d2v2_refined_h14")
```

An example how to use torchhub models for a k-NN classifier can be
found [here](https://github.com/ml-jku/MIM-Refiner/blob/main/eval_knn_torchhub.py).

`python eval_knn_torchhub.py --model mae_refined_l16 --data_train /imagenet/train/ --data_test /imagenet/val`

Note that the results of this script can differ slightly from the the paper results as the paper results remove the last
LayerNorm of pre-norm ViTs and use bfloat16 precision.


# Train your own models

Instructions to setup the codebase on your own environment are provided in
[SETUP_CODE](https://github.com/ml-jku/MIM-Refiner/blob/main/SETUP_CODE.md),
[SETUP_DATA](https://github.com/ml-jku/MIM-Refiner/blob/main/SETUP_DATA.md) and
[SETUP_MODELS](https://github.com/ml-jku/MIM-Refiner/blob/main/SETUP_MODELS.md).

A video to motivate design choices of the codebase and give an overview of the codebase can be
found [here](https://www.youtube.com/watch?v=80kc3hscTTg).

Configurations to train, evaluate or analyze models can be
found [here](https://github.com/ml-jku/MIM-Refiner/tree/main/src/yamls).

# Citation

If you like our work, please consider giving it a star :star: and cite us

```
@article{alkin2024mimrefiner,
      title={MIM-Refiner: A Contrastive Learning Boost from Intermediate Pre-Trained Representations}, 
      author={Benedikt Alkin and Lukas Miklautz and Sepp Hochreiter and Johannes Brandstetter},
      journal={arXiv preprint arXiv:2402.10093},
      year={2024}
}
```