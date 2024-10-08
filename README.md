# MIM-Refiner

[[`#1 ImageNet-1K SSL (without extra data)`](https://paperswithcode.com/sota/self-supervised-image-classification-on?p=mim-refiner-a-contrastive-learning-boost-from)]
[[`#1 ImageNet-1K Clustering (without extra data)`](https://paperswithcode.com/sota/image-clustering-on-imagenet?p=mim-refiner-a-contrastive-learning-boost-from)]

[[`Project Page`](https://ml-jku.github.io/MIM-Refiner)] [[`Paper`](https://arxiv.org/abs/2402.10093)] [[`Models`](https://github.com/ml-jku/MIM-Refiner#pre-trained-models)] [[`Codebase
Demo
Video`](https://youtu.be/80kc3hscTTg)] [[`Model Training Demo Video`](https://youtu.be/mS_Lsjfxfko)] [[`BibTeX`](https://github.com/ml-jku/MIM-Refiner#citation)]

Pytorch implementation and pre-trained models of MIM-Refiner.



<p align="center">
<img width="90%" alt="mimrefiner_schematic" src="https://raw.githubusercontent.com/ml-jku/MIM-Refiner/main/docs/imgs/schematic.svg">
</p>


MIM-Refiner efficiently combines the advantages of MIM and ID models and surpasses previous state-of-the-art methods
while being easy to scale up to extremely large models.


<p align="center">
<img width="50%" alt="mimrefiner_spider" src="https://raw.githubusercontent.com/ml-jku/MIM-Refiner/main/docs/imgs/spider.svg">
</p>

# Pre-trained Models

Pre-trained models can be found [here](https://ml.jku.at/research/mimrefiner/download/)

They can also be loaded via torchhub:

```
import torch

# MAE
model = torch.hub.load("ml-jku/MIM-Refiner", "mae_refined_l16")
model = torch.hub.load("ml-jku/MIM-Refiner", "mae_refined_h14")
model = torch.hub.load("ml-jku/MIM-Refiner", "mae_refined_twob14")
# D2V2
model = torch.hub.load("ml-jku/MIM-Refiner", "d2v2_refined_l16")
model = torch.hub.load("ml-jku/MIM-Refiner", "d2v2_refined_h14")
# dBOT
model = torch.hub.load("ml-jku/MIM-Refiner", "dbot_refined_l16")
model = torch.hub.load("ml-jku/MIM-Refiner", "dbot_refined_h14")
# CrossMAE
model = torch.hub.load("ml-jku/MIM-Refiner", "crossmae_refined_l16")
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
found [here](https://github.com/ml-jku/MIM-Refiner/tree/main/src/yamls). Note that MIM-Refiner is trained in 2 stages. "stage 2" trains only the ID heads with a frozen encoder, to ensure a good and stable learning signal for "stage 3" where the encoder is then trained. "stage 2" needs significantly less compute resources and can also be used to get a quick estimate if hyperparameters are suited (temperature, head learning rate, ...).

A demo that showcases how to train models can be found [here](https://youtu.be/mS_Lsjfxfko).

As the trained models are quite large, they also need a bunch of memory and therefore need multi-GPU/multi-node training. If memory is a bottleneck, there are multiple tradeoffs (see [this](https://github.com/ml-jku/MIM-Refiner/issues/11) issue).

# External Evaluation Frameworks

The evaluations of VTAB-1K were done with [this](https://github.com/BenediktAlkin/vtab1k-pytorch) codebase by loading the pre-trained models from torchhub.

The evaluations for ADE20K semantic segmentation were done with [this](https://github.com/BenediktAlkin/KappaSegmentation) codebase by loading the pre-trained models from torchhub.

# Citation

If you like our work, please consider giving it a star :star: and cite us

```
@article{alkin2024mimrefiner,
      title={{MIM-Refiner}: A Contrastive Learning Boost from Intermediate Pre-Trained Representations}, 
      author={Benedikt Alkin and Lukas Miklautz and Sepp Hochreiter and Johannes Brandstetter},
      journal={arXiv preprint arXiv:2402.10093},
      year={2024}
}
```
