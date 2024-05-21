[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mim-refiner-a-contrastive-learning-boost-from/self-supervised-image-classification-on)](https://paperswithcode.com/sota/self-supervised-image-classification-on?p=mim-refiner-a-contrastive-learning-boost-from) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mim-refiner-a-contrastive-learning-boost-from/image-clustering-on-imagenet)](https://paperswithcode.com/sota/image-clustering-on-imagenet?p=mim-refiner-a-contrastive-learning-boost-from)



[[Code](https://github.com/ml-jku/MIM-Refiner)] [[Paper](https://arxiv.org/abs/2402.10093)] [[Models](https://github.com/ml-jku/MIM-Refiner#pre-trained-models)] [[Codebase Demo Video](https://youtu.be/80kc3hscTTg)] [[BibTeX](https://github.com/ml-jku/MIM-Refiner#citation)]




MIM-Refiner refines the representation of pre-trained **M**asked **I**mage **M**odels (MIM) by attaching 
**I**nstance **D**iscrimination (ID) heads to multiple intermediate heads. This setup is then trained for a few epochs with
with our **N**earest **N**eighbor **A**lignment (NNA) objective.


<p align="center">
<img width="100%" alt="mimrefiner_schematic" src="https://raw.githubusercontent.com/ml-jku/MIM-Refiner/main/docs/imgs/schematic.svg">
</p>


MIM-Refiner drastically advances state-of-the-art in ImageNet-1K linear probing.
It achieves an improvement of +2.5% over the previous state-of-the-art.
In comparison, over the last 4 years, state-of-the-art improved by +2.6%.


<p align="center">
<img width="80%" alt="mimrefiner_timeline" src="https://raw.githubusercontent.com/ml-jku/MIM-Refiner/main/docs/imgs/timeline.svg">
</p>



This improvement in linear probing correlates with strong improvements in k-NN accuracy and low-shot classification accuracies on ImageNet-1K.

<p align="center">
<img width="80%" alt="mimrefiner_timeline" src="https://raw.githubusercontent.com/ml-jku/MIM-Refiner/main/docs/imgs/in1k_lowshot.svg">
</p>



MIM-Refiner learns general purpose features that can be easily transferred to a diverse set of datasets.

<p align="center">
<img width="80%" alt="mimrefiner_timeline" src="https://raw.githubusercontent.com/ml-jku/MIM-Refiner/main/docs/imgs/transfer.svg">
</p>



An improvement is also visible (although smaller) when fully fine-tuning the models on large amounts of data.

<p align="center">
<img width="80%" alt="mimrefiner_timeline" src="https://raw.githubusercontent.com/ml-jku/MIM-Refiner/main/docs/imgs/finetuning.svg">
</p>


