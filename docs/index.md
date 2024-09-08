[[#1 ImageNet-1K SSL (without extra data)](https://paperswithcode.com/sota/self-supervised-image-classification-on?p=mim-refiner-a-contrastive-learning-boost-from)]
[[#1 ImageNet-1K Clustering (without extra data)](https://paperswithcode.com/sota/image-clustering-on-imagenet?p=mim-refiner-a-contrastive-learning-boost-from)]


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
<img width="60%" alt="mimrefiner_timeline" src="https://raw.githubusercontent.com/ml-jku/MIM-Refiner/main/docs/imgs/timeline.svg">
</p>


MIM-Refiner efficiently combines the advantages of MIM and ID models and surpasses previous state-of-the-art methods
while being easy to scale up to extremely large models.


<p align="center">
<img width="60%" alt="mimrefiner_spider" src="https://raw.githubusercontent.com/ml-jku/MIM-Refiner/main/docs/imgs/spider.svg">
</p>




