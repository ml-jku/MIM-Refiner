# Models Setup

URLs for downloading pre-trained models for refining or evaluation.
Download models into the path defined in the field "model_path" from your `static_config.yaml`.
Afterwards you can load it via:
```
initializers:
  - kind: pretrained_initializer
    weights_file: mae_large16.pth
    use_checkpoint_kwargs: true
```


```bash 
# MAE
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth -O mae_base16.pth
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth -O mae_large16.pth
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth -O mae_huge14.pth
wget https://dl.fbaipublicfiles.com/maws/pretrain/mae_in1k/vit_2b14.pt -O mae_twob14.pt

# Data2Vec 2.0
wget https://dl.fbaipublicfiles.com/fairseq/data2vec2/base_imagenet.pt -O data2vec2_base16.pt
wget https://dl.fbaipublicfiles.com/fairseq/data2vec2/large_imagenet.pt -O data2vec2_large16.pt
wget https://dl.fbaipublicfiles.com/fairseq/data2vec2/huge_imagenet.pt -O data2vec2_huge14.pt

# MAE-CT-aug
wget https://ml.jku.at/research/maect/download/maectaug_base16.th -O maectaug_base16.th
wget https://ml.jku.at/research/maect/download/maectaug_large16.th -O maectaug_large16.th
wget https://ml.jku.at/research/maect/download/maectaug_huge16.th -O maectaug_huge16.th
wget https://ml.jku.at/research/maect/download/maectaug_huge14.th -O maectaug_huge14.th

# LayerGrafting
wget https://www.dropbox.com/sh/e9czo0xtivdqvff/AAC730kZx8Bj6pEIhFswpvVla/checkpoint_final.pth.tar -O layergrafting_base16.pth.tar
wget https://www.dropbox.com/sh/fk92wphgu8fq772/AABklx8vjQmDZgz8vg6BbTPWa/checkpoint_final.pth.tar -O layergrafting_large16.pth.tar

# MUGS
wget https://huggingface.co/zhoupans/Mugs/resolve/main/pretrained%20models/vit_large_backbone_250ep.pth -O mugs_large16.pth

# I-JEPA
wget https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar -O ijepa_huge14.pth.tar
wget https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16-448px-300e.pth.tar -O ijepa_huge16res448.pth.tar

# IBOT
wget https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16_rand_mask/checkpoint.pth -O ibot_large16_rand.pth
wget https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16_pt22k/checkpoint.pth -O ibot_large16_in22k.pth

# DINO
wget https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth -O dino_base8.pth
```