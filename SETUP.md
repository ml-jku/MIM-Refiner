# Data

## iNat18
### create lowshot dataset
python analysis/data/create_inat18_lowshot_dataset.py --src /localdata/inat18 --dst /localdata/inat18_1shot_split1 --shots 1 --seed 1
python analysis/data/create_inat18_lowshot_dataset.py --src /localdata/inat18 --dst /localdata/inat18_1shot_split2 --shots 1 --seed 2
python analysis/data/create_inat18_lowshot_dataset.py --src /localdata/inat18 --dst /localdata/inat18_1shot_split3 --shots 1 --seed 3
python analysis/data/create_inat18_lowshot_dataset.py --src /localdata/inat18 --dst /localdata/inat18_5shot_split1 --shots 5 --seed 1
python analysis/data/create_inat18_lowshot_dataset.py --src /localdata/inat18 --dst /localdata/inat18_5shot_split2 --shots 5 --seed 2
python analysis/data/create_inat18_lowshot_dataset.py --src /localdata/inat18 --dst /localdata/inat18_5shot_split3 --shots 5 --seed 3
python analysis/data/create_inat18_lowshot_dataset.py --src /localdata/inat18 --dst /localdata/inat18_10shot_split1 --shots 10 --seed 1
python analysis/data/create_inat18_lowshot_dataset.py --src /localdata/inat18 --dst /localdata/inat18_10shot_split2 --shots 10 --seed 2
python analysis/data/create_inat18_lowshot_dataset.py --src /localdata/inat18 --dst /localdata/inat18_10shot_split3 --shots 10 --seed 3
### per-class zips for fast copying to compute nodes (uses KappaData's main_create_zips.py file)
python main_create_zips.py --src /localdata/inat18/train --dst /localdata/inat18/train_zip --zips --image_folder
python main_create_zips.py --src /localdata/inat18/val --dst /localdata/inat18/val_zip --zips --image_folder
python main_create_zips.py --src /localdata/inat18_1shot_split1/train --dst /localdata/inat18_1shot_split1/train_zip --zips --image_folder
python main_create_zips.py --src /localdata/inat18_1shot_split2/train --dst /localdata/inat18_1shot_split2/train_zip --zips --image_folder
python main_create_zips.py --src /localdata/inat18_1shot_split3/train --dst /localdata/inat18_1shot_split3/train_zip --zips --image_folder
python main_create_zips.py --src /localdata/inat18_5shot_split1/train --dst /localdata/inat18_5shot_split1/train_zip --zips --image_folder
python main_create_zips.py --src /localdata/inat18_5shot_split2/train --dst /localdata/inat18_5shot_split2/train_zip --zips --image_folder
python main_create_zips.py --src /localdata/inat18_5shot_split3/train --dst /localdata/inat18_5shot_split3/train_zip --zips --image_folder
python main_create_zips.py --src /localdata/inat18_10shot_split1/train --dst /localdata/inat18_10shot_split1/train_zip --zips --image_folder
python main_create_zips.py --src /localdata/inat18_10shot_split2/train --dst /localdata/inat18_10shot_split2/train_zip --zips --image_folder
python main_create_zips.py --src /localdata/inat18_10shot_split3/train --dst /localdata/inat18_10shot_split3/train_zip --zips --image_folder


# caltech101
wget https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip
unzip caltech-101.zip
rm caltech-101.zip
rm -rf __MACOSX/
cd caltech-101
tar -xvzf 101_ObjectCategories.tar.gz
rm 101_ObjectCategories.tar.gz
rm caltech-101/101_ObjectCategories/BACKGROUND_Google/tmp
rm Annotations.tar
rm show_annotation.m

# oxford-pets
mkdir oxford-pets
cd oxford-pets
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
tar -xvzf images.tar.gz
rm images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xvzf annotations.tar.gz
rm annotations.tar.gz
rm -rf annotations/trimaps
rm -rf annotations/xmls

# oxford-flowers
mkdir oxford-flowers
cd oxford-flowers
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
tar -xvzf 102flowers.tgz
rm 102flowers.tgz
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat

# sun397
mkdir sun-397
wget http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
tar -xvzf SUN397.tar.gz
rm SUN397.tar.gz

# DTD
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xvzf dtd-r1.0.1.tar.gz
rm dtd-r1.0.1.tar.gz

# SVHN
mkdir svhn
python
from torchvision.datasets import SVHN
SVHN(root=".", split="train", download=True)
SVHN(root=".", split="test", download=True)
# Models

```bash
# MAE
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth -O mae_base16.pth
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth -O mae_large16.pth
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth -O mae_huge14.pth
wget https://dl.fbaipublicfiles.com/maws/pretrain/mae_in1k/vit_2b14.pt -O mae_twob14.pt
# MAE IG3B
wget https://dl.fbaipublicfiles.com/maws/pretrain/mae/vit_l16.pt -O mae_ig3b_large16.pt
wget https://dl.fbaipublicfiles.com/maws/pretrain/mae/vit_h14.pt -O mae_ig3b_huge14.pt
wget https://dl.fbaipublicfiles.com/maws/pretrain/mae/vit_2b14.pt -O mae_ig3b_twob14.pt
# MAE-WS IG3B
wget https://dl.fbaipublicfiles.com/maws/pretrain/maws/vit_l16.pt -O maews_ig3b_large16.pt
wget https://dl.fbaipublicfiles.com/maws/pretrain/maws/vit_h14.pt -O maews_ig3b_huge14.pt
wget https://dl.fbaipublicfiles.com/maws/pretrain/maws/vit_2b14.pt -O maews_ig3b_twob14.pt
# MAE finetuned
wget https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth -O mae_base16_finetuned.pth
wget https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_large.pth -O mae_large16_finetuned.pth
wget https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_huge.pth -O mae_huge14_finetuned.pth
# MAE from MAE-CT
wget https://ml.jku.at/research/maect/download/mae_reimpl_base16.th -O maereimpl_base16.th
wget https://ml.jku.at/research/maect/download/mae_reimpl_large16.th -O maereimpl_large16.th
wget https://ml.jku.at/research/maect/download/mae_reimpl_huge16.th -O maereimpl_huge16.th
# long sequence MAE
wget https://dl.fbaipublicfiles.com/long_seq_mae/pretrained_models/in1k/vitb_dec384d12h8b_800ep_img448_crop0.2-1.0_maskds2.pth -O mae_base16res448e800.pth
wget https://dl.fbaipublicfiles.com/long_seq_mae/pretrained_models/in1k/vitl_dec512d16h8b_800ep_img448_crop0.2-1.0_maskds2.pth -O mae_large16res448e800.pth
wget https://dl.fbaipublicfiles.com/long_seq_mae/pretrained_models/in1k/vitb_dec384d12h8b_1600ep_img448_crop0.2-1.0_maskds2.pth -O mae_base16res448.pth
wget https://dl.fbaipublicfiles.com/long_seq_mae/pretrained_models/in1k/vitl_dec512d16h8b_1600ep_img448_crop0.2-1.0_maskds2.pth -O mae_large16res448.pth

# Data2Vec 2.0
wget https://dl.fbaipublicfiles.com/fairseq/data2vec2/base_imagenet.pt -O data2vec2_base16.pt
wget https://dl.fbaipublicfiles.com/fairseq/data2vec2/large_imagenet.pt -O data2vec2_large16.pt
wget https://dl.fbaipublicfiles.com/fairseq/data2vec2/huge_imagenet.pt -O data2vec2_huge14.pt
# Data2Vec 2.0 finetuned
wget https://dl.fbaipublicfiles.com/fairseq/data2vec2/large_imagenet_ft.pt -O data2vec2_large16_finetuned.pt
wget https://dl.fbaipublicfiles.com/fairseq/data2vec2/huge_imagenet_ft.pt -O data2vec2_huge14_finetuned.pt

# MAE-CT
wget https://ml.jku.at/research/maect/download/maect_base16.th -O maect_base16.th
wget https://ml.jku.at/research/maect/download/maect_large16.th -O maect_large16.th
wget https://ml.jku.at/research/maect/download/maect_huge16.th -O maect_huge16.th
wget https://ml.jku.at/research/maect/download/maect_huge14.th -O maect_huge14.th
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