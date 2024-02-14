# Data

Instructions for setting up datasets for evaluations.


## iNat18
### create lowshot dataset
```bash
python analysis/data/create_inat18_lowshot_dataset.py --src /localdata/inat18 --dst /localdata/inat18_1shot_split1 --shots 1 --seed 1
python analysis/data/create_inat18_lowshot_dataset.py --src /localdata/inat18 --dst /localdata/inat18_1shot_split2 --shots 1 --seed 2
python analysis/data/create_inat18_lowshot_dataset.py --src /localdata/inat18 --dst /localdata/inat18_1shot_split3 --shots 1 --seed 3
python analysis/data/create_inat18_lowshot_dataset.py --src /localdata/inat18 --dst /localdata/inat18_5shot_split1 --shots 5 --seed 1
python analysis/data/create_inat18_lowshot_dataset.py --src /localdata/inat18 --dst /localdata/inat18_5shot_split2 --shots 5 --seed 2
python analysis/data/create_inat18_lowshot_dataset.py --src /localdata/inat18 --dst /localdata/inat18_5shot_split3 --shots 5 --seed 3
python analysis/data/create_inat18_lowshot_dataset.py --src /localdata/inat18 --dst /localdata/inat18_10shot_split1 --shots 10 --seed 1
python analysis/data/create_inat18_lowshot_dataset.py --src /localdata/inat18 --dst /localdata/inat18_10shot_split2 --shots 10 --seed 2
python analysis/data/create_inat18_lowshot_dataset.py --src /localdata/inat18 --dst /localdata/inat18_10shot_split3 --shots 10 --seed 3
```

# caltech101
```bash
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
```

# oxford-pets
```bash
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
```

# oxford-flowers
```bash
mkdir oxford-flowers
cd oxford-flowers
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
tar -xvzf 102flowers.tgz
rm 102flowers.tgz
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
```

# sun397
```bash
mkdir sun-397
wget http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz
tar -xvzf SUN397.tar.gz
rm SUN397.tar.gz
```

# DTD
```bash
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xvzf dtd-r1.0.1.tar.gz
rm dtd-r1.0.1.tar.gz
```

# SVHN
```bash
mkdir svhn
python
```
```python
from torchvision.datasets import SVHN
SVHN(root=".", split="train", download=True)
SVHN(root=".", split="test", download=True)
```