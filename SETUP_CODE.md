# Code Setup

## Environment

Create environments by using the provided environment files.

`conda env create --file environment_<OS>.yml --name <NAME>`


After creating the environment, you can check the installed pytorch and cuda version with:

```
import torch
torch.__version__ 
torch.version.cuda 
```

If you need a different version for your setup, install it via: 

`pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 --index-url https://download.pytorch.org/whl/cu121`



## Optional: cyanure for logistic regression
For logistic regression we use the [cyanure](https://github.com/inria-thoth/cyanure) library. Install it via

`pip install cyanure-mkl` (only available on linux)

# Configuration

Training models with this codebase requires some configurations such as where the code can find datasets or where to put logs.

## static_config.yaml

The file `static_config.yaml` defines all kinds of paths that are specific to your setup:
- where to store checkpoints/logs (`output_path`)
- from where to load pre-trained models (`model_path`)
- from where to load data (`global_dataset_paths`)

Some additional configurations are contained:
- `local_dataset_path` if this is defined, data will be copied before training to this location. This is typically used 
if there is a "slow" global storage and compute nodes have a fast local storage (such as a fast SSD).
- `default_wandb_mode` how you want to log to [Weights and Biases](https://wandb.ai/)
  - `disabled` dont log to W&B
  - `online` use W&B in the "online" mode, i.e. such that you can see live updates in the web interface
  - `offline` use W&B in the "offline" mode. This has to be used if compute nodes dont have internet access. You can 
sync the W&B logs after the run has finished to inspect it via the web interface.
  - if `online` or `offline` is used you will need to create a `wandb_config` (see below)

An example is provided in `template_static_config.yaml` which you can copy and adapt to your setup.

## W&B config

You can log to W&B by setting a `wandb_mode`. Set it in the `static_config.yaml` via `default_wandb_mode`. 
You can define to which W&B project you want to log to via a `wandb: <CONFIG_NAME>` field in a yaml file that defines your run.

All provided yamls by default use the name `v4` as `<CONFIG_NAME>`. To use the same config as defined in the provided 
yamls create a folder `wandb_configs`, copy the `template_wandb_config.yaml` into this folder, change 
`entity`/`project` in this file and rename it to `v4.yaml`.
Every run that defines `wandb: v4` will now fetch the details from this file and log your metrics to this W&B project.


## Run

### Runs require the following arguments

- `--hp <YAML>` e.g. `--hp hyperparams.yaml` define what to run
- `--devices <DEVICES>` e.g. `--devices 0` to run on GPU0 or `--devices 0,1,2,3` to run on 4 GPUs

### Run with SLURM

`python main_sbatch.py --time 2-00:00:00 --qos default --nodes 1 ADDITIONAL_ARGUMENTS`
`python main_sbatch.py --time 2-00:00:00 --qos default --nodes 1 --hp <HP> --name <NAME>`

### Optional arguments (most important ones)

- `--name <NAME>` what name to assign in wandb
- `--wandb_config <YAML>` what wandb configuration to use (by default the `wandb_config.yaml` in the MLPlayground
  directory will be used)
    - only required if you have either `default_wandb_mode` to `online`/`offline` or pass `--wandb_mode <WANDB_MODE>`
      which is `online`/`offline` (a warning will be logged if you specify it with  `wandb_mode=disabled`)
- `--num_workers` specify how many workers will be used for data loading
    - by default `num_workers` will be `number_of_cpus / number_of_gpus`

### Development arguments

- `--accelerator cpu` runs on cpu (can still use multiple devices for debugging multi-gpu runs but with cpu)
- `--mindatarun` adjusts datasets length, epochs, logger intervals and batchsize to a minimum
- `--minmodelrun` replaces all values in the hp yaml of the pattern `${select:model_key:${yaml:models/...}}`
  with `${select:debug:${yaml:models/...}}`
    - you can define your model size with a model key and it will automatically replace it with a minimal model
    - e.g. `encoder_model_key: tiny` for a ViT-T as encoder
      with `encoder_params: ${select:${vars.encoder_model_key}:${yaml:models/vit}}` will select a very light-weight ViT
- `--testrun` combination of `--mindatarun` and `--minmodelrun`

## Data setup

#### data_loading_mode == "local"

- ImageFolder datasets can be stored as zip files (see SETUP.md for creating these)
  - 1 zip per split (slow unpacking): ImageNet/train -> ImageNet/train.zip
  - 1 zip per class per split (fast unpacking): ImageNet/train/n1830348 -> ImageNet/train/n1830348.zip
- sync zipped folders to other servers `rsync -r /localdata/imagenet1k host:/data/`

## Resume run

### Via CLI
- `--resume_stage_id <STAGGE_ID>` resume from `cp=latest`
- `--resume_stage_id <STAGGE_ID> --resume_checkpoint E100` resume from epoch 100
- `--resume_stage_id <STAGGE_ID> --resume_checkpoint U100` resume from update 100
- `--resume_stage_id <STAGGE_ID> --resume_checkpoint S1024` resume from sample 1024

### Via yaml
add a resume initializer to the trainer

```
trainer:
  ...
  initializer:
    kind: resume_initializer
    stage_id: ???
    checkpoint:
      epoch: 100
```