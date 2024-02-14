# MLPlayground

## Setup

#### environment

`conda env create --file environment_<OS>.yml --name <NAME>`
this will most likely install pytorch 2.0.0 with some old cuda version -> install newer cuda version
`pip install torch==2.0.0+cu117 torchvision==0.15.0+cu117 --index-url https://download.pytorch.org/whl/cu117`
`pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 --index-url https://download.pytorch.org/whl/cu118`
`pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 --index-url https://download.pytorch.org/whl/cu121`

you can check the installed version with:

```
import torch
torch.__version__ 
torch.version.cuda 
```

#### Optional: special libraries

- `pip install cyanure-mkl` (logistic regression; only available on linux; make sure you are on a GPU node to install)

#### configuration

#### static_config.yaml

choose one of the two options:

- copy a template and adjust values to your setup `cp template_static_config_iml.yaml static_config.yaml`
- create a file `static_config.yaml` with the first line `template: ${yaml:template_static_config_iml}`
    - overwrite values in the template by adding lines of the format `template.<property_to_overwrite>: <value>`
        - `template.account_name: <ACCOUNT_NAME>`
        - `template.output_path: <OUTPUT_PATH>`
    - add new values
        - `model_path: <MODEL_PATH>`
    - example:
  ```
  template: ${yaml:template_static_config_iml}
  template.account_name: <ACCOUNT_NAME>
  template.output_path: /system/user/publicwork/ssl/save
  template.local_dataset_path: /localdata
  ```

#### optional configs configs
- create wandb config(s) (use via `--wandb_config <WANDB_CONFIG_NAME>` in CLI or `wandb: <WANDB_CONFIG_NAME` in yaml)
    - `cp template_wandb_config.yaml wandb_configs/<WANDB_CONFIG_NAME>.yaml`
    - adjust values to your setup
- create a default wandb config (this will be used when no wandb config is defined)
    - `cp template_wandb_config.yaml wandb_config.yaml`
    - adjust values to your setup
- create `sbatch_config.yaml` (only required for `main_sbatch.py` on slurm clusters)
    - `cp template_config_sbatch.yaml sbatch_config.yaml`
- create `template_sbatch_nodes.yaml` (only required for running `main_sbatch.py --nodes <NODES>` on slurm clusters)
    - `cp template_sbatch_nodes_<HPC>.yaml template_sbatch_nodes.yaml`
- create `template_sbatch_gpus.yaml` (only required for running `main_sbatch.py --gpus <GPUS>` on slurm clusters)
    - `cp template_sbatch_gpus_<HPC>.yaml template_sbatch_gpus.yaml`

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