# Domain Adaptive Diffusion Policy (DADP)

Welcome to the official repository for Domain Adaptive Diffusion Policy. This repository provides implementations for training both domain-specific embeddings and diffusion-based planning/policy models across mixed environments.

## Getting Started

### 1. Construct the Conda Environment

We provide an `environment.yml` to help you set up the python environment correctly. Use the following commands to create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate dadp310
```

*Note: Ensure you are running this on a machine with CUDA devices available.*

### 2. Configure MuJoCo

Before proceeding, you must ensure that MuJoCo (mujoco210) is properly installed and configured on your system, as it is required by the environments. Follow the [official d4rl installation instructions](https://github.com/Farama-Foundation/D4RL#installation) or set up your `~/.mujoco/mujoco210` folder correctly and make sure your `LD_LIBRARY_PATH` includes the MuJoCo binaries.

*Note: For many distributions, this looks like `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin`.*

### 3. Download Datasets and Pre-trained Models

Before training or evaluating, you need to download the required datasets and pre-trained models from Hugging Face:

- **Models:** [https://huggingface.co/qinghangliu/DADP](https://huggingface.co/qinghangliu/DADP)
- **Datasets:** [https://huggingface.co/datasets/qinghangliu/DADP](https://huggingface.co/datasets/qinghangliu/DADP)

Once you have downloaded the datasets, extract and move them into your local Minari datasets directory. If you are using Minari for the first time, you may need to manually create this folder path (`mkdir -p ~/.minari/datasets/`). The expected directory structure should look like this:

```text
~/.minari/
└── datasets/
    ├── RandomAnt/
    ├── RandomWalker2d/
    ├── Adroit/
    └── ...
```




For the pre-trained models, place the downloaded checkpoints inside the `./dadp/embedding/logs/ckpt/` directory within this repository. The expected folder structure should look like this:

```text
DomainAdaptiveDiffusionPolicy/
└── dadp/
    └── embedding/
        └── logs/
            └── ckpt/
                ├── Adroit_door_shrink_combined-v0/
                │   └── best_model.zip
                ├── RandomAnt_28dynamics-v0/
                │   └── best_model.zip
                └── ...
```

### 4. Training Models

This repository provides single-run bash scripts located in the `scripts/` directory to streamline training.

#### Option A: Train Diffusion Only

If you have precomputed embeddings properly configured, or just want to train the main diffusion/planning model for a single environment, use the `train_diffusion.sh` wrapper (which maps to `run_single_seed.sh`). 

The script allows referring to environments via a simple `env_key` mapping.

**Selectable `env_key` variables:**
`ant`, `halfcheetah`, `walker`, `hopper`, `door`, `relocate`

```bash
# Example: Train diffusion for 'ant' on GPU 0 with seed 0
bash scripts/train_diffusion.sh ant --seed 0 --gpu 0

# Example: Train and run evaluations post-train
bash scripts/train_diffusion.sh walker --seed 1 --gpu 1 --eval both
```

#### Option B: Train Embeddings

If you want to newly train the latent environment embeddings (which capture dynamic shifts across tasks), use the `train_embedding.sh` script.

This uses a similar architecture to pass configs and parameters to `train_embedding.py`.

```bash
# Example: Train embedding for 'ant' on GPU 0 with seed 42
bash scripts/train_embedding.sh ant --seed 42 --gpu 0

# Example: Override epochs to 20 for 'walker'
bash scripts/train_embedding.sh walker --seed 1 --gpu 1 --num_epochs 20
```

After training the embeddings, you need to extract them by running the `extract_embedding.sh` script, which will default to extracting embeddings using the `best_model.zip` found in your config's `log_dir`:

```bash
# Example: Extract embedding for 'ant' on GPU 0
bash scripts/extract_embedding.sh ant --gpu 0

# Example: Extract embedding for a specific checkpoint
bash scripts/extract_embedding.sh ant --gpu 0 --checkpoint path/to/your/best_model.zip
``` 


