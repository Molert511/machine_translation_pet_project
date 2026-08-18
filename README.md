# Machine Translation Experiment Pipeline

A machine translation project with implementations of **RNN** and **Transformer** models.

The project supports:

* Training RNN and Transformer models
* Configuring experiments with YAML files
* Training on custom datasets
* Saving and loading model checkpoints
* Running inference with trained models
* Plotting training losses and evaluation metrics

## Requirements

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

## Installation

Clone the repository and install the dependencies:

```bash
git clone git@github.com:Molert511/machine_translation_pet_project.git
cd machine_translation_pet_project
uv sync
```

## Configuration

Training parameters are specified in YAML configuration files. You can find example configurations in the [`configs`](configs) directory.

### General

| Parameter         | Description                                                     |
| ----------------- | --------------------------------------------------------------- |
| `device`          | Device used for training: `cuda` or `cpu`                       |
| `model_name`      | Model architecture: `rnn` or `transformer`                      |
| `checkpoint_path` | Path where model checkpoints and training information are saved |

### Dataset

| Parameter    | Description                                               |
| ------------ | --------------------------------------------------------- |
| `src_train`  | Path to the source-language training file                 |
| `tgt_train`  | Path to the corresponding target-language training file   |
| `src_val`    | Path to the source-language validation file               |
| `tgt_val`    | Path to the corresponding target-language validation file |
| `max_length` | Maximum sequence length used by the dataset               |

Source and target files should contain corresponding sentences on the same lines.

### Training

| Parameter    | Description               |
| ------------ | ------------------------- |
| `batch_size` | Training batch size       |
| `epochs_cnt` | Number of training epochs |

### RNN

The following parameters are used for the RNN model:

| Parameter     | Description                           |
| ------------- | ------------------------------------- |
| `embed_size`  | Dimension of the input embeddings     |
| `hidden_size` | Dimension of the LSTM hidden states   |
| `num_layers`  | Number of LSTM layers                 |
| `max_length`  | Maximum length of generated sequences |

### Transformer

The following parameters are used for the Transformer model:

| Parameter            | Description                           |
| -------------------- | ------------------------------------- |
| `d_model`            | Embedding dimension                   |
| `d_ff`               | Feed-forward network hidden dimension |
| `num_encoder_layers` | Number of encoder layers              |
| `num_decoder_layers` | Number of decoder layers              |
| `nhead`              | Number of attention heads             |
| `dropout`            | Dropout probability                   |
| `max_length`         | Maximum length of generated sequences |

## Training

To train a model, run the following command from the project root:

```bash
uv run train.py --config=configs/your_config.yaml
```

The --config argument specifies the path to the YAML configuration file

You can find training and evaluation plots in the [`plots`](plots) directory. They include:

* Training and validation losses
* Evaluation metrics

## Inference

To translate sentences with a trained model, run:

```bash
uv run inference.py \
    --checkpoint=path/to/checkpoint \
    --src_filepath=path/to/input.txt \
    --output_path=path/to/output.txt
```

Where:

| Argument         | Description                            |
| ---------------- | -------------------------------------- |
| `--checkpoint`   | Path to the model checkpoint           |
| `--src_filepath` | Input file with sentences to translate |
| `--output_path`  | File where translations will be saved  |

## IWSLT14

The project can be tested on the **IWSLT14 German-English** dataset.

To download and prepare the data, run the following command:

```bash
uv run prepare_data.py
```

After that, you can use one of the configs in [`configs`](configs) to train a model

## Transformer

The Transformer implementation is based on: [arXiv](https://arxiv.org/abs/1706.03762)

## TODO

* [ ] Implement beam search
* [ ] Implement KV cache
* [ ] Add configurable tokenizer selection
* [ ] Make optimizer, learning-rate scheduler, and loss function configurable
