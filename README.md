# InferSent: Learning universal sentence representations

[![Paper](http://img.shields.io/badge/paper-arxiv.1705.02364-B31B1B.svg)](https://arxiv.org/abs/1705.02364)
[![Conference](http://img.shields.io/badge/EMNLP-2017-4b44ce.svg)](https://www.aclweb.org/anthology/events/emnlp-2017/)

This repository implements and experiments with several models for supervised learning with NLI data for learning universal sentence representations.
#### Results

| Model          | snli-dev | snli-test | senteval-micro | senteval-macro |
|----------------|----------|-----------|----------------|----------------|
| MeanEmbedding  | 69.5     | 69.1      | 77.31          | 77.92          |
| LSTM           | 80.5     | 80.2      | 70.467         | 70.282         |
| BiLSTM         | 80.00    | 80.08     | 71.997         | 71.531         |
| BiLSTM-maxpool | 86.50    | 85.87     | 79.075         | 78.831         |


## Organization
This repository containes the  following major components:

* `models.py` - Pytorch modules for the encoder and classifier models.
* `data.py` - `SNLIData` class for preparing data for training and evaluation.
* `train.py` - Pytorch Lightning model and training CLI interface for training with different encoders.
* `eval.py` - CLI interface that takes model checkpoint and runs evaluation on SNLI and SentEval tasks.
* `demo.ipynb` - Jupyter notebook for testing model inference and analyzing the results.

## Setup

```shell
# Using pip
pip install -r requirements.txt
# Using conda
conda env create -f environment.yml
# Download english model for SpaCy tokenizer
python -m spacy download en_core_web_sm

```

To run evaluation with SentEval, proceed as follows:

```shell
git clone https://github.com/facebookresearch/SentEval.git
cd SentEval/ && python setup.py install
# Download datasets
cd SentEval/data/downstream/ && ./get_transfer_data.bash
```
## Training

Run `train.py` with one of the following encoder types: `MeanEmbedding`, `LSTM`, `BiLSTM`, `BiLSTM-maxpool`. The training process will create model checkpoints, TensorBoard logs and hyperparams file `hparams.yaml` in the `./logs` directory.

```shell
python train.py --encoder_type='BiLSTM'
```

## Evaluation

Run `eval.py` with a model checkpoint flag to run evaluation tasks on SNLI and SentEval.

```shell
python eval.py --checkpoint_path='./logs/MeanEmbedding/version_0/checkpoints/epoch=2-step=12875.ckpt'
```

## Pre-trained models

The model checkpoints and TensorBoard logs can be found here: https://drive.google.com/drive/folders/1Ebjyf0wj31EZMPEBiG1nHW-1JOMMl1IY?usp=sharing