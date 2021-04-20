# InferSent: Learning universal sentence representations

[![Paper](http://img.shields.io/badge/paper-arxiv.1705.02364-B31B1B.svg)](https://arxiv.org/abs/1705.02364)
[![Conference](http://img.shields.io/badge/EMNLP-2017-4b44ce.svg)](https://www.aclweb.org/anthology/events/emnlp-2017/)

This repository contains PyTorch implementation and experiment interface for supervised NLI task with different models for learning universal sentence representations.
#### Results

A baseline model `MeanEmbedding` and three LSTM based models `LSTM`, `BiLSTM` and `BiLSTM-maxpool` are trained on NLI task using [SNLI](https://nlp.stanford.edu/projects/snli/) data. The sentence embeddings are evaluated on 8 transfer tasks using [SentEval](https://github.com/facebookresearch/SentEval) framework.

The micro and macro metric for SentEval tasks are computed as defined in Section 5 of the InferSent paper [1]. The results are tabulated below:

| Model          | snli-dev | snli-test | senteval-micro | senteval-macro |
|----------------|----------|-----------|----------------|----------------|
| MeanEmbedding  | 69.5     | 69.1      | 77.31          | 77.92          |
| LSTM           | 80.5     | 80.2      | 70.467         | 70.282         |
| BiLSTM         | 80.00    | 80.08     | 71.997         | 71.531         |
| BiLSTM-maxpool | 86.50    | 85.87     | 79.075         | 78.831         |


## Organization
This repository is organized into the following major components:

* `models.py` - Pytorch modules for the encoder and classifier models.
* `data.py` - `SNLIData` class for preparing data for training and evaluation.
* `train.py` - Pytorch Lightning model and training CLI for training with different encoders.
* `eval.py` - CLI that takes model checkpoint and runs evaluation on SNLI and SentEval tasks.
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

To run evaluation with SentEval, prepare SentEval installation as follows:

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

The model checkpoints and TensorBoard logs are public and can be found here: https://drive.google.com/drive/folders/1Ebjyf0wj31EZMPEBiG1nHW-1JOMMl1IY?usp=sharing

## References

[1] A. Conneau, D. Kiela, H. Schwenk, L. Barrault, A. Bordes, Supervised Learning of Universal Sentence Representations from Natural Language Inference Data