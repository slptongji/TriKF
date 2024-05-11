# TriKF: Triple-Perspective Knowledge Fusion Network for Empathetic Question Generation

This work has been submitted to _IEEE Transactions on Computational Social System_.

Please cite our paper if you find our work helpful.

## Environment Settings

Our basic environment configurations are as follows:

- Operating System: Ubuntu 18.04
- CUDA: 10.1.105
- Python: 3.8.5
- PyTorch: 1.7.0

## Usage
- Download the GloVe embedding file `glove.6B.300d.txt` and put it into `/data`.
- Download the [COMET-BART model](https://github.com/allenai/comet-atomic-2020) `pytorch_model.bin` and put it into `/data/Comet`. 
- Execute the source code through the following commands:
  - for EQT dataset: `python main.py`
  - for EQ-EMAC dataset: `python main_emac.py`
