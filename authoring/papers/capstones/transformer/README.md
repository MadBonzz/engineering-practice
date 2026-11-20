# Train a Transformer translator

Translate a sequence of five input symbols by reversing it, shifting each symbol forward by one modulo ten, and appending EOS. During training the decoder receives BOS followed by the translated symbols. During inference it receives its own previous predictions.

## Task

Implement `DecoderLayer.forward` in `model.py`. Finish masked self-attention, cross-attention, and the position-wise feed-forward residual. The source encoder and sinusoidal position function are provided. For a batch with B examples, target length T, and vocabulary size 12, return logits of shape [B,T,12].

## Worked example

Source [2, 4, 11] maps to target [2, 5, 3, 0]. The reversed source is [11, 4, 2]. Shifting the first element wraps token 11 to 2. Token 1 is BOS and token 0 is EOS.

## Project layout

- `data.py` generates a repeatable stream of synthetic training batches.
- `model.py` contains the architecture and the methods to implement.
- `train.py` computes the training objective, updates the parameters, evaluates a held-out batch, and runs the model.
- `test_model.py` exercises numerical and architectural invariants.

## Run locally

Use Python 3.10 or later in an environment with PyTorch 2.x. The examples use the CPU. Start in this project directory.

```shell
python -m pip install torch
python -m unittest -v test_model
python train.py --steps 2 --seed 7
```

The first test run on the starter raises `NotImplementedError` in the methods above. After implementation, each step prints a training loss. The held-out examples use their own fixed random seed.

## Experiment

Change a token in the target suffix and check that earlier logits stay identical. Change a source token and check that the first target prediction can change. Inspect the generated sequence after 2 and 150 steps.

## Reference run

At seed 7 the reference reached held-out cross-entropy 1.958 after 150 updates and 0.219 after 800. At 800 updates, greedy decoding still transposed two symbols in the printed example. Compare exact sequence accuracy with token loss when diagnosing the error. These are CPU measurements with PyTorch 2.6.0.

## Paper scope

One post-normalized encoder layer and one decoder layer preserve the structure of the 2017 translation model on a synthetic vocabulary. The paper used a much larger stack, tokenized natural language, dropout, and a different learning-rate schedule.

[Vaswani et al., Attention Is All You Need](https://arxiv.org/abs/1706.03762)
