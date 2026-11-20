# Train a GPT-2-style autoregressive decoder

Learn modular arithmetic sequences from next-token supervision. Build causal multi-head attention and the pre-normalized residual block inside a two-layer decoder. Generation appends one token at a time.

## Task

Implement `CausalAttention.forward and Block.forward` in `model.py`. Build the attention mask, compute per-head weighted sums, reassemble features, and connect the two residual branches. The model provides learned position embeddings, a final normalization and a token matrix shared between embeddings and output logits.

## Worked example

For a progression starting at 10 with stride 2 and vocabulary size 13, input [10, 12, 1] has next-token targets [12, 1, 3]. For B input sequences of length T the model returns [B,T,13] logits.

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

The numerical identity-projection test pins down score scaling, axis order, and mask orientation. Compare generated continuations and held-out loss after 2 and 150 optimizer steps.

## Reference run

At seed 7 the reference produced held-out cross-entropy 0.462 after 150 updates. Greedy generation extended [2, 4, 6] with [8, 10, 12, 1]. These are CPU measurements with PyTorch 2.6.0.

## Paper scope

This is a small decoder with GPT-2's causal attention, pre-normalization, learned positions and tied embeddings. It uses integer tokens, tiny dimensions and a synthetic task in place of web-text training.

[Radford et al., Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
