# Train a residual image classifier

Classify noisy synthetic images containing a horizontal or vertical stripe. Route each image through an identity block, a stride-two projection block, and a final identity block.

## Task

Implement `Residual.forward and Model.forward` in `model.py`. Apply the shortcut to the original input and add the completed residual transform before the rectifier. Globally average spatial features so the classifier accepts rectangular images.

## Worked example

A 16 by 16 image with a bright stripe in columns 4 and 5 has label 1 (vertical). A horizontal stripe has label 0. A batch of 12 images produces logits with shape [12,2].

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

Zero the residual transform and inspect an identity block on a signed input. The expected result is its positive part. Repeat with a stride-two projection and trace both branches. Check performance on new synthetic stripes.

## Reference run

At seed 7 the reference held-out cross-entropy fell from 0.705 after two updates to 0.00151 after 100. The six printed classes after 100 steps were [0, 1, 1, 0, 1, 0], matching the sampled labels. These are CPU measurements with PyTorch 2.6.0.

## Paper scope

This project uses ResNet's two-convolution basic block and a learned projection for stage changes. Its smaller stem, widths and depth keep CIFAR-sized experiments practical on a CPU.

[He et al., Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
