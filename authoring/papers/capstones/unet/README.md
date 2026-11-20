# Train a U-Net segmentation model

Mark every pixel belonging to a bright square in a noisy grayscale image. The encoder loses spatial resolution while learning features. The decoder can recover boundaries using feature maps captured before pooling.

## Task

Implement `Up.forward and Model.forward` in `model.py`. Implement the upsample and skip concatenation, then connect the two encoder and decoder levels. The training script uses a positive-weighted binary cross-entropy to counter the small foreground area.

## Worked example

For an image with a four-by-four square beginning at row 3, column 7, the target has ones in rows 3–6 and columns 7–10. All other pixels are zero. The model should emit [B,1,H,W] raw logits.

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

Inspect the gradient test for both the high-resolution and bridge convolutions. Track held-out intersection-over-union after 2 and 100 steps. As an ablation, replace skip activations with zeros and compare under the same seed.

## Reference run

At seed 7 the reference reached held-out pixel intersection-over-union 0.805 after 100 updates. At two updates it predicted no foreground on the same evaluation batch. These are CPU measurements with PyTorch 2.6.0.

## Paper scope

Two levels reproduce the contracting and expanding structure. Padded convolutions keep output and input aligned. The original U-Net used valid convolutions and cropped its skip activations.

[Ronneberger et al., U-Net](https://arxiv.org/abs/1505.04597)
