# Train a diffusion model on a two-cluster distribution

Learn to predict the noise added to two-dimensional points at different timesteps. Then iteratively denoise a Gaussian sample using the learned network and the supplied schedule.

## Task

Implement `Model.forward, Model.add_noise and Model.reverse` in `model.py`. Implement timestep conditioning, closed-form forward noising, and reverse posterior sampling. Training samples per-example timesteps and minimizes mean squared error between the predicted and supplied Gaussian noise. The supplied sampler walks all steps in descending order.

## Worked example

For a single diffusion step with cumulative signal coefficient 0.81, clean [1, 0] and noise [0, 1], the forward equation gives [0.9, sqrt(0.19)]. The reverse step at index zero ignores the new noise draw.

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

The tests pin down coefficients for two different timesteps in one batch. Train for 2 steps to exercise the full path, then try 300 steps and sample more points. Check whether both target clusters receive probability mass.

## Reference run

At seed 7 the reference reached held-out noise MSE 0.374 after 300 updates. Its three printed samples occupied both target clusters. Use a larger sample to quantify mode coverage. These are CPU measurements with PyTorch 2.6.0.

## Paper scope

A timestep-conditioned MLP acts on two-dimensional points. The short linear beta schedule, noise-prediction objective and fixed posterior variance make the forward and reverse processes observable on a CPU.

[Ho et al., Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
