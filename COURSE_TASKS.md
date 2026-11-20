# Course tasks

The Course tasks tab (`/?track=courses`) brings together 22 new implementation labs and 28 existing labs. Each new task has three cumulative parts, public tests, additional grading workloads, progressive hints, source references and an editable experiment notebook. Shared subjects keep one task ID and one progress history.

## How to use the tasks

1. Choose a course and start with the suggested preparation links.
2. Read the problem, supplied gateway and part contracts.
3. Write a prediction in the experiment notebook before changing a configuration.
4. Implement the parts and run public tests. Add your own regression tests.
5. Submit a snapshot for additional checks, compare controlled runs and explain the result.
6. Export the project to continue locally. `python experiment.py` runs the provided experiment fixtures.

The numerical tests grade code behavior. Written predictions and explanations are saved with the attempt but require self-review. Every new lab runs with standard-library Python in the browser or locally; no external compute account is required.

## New implementation labs

| Task | Topic | Courses |
|---|---|---|
| [Train a stable softmax classifier](public/courses/course-classifier/exercise.json) | Foundations | DL Systems, CS336 |
| [Repair a reverse-mode differentiation engine](public/courses/course-autodiff/exercise.json) | Foundations | DL Systems |
| [Implement tensor views and their backward scatter](public/courses/course-tensor-views/exercise.json) | Foundations | DL Systems, CS336 |
| [Make training and evaluation state agree](public/courses/course-normalization/exercise.json) | Foundations | DL Systems, Alchemy |
| [Resume a correct AdamW training update](public/courses/course-optimizer/exercise.json) | Training mechanics | DL Systems, CS336, Alchemy |
| [Connect an image filter to backpropagation](public/courses/course-convolution-backward/exercise.json) | Foundations | DL Systems |
| [Build an edge-safe tiled matrix operator](public/courses/course-tiled-matmul/exercise.json) | Training systems | DL Systems, CS336 |
| [Recover training after a mixed-precision overflow](public/courses/course-loss-scaling/exercise.json) | Training systems | DL Systems, CS336 |
| [Fit an activation replay plan into a memory budget](public/courses/course-checkpoint-planner/exercise.json) | Training systems | DL Systems, CS336 |
| [Shard optimizer ownership without changing the update](public/courses/course-sharded-state/exercise.json) | Training systems | DL Systems, CS336 |
| [Build a reversible byte-pair tokenizer](public/courses/course-byte-tokenizer/exercise.json) | Language-model pipeline | CS336 |
| [Deduplicate training data without leaking evaluation text](public/courses/course-data-curation/exercise.json) | Language-model pipeline | CS336, Alchemy |
| [Pack language-model batches and score only valid targets](public/courses/course-packed-evaluation/exercise.json) | Language-model pipeline | CS336, Alchemy |
| [Choose a training plan under a compute budget](public/courses/course-scaling-budget/exercise.json) | Empirical deep learning | CS336, Alchemy |
| [Audit a group-relative policy update](public/courses/course-policy-objective/exercise.json) | Language-model pipeline | CS336, Alchemy |
| [Compare experiments without changing the question](public/courses/course-controlled-sweeps/exercise.json) | Empirical deep learning | Alchemy, CS336 |
| [Investigate curvature, momentum and learning-rate stability](public/courses/course-loss-landscape/exercise.json) | Empirical deep learning | Alchemy, DL Systems |
| [Measure signal and sensitivity through a residual stack](public/courses/course-depth-signal/exercise.json) | Empirical deep learning | Alchemy, DL Systems |
| [Compare factorized output heads under a parameter budget](public/courses/course-head-capacity/exercise.json) | Empirical deep learning | Alchemy, CS336 |
| [Dispatch sparse experts without losing capacity accounting](public/courses/course-expert-capacity/exercise.json) | Empirical deep learning | Alchemy, CS336 |
| [Run a tiny training and transfer-learning experiment](public/courses/course-generalization/exercise.json) | Empirical deep learning | Alchemy, DL Systems |
| [Detect a reward specification that favors the wrong behavior](public/courses/course-reward-audit/exercise.json) | Empirical deep learning | Alchemy, CS336 |

## Course coverage

Material reviewed on 2026-11-20. DL Systems and Alchemy use the currently published course material. CS336 specifically uses the Spring 2025 offering. Some CS336 default-branch PDF names changed; the task references pin accessible versions of the original 2025 handouts.

### Deep Learning Systems

[Course source](https://dlsyscourse.org/lectures/)

Foundations, reverse-mode differentiation, tensor layouts, training state, convolution, hardware cost models, distributed state, sequence models and generative architectures.

New CPU-scale implementations fill the framework and training gaps. Existing CNN, RNN, Transformer, generative-model, inference and adaptation labs cover overlapping material. GPU timing and full framework construction remain extensions.

| Source unit | Practice tasks |
|---|---|
| Lectures 2–5 · classification, backpropagation and autodiff | Train a stable softmax classifier; Repair a reverse-mode differentiation engine |
| Lectures 6–9 · optimization and library state | Resume a correct AdamW training update; Make training and evaluation state agree; Run a tiny training and transfer-learning experiment |
| Lectures 10, 12–13 · array backends and acceleration | Implement tensor views and their backward scatter; Build an edge-safe tiled matrix operator; Blockwise attention |
| Lectures 11, 14 · convolution | Connect an image filter to backpropagation; ResNet · Route identity and projected shortcuts; VGG · Execute a configurable 3×3 stage stack; U-Net · Align a decoder with a cropped skip |
| Lectures 15–18 · sequence models and attention | Vanilla RNN · backpropagate through a delayed label; LSTM · carry hidden and cell state through padded streams; GRU · encode a chunked phrase with reset-before gating; Transformer · translate with encoder–decoder attention; GPT-2 · score a text continuation |
| Lecture 19 · large-model training | Recover training after a mixed-precision overflow; Fit an activation replay plan into a memory budget; Shard optimizer ownership without changing the update; Distributed data-parallel SGD |
| Lectures 20–21 · generative models | VAE · train with a reparameterized latent; GAN · alternate discriminator and generator updates; DDPM · corrupt a signal and run its reverse chain |
| Lectures 22–23 · adaptation and deployment | LoRA linear layer; Chat supervised fine-tuning; Quantized linear inference; Per-layer KV cache |

### Deep Learning Alchemy · CS312

[Course source](https://deep-learning-alchemy.github.io/)

Controlled sweeps, budget-dependent conclusions, optimizer trajectories, depth diagnostics, capacity constraints, small-data learning, transfer and reward specification.

Original prediction prompts use small reproducible experiments and saved notebooks. They do not reproduce the course answer keys or claim toy experiments establish the outcomes of its large training runs. Full µP/Muon studies, Hessian eigensolvers and GPU sweeps are extension reading.

| Source unit | Practice tasks |
|---|---|
| A1 · Basics; prediction quiz | Compare experiments without changing the question; Choose a training plan under a compute budget; Release comparison; Paired statistical comparison |
| A2 · Hyperparameter scaling; prediction quiz | Resume a correct AdamW training update; Investigate curvature, momentum and learning-rate stability; Run a tiny training and transfer-learning experiment |
| A3 · Optimization and loss landscapes; prediction quiz | Investigate curvature, momentum and learning-rate stability; Compare experiments without changing the question |
| A4 · Stability through depth and time; prediction quiz | Measure signal and sensitivity through a residual stack; Make training and evaluation state agree; ResNet · Route identity and projected shortcuts; Vanilla RNN · backpropagate through a delayed label; LSTM · carry hidden and cell state through padded streams; GRU · encode a chunked phrase with reset-before gating; Gated DeltaNet · implement the recurrent token mixer; Transformer · translate with encoder–decoder attention; GPT-2 · score a text continuation; LLaMA · assemble a rotary decoder block |
| A5 · Capacity; prediction quiz | Compare factorized output heads under a parameter budget; Dispatch sparse experts without losing capacity accounting; DeepSeek-V3 · decode from latent KV into sparse experts |
| A6 · Generalization and transfer | Run a tiny training and transfer-learning experiment; Deduplicate training data without leaking evaluation text; Pack language-model batches and score only valid targets |
| A7 · Reinforcement learning and reward design | Audit a group-relative policy update; Detect a reward specification that favors the wrong behavior |

### CS336 · Language Modeling from Scratch · Spring 2025

[Course source](https://cs336.stanford.edu/spring2025/)

Tokenization, numerical training components, model architecture, systems accounting, scaling, corpus preparation, likelihood evaluation, supervised adaptation and policy objectives.

The 2025 assignment handouts inspire scoped tasks with explicit contracts. Existing model and serving labs are reused. Real Triton kernels, distributed network timing, Common Crawl processing and large-model RL runs require local hardware/data beyond this browser runner.

| Source unit | Practice tasks |
|---|---|
| A1 · Basics | Build a reversible byte-pair tokenizer; Train a stable softmax classifier; Implement tensor views and their backward scatter; Resume a correct AdamW training update; Pack language-model batches and score only valid targets; Transformer · translate with encoder–decoder attention; GPT-2 · score a text continuation; LLaMA · assemble a rotary decoder block |
| A2 · Systems | Build an edge-safe tiled matrix operator; Recover training after a mixed-precision overflow; Fit an activation replay plan into a memory budget; Shard optimizer ownership without changing the update; Blockwise attention; Distributed data-parallel SGD |
| A3 · Scaling and architecture | Choose a training plan under a compute budget; Compare factorized output heads under a parameter budget; Dispatch sparse experts without losing capacity accounting; DeepSeek-V3 · decode from latent KV into sparse experts |
| A4 · Data | Deduplicate training data without leaking evaluation text; Build a reversible byte-pair tokenizer; Pack language-model batches and score only valid targets |
| A5 · Alignment and reasoning RL | Chat supervised fine-tuning; Audit a group-relative policy update; Detect a reward specification that favors the wrong behavior; DPO preference loss; LoRA linear layer |
| Inference and evaluation lectures | Per-layer KV cache; Live continuous batching; Paged KV-cache allocator; Shared-prefix caching; Speculative decoding; Quantized linear inference; Paired statistical comparison; LLM-judge validation; Release comparison |

## Assignment and quiz inspiration

- DL Systems HW0–HW4 supply the progression from numerical training through autodiff, library state, strided array backends and convolution/sequence models. See the [homework index](https://dlsyscourse.org/assignments/).
- Alchemy assignments and public prediction quizzes inform controlled comparisons, explicit interventions, matched experimental budgets, before/after predictions and explanations. Relevant handouts and quizzes are linked from each new lab.
- CS336 assignments inform tokenizer reversibility, numerical stability, workload accounting, corpus filtering, resource-constrained model choices and masked policy objectives. The new tasks define their own smaller contracts; they are not drop-in implementations for the official assignment adapters.

## Deliberate adaptations and omissions

- Existing Transformer, recurrent, convolutional, generative, inference, fine-tuning and evaluation exercises are reused rather than copied into parallel versions.
- GPU kernels, CUDA/Triton benchmarking, network communication timing and allocator profiling are represented only where a deterministic CPU model preserves the reasoning. The app does not claim an analytical estimate measures real hardware performance.
- Full Needle framework assembly, large language-model pretraining, Common Crawl ingestion, multi-GPU training, large DNA/physics experiments and model-based RL remain local extensions. Existing downloadable PyTorch projects provide a separate path for selected architectures.
- µP/Muon transfer experiments, full Hessian spectra, learned quality/language classifiers, special-token/regex tokenizer policies and other implementation-heavy details are not claimed as completed by a simpler related task. Follow the source assignments for those extensions.
- Administrative material, participation rules, live quizzes, project presentations and open-ended unpublished material are omitted.
- Course phrasing, code, datasets and answer keys are not bundled. The tasks, small fixtures, references and tests here are original.

## Rebuilding and validation

See [the authoring guide](authoring/courses/README.md). The generated files are committed so ordinary site builds do not need to run the Python generators.
