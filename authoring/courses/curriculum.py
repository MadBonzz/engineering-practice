"""Traceable overlap map. Reused tasks retain their original URLs and progress keys."""
import json
from common import DLSYS, ALCHEMY, CS336, A1, A2, A4, A5

REUSED={
 'release-comparison':['alchemy','cs336'],
 'kv-cache':['dlsys','cs336'], 'live-batching':['cs336'], 'paged-cache':['cs336'],
 'shared-prefix':['cs336'], 'speculative-decode':['cs336'], 'block-attention':['dlsys','cs336'],
 'quantized-linear':['dlsys','cs336'], 'chat-sft':['dlsys','cs336'], 'lora-layer':['dlsys','cs336'],
 'dpo-loss':['cs336'], 'data-parallel':['dlsys','cs336'],
 'paired-bootstrap':['alchemy','cs336'], 'judge-validation':['cs336'],
 'paper-transformer':['dlsys','alchemy','cs336'], 'paper-gpt2':['dlsys','alchemy','cs336'],
 'paper-llama':['alchemy','cs336'], 'paper-deepseek-v3':['alchemy','cs336'],
 'paper-resnet':['dlsys','alchemy'], 'paper-vgg':['dlsys'], 'paper-unet':['dlsys'],
 'paper-rnn':['dlsys','alchemy'], 'paper-lstm':['dlsys','alchemy'], 'paper-gru':['dlsys','alchemy'],
 'paper-gated-deltanet':['alchemy'], 'paper-vae':['dlsys'], 'paper-gan':['dlsys'], 'paper-ddpm':['dlsys'],
}

COURSES=[
 dict(id='dlsys',title='Deep Learning Systems',url=DLSYS,
      coverage='Foundations, reverse-mode differentiation, tensor layouts, training state, convolution, hardware cost models, distributed state, sequence models and generative architectures.',
      adaptations='New CPU-scale implementations fill the framework and training gaps. Existing CNN, RNN, Transformer, generative-model, inference and adaptation labs cover overlapping material. GPU timing and full framework construction remain extensions.',
      units=[
       dict(title='Lectures 2–5 · classification, backpropagation and autodiff',tasks=['course-classifier','course-autodiff']),
       dict(title='Lectures 6–9 · optimization and library state',tasks=['course-optimizer','course-normalization','course-generalization']),
       dict(title='Lectures 10, 12–13 · array backends and acceleration',tasks=['course-tensor-views','course-tiled-matmul','block-attention']),
       dict(title='Lectures 11, 14 · convolution',tasks=['course-convolution-backward','paper-resnet','paper-vgg','paper-unet']),
       dict(title='Lectures 15–18 · sequence models and attention',tasks=['paper-rnn','paper-lstm','paper-gru','paper-transformer','paper-gpt2']),
       dict(title='Lecture 19 · large-model training',tasks=['course-loss-scaling','course-checkpoint-planner','course-sharded-state','data-parallel']),
       dict(title='Lectures 20–21 · generative models',tasks=['paper-vae','paper-gan','paper-ddpm']),
       dict(title='Lectures 22–23 · adaptation and deployment',tasks=['lora-layer','chat-sft','quantized-linear','kv-cache'])]),
 dict(id='alchemy',title='Deep Learning Alchemy · CS312',url=ALCHEMY,
      coverage='Controlled sweeps, budget-dependent conclusions, optimizer trajectories, depth diagnostics, capacity constraints, small-data learning, transfer and reward specification.',
      adaptations='Original prediction prompts use small reproducible experiments and saved notebooks. They do not reproduce the course answer keys or claim toy experiments establish the outcomes of its large training runs. Full µP/Muon studies, Hessian eigensolvers and GPU sweeps are extension reading.',
      units=[
       dict(title='A1 · Basics; prediction quiz',tasks=['course-controlled-sweeps','course-scaling-budget','release-comparison','paired-bootstrap']),
       dict(title='A2 · Hyperparameter scaling; prediction quiz',tasks=['course-optimizer','course-loss-landscape','course-generalization']),
       dict(title='A3 · Optimization and loss landscapes; prediction quiz',tasks=['course-loss-landscape','course-controlled-sweeps']),
       dict(title='A4 · Stability through depth and time; prediction quiz',tasks=['course-depth-signal','course-normalization','paper-resnet','paper-rnn','paper-lstm','paper-gru','paper-gated-deltanet','paper-transformer','paper-gpt2','paper-llama']),
       dict(title='A5 · Capacity; prediction quiz',tasks=['course-head-capacity','course-expert-capacity','paper-deepseek-v3']),
       dict(title='A6 · Generalization and transfer',tasks=['course-generalization','course-data-curation','course-packed-evaluation']),
       dict(title='A7 · Reinforcement learning and reward design',tasks=['course-policy-objective','course-reward-audit'])]),
 dict(id='cs336',title='CS336 · Language Modeling from Scratch · Spring 2025',url=CS336,
      coverage='Tokenization, numerical training components, model architecture, systems accounting, scaling, corpus preparation, likelihood evaluation, supervised adaptation and policy objectives.',
      adaptations='The 2025 assignment handouts inspire scoped tasks with explicit contracts. Existing model and serving labs are reused. Real Triton kernels, distributed network timing, Common Crawl processing and large-model RL runs require local hardware/data beyond this browser runner.',
      units=[
       dict(title='A1 · Basics',url=A1,tasks=['course-byte-tokenizer','course-classifier','course-tensor-views','course-optimizer','course-packed-evaluation','paper-transformer','paper-gpt2','paper-llama']),
       dict(title='A2 · Systems',url=A2,tasks=['course-tiled-matmul','course-loss-scaling','course-checkpoint-planner','course-sharded-state','block-attention','data-parallel']),
       dict(title='A3 · Scaling and architecture',url=CS336,tasks=['course-scaling-budget','course-head-capacity','course-expert-capacity','paper-deepseek-v3']),
       dict(title='A4 · Data',url=A4,tasks=['course-data-curation','course-byte-tokenizer','course-packed-evaluation']),
       dict(title='A5 · Alignment and reasoning RL',url=A5,tasks=['chat-sft','course-policy-objective','course-reward-audit','dpo-loss','lora-layer']),
       dict(title='Inference and evaluation lectures',url=CS336,tasks=['kv-cache','live-batching','paged-cache','shared-prefix','speculative-decode','quantized-linear','paired-bootstrap','judge-validation','release-comparison'])]),
]


def write_curriculum(root,new_catalog):
    public=root/'public'
    existing={r['slug']:r for name in ['laboratory','paper-catalog','systems-catalog'] for r in json.loads((public/(name+'.json')).read_text())}
    reused=[dict(existing[slug],courses=courses,kind='existing',order=len(new_catalog)+i+1) for i,(slug,courses) in enumerate(REUSED.items())]
    combined=new_catalog+reused
    assert len({x['slug'] for x in combined})==len(combined)
    all_tasks={x['slug']:x for x in combined}
    for course in COURSES:
        for unit in course['units']:
            for slug in unit['tasks']:
                assert slug in all_tasks,(course['id'],slug)
                assert course['id'] in all_tasks[slug]['courses'],(course['id'],slug,'missing course tag')
    for lab in new_catalog:
        for slug in lab['prerequisites']:assert slug in all_tasks,(lab['slug'],slug)
    (public/'course-catalog.json').write_text(json.dumps(combined,separators=(',',':')))
    (public/'course-map.json').write_text(json.dumps(dict(reviewed='2026-11-20',courses=COURSES,new_tasks=len(new_catalog),reused_tasks=len(reused)),indent=2)+'\n')
    return reused
