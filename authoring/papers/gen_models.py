"""Original small-scale paper labs with independent public examples."""
from math import log, sqrt

PAPERS = []


def part(title, file, funcs, requirement, *hints):
    return dict(title=title, file=file, funcs=funcs.split(), requirement=requirement, hints=list(hints))


def ex(request, result, why):
    return {"input": request, "output": result, "why": why}


def check(part_number, function, args, name):
    return dict(part=part_number, fn=function, args=args, title=name)


def source(title, paper):
    return [{"title":title,"url":"https://arxiv.org/abs/"+paper}]


def integrations(number, requests):
    return [check(number,"gateway.handle",[request],label) for label,request in requests]


VAE = {
"posterior.py": '''import math


def sample_latent(x, mean_weight, variance_weight, epsilon):
    """Return [mean, log_variance, z] for one scalar x and supplied Gaussian draw."""
    mean = mean_weight * x
    log_variance = variance_weight * x
    z = mean + math.exp(0.5 * log_variance) * epsilon
    return [mean, log_variance, z]
''',
"bound.py": '''import math


def loss_terms(x, z, mean, log_variance, decoder_weight):
    """Return [Bernoulli reconstruction NLL, KL(q||N(0,1))] in nats."""
    logit = decoder_weight * z
    recon = max(logit, 0.0) - x * logit + math.log1p(math.exp(-abs(logit)))
    kl = 0.5 * (mean * mean + math.expm1(log_variance) - log_variance)
    return [recon, kl]
''',
"updates.py": '''import math


def update(x, epsilon, weights, posterior, lr):
    """Return all three encoder/decoder weights after one simultaneous SGD update."""
    mean, log_variance, z = posterior
    decoder_weight = weights[2]
    logit = decoder_weight * z
    if logit >= 0:
        probability = 1.0 / (1.0 + math.exp(-logit))
    else:
        exp_logit = math.exp(logit)
        probability = exp_logit / (1.0 + exp_logit)
    residual = probability - x
    d_z = residual * decoder_weight
    d_mean = d_z + mean
    d_log_variance = 0.5 * (d_z * math.exp(0.5*log_variance) * epsilon + math.expm1(log_variance))
    grads = [d_mean * x, d_log_variance * x, residual * z]
    return [w-lr*g for w,g in zip(weights,grads)]
''',
"gateway.py": '''from posterior import sample_latent
from bound import loss_terms
from updates import update


def handle(request):
    x, eps, weights = request["x"], request["epsilon"], request["weights"]
    posterior = sample_latent(x, weights[0], weights[1], eps)
    mean, lv, z = posterior
    recon, kl = loss_terms(x, z, mean, lv, weights[2])
    return {"z":z, "reconstruction":recon, "kl":kl,
            "weights":update(x, eps, weights, posterior, request["lr"])}
'''}

PAPERS.append(dict(slug="paper-vae", title="VAE · train with a reparameterized latent", group="Generative models", level="Intermediate", duration="40–55 min",
mission="A toy binary sensor autoencoder samples a latent before predicting the sensor reading. Wire its encoder, likelihood and optimizer together so a reconstruction error can update both the decoder and the approximate posterior.",
inputs="x is one binary observation (0 or 1). weights is [mean_weight, variance_weight, decoder_weight]. epsilon is a fixed standard-normal draw supplied by the caller and lr is a nonnegative learning rate.",
outputs="Return the sampled scalar z, reconstruction NLL and analytic kl before the update, plus the three weights after one step. All logarithms are natural.",
constraints="The encoder has no biases. Its two heads produce mean = mean_weight*x and log_variance = variance_weight*x. The prior is a standard normal. The decoder Bernoulli logit is decoder_weight*z. Use simultaneous unconstrained SGD with KL coefficient one. Log-variance is in [−10,10]. Logits may have magnitude 1000.",
scope="Train a complete one-observation scalar VAE using a linear encoder, Bernoulli decoder, analytic Gaussian KL and pathwise gradients. The fixed noise input makes the stochastic update repeatable. Larger image networks and sampled multi-example training are outside this lab.",
source=source("Kingma and Welling · Auto-Encoding Variational Bayes","1312.6114"),modules=VAE,
parts=[part("Sample a differentiable latent","posterior.py","sample_latent","Compute the two encoder heads and z = mean + exp(log_variance/2) × epsilon.","The encoder reports log variance.","The noise scale is a standard deviation."),
part("Evaluate the variational bound","bound.py","loss_terms","Compute softplus(logit) − x·logit and ½(mean² + exp(log_variance) − 1 − log_variance). Return each term separately.","A decoder output of zero gives log(2) nats.","Use a stable softplus expression at extreme logits."),
part("Send gradients through the sample","updates.py","update","Differentiate reconstruction + KL at the supplied posterior. Return updated weights in their input order using only pre-update values.","The decoder-logit gradient is sigmoid(logit) − x.","The derivative of z with respect to log variance is ε·exp(log_variance/2)/2.")],
examples=[ex(dict(x=1,epsilon=0,weights=[0,0,1],lr=1),dict(z=0,reconstruction=log(2),kl=0,weights=[0.5,0,1]),"A zero sample yields probability 0.5. The decoder weight sees zero input, while the mean head receives gradient −0.5."),
ex(dict(x=0,epsilon=2,weights=[0,0,0],lr=0.25),dict(z=2,reconstruction=log(2),kl=0,weights=[0,0,-0.25]),"With x equal to zero the encoder heads receive no weight gradient. The decoder gradient is 0.5 × 2 = 1.")],
checks=[check(1,"posterior.sample_latent",[1,-2,log(4),1.5],"Use standard deviation to scale noise"),check(1,"posterior.sample_latent",[0,2,3,-2],"Zero observation still samples the prior"),check(2,"bound.loss_terms",[1,1000,1,0,-1],"Large negative decoder logit"),check(2,"bound.loss_terms",[0,-3,-0.4,0.7,2],"Nonsymmetric posterior KL"),check(3,"updates.update",[1,1,[2,0,1],[2,0,3],0.2],"Use a shared pre-update posterior"),check(3,"updates.update",[0,-1,[0,0,1000],[0,0,-1],0.5],"Stable sigmoid at a saturated logit")] +integrations(3,[("A shifted posterior",dict(x=1,epsilon=-0.4,weights=[0.8,-0.5,2],lr=0.06)),("A frozen optimizer",dict(x=1,epsilon=0.3,weights=[-0.7,0.2,1],lr=0)),("A prior draw for a negative observation",dict(x=0,epsilon=-1.5,weights=[2,-1,3],lr=.2))])))


GAN = {
"layers.py": '''import math


def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return e / (1.0 + e)


def synthesize(noise, generator):
    return [generator[0]*z + generator[1] for z in noise]
''',
"critic.py": '''from layers import sigmoid


def discriminator_step(real, fake, discriminator, lr):
    """Return updated [slope,bias] for mean real BCE plus mean fake BCE."""
    slope, bias = discriminator
    grad_w = grad_b = 0.0
    for batch, label in ((real,1), (fake,0)):
        for value in batch:
            derivative = (sigmoid(slope*value+bias)-label)/len(batch)
            grad_w += derivative * value
            grad_b += derivative
    return [slope-lr*grad_w, bias-lr*grad_b]
''',
"producer.py": '''from layers import sigmoid


def generator_step(noise, generator, discriminator, lr):
    """Return updated [slope,bias] under the mean non-saturating generator loss."""
    g, c = generator
    d, b = discriminator
    grad_g = grad_c = 0.0
    for z in noise:
        fake = g*z+c
        grad_fake = (sigmoid(d*fake+b)-1.0)*d/len(noise)
        grad_g += grad_fake*z
        grad_c += grad_fake
    return [g-lr*grad_g, c-lr*grad_c]
''',
"gateway.py": '''from layers import synthesize
from critic import discriminator_step
from producer import generator_step


def handle(request):
    g, d = request["generator"], request["discriminator"]
    for batch in request["rounds"]:
        noise = batch["noise"]
        d = discriminator_step(batch["real"], synthesize(noise, g), d, request["lr_d"])
        g = generator_step(noise, g, d, request["lr_g"])
    return {"generator":g,"discriminator":d,"samples":synthesize(request["probe"],g)}
'''}

def adversary(g,d,lr_d,lr_g,rounds,probe):
    return dict(generator=g,discriminator=d,lr_d=lr_d,lr_g=lr_g,rounds=rounds,probe=probe)

PAPERS.append(dict(slug="paper-gan",title="GAN · alternate discriminator and generator updates",group="Generative models",level="Intermediate",duration="40–55 min",
mission="A one-dimensional data generator is frozen whenever the discriminator trains. Implement both optimizers so a complete round moves the generator toward examples the newly updated discriminator would call real.",
inputs="generator and discriminator are [slope,bias] pairs. Each entry in rounds holds a nonempty real batch and a nonempty noise batch. lr_d and lr_g are independent learning rates. probe lists noise values for a final synthesis.",
outputs="Return both final parameter pairs and samples generated from probe after all rounds. An empty list of rounds returns the original model predictions.",
constraints="G(z)=g*z+c and D(v)=sigmoid(d*v+b). Minimize the sum of mean binary cross-entropies for real and fake batches in the discriminator step. Use −mean(log D(G(z))) for the generator step. Reuse that round’s noise and the updated discriminator. Gradients within either step are simultaneous.",
scope="Run the complete alternating, non-saturating GAN training loop on one-dimensional affine models. Fixed noise batches expose update order, class weighting and saturation without requiring image data.",
source=source("Goodfellow et al. · Generative Adversarial Networks","1406.2661"),modules=GAN,
parts=[part("Separate the discriminator's two populations","critic.py","discriminator_step","Compute the discriminator gradient for real labels 1 and fake labels 0. Average each population separately before summing the two gradients.","For each logit the BCE derivative is p − label.","Unequal batch sizes should still give equal population weighting."),
part("Backpropagate through a frozen discriminator","producer.py","generator_step","Differentiate the non-saturating generator loss through the fixed discriminator slope. Return new generator parameters only.","The generator treats each synthetic sample as a target-1 prediction.","A zero discriminator slope blocks every generator gradient.")],
examples=[ex(adversary([0,0],[0,0],1,1,[dict(real=[2],noise=[0])],[-1,2]),dict(generator=[0,0.5],discriminator=[1,0],samples=[0.5,0.5]),"The discriminator starts at probability 0.5 and learns slope 1. With that new slope the generator bias moves to 0.5. Noise zero gives the generator slope no gradient."),
ex(adversary([2,-1],[0,0],1,1,[dict(real=[1,1],noise=[1])],[0,3]),dict(generator=[2,-1],discriminator=[0,0],samples=[-1,5]),"Every real and fake sample equals 1. The population-mean discriminator gradients cancel, so the zero discriminator slope leaves the generator unchanged.")],
checks=[check(1,"critic.discriminator_step",[[1,-2],[0],[0.3,-0.1],0.2],"Unequal population sizes"),check(1,"critic.discriminator_step",[[-1],[2],[1000,0],1],"Saturated discriminator logits"),
check(2,"producer.generator_step",[[1,-3],[0.2,1],[-0.7,0.4],0.3],"Non-saturating gradient sign"),check(2,"producer.generator_step",[[4],[1,-1],[0,2],0.9],"Frozen flat discriminator")] +integrations(2,[("Zero training rounds",adversary([2,3],[-1,0],1,1,[],[-1,2])),("Successive optimizer states",adversary([.4,-.1],[.2,.3],.2,.1,[dict(real=[1,2],noise=[0,-1]),dict(real=[3],noise=[1,2,0])],[2])),("Discriminator-only training",adversary([1,0],[-.5,0],.1,0,[dict(real=[1],noise=[-1,3])],[]))])))

DDPM = {
"noise_path.py": '''import math


def products(betas):
    """Return length T+1 with alpha_bar[0]=1 and cumulative (1-beta) thereafter."""
    history = [1.0]
    for beta in betas:
        history.append(history[-1]*(1.0-beta))
    return history


def corrupt(clean, epsilon, alpha_bar):
    """Return a noisy sample from clean and epsilon vectors at one signal fraction."""
    signal, noise = math.sqrt(alpha_bar), math.sqrt(1.0-alpha_bar)
    return [signal*x+noise*e for x,e in zip(clean,epsilon)]
''',
"reverse_path.py": '''import math


def reverse_step(current, predicted_noise, noise, beta, previous_bar, current_bar):
    """Return one DDPM posterior-variance step. Inputs are aligned vectors."""
    scale = beta / math.sqrt(1.0-current_bar)
    variance = beta * (1.0-previous_bar)/(1.0-current_bar)
    return [(value-scale*eps)/math.sqrt(1.0-beta)+math.sqrt(variance)*draw
            for value,eps,draw in zip(current,predicted_noise,noise)]
''',
"noise_net.py": '''def predict(state, coefficients):
    """A tiny time-conditioned affine noise model for the exercise."""
    weight, bias = coefficients
    return [weight*x+bias for x in state]
''',
"gateway.py": '''from noise_path import products, corrupt
from noise_net import predict
from reverse_path import reverse_step


def handle(request):
    betas, net = request["betas"], request["denoiser"]
    bars = products(betas)
    if request["mode"] == "loss":
        t, noise = request["t"], request["epsilon"]
        noisy = corrupt(request["clean"], noise, bars[t])
        predicted = predict(noisy,net[t-1])
        return {"noisy":noisy,"mse":sum((a-b)**2 for a,b in zip(noise,predicted))/len(noise)}
    state = request["terminal"][:]
    trajectory = []
    for t in range(len(betas),0,-1):
        eps = predict(state, net[t-1])
        state = reverse_step(state, eps, request["draws"][t-1], betas[t-1], bars[t-1], bars[t])
        trajectory.append(state)
    return {"trajectory":trajectory,"sample":state}
'''}

def denoise(betas,net,terminal,draws):
    return dict(mode="sample",betas=betas,denoiser=net,terminal=terminal,draws=draws)


def noise_loss(betas,net,t,clean,epsilon):
    return dict(mode="loss",betas=betas,denoiser=net,t=t,clean=clean,epsilon=epsilon)

PAPERS.append(dict(slug="paper-ddpm",title="DDPM · corrupt a signal and run its reverse chain",group="Generative models",level="Advanced",duration="45–65 min",
mission="A diffusion experiment uses different noise schedules for training and sampling. Build both sides from a single schedule so the loss probe and the reverse sampler agree on what timestep t means.",
inputs="betas lists T step variances and denoiser has T [weight,bias] pairs, indexed from t=1. In loss mode supply the clean vector, epsilon vector and t. In sample mode supply terminal x_T and draws[t−1] for each reverse step.",
outputs="Loss mode returns noisy x_t and the mean squared error between the predicted and supplied noise. Sample mode returns the trajectory [x_(T−1), …, x_0] and final sample. Each state is a vector.",
constraints="T ≥ 1 and every beta lies strictly between zero and one. Use alpha_t=1−beta_t and bar_t=product(alpha_1…alpha_t), with bar_0=1. The reverse step uses posterior variance beta_t(1−bar_(t−1))/(1−bar_t). All vectors are nonempty and equally sized.",
scope="Implement DDPM’s closed-form corruption, epsilon-prediction loss path, and full ancestral sampler using the posterior variance choice. A supplied per-timestep affine denoiser replaces the learned image U-Net. Noise draws are explicit fixtures.",
source=source("Ho et al. · Denoising Diffusion Probabilistic Models","2006.11239"),modules=DDPM,
parts=[part("Construct the training-time noisy observation","noise_path.py","products corrupt","Build the cumulative signal schedule with an explicit t=0 sentinel. Form x_t = sqrt(bar_t)x_0 + sqrt(1−bar_t)ε.","A scalar beta describes only one step.","The schedule stores squared amplitudes, so take square roots before mixing."),
part("Propagate the reverse posterior","reverse_path.py","reverse_step","Compute (x_t − beta_t·epsilon_hat/sqrt(1−bar_t))/sqrt(1−beta_t) plus the supplied draw times the posterior standard deviation. The final step adds zero noise.","Use the current and previous cumulative products in the variance.","Set previous_bar to 1 for the last step and inspect the variance numerator.")],
examples=[ex(noise_loss([.75],[[2,0]],1,[2],[0]),dict(noisy=[1],mse=4),"The signal amplitude is sqrt(0.25) = 0.5. The denoiser predicts 2 on the resulting state of 1, so the squared error relative to zero is 4."),
ex(denoise([.75,.75],[[0,0],[0,0]],[0],[[999],[1]]),dict(trajectory=[[sqrt(.6)],[2*sqrt(.6)]],sample=[2*sqrt(.6)]),"At t=2 the posterior variance is 0.75 × 0.75 / 0.9375 = 0.6. At t=1 the reverse mean doubles the state and the draw 999 has zero weight.")],
checks=[check(1,"noise_path.products",[[.1,.2,.3]],"Multiply independent retention fractions"),check(1,"noise_path.corrupt",[[3,-2],[4,5],.64],"Mix vector signal and noise"),check(2,"reverse_path.reverse_step",[[1,-2],[.5,0],[9,-1],.2,1,.8],"Final step suppresses sampling noise"),check(2,"reverse_path.reverse_step",[[1],[.7],[-1],.4,.75,.45],"Evaluate posterior mean and variance")] +integrations(2,[("Two-coordinate training loss",noise_loss([.2,.3],[[1,0],[-.25,.2]],2,[1,-1],[.4,.7])),("Three-step reversed traversal",denoise([.2,.3,.1],[[0,1],[.4,-.1],[.3,.2]],[1,-2],[[3,4],[1,-1],[.2,-.5]])),("Training at the earliest step",noise_loss([.36,.5],[[1,-1],[0,2]],1,[0,2],[-1,0]))])))


LLADA = {
"token_oracle.py": '''import math


def scores(tokens, bias, context):
    """Fixture denoiser. Each visible token contributes its row to every position."""
    vocab = len(context)
    signal = [sum(context[t][k] for t in tokens if t != -1) for k in range(vocab)]
    return [[b+s for b,s in zip(row,signal)] for row in bias]


def logsumexp(logits):
    peak = max(logits)
    return peak + math.log(sum(math.exp(x-peak) for x in logits))
''',
"masked_objective.py": '''from token_oracle import scores, logsumexp


def corruption_loss(clean, draws, t, bias, context):
    """Return masked IDs and inverse-t weighted NLL per original token."""
    masked = [-1 if u<t else token for token,u in zip(clean,draws)]
    logits = scores(masked,bias,context)
    total = sum(logsumexp(row)-row[target] for token,target,row in zip(masked,clean,logits) if token==-1)
    return {"masked":masked,"loss":total/(t*len(clean))}
''',
"unmask.py": '''import math
from token_oracle import scores, logsumexp


def commit(tokens, logits, quota):
    """Return a copy with up to quota masked positions frozen by predicted probability."""
    ranked = []
    for position, (token,row) in enumerate(zip(tokens,logits)):
        if token == -1:
            best = max(range(len(row)),key=row.__getitem__)
            confidence = row[best] - logsumexp(row)
            ranked.append((-confidence,position,best))
    result = tokens[:]
    for _,position,new_token in sorted(ranked)[:quota]:
        result[position] = new_token
    return result


def generate(tokens, steps, bias, context):
    """Return tokens and the post-commit trace for exactly steps parallel rounds."""
    count, extra = divmod(tokens.count(-1),steps)
    trace = []
    state = tokens[:]
    for i in range(steps):
        state = commit(state, scores(state,bias,context), count+int(i<extra))
        trace.append(state)
    return {"tokens":state,"trace":trace}
''',
"gateway.py": '''from masked_objective import corruption_loss
from unmask import generate


def handle(request):
    if request["mode"] == "loss":
        return corruption_loss(request["tokens"],request["draws"],request["t"],request["bias"],request["context"])
    return generate(request["tokens"],request["steps"],request["bias"],request["context"])
'''}

def fill(tokens, steps, bias, context):
    return dict(mode="generate",tokens=tokens,steps=steps,bias=bias,context=context)


def masked(tokens,draws,t,bias,context):
    return dict(mode="loss",tokens=tokens,draws=draws,t=t,bias=bias,context=context)

PAPERS.append(dict(slug="paper-llada",title="LLaDA · learn and decode by unmasking",group="Generative models",level="Advanced",duration="50–70 min",
mission="An editor asks a discrete diffusion model to fill several holes in one sentence. Implement a training loss for randomly masked tokens, then let the decoder commit its most confident predictions and reconsider the rest.",
inputs="tokens contains vocabulary IDs and, for generation, −1 holes. bias is one row of logits per position. context is a vocabulary-by-vocabulary contribution table used by the supplied denoiser. Loss mode also receives t and one uniform draw per token. Generate mode receives a positive steps count.",
outputs="Loss mode returns masked tokens and inverse-t-weighted loss. Generate mode returns the completed tokens and a trace of the entire sequence after every update, including rounds with quota zero.",
constraints="Clean sequences are nonempty. Mask when draw < t with 0 < t ≤ 1. Divide the sum of masked cross-entropies by t times the original length. Predict greedily, breaking token ties toward smaller IDs. Rank only masked positions by their best-token softmax probability, breaking ties toward the earlier position. Split the initial mask count evenly across rounds, assigning remainder slots to earlier rounds.",
scope="Implement the masking objective and deterministic, confidence-based reverse sampler for one whole-sequence block. The provided bidirectional logit fixture replaces LLaDA’s learned Transformer. Temperature sampling, classifier-free guidance and large-scale training are omitted.",
source=source("Nie et al. · Large Language Diffusion Models","2502.09992")+[dict(title="LLaDA authors · generation code",url="https://github.com/ML-GSAI/LLaDA/blob/main/generate.py")],modules=LLADA,
parts=[part("Score only corrupted positions","masked_objective.py","corruption_loss","Build a corrupted copy, run the supplied scores on that copy and compute cross-entropy against the original IDs at masked positions. Apply the specified normalization.","Retain target IDs before corruption.","The denominator uses the clean length even when a single token is masked."),
part("Resolve high-confidence holes in parallel","unmask.py","commit generate","Compute all proposals from the current sequence before committing any position. Use normalized log probability or probability to rank proposals across positions. Recompute scores after every round.","Two equal maximum logits can correspond to different confidences.","If there are three holes and two steps, commit two positions first.")],
examples=[ex(fill([1,-1,-1],2,[[0,0],[1,0],[0,3]],[[0,0],[0,1]]),dict(tokens=[1,1,1],trace=[[1,-1,1],[1,1,1]]),"The prompt adds one to the logit for token 1. The last position is most confident and commits first. Its visible 1 then changes the middle logits to [1,2]."),
ex(masked([0,1],[.45,.72],.5,[[0,0],[0,0]],[[0,0],[0,0]]),dict(masked=[-1,1],loss=log(2)),"Only position 0 is hidden. Both vocabulary logits are equal and t times the length equals 1, giving log(2) nats.")],
checks=[check(1,"masked_objective.corruption_loss",[[1,0],[.5,.9],.5,[[0,1],[2,0]],[[0,0],[0,0]]],"Strict masking boundary and zero selected"),check(1,"masked_objective.corruption_loss",[[1,0],[0,.7],1,[[1000,-1000],[0,0]],[[4,-1],[1,2]]],"Stable cross-entropy for extreme logits"),check(2,"unmask.commit",[[-1,-1,1],[[99,99],[0,2],[1,0]],1],"Compare normalized confidences across positions"),check(2,"unmask.generate",[[-1,-1,-1],2,[[0,0],[0,0],[0,0]],[[0,0],[0,0]]],"Assign the remainder and ties to early positions")]+integrations(2,[("A fully specified prompt",fill([0,1],2,[[1,2],[0,0]],[[0,0],[0,1]])),("More rounds than holes",fill([-1,0],3,[[0,0],[1,2]],[[3,0],[0,0]])),("Train with visible supporting context",masked([2,1,0],[.2,.7,.1],.4,[[0,0,0]]*3,[[1,0,0],[0,2,1],[0,0,2]]))])))

TRM = {
"shared_cell.py": '''def network(values, weights):
    """Compute W_out · ReLU(W_in · values) using row-major matrices without biases."""
    hidden = [max(0.0, sum(w*v for w,v in zip(row,values))) for row in weights["in"]]
    return [sum(w*h for w,h in zip(row,hidden)) for row in weights["out"]]
''',
"recursion.py": '''from shared_cell import network


def cycle(x, y, z, n, weights):
    """Return [new_answer, new_latent] after n latent updates and one answer update."""
    for _ in range(n):
        z = network([a+b+c for a,b,c in zip(x,y,z)],weights)
    y = network([b+c for b,c in zip(y,z)],weights)
    return [y,z]
''',
"reasoner.py": '''from recursion import cycle


def infer(x, y, z, n, cycles, max_steps, weights, halt):
    """Return last states and one prediction per outer step, stopping for q>0."""
    predictions = []
    for _ in range(max_steps):
        for _ in range(cycles):
            y,z = cycle(x,y,z,n,weights)
        predictions.append(max(range(len(y)),key=y.__getitem__))
        logit = sum(w*v for w,v in zip(halt[:-1],y))+halt[-1]
        if logit > 0:
            break
    return {"answer":y,"latent":z,"predictions":predictions}
''',
"gateway.py": '''from reasoner import infer


def handle(request):
    return infer(request["x"],request["y"],request["z"],request["n"],request["cycles"],
                 request["max_steps"],request["cell"],request["halt"])
'''}

IDENTITY2={"in":[[1,0],[0,1]],"out":[[1,0],[0,1]]}
IDENTITY1={"in":[[1]],"out":[[1]]}
def recur(x,y,z,n,c,s,net,halt):
    return dict(x=x,y=y,z=z,n=n,cycles=c,max_steps=s,cell=net,halt=halt)

PAPERS.append(dict(slug="paper-trm",title="Tiny Recursive Model · refine latent and answer states",group="Sequence models and embeddings",level="Advanced",duration="40–60 min",
mission="A reasoning service can spend several cheap iterations improving a tentative answer before emitting it. Implement its shared two-layer cell, its two update rates and a halting controller. A trace of predictions should reveal exactly when a new answer became available.",
inputs="x, y and z are equal-width vectors representing input, initial answer and initial latent. cell.in maps model width to hidden width and cell.out maps back. n sets inner latent updates, cycles sets complete cycles per prediction, and max_steps bounds the number of predictions. halt contains one coefficient per answer dimension plus a final bias.",
outputs="Return final answer and latent vectors, and predictions containing the argmax class for each completed outer step. Ties use the lowest class index.",
constraints="All iteration counts are positive. Define F(v)=W_out·ReLU(W_in·v). Within each cycle keep y fixed for n updates z←F(x+y+z), then update y←F(y+z) exactly once. Reuse the same F each time. After cycles such cycles compute the class and halt if dot(halt_weights,y)+bias > 0. Continue from the current states.",
scope="Reproduce TRM’s inference-time latent/answer update schedule with a shared two-layer dense cell, an identity class head and a linear halt head. Use small numeric vectors in place of puzzle embeddings. Deep supervision and training-time gradient truncation are outside this inference lab.",
source=source("Jolicoeur-Martineau · Less is More: Recursive Reasoning with Tiny Networks","2510.04871"),modules=TRM,
parts=[part("Build the reusable network","shared_cell.py","network","Accept rectangular matrices and return one output per row of W_out. Apply ReLU only between the two products.","W_in and W_out use rows for output features.","A negative output is valid when its weight is negative."),
part("Separate reasoning from answer refinement","recursion.py","cycle","Advance z exactly n times before updating y. Return the new pair without mutating input vectors.","Include the input embedding in every latent update.","Only the latent update uses x."),
part("Control inference compute","reasoner.py","infer","Run the prescribed cycles, decode one class, then test halting. Return both final states and every emitted class.","The halt head is evaluated at an outer-step boundary.","A zero logit keeps the computation running.")],
examples=[ex(recur([1,0],[0,1],[0,0],2,1,3,IDENTITY2,[.5,0,-2]),dict(answer=[10,11],latent=[8,8],predictions=[1,1]),"The first cycle visits z=[1,1] and [2,2], and produces y=[2,3]. Its halt logit is −1. The next cycle produces y=[10,11] and a positive halt logit of 3."),
ex(recur([-1],[0],[0],1,1,2,IDENTITY1,[1,0]),dict(answer=[0],latent=[0],predictions=[0,0]),"ReLU zeros the negative input, leaving the answer unchanged and the halt logit exactly zero. The step budget is exhausted.")],
checks=[check(1,"shared_cell.network",[[2,-1],{"in":[[1,0],[0,1],[1,1]],"out":[[-1,3,0],[1,0,2]]}],"Unequal hidden width and signed output"),check(1,"shared_cell.network",[[-3],IDENTITY1],"Hidden ReLU blocks negative preactivation"),
check(2,"recursion.cycle",[[1],[2],[3],2,IDENTITY1],"Hold the answer during latent iterations"),check(2,"recursion.cycle",[[2,-1],[0,1],[1,0],1,IDENTITY2],"Input enters only latent recurrence"),
check(3,"reasoner.infer",[[1],[0],[0],1,2,3,IDENTITY1,[0,1]],"Evaluate halt only after all inner cycles"),check(3,"reasoner.infer",[[0,0],[1,1],[0,0],1,1,1,IDENTITY2,[0,0,-1]],"Argmax tie selects the first class")]+integrations(3,[("First decision halts",recur([2],[0],[1],1,1,4,IDENTITY1,[0,2])),("Cycle state survives supervision",recur([1],[0],[0],1,2,2,IDENTITY1,[0,-1])),("A shared channel swap",recur([1,-1],[0,1],[1,0],2,1,2,{"in":[[0,1],[1,0]],"out":[[1,0],[0,.5]]},[0,1,-1]))])))


BERT = {
"corruption.py": '''def corrupt(tokens, selections, mask_id):
    """Return altered IDs and original target IDs following each supplied selection."""
    encoded = tokens[:]
    targets = []
    for position, draw, random_token in selections:
        targets.append(tokens[position])
        if draw < 0.8:
            encoded[position] = mask_id
        elif draw < 0.9:
            encoded[position] = random_token
    return [encoded,targets]
''',
"attention_core.py": '''import math


def attend(rows, valid):
    """Return identity-QKV bidirectional scaled attention for the first valid rows."""
    active = rows[:valid]
    width = len(rows[0])
    outputs = []
    for query in active:
        scores = [sum(a*b for a,b in zip(query,key))/math.sqrt(width) for key in active]
        peak = max(scores)
        masses = [math.exp(s-peak) for s in scores]
        total = sum(masses)
        outputs.append([sum(mass*v[j] for mass,v in zip(masses,active))/total for j in range(width)])
    return outputs
''',
"tiny_block.py": '''import math
from attention_core import attend


def norm(v):
    mean = sum(v)/len(v)
    variance = sum((x-mean)**2 for x in v)/len(v)
    return [(x-mean)/math.sqrt(variance+1e-12) for x in v]


def gelu(x):
    return 0.5*x*(1+math.erf(x/math.sqrt(2)))


def encode(rows, valid):
    rows = [norm(v) for v in rows]
    context = attend(rows, valid)
    attention = [norm([x+c for x,c in zip(v,ctx)]) for v,ctx in zip(rows,context)]
    return [norm([x+gelu(x) for x in v]) for v in attention]
''',
"token_head.py": '''import math
from tiny_block import gelu, norm


def predict(hidden, table, positions, targets):
    """Return tied-vocabulary ID predictions and mean selected-position NLL."""
    loss = 0.0
    predictions = []
    for pos,target in zip(positions,targets):
        projected = norm([gelu(v) for v in hidden[pos]])
        logits = [sum(a*b for a,b in zip(projected,word)) for word in table]
        best = max(range(len(logits)),key=logits.__getitem__)
        predictions.append(best)
        peak = logits[best]
        loss += (peak-logits[target]) + math.log(sum(math.exp(v-peak) for v in logits))
    return [predictions,loss/len(targets) if targets else 0.0]
''',
"gateway.py": '''from corruption import corrupt
from tiny_block import encode
from token_head import predict


def handle(request):
    table, tokens = request["table"], request["tokens"]
    selected = request["selections"]
    altered,targets = corrupt(tokens,selected,len(table)-1)
    zeros = [0.0]*len(table[0])
    pos = request.get("positions",[zeros]*len(tokens))
    types = request.get("types",[zeros]*len(tokens))
    rows = [[a+b+c for a,b,c in zip(table[t],p,s)] for t,p,s in zip(altered,pos,types)]
    hidden = encode(rows,request["valid"])
    guesses,loss = predict(hidden,table,[s[0] for s in selected],targets)
    return {"input_ids":altered,"predictions":guesses,"mlm_loss":loss}
'''}

VOCAB=[[1,-1],[-1,1],[0,0]]
def bert(tokens, selections, valid, table=VOCAB, **extras):
    return dict(tokens=tokens,selections=selections,valid=valid,table=table,**extras)

PAPERS.append(dict(slug="paper-bert",title="BERT · pretrain a tiny bidirectional masked encoder",group="Transformers and LLMs",level="Advanced",duration="55–75 min",
mission="A masked-language-model pretrainer is being debugged on one short sequence. Assemble its deterministic input corruption, one bidirectional encoder block and the masked vocabulary head. Reserve the original token labels before changing any input.",
inputs="tokens is a vector of IDs into table, whose final row is [MASK]. valid gives the active prefix length. Each distinct selection is [position, draw, random_id] at an active position. Optional positions and types supply per-position embedding vectors to add to the token embeddings. Their defaults are zero.",
outputs="Return the corrupted input_ids including padding, predictions in selection order, and mlm_loss as mean natural-log cross-entropy on the original selected IDs. Empty selections have zero loss.",
constraints="Replace a selected token with [MASK] for draw < 0.8, with random_id for 0.8 ≤ draw < 0.9, and otherwise keep it visible. Every active position attends to every active key using dot(q,k)/sqrt(width). Exclude padding. The supplied post-norm block uses one head, identity projections and GELU. The MLM head uses LayerNorm(GELU(hidden)), tied vocabulary weights and zero biases. Break argmax ties toward smaller IDs.",
scope="Run a complete forward masked-language objective through a single tiny BERT-style encoder. Position and segment embedding contributions are supplied. Projection matrices are identities, dropout is disabled, and next-sentence prediction and optimizer updates are outside the exercise.",
source=source("Devlin et al. · BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding","1810.04805"),modules=BERT,
parts=[part("Construct masked-token inputs and targets","corruption.py","corrupt","Apply the prescribed 80/10/10 branch to each selected position and preserve the original labels in selection order. Return a new token list.","Draw 0.8 starts the random-token branch.","Keep a target even when the visible input did not change."),
part("Run a bidirectional attention head","attention_core.py","attend","Use all valid keys and values for each valid query. Here Q, K and V each equal the input rows. Stabilize softmax independently per query.","A token may attend to a later position.","Padding contributes neither scores nor denominator mass."),
part("Score the original selected tokens","token_head.py","predict","Apply the supplied GELU and population layer normalization to each selected hidden vector. Project against every original token embedding, including [MASK]. Return greedy IDs and mean stable negative log likelihood.","The output table is also the input embedding table.","Subtract the maximum logit when computing the log partition.")],
examples=[ex(bert([0,1],[[0,0,1]],2),dict(input_ids=[2,1],predictions=[1],mlm_loss=log(1+__import__("math").exp(2)+__import__("math").exp(4))),"The masked query is zero and averages itself with the token on its right. After normalization its head is approximately [−1,1], giving logits near [−2,2,0]. The correct label is token 0, so its loss is about 4.143 nats."),
ex(bert([1,0,0],[],2),dict(input_ids=[1,0,0],predictions=[],mlm_loss=0),"The prefix still passes through the encoder. With no reconstruction targets the head emits no predictions.")],
checks=[check(1,"corruption.corrupt",[[0,1,0],[[2,.8,1],[0,.9,1],[1,.799,0]],2],"Boundary draws and selection order"),check(1,"corruption.corrupt",[[1,0],[],2],"Empty selected set"),
check(2,"attention_core.attend",[[[0,0],[1,3],[100,100]],2],"Right context enters and padding stays out"),check(2,"attention_core.attend",[[[1000,-1000],[-1000,1000]],2],"Large query-key dot products"),
check(3,"token_head.predict",[[[0,0]],VOCAB,[0],[1]],"Uniform logits and tie-breaking"),check(3,"token_head.predict",[[[1,-1],[-1,1]],VOCAB,[1,0],[0,1]],"Tied projection and selected order")]+integrations(3,[("Exposed target remains supervised",bert([0,1,0],[[1,.95,0],[0,.8,1]],3)),("Extreme padding embedding",bert([0,1,2],[[0,.2,1]],2,[[2,0,-1],[0,5,1],[200,-100,100]])),("Optional embedding contributions",bert([0,1],[[0,.3,1]],2,positions=[[0,2],[0,0]],types=[[.5,0],[1,0]]))])))
