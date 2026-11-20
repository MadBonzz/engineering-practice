def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def linear(weight, x):
    return [dot(row, x) for row in weight]


def forward(weight, a, b, x, alpha):
    """Return W x + (alpha/r) B A x. W is [out,in], A [r,in], B [out,r], r>0. All arrays are ordinary lists; no bias. Return a vector without editing the inputs."""
    scale = alpha/len(a)
    delta = linear(b, linear(a,x))
    return [v+scale*d for v,d in zip(linear(weight,x), delta)]


def update(weight, a, b, x, target, alpha, lr):
    """One SGD step for loss 0.5*sum((prediction-target)^2). Keep W frozen. Return [new_A,new_B] using gradients from the same pre-update parameters. Do not mutate inputs."""
    scale = alpha/len(a)
    error = [v-y for v,y in zip(forward(weight,a,b,x,alpha),target)]
    down = linear(a,x)
    back = [sum(row[k]*e for row,e in zip(b,error)) for k in range(len(a))]
    na = [[v-lr*scale*back[k]*xi for v,xi in zip(row,x)] for k,row in enumerate(a)]
    nb = [[v-lr*scale*e*h for v,h in zip(row,down)] for row,e in zip(b,error)]
    return [na,nb]


def merge(weight, a, b, alpha):
    """Export a standalone weight matrix whose ordinary linear() output agrees with the adapter for any input. Return a new [out,in] matrix; preserve the frozen weights."""
    scale = alpha/len(a)
    return [[v+scale*sum(b[i][k]*a[k][j] for k in range(len(a))) for j,v in enumerate(row)] for i,row in enumerate(weight)]

# AUTHORING
TITLE = "LoRA linear layer"
GROUP = "Training"
INTRO = "Extend a frozen projection with a trainable adapter, update it on one labeled example, and export a merged projection for serving."
PARTS = ["forward", "update", "merge"]
LABELS = ["Run the adapter", "Train only the adapter", "Merge for deployment"]
HINTS = ["Write the shapes beside each multiply.|Rank controls the scaling.", "Compute both gradients at the old parameters.|A zero B has different consequences for the two gradients.", "Check the shape of the merged projection.|Compare outputs on each input basis vector."]
def cases(n):
    r, dim, out = 1+n%3, 2+n%2, 1+n%3
    w = [[(i-j+n)*.3 for j in range(dim)] for i in range(out)]
    a = [[(j+1+k)*.2 for j in range(dim)] for k in range(r)]
    b = [[0. if n%5==0 else (i-k+1)*.4 for k in range(r)] for i in range(out)]
    x = [(-1)**j*(j+n/5) for j in range(dim)]
    alpha = (n%4)+.5
    return [[w,a,b,x,alpha],[w,a,b,x,[.5]*out,alpha,.1],[w,a,b,alpha]]
MUTANTS = [("alpha/len(a)","alpha"),("v-y for v,y","y-v for v,y"),("v-lr*scale*e*h","v-lr*e*h"),("v+scale*sum","v-scale*sum")]

