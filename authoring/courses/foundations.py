"""Original CPU-scale tasks motivated by DL Systems and CS336 foundations."""
import math
import copy
import itertools
from collections import Counter
from common import make, source, DLSYS, A1

LABS=[]


def log_probs(logits):
    """Return log-softmax of a nonempty finite vector. Preserve accuracy for logits near ±1000; subtracting a shared constant must not change results."""
    maximum=max(logits)
    normalizer=math.log(sum(math.exp(x-maximum) for x in logits))
    return [x-maximum-normalizer for x in logits]


def classifier_gradient(features, weights, labels):
    """Return {loss, gradient} for mean cross-entropy: logits=XW, X is B×D, W is D×C, labels has B class IDs. B,D,C are positive. Gradient has W's shape; no bias term or regularizer."""
    b,d,c=len(features),len(weights),len(weights[0])
    gradient=[[0.0]*c for _ in range(d)];loss=0.0
    for x,y in zip(features,labels):
        lp=log_probs([sum(x[k]*weights[k][j] for k in range(d)) for j in range(c)])
        loss-=lp[y]/b
        for k in range(d):
            for j in range(c):gradient[k][j]+=x[k]*(math.exp(lp[j])-(j==y))/b
    return dict(loss=loss,gradient=gradient)


def train_classifier(features, weights, labels, lr, steps):
    """Run steps full-batch SGD updates at nonnegative lr. Return weights, loss history before each update, and final_loss. Copy inputs; steps may be zero."""
    weights=copy.deepcopy(weights);history=[]
    for _ in range(steps):
        state=classifier_gradient(features,weights,labels);history.append(state['loss'])
        weights=[[v-lr*g for v,g in zip(w,dw)] for w,dw in zip(weights,state['gradient'])]
    return dict(weights=weights,history=history,final_loss=classifier_gradient(features,weights,labels)['loss'])

LABS.append(make('classifier','Train a stable softmax classifier','Foundations',['dlsys','cs336'],
 [(log_probs,'Stabilize the probability boundary',['A common logit shift leaves probabilities unchanged.','Separate the maximum logit from the exponential sum.']),
  (classifier_gradient,'Backpropagate the batch loss',['Trace the batch reduction before updating a weight.','Check the gradient against a central finite difference.']),
  (train_classifier,'Integrate and diagnose learning',['Log loss before applying an update.','A zero-step request still evaluates the current model.'])],
 [ [([[0,0]],[-math.log(2)]*2,'Equal logits'),([[5]], [0],'Single class'),([[1000,999]],None,'Large positive logits'),([[-1000,-1000]],[-math.log(2)]*2,'Large negative logits')],
   [([[[1]],[[0,0]],[0]],{'loss':math.log(2),'gradient':[[-.5,.5]]},'One training example'),([[[0]],[[3,-2]],[1]],{'loss':math.log(2),'gradient':[[0,0]]},'Zero feature'),([[[1],[1]],[[0,0]],[0,1]],{'loss':math.log(2),'gradient':[[0,0]]},'Opposite labels cancel'),([[[2,-1],[-1,3]],[[2,-2],[1,0]],[1,0]],None,'Cross-feature gradients')],
   [([[[1]],[[0,0]],[0],0,1],{'weights':[[0,0]],'history':[math.log(2)],'final_loss':math.log(2)},'Frozen learning rate'),([[[1]],[[0,0]],[0],.2,0],{'weights':[[0,0]],'history':[],'final_loss':math.log(2)},'No updates'),([[[1],[-1]],[[0,0]],[0,1],.2,3],None,'Repeated updates'),([[[1,2]],[[1,0],[0,1]],[1],.1,2],None,'Nonzero initial weights')]],
 [(dict(features=[[1]],weights=[[0,0]],labels=[0],lr=.2,steps=1),None,'One update favors the observed label.'),
  (dict(features=[[1],[-1]],weights=[[0,0]],labels=[0,1],lr=.5,steps=3),None,'Opposite inputs require opposite class evidence.'),
  (dict(features=[[2,1],[-1,2]],weights=[[0,0,0],[0,0,0]],labels=[2,1],lr=.1,steps=4),None,'Three-class integration.'),
  (dict(features=[[0]],weights=[[1000,-1000]],labels=[1],lr=0,steps=0),None,'Zero signal and no updates.')],
 'A training service intermittently returns NaN and reports inconsistent gradients when batch size changes. Repair its numerical and update path.',
 ('Dense feature rows, a weight matrix, integer labels, learning rate and update count.','Model weights, pre-update losses, and final cross-entropy.','Valid rectangular shapes; finite inputs; no mutation. Loss is a mean across examples, never across classes.'),
 ['Predict what a shared +1000 shift to the logits does to loss and gradients.','Compare lr=0.1, 1 and 5 for ten updates. Does a larger step always improve final loss?'],
 [source('DL Systems · softmax regression and manual backpropagation',DLSYS),source('CS336 2025 · A1, cross-entropy and training',A1)],
 [('log_probs.py','maximum=max(logits)','maximum=0'),('classifier_gradient.py','loss-=lp[y]/b','loss-=lp[y]'),('classifier_gradient.py','-(j==y)','+(j==y)'),('train_classifier.py','v-lr*g','v+lr*g')]))


def forward_graph(nodes):
    """Evaluate a topologically ordered scalar graph. Nodes are {op:'input',value}, {op:'add'|'mul',parents:[i,j]}, or {op:'tanh',parents:[i]}; parent indices are smaller than node index. Return all values."""
    values=[]
    for node in nodes:
        op=node['op'];p=node.get('parents',[])
        if op=='input':v=node['value']
        elif op=='add':v=values[p[0]]+values[p[1]]
        elif op=='mul':v=values[p[0]]*values[p[1]]
        else:v=math.tanh(values[p[0]])
        values.append(v)
    return values


def reverse_graph(nodes, output, seed):
    """Return d(seed*nodes[output])/d(each node value), treating each node as an intermediate. Accumulate every incoming edge, including duplicate parents. Disconnected nodes have zero gradient."""
    values=forward_graph(nodes);grads=[0.0]*len(nodes);grads[output]=seed
    for i in range(len(nodes)-1,-1,-1):
        n=nodes[i];p=n.get('parents',[]);g=grads[i]
        if n['op']=='add':
            for parent in p:grads[parent]+=g
        elif n['op']=='mul':
            grads[p[0]]+=g*values[p[1]];grads[p[1]]+=g*values[p[0]]
        elif n['op']=='tanh':grads[p[0]]+=g*(1-values[i]**2)
    return grads


def graph_report(nodes, outputs):
    """Return {values, gradients}, where outputs is a list of {node,seed} and gradients is the sum of their reverse passes. Empty outputs gives all zeros; repeated roots count repeatedly."""
    gradients=[0.0]*len(nodes)
    for root in outputs:
        g=reverse_graph(nodes,root['node'],root['seed'])
        gradients=[a+b for a,b in zip(gradients,g)]
    return dict(values=forward_graph(nodes),gradients=gradients)

G=[{'op':'input','value':3},{'op':'mul','parents':[0,0]},{'op':'add','parents':[1,0]}]
H=[{'op':'input','value':0},{'op':'tanh','parents':[0]},{'op':'input','value':7}]
LABS.append(make('autodiff','Repair a reverse-mode differentiation engine','Foundations',['dlsys'],
 [(forward_graph,'Evaluate a shared computation graph',['A node is evaluated once even if several edges consume it.','The supplied order already satisfies dependencies.']),
  (reverse_graph,'Accumulate vector–Jacobian products',['A repeated parent represents two contributions.','Work backward and add contributions; do not overwrite them.']),
  (graph_report,'Support multiple loss roots',['Each root has an independent upstream seed.','Linearity lets you combine reverse passes.'])],
 [[([G],[3,9,12],'Shared square'),([H],[0,0,7],'Disconnected input'),([[{'op':'input','value':-2},{'op':'add','parents':[0,0]}]],[-2,-4],'Negative duplicate'),([[{'op':'input','value':2},{'op':'tanh','parents':[0]}]],None,'Nonzero tanh')],
  [([G,2,1],[7,1,1],'Branch accumulation'),([H,1,2],[2,2,0],'Seed scaling'),([G,1,-2],[-12,-2,0],'Intermediate root'),([G,0,0],[0,0,0],'Zero seed'),([[{'op':'input','value':.7},{'op':'tanh','parents':[0]}],1,2], [2*(1-math.tanh(.7)**2),2], 'Nonzero tanh derivative')],
  [([G,[]],{'values':[3,9,12],'gradients':[0,0,0]},'No roots'),([H,[{'node':1,'seed':1}]],{'values':[0,0,7],'gradients':[1,1,0]},'One root'),([G,[{'node':1,'seed':1},{'node':2,'seed':2}]],None,'Overlapping roots'),([G,[{'node':2,'seed':1},{'node':2,'seed':-1}]],{'values':[3,9,12],'gradients':[0,0,0]},'Cancellation')]],
 [(dict(nodes=G,outputs=[{'node':2,'seed':1}]),None,'The squared input reaches the loss along more than one edge.'),(dict(nodes=H,outputs=[{'node':1,'seed':2}]),None,'A disconnected input contributes nothing.'),(dict(nodes=G,outputs=[{'node':1,'seed':1},{'node':2,'seed':1}]),None,'Two objectives share intermediates.'),(dict(nodes=H,outputs=[]),None,'An empty objective has zero gradient.')],
 'A small differentiable simulator reuses intermediate values. Its current engine loses gradient contributions when branches meet. Complete the graph evaluator and reverse pass.',
 ('A nonempty valid DAG and seeded output nodes.','Forward values and accumulated gradients for every node.','Only input, add, mul and tanh are supported; every parent precedes its child. Inputs stay unchanged.'),
 ['Predict the input gradient if a square uses the same input twice.','Add a disconnected subgraph and explain which derivatives should remain unchanged.'],
 [source('DL Systems · lectures 4–5, automatic differentiation','https://dlsyscourse.org/slides/4-automatic-differentiation.pdf')],
 [('forward_graph.py',"values[p[0]]*values[p[1]]","values[p[0]]+values[p[1]]"),('reverse_graph.py','grads[parent]+=g','grads[parent]=g'),('reverse_graph.py','g*(1-values[i]**2)','g*(1-values[i])'),('graph_report.py','a+b for a,b','b for a,b')],['course-classifier']))


def view_offsets(shape, strides, offset):
    """Return storage indices in logical row-major order for a strided view. Rank is 1–4; dimensions are nonnegative; strides may be zero or negative. All addressed storage indices are valid. Scalar views are out of scope."""
    return [offset+sum(i*s for i,s in zip(index,strides)) for index in itertools.product(*(range(d) for d in shape))]


def gather_view(storage, shape, strides, offset):
    """Return flat logical values using the view's offsets. Broadcasting may reuse one storage element; do not infer strides from shape."""
    return [storage[i] for i in view_offsets(shape,strides,offset)]


def view_vjp(storage, shape, strides, offset, upstream):
    """Return {values, storage_gradient}. upstream has product(shape) entries in logical order. Scatter-add into a zero gradient of length len(storage), including aliased/broadcast entries."""
    gradient=[0.0]*len(storage)
    for i,g in zip(view_offsets(shape,strides,offset),upstream):gradient[i]+=g
    return dict(values=gather_view(storage,shape,strides,offset),storage_gradient=gradient)

LABS.append(make('tensor-views','Implement tensor views and their backward scatter','Foundations',['dlsys','cs336'],
 [(view_offsets,'Interpret shape and strides',['Enumerate logical coordinates, then map each to storage.','A negative stride is valid when the base offset points to the far end.']),
  (gather_view,'Read transposed and broadcast views',['Logical order and physical storage order can differ.','Zero strides repeat the same value.']),
  (view_vjp,'Backpropagate through aliasing',['Gather becomes scatter in the reverse pass.','Repeated storage offsets need addition, not assignment.'])],
 [[([[2,2],[1,2],0],[0,2,1,3],'Transpose offsets'),([[3],[-1],2],[2,1,0],'Reverse offsets'),([[2,3],[0,1],0],[0,1,2,0,1,2],'Broadcast offsets'),([[0,2],[2,1],0],[],'Empty dimension')],
  [([[10,20,30,40],[2,2],[1,2],0],[10,30,20,40],'Transpose gather'),([[4,5],[3],[0],1],[5,5,5],'Scalar broadcast'),([[0,1,2,3,4],[2],[2],1],[1,3],'Strided slice'),([[1],[0],[1],0],[],'Empty read')],
  [([[2],[3],[0],0,[1,2,3]],{'values':[2,2,2],'storage_gradient':[6]},'Broadcast accumulation'),([[1,2],[2],[-1],1,[3,4]],{'values':[2,1],'storage_gradient':[4,3]},'Reverse VJP'),([[1,2,3],[2,2],[1,1],0,[1,2,3,4]],{'values':[1,2,2,3],'storage_gradient':[1,5,4]},'Overlapping window'),([[1,2],[0],[1],0,[]],{'values':[],'storage_gradient':[0,0]},'Empty VJP')]],
 [(dict(storage=[1,2,3,4],shape=[2,2],strides=[1,2],offset=0,upstream=[1,2,3,4]),None,'A transpose changes the placement of the gradient.'),(dict(storage=[7,8],shape=[2,2],strides=[0,1],offset=0,upstream=[1,2,3,4]),None,'Broadcasting sums contributions across rows.'),(dict(storage=[1,2,3],shape=[3],strides=[-1],offset=2,upstream=[2,0,-2]),None,'Reverse indexing.'),(dict(storage=[1],shape=[0,2],strides=[2,1],offset=0,upstream=[]),None,'Zero-sized view.')],
 'A tensor backend must expose transpose, slicing and broadcasting without copying storage. Implement the view boundary and its gradient transport.',
 ('A flat storage buffer, shape, strides, base offset and upstream derivative.','Logical view values and a gradient over the original storage.','All shapes/strides have equal rank, and all accessed indices are valid. Zero-sized dimensions are legal; inputs are immutable.'),
 ['Predict why a broadcast read can require a reduction in backward.','Compare contiguous and transposed views with identical storage. Which values and gradients change order?'],
 [source('DL Systems · hardware acceleration and NDArray implementation',DLSYS),source('CS336 2025 · tensors and resource accounting','https://cs336.stanford.edu/spring2025/')],
 [('view_offsets.py','offset+sum','0+sum'),('gather_view.py','storage[i]','storage[0]'),('view_vjp.py','gradient[i]+=g','gradient[i]=g'),('view_offsets.py','i*s for i,s','i*abs(s) for i,s')],['course-autodiff']))


def batch_stats(rows):
    """Return [column_means, population_variances] for nonempty B×D rows. Divide variance by B (not B-1), including B=1."""
    n=len(rows);d=len(rows[0]);mean=[sum(r[j] for r in rows)/n for j in range(d)]
    return [mean,[sum((r[j]-mean[j])**2 for r in rows)/n for j in range(d)]]


def normalize_batch(rows, state, training, momentum, epsilon):
    """Return {values,state}. state has mean/variance vectors. Training uses current batch statistics and new_state=(1-momentum)*old+momentum*batch; eval uses old state unchanged. Normalize (x-mean)/sqrt(variance+epsilon). 0<=momentum<=1, epsilon>0; no affine gain/bias."""
    mean,var=batch_stats(rows) if training else [state['mean'],state['variance']]
    values=[[(x-m)/math.sqrt(v+epsilon) for x,m,v in zip(row,mean,var)] for row in rows]
    new=copy.deepcopy(state)
    if training:
        new={'mean':[(1-momentum)*a+momentum*b for a,b in zip(state['mean'],mean)],'variance':[(1-momentum)*a+momentum*b for a,b in zip(state['variance'],var)]}
    return dict(values=values,state=new)


def norm_session(batches, state, momentum, epsilon):
    """Process batches containing rows, training, keep and uniform. Normalize then inverted-dropout: in training keep a coordinate iff uniform<keep, scaling by 1/keep. In eval ignore uniform and dropout. keep is in (0,1]; uniform has rows' shape. Return outputs per batch and final running state."""
    outputs=[]
    for batch in batches:
        out=normalize_batch(batch['rows'],state,batch['training'],momentum,epsilon);state=out['state']
        if batch['training']:
            keep=batch['keep'];out['values']=[[x/keep if u<keep else 0.0 for x,u in zip(row,draws)] for row,draws in zip(out['values'],batch['uniform'])]
        outputs.append(out['values'])
    return dict(outputs=outputs,state=state)

S={'mean':[0.0],'variance':[1.0]}
B={'rows':[[1],[3]],'training':True,'keep':1.0,'uniform':[[0],[.5]]}
LABS.append(make('normalization','Make training and evaluation state agree','Foundations',['dlsys','alchemy'],
 [(batch_stats,'Measure per-feature statistics',['The batch axis is reduced; feature columns remain separate.','A singleton batch has zero population variance.']),
  (normalize_batch,'Separate batch and running statistics',['Training output uses current statistics, not the moving average.','Evaluation must not update buffers.']),
  (norm_session,'Compose state and stochastic layers',['Use supplied draws for reproducibility.','Inverted dropout scales retained values only during training.'])],
 [[([[[1],[3]]],[[2],[1]],'One feature'),([[[2,4]]],[[2,4],[0,0]],'Singleton'),([[[1,3],[3,7]]],[[2,5],[1,4]],'Two features'),([[[5],[5],[5]]],[[5],[0]],'Constant batch')],
  [([[[1],[3]],S,True,.5,3],{'values':[[-.5],[.5]],'state':{'mean':[1],'variance':[1]}},'Training state update'),([[[2]],S,False,.5,3],{'values':[[1]],'state':S},'Evaluation'),([[[9]],S,True,0,1],{'values':[[0]],'state':S},'Frozen buffers'),([[[1],[5]],S,True,1,5],None,'Replace buffers')],
  [([[],S,.5,1],{'outputs':[],'state':S},'Empty session'),([[B],S,1,3],{'outputs':[[[-.5],[.5]]],'state':{'mean':[2],'variance':[1]}},'No dropout'),([[dict(B,keep=.5,uniform=[[.5],[.1]])],S,.5,3],None,'Threshold equality'),([[B,dict(B,training=False)],S,.5,3],None,'Train then eval')]],
 [(dict(batches=[B],state=S,momentum=.5,epsilon=3),None,'Statistics are updated during training.'),(dict(batches=[B,dict(B,training=False,keep=.2)],state=S,momentum=.5,epsilon=3),None,'Evaluation reads the accumulated state.'),(dict(batches=[dict(B,keep=.5,uniform=[[.5],[.1]])],state=S,momentum=1,epsilon=3),None,'Dropout uses a strict threshold.'),(dict(batches=[],state=S,momentum=0,epsilon=1),None,'No batches leave state unchanged.')],
 'A model behaves differently after export because its normalization buffers and dropout mode are mishandled. Build a reproducible training/evaluation session.',
 ('Batch records, initial running mean/variance, momentum and epsilon.','Normalized and optionally dropped values plus the final state.','No random generator is needed: uniform draws are supplied. Batch sizes may vary. Valid nonempty feature rows; do not mutate state.'),
 ['Predict the result when an evaluation example is placed beside a different example.','Hold draws fixed and compare keep=1 and keep=0.5. Why does train-mode output differ from eval-mode output?'],
 [source('DL Systems · normalization and regularization','https://dlsyscourse.org/slides/norm_reg.pdf')],
 [('batch_stats.py','/n for j in range(d)]]','/max(1,n-1) for j in range(d)]]'),('normalize_batch.py','if training else','if not training else'),('norm_session.py','u<keep','u<=keep'),('norm_session.py','x/keep if','x if')],['course-classifier']))


def clip_gradient(gradient, limit):
    """Return a copied gradient scaled to Euclidean norm at most positive limit. A zero gradient stays zero; clip globally, not per coordinate."""
    norm=math.sqrt(sum(g*g for g in gradient));scale=min(1.0,limit/norm) if norm else 1.0
    return [g*scale for g in gradient]


def adam_step(weights, gradient, state, lr, decay, beta1, beta2, epsilon):
    """Return {weights,state} after one AdamW step. state has m,v,t (t is completed updates). Increment t; update moments; divide by 1-beta**t; apply w*(1-lr*decay)-lr*m_hat/(sqrt(v_hat)+epsilon). Inputs unchanged; beta in [0,1), epsilon>0."""
    t=state['t']+1;m=[beta1*a+(1-beta1)*g for a,g in zip(state['m'],gradient)];v=[beta2*a+(1-beta2)*g*g for a,g in zip(state['v'],gradient)]
    new=[w*(1-lr*decay)-lr*(a/(1-beta1**t))/(math.sqrt(b/(1-beta2**t))+epsilon) for w,a,b in zip(weights,m,v)]
    return dict(weights=new,state=dict(m=m,v=v,t=t))


def optimize(weights, batches, lr, decay, limit, beta1, beta2, epsilon):
    """Each batch is a list of {sum: gradient_sum_vector, count: positive example_count}. Divide the sum of all vectors by total count, clip once, then perform one AdamW step. Initialize m/v/t to zero and return final weights/state and used_gradients. No batches means no update."""
    state=dict(m=[0.0]*len(weights),v=[0.0]*len(weights),t=0);used=[];weights=weights[:]
    for batch in batches:
        count=sum(x['count'] for x in batch)
        gradient=clip_gradient([sum(x['sum'][j] for x in batch)/count for j in range(len(weights))],limit);used.append(gradient)
        result=adam_step(weights,gradient,state,lr,decay,beta1,beta2,epsilon);weights,state=result['weights'],result['state']
    return dict(weights=weights,state=state,used_gradients=used)

Z=dict(m=[0.0],v=[0.0],t=0)
O=dict(weights=[1.0],batches=[[{'sum':[2.0],'count':2}]],lr=.1,decay=.2,limit=10,beta1=0,beta2=0,epsilon=1)
LABS.append(make('optimizer','Resume a correct AdamW training update','Training mechanics',['dlsys','cs336','alchemy'],
 [(clip_gradient,'Clip a global gradient',['Compare the vector norm with the threshold.','Clipping should preserve the direction.']),
  (adam_step,'Update optimizer state',['The first update uses t=1 for bias correction.','Decoupled decay acts on the weight, outside the moments.']),
  (optimize,'Combine unequal microbatches',['A mean of microbatch means is wrong when counts differ.','Accumulate sums and counts, then clip once.'])],
 [[([[3,4],1],[.6,.8],'Global norm'),([[0,0],1],[0,0],'Zero vector'),([[.1,-.2],1],[.1,-.2],'Below threshold'),([[-6,8],5],[-3,4],'Signed clipping')],
  [([[1],[1],Z,.1,.2,0,0,1],{'weights':[.93],'state':{'m':[1],'v':[1],'t':1}},'First step'),([[2],[0],Z,.1,.5,0,0,1],{'weights':[1.9],'state':{'m':[0],'v':[0],'t':1}},'Decay with zero gradient'),([[1],[2],Z,.1,0,.9,.99,1],None,'Bias correction'),([[1],[-1],dict(m=[1],v=[1],t=2),.1,0,.5,.5,.01],None,'Resumed moments')],
  [([*O.values()],None,'Single batch'),([[1],[],.1,0,1,.9,.99,1e-8],{'weights':[1],'state':Z,'used_gradients':[]},'No updates'),([[1],[[{'sum':[2],'count':1},{'sum':[0],'count':3}]],.1,0,10,0,0,1],None,'Unequal microbatches'),([[1,2],[[{'sum':[3,4],'count':1}],[{'sum':[-1,2],'count':1}]],.1,.1,1,.9,.99,1e-8],None,'Two optimizer steps')]],
 [(O,None,'The first step includes decoupled weight decay.'),(dict(O,batches=[[{'sum':[2.0],'count':1},{'sum':[0.0],'count':3}]]),None,'Unequal microbatches are weighted by their counts.'),(dict(O,beta1=.9,beta2=.99,batches=O['batches']*3),None,'Moments carry across steps.'),(dict(O,batches=[]),None,'No optimizer step means no decay.')],
 'A trainer must match a large batch while accumulating smaller batches, and preserve Adam state across updates. Repair the reduction, clipping and optimizer path.',
 ('Weights, gradient-sum microbatches, AdamW parameters and a global clipping limit.','Final weights, moments, update count and the clipped gradient used per update.','All vectors have equal positive length. Each update has at least one positive-count microbatch. Rates and decay are nonnegative.'),
 ['Split the same batch into unequal microbatches. Predict whether the update changes.','Compare zero gradient with no optimizer step. Does weight decay behave identically?'],
 [source('CS336 2025 · A1, AdamW and gradient clipping',A1),source('DL Systems · optimization',DLSYS)],
 [('clip_gradient.py','sum(g*g for g in gradient)','sum(abs(g) for g in gradient)'),('adam_step.py',"state['t']+1","1"),('adam_step.py','w*(1-lr*decay)','w'),('optimize.py',"sum(x['count'] for x in batch)",'len(batch)')],['course-classifier']))


def image_patches(image, kernel, stride, padding):
    """Single-channel cross-correlation: return flat row-major K×K patches for a rectangular image, zero-padding all sides. Output H=floor((H+2P-K)/S)+1, similarly W; K,S>0, P>=0 and outputs have positive dimensions."""
    h,w=len(image),len(image[0]);patches=[]
    for y in range(0,h+2*padding-kernel+1,stride):
        for x in range(0,w+2*padding-kernel+1,stride):
            patches.append([image[yy-padding][xx-padding] if 0<=yy-padding<h and 0<=xx-padding<w else 0 for yy in range(y,y+kernel) for xx in range(x,x+kernel)])
    return patches


def filter_forward(image, weights, stride, padding):
    """Return a 2-D single-channel cross-correlation using square weights, no kernel flip, no bias. Reuse image_patches; preserve output row-major order."""
    k=len(weights);flat=[v for row in weights for v in row];w=(len(image[0])+2*padding-k)//stride+1
    values=[sum(a*b for a,b in zip(p,flat)) for p in image_patches(image,k,stride,padding)]
    return [values[i:i+w] for i in range(0,len(values),w)]


def filter_backward(image, weights, stride, padding, upstream):
    """Return {output, image_gradient, weight_gradient} for sum(output*upstream). Accumulate overlapping receptive fields; skip padded cells in image_gradient. Upstream has output shape."""
    h,w=len(image),len(image[0]);k=len(weights);dx=[[0.0]*w for _ in range(h)];dw=[[0.0]*k for _ in range(k)]
    for oy,row in enumerate(upstream):
        for ox,g in enumerate(row):
            for ky in range(k):
                for kx in range(k):
                    y=oy*stride+ky-padding;x=ox*stride+kx-padding
                    if 0<=y<h and 0<=x<w:
                        dx[y][x]+=g*weights[ky][kx];dw[ky][kx]+=g*image[y][x]
    return dict(output=filter_forward(image,weights,stride,padding),image_gradient=dx,weight_gradient=dw)

C=dict(image=[[1,2],[3,4]],weights=[[1,0],[0,-1]],stride=1,padding=0,upstream=[[2]])
LABS.append(make('convolution-backward','Connect an image filter to backpropagation','Foundations',['dlsys'],
 [(image_patches,'Lower receptive fields',['Padding changes coordinates but does not add trainable image pixels.','Keep the order inside each patch consistent with flattened weights.']),
  (filter_forward,'Apply a shared filter',['Cross-correlation does not reverse the kernel.','Derive output width before reshaping the flat result.']),
  (filter_backward,'Accumulate both derivatives',['One image pixel may affect many output locations.','Check image and weight derivatives separately using finite differences.'])],
 [[([[[1,2],[3,4]],2,1,0],[[1,2,3,4]],'One patch'),([[[2]],1,1,0],[[2]],'Point filter'),([[[1,2,3],[4,5,6]],2,1,0],[[1,2,4,5],[2,3,5,6]],'Overlapping patches'),([[[1]],2,1,1],[[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]],'Border padding')],
  [([C['image'],C['weights'],1,0],[[-3]],'Asymmetric filter'),([[[2]],[[3]],1,0],[[6]],'Pointwise'),([[[1,2,3],[4,5,6],[7,8,9]],[[1,1],[1,1]],2,0],[[12]],'Stride two'),([[[1]],[[1,2],[3,4]],1,1],[[4,3],[2,1]],'Padded cross-correlation')],
  [([*C.values()],{'output':[[-3]],'image_gradient':[[2,0],[0,-2]],'weight_gradient':[[2,4],[6,8]]},'Both derivatives'),([[[2]],[[3]],1,0,[[0]]],{'output':[[6]],'image_gradient':[[0]],'weight_gradient':[[0]]},'Zero upstream'),([[[1,2,3],[4,5,6],[7,8,9]],[[1,1],[1,1]],1,0,[[1,1],[1,1]]],None,'Overlap sums'),([[[1]],[[1,2],[3,4]],1,1,[[1,2],[3,4]]],None,'Padding gradient')]],
 [(C,None,'A single receptive field exposes both derivatives.'),(dict(image=[[2]],weights=[[3]],stride=1,padding=0,upstream=[[1]]),None,'A point filter is a multiplication.'),(dict(image=[[1,2,3],[4,5,6],[7,8,9]],weights=[[1,1],[1,1]],stride=1,padding=0,upstream=[[1,1],[1,1]]),None,'The center pixel receives overlapping contributions.'),(dict(image=[[1]],weights=[[1,2],[3,4]],stride=1,padding=1,upstream=[[1,2],[3,4]]),None,'Padding has no storage gradient.')],
 'An image pipeline already evaluates filters but cannot train them. Implement receptive-field extraction and the backward transport needed by the optimizer.',
 ('A rectangular one-channel image, square filter, stride, padding and output derivative.','Forward image and gradients with respect to the original image and filter.','Dimensions satisfy the output formula; gradients are sums, not averages. Inputs remain unchanged.'),
 ['Predict the input-gradient pattern for an all-ones kernel and upstream image.','Change stride from one to two. Which input pixels stop receiving gradient?'],
 [source('DL Systems · convolution implementation',DLSYS)],
 [('image_patches.py','h+2*padding-kernel+1','h+2*padding-kernel'),('filter_forward.py','flat=[v for row in weights for v in row]','flat=[v for row in weights[::-1] for v in row[::-1]]'),('filter_backward.py','dx[y][x]+=','dx[y][x]='),('filter_backward.py','dw[ky][kx]+=g*image[y][x]','dw[ky][kx]+=image[y][x]')],['course-autodiff','course-tensor-views']))
