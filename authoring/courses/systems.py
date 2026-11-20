"""Deterministic systems contracts; these do not claim to benchmark GPU hardware."""
import math
import copy
import itertools
from collections import Counter
from common import make, source, DLSYS, A2

LABS=[]


def tile_ranges(length, tile):
    """Partition [0,length) into [start,end] pairs of at most tile elements. length>=0 and tile>0. End is exclusive and the final tile may be short."""
    return [[i,min(length,i+tile)] for i in range(0,length,tile)]


def tiled_product(a, b, tile):
    """Return A@B for nonempty rectangular M×K and K×N matrices using edge-safe tiles along all three axes. Numerical results must match dense multiplication; no padded values enter sums."""
    m,k,n=len(a),len(b),len(b[0]);out=[[0.0]*n for _ in range(m)]
    for i0,i1 in tile_ranges(m,tile):
        for j0,j1 in tile_ranges(n,tile):
            for k0,k1 in tile_ranges(k,tile):
                for i in range(i0,i1):
                    for j in range(j0,j1):
                        out[i][j]+=sum(a[i][p]*b[p][j] for p in range(k0,k1))
    return out


def matmul_plan(a, b, tile, bytes_per_element, bandwidth, flops_per_second):
    """Return output, flops, bytes, intensity, seconds, bottleneck for an idealized tiled plan. For each (M,N,K) tile load its A and B once; no reuse across output tiles. Write C once after all K tiles, with no C reads. FLOPs=2*M*N*K. seconds=max(bytes/bandwidth,FLOPs/flops_per_second). Tie => 'balanced', otherwise 'memory' or 'compute'. Positive byte size and rates."""
    m,k,n=len(a),len(b),len(b[0]);elements=m*n
    for i0,i1 in tile_ranges(m,tile):
        for j0,j1 in tile_ranges(n,tile):
            for k0,k1 in tile_ranges(k,tile):elements+=(i1-i0)*(k1-k0)+(k1-k0)*(j1-j0)
    byte_count=elements*bytes_per_element;flops=2*m*n*k
    memory=byte_count/bandwidth;compute=flops/flops_per_second
    return dict(output=tiled_product(a,b,tile),flops=flops,bytes=byte_count,intensity=flops/byte_count,seconds=max(memory,compute),bottleneck='balanced' if memory==compute else 'memory' if memory>compute else 'compute')

T=dict(a=[[1,2],[3,4]],b=[[1,0],[0,1]],tile=2,bytes_per_element=4,bandwidth=48,flops_per_second=16)
LABS.append(make('tiled-matmul','Build an edge-safe tiled matrix operator','Training systems',['dlsys','cs336'],
 [(tile_ranges,'Partition logical dimensions',['Use exclusive ends.','Do not discard a partial final tile.']),
  (tiled_product,'Accumulate partial products',['Reset each output only once.','Reduction tiles contribute to the same output cell.']),
  (matmul_plan,'Account for movement and arithmetic',['Count transfers under the stated reuse policy.','The slower idealized time determines the bound.'])],
 [[([5,2],[[0,2],[2,4],[4,5]],'Partial final tile'),([0,3],[],'Empty partition'),([1,9],[[0,1]],'Oversized tile'),([6,3],[[0,3],[3,6]],'Exact division')],
  [([[[1,2]],[[3],[4]],1],[[11]],'Dot product'),([T['a'],T['b'],2],T['a'],'Identity'),([[[1,2,3]],[[1,0],[0,1],[2,3]],2],[[7,11]],'Reduction tail'),([[[1],[-2],[3]],[[4,5]],2],[[4,5],[-8,-10],[12,15]],'Output tail')],
  [([*T.values()],{'output':T['a'],'flops':16,'bytes':48,'intensity':1/3,'seconds':1,'bottleneck':'balanced'},'Balanced roofline'),([[[2]],[[3]],1,4,12,1],{'output':[[6]],'flops':2,'bytes':12,'intensity':1/6,'seconds':2,'bottleneck':'compute'},'Compute bound'),([T['a'],T['b'],1,4,16,100],None,'Memory-bound small tiles'),([[[1,2,3]],[[1],[2],[3]],2,2,10,10],None,'Edge accounting')]],
 [(T,None,'An identity product with balanced idealized costs.'),(dict(T,tile=1),None,'Smaller tiles reload operands more often under this model.'),(dict(T,bandwidth=8),None,'Reducing bandwidth makes movement dominate.'),(dict(T,a=[[1,2,3]],b=[[1],[2],[3]]),None,'A partial reduction tile still contributes.')],
 'A kernel planner needs correct outputs and reproducible operation counts before hardware benchmarking. Implement tiled multiplication and a transparent traffic model.',
 ('Two compatible matrices, tile size, element byte size and positive hardware-rate assumptions.','Matrix product and analytical compute/traffic estimates.','Tile tails must work. Rates use bytes/second and FLOPs/second. This model excludes launch latency, cache effects, occupancy and GPU timing.'),
 ['Predict which traffic counts change when the tile grows.','Can a lower modeled time prove a real GPU speedup? List the missing measurements.'],
 [source('DL Systems · hardware acceleration','https://dlsyscourse.org/slides/11-hardware-acceleration.pdf'),source('CS336 2025 · A2, benchmarking and kernels',A2)],
 [('tile_ranges.py','min(length,i+tile)','i+tile'),('tiled_product.py','out[i][j]+=','out[i][j]='),('matmul_plan.py','elements=m*n','elements=0'),('matmul_plan.py','seconds=max(memory,compute)','seconds=min(memory,compute)')],['course-tensor-views']))


def unscale_gradient(scaled, scale, limit):
    """Return {finite,gradient}. A scaled entry may be null to denote nonfinite hardware output. If any is null or abs(entry)>limit, return finite=false and gradient=null. Equality to limit is valid. Otherwise divide each entry by positive scale. This checks overflow before division."""
    finite=all(g is not None and abs(g)<=limit for g in scaled)
    return dict(finite=finite,gradient=[g/scale for g in scaled] if finite else None)


def scaler_update(state, finite, interval):
    """state={scale,good}. On overflow halve scale with minimum 1 and reset good=0. On success increment good; at interval successes double scale and reset good. interval>0. Return a new state."""
    if not finite:return dict(scale=max(1,state['scale']/2),good=0)
    good=state['good']+1
    return dict(scale=state['scale']*2,good=0) if good>=interval else dict(scale=state['scale'],good=good)


def precision_session(weights, batches, state, lr, limit, interval):
    """Each batch contains unscaled microbatch gradient vectors (possibly null entries). Simulate scaling each vector with the current scale; if any scaled microbatch overflows, skip the whole optimizer step. Otherwise sum the unscaled vectors and apply SGD. Always update scaler once per batch. Return weights,state,applied boolean list; skipped steps do not change weights."""
    weights=weights[:];applied=[]
    for batch in batches:
        grad=[0.0]*len(weights);finite=True
        for micro in batch:
            record=unscale_gradient([g*state['scale'] if g is not None else None for g in micro],state['scale'],limit)
            finite=finite and record['finite']
            if record['finite']:grad=[a+b for a,b in zip(grad,record['gradient'])]
        if finite:weights=[w-lr*g for w,g in zip(weights,grad)]
        applied.append(finite);state=scaler_update(state,finite,interval)
    return dict(weights=weights,state=state,applied=applied)

P=dict(weights=[1],batches=[[[2]],[[9]],[[2]]],state={'scale':4,'good':0},lr=.1,limit=16,interval=2)
LABS.append(make('loss-scaling','Recover training after a mixed-precision overflow','Training systems',['dlsys','cs336'],
 [(unscale_gradient,'Validate scaled gradients',['An overflow can be hidden by dividing first.','A nonfinite marker invalidates the whole vector.']),
  (scaler_update,'Maintain the scale controller',['Count consecutive successful steps.','Overflow resets the growth window.']),
  (precision_session,'Make each optimizer step atomic',['Inspect every microbatch before applying any update.','Discard all accumulated gradients on a failed step.'])],
 [[([[8,-4],4,8],{'finite':True,'gradient':[2,-1]},'Finite boundary'),([[None,1],4,8],{'finite':False,'gradient':None},'Nonfinite marker'),([[9],4,8],{'finite':False,'gradient':None},'Before unscale'),([[-9],4,8],{'finite':False,'gradient':None},'Negative overflow')],
  [([{'scale':4,'good':0},True,2],{'scale':4,'good':1},'Successful step'),([{'scale':4,'good':1},True,2],{'scale':8,'good':0},'Growth'),([{'scale':4,'good':1},False,2],{'scale':2,'good':0},'Backoff'),([{'scale':1,'good':0},False,3],{'scale':1,'good':0},'Scale floor')],
  [([*P.values()],{'weights':[.6],'state':{'scale':2,'good':1},'applied':[True,False,True]},'Recover after overflow'),([[1],[],{'scale':4,'good':0},.1,16,2],{'weights':[1],'state':{'scale':4,'good':0},'applied':[]},'No steps'),([[1],[[[2],[None]]],{'scale':4,'good':0},.1,16,2],None,'Atomic skip'),([[1],[[[None],[2]]],{'scale':4,'good':0},.1,16,2],{'weights':[1],'state':{'scale':2,'good':0},'applied':[False]},'Early microbatch overflow'),([[1],[[[2]],[[2]],[[2]]],{'scale':4,'good':0},.1,16,2],None,'Scale growth changes boundary')]],
 [(P,None,'One overflowing step is skipped; the following step can recover.'),(dict(P,batches=[[[2],[None]]]),None,'A later bad microbatch invalidates the entire step.'),(dict(P,batches=[[[2]],[[2]],[[2]]]),None,'Consecutive successes grow the scale.'),(dict(P,batches=[[[5]],[[5]],[[5]]]),None,'Backoff can make the same gradient representable.')],
 'A mixed-precision worker must preserve model state when any microbatch overflows. Implement the loss-scale controller and atomic update boundary.',
 ('Master weights, unscaled gradient microbatches, scaler state and a simulated finite-range limit.','Applied/skipped decisions, master weights and final scale.','This models overflow control, not IEEE FP16 rounding. Supplied gradients are sums; no weight decay or Adam moments are modeled. Values are finite numbers or null.'),
 ['Predict the effect of an overflow in the final microbatch.','Keep gradients fixed and vary the initial scale. Does a larger scale always apply more updates?'],
 [source('CS336 2025 · A2, mixed precision',A2),source('DL Systems · large-model training',DLSYS)],
 [('unscale_gradient.py','abs(g)<=limit','g<=limit'),('scaler_update.py',"good=state['good']+1","good=state['good']"),('precision_session.py','if finite:weights=','if True:weights='),('precision_session.py',"finite=finite and record['finite']","finite=record['finite']")],['course-optimizer']))


def segment_cost(activation_bytes, start, end):
    """For a chain with activation sizes a[0..L], a checkpoint segment [start,end] replays layers start..end-1. Return sum(a[start+1:end+1]), the replay tape bytes. 0<=start<end<=L."""
    return sum(activation_bytes[start+1:end+1])


def checkpoint_cost(activation_bytes, boundaries):
    """Boundaries are strictly increasing, start at 0 and end at L. Persistent bytes=sum(a[b] for boundaries). Scratch is the largest segment replay tape. Return persistent,scratch,peak=persistent+scratch,recomputed=L. This conservative model counts boundary outputs in replay scratch too; parameter/gradient memory is excluded."""
    persistent=sum(activation_bytes[b] for b in boundaries)
    scratch=max(segment_cost(activation_bytes,a,b) for a,b in zip(boundaries,boundaries[1:]))
    return dict(persistent=persistent,scratch=scratch,peak=persistent+scratch,recomputed=len(activation_bytes)-1)


def choose_checkpoints(activation_bytes, budget):
    """Enumerate all checkpoint subsets for 1<=L<=10. Among plans with modeled peak<=budget, choose the fewest persistent bytes, then smaller peak, then lexicographically smallest boundaries. Return {boundaries,cost}, or null if none fits. Positive activation sizes and nonnegative budget."""
    length=len(activation_bytes)-1;plans=[]
    for mask in range(1<<(length-1)):
        boundaries=[0]+[i for i in range(1,length) if mask&(1<<(i-1))]+[length]
        cost=checkpoint_cost(activation_bytes,boundaries)
        if cost['peak']<=budget:plans.append((cost['persistent'],cost['peak'],boundaries,cost))
    if not plans:return None
    _,_,boundaries,cost=min(plans)
    return dict(boundaries=boundaries,cost=cost)

Q=dict(activation_bytes=[2,2,2,2,2],budget=12)
LABS.append(make('checkpoint-planner','Fit an activation replay plan into a memory budget','Training systems',['dlsys','cs336'],
 [(segment_cost,'Account for a replay tape',['The segment input is already checkpointed.','Include every replayed output, including its endpoint.']),
  (checkpoint_cost,'Measure simultaneous live state',['Persistent and replay allocations coexist.','Use the largest replay segment, not the average.']),
  (choose_checkpoints,'Search feasible checkpoint plans',['An infeasible plan must never win on another metric.','Apply the documented tie-breakers in order.'])],
 [[([[2,3,5],0,2],8,'Full replay'),([[2,3,5],1,2],5,'Last layer'),([[1,9,1,9],0,1],9,'Uneven activation'),([[1,9,1,9],1,3],10,'Two-layer tape')],
  [([[2,2,2],[0,2]],{'persistent':4,'scratch':4,'peak':8,'recomputed':2},'One segment'),([[2,2,2],[0,1,2]],{'persistent':6,'scratch':2,'peak':8,'recomputed':2},'Every boundary'),([[1,9,1,9],[0,1,3]],None,'Uneven segments'),([[3,7],[0,1]],{'persistent':10,'scratch':7,'peak':17,'recomputed':1},'Single layer')],
  [([[2,2,2],8],{'boundaries':[0,2],'cost':{'persistent':4,'scratch':4,'peak':8,'recomputed':2}},'Fewest persistent bytes'),([[2,2,2],7],None,'No fit'),([[1,9,1,9],22],None,'Uneven budget'),([[2,2,2,2,2],10],None,'Interior checkpoint')]],
 [(Q,None,'The plan chooses a feasible persistent/replay tradeoff.'),(dict(Q,budget=10),None,'A tighter budget can require an interior checkpoint.'),(dict(Q,budget=1),None,'No plan fits this budget.'),(dict(activation_bytes=[1,8,1,8,1],budget=20),None,'Uneven layers make equal spacing unreliable.')],
 'A trainer has a chain of layers with uneven activation sizes. Build a planner that explicitly accounts for saved boundaries and replay storage.',
 ('Activation byte sizes for input and layer outputs, plus a memory budget.','A feasible checkpoint plan and its accounting, or null.','Use exactly the stated conservative liveness model. This is not a claim about PyTorch allocator peaks. L<=10 makes exhaustive search acceptable.'),
 ['Predict whether checkpointing every layer always minimizes peak memory in this model.','Compare uniform and alternating large/small activations under the same budget.'],
 [source('CS336 2025 · A2, memory profiling and distributed training',A2),source('DL Systems · training large models','https://dlsyscourse.org/slides/15-training-large-models.pdf')],
 [('segment_cost.py','start+1:end+1','start:end'),('checkpoint_cost.py','persistent+scratch','max(persistent,scratch)'),('choose_checkpoints.py',"cost['peak']<=budget","cost['peak']<budget"),('choose_checkpoints.py','if not plans:return None','if not plans:return {}')],['course-tensor-views']))


def shard_ranges(size, ranks):
    """Partition size parameters into contiguous nearly equal shards across positive ranks. First size%ranks ranks get one extra element. Return [start,end] for every rank, including empty shards."""
    q,r=divmod(size,ranks);start=0;out=[]
    for rank in range(ranks):
        end=start+q+(rank<r);out.append([start,end]);start=end
    return out


def reduce_scatter(gradients, counts):
    """Each rank supplies a full gradient SUM and a positive example count. Sum across ranks and divide by total examples; then return each rank's contiguous shard. Shapes are R×P; P may be zero. Rank order is stable."""
    total=sum(counts);mean=[sum(g[i] for g in gradients)/total for i in range(len(gradients[0]))]
    return [mean[a:b] for a,b in shard_ranges(len(mean),len(gradients))]


def sharded_step(weights, gradients, counts, lr, bytes_per_element):
    """Return shards after owner-only SGD, all-gathered weights, and state_bytes per rank. Each owner stores 4 arrays per owned parameter (master weight, gradient, first and second moment); model replicas and communication buffers excluded. No Adam update is performed. lr>=0, bytes_per_element>0."""
    ranges=shard_ranges(len(weights),len(gradients));reduced=reduce_scatter(gradients,counts)
    shards=[[w-lr*g for w,g in zip(weights[a:b],grad)] for (a,b),grad in zip(ranges,reduced)]
    return dict(shards=shards,weights=[w for shard in shards for w in shard],state_bytes=[4*(b-a)*bytes_per_element for a,b in ranges])

R=dict(weights=[1,2,3],gradients=[[2,4,6],[0,0,0]],counts=[1,3],lr=.1,bytes_per_element=4)
LABS.append(make('sharded-state','Shard optimizer ownership without changing the update','Training systems',['dlsys','cs336'],
 [(shard_ranges,'Assign unique parameter ownership',['Remainders belong to the first ranks.','Some ranks can own no parameters.']),
  (reduce_scatter,'Reduce example-weighted gradients',['The supplied arrays are sums, not local means.','Normalize before taking each owned slice.']),
  (sharded_step,'Update owners and reconstruct replicas',['Gather shards in rank order.','Memory accounting must use actual shard sizes.'])],
 [[([5,3],[[0,2],[2,4],[4,5]],'Uneven split'),([2,4],[[0,1],[1,2],[2,2],[2,2]],'Empty owners'),([0,2],[[0,0],[0,0]],'No parameters'),([4,2],[[0,2],[2,4]],'Exact split')],
  [([[[2,4],[0,0]],[1,3]],[[.5],[1]],'Unequal local counts'),([[[1,2,3]],[1]],[[1,2,3]],'One rank'),([[[3],[-1],[2]],[1,1,2]],[[1],[],[]],'More ranks than parameters'),([[[],[]],[1,2]],[[],[]],'Empty vectors')],
  [([*R.values()],{'shards':[[.95,1.9],[2.85]],'weights':[.95,1.9,2.85],'state_bytes':[32,16]},'Gather owner updates'),([[1],[[1],[1]],[1,1],0,2],{'shards':[[1],[]],'weights':[1],'state_bytes':[8,0]},'Zero step'),([[1,2],[[2,4]],[2],.5,4],None,'Single worker'),([[],[[],[]],[1,1],.1,4],{'shards':[[],[]],'weights':[],'state_bytes':[0,0]},'Empty model')]],
 [(R,None,'Example counts differ across ranks.'),(dict(R,lr=0),None,'A zero learning rate preserves gathered parameters.'),(dict(weights=[1],gradients=[[2],[0],[0]],counts=[1,1,2],lr=.2,bytes_per_element=4),None,'Empty owners remain part of the collective.'),(dict(weights=[],gradients=[[],[]],counts=[1,1],lr=.1,bytes_per_element=4),None,'Empty shards require no optimizer storage.')],
 'A distributed trainer is moving optimizer state off replicated workers. Implement ownership, weighted reduce-scatter and reconstruction while preserving a single-worker SGD result.',
 ('Weights, each rank’s gradient sums and counts, step size and storage element size.','Owner shards, reconstructed weights and per-rank optimizer-state bytes.','All ranks participate; local counts are positive. Communication is simulated deterministically, not measured. Only an SGD update is tested.'),
 ['Predict the failure caused by averaging rank means when rank batch sizes differ.','Does doubling ranks exactly halve peak owner memory for a five-parameter model?'],
 [source('CS336 2025 · A2, sharded optimizer state',A2),source('DL Systems · training large models',DLSYS)],
 [('shard_ranges.py','rank<r','rank<=r'),('reduce_scatter.py','total=sum(counts)','total=len(counts)'),('sharded_step.py','w-lr*g','w+lr*g'),('sharded_step.py','4*(b-a)*bytes_per_element','4*len(weights)*bytes_per_element')],['course-optimizer','data-parallel']))
