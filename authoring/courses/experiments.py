"""Small reproducible experiments, inspired by Alchemy assignments and prediction quizzes."""
import math
import copy
import itertools
from collections import Counter
from common import make, source, alchemy, CS336

LABS=[]


def final_records(records, budget):
    """Records have recipe,seed,step,validation,test. Keep only exact step==budget; for repeated (recipe,seed) records at that step, the LAST record wins. Return rows sorted by recipe then integer seed. Do not select each run's best checkpoint."""
    chosen={}
    for row in records:
        if row['step']==budget:chosen[(row['recipe'],row['seed'])]=dict(row)
    return [chosen[k] for k in sorted(chosen)]


def paired_effect(records, control, candidate, budget):
    """Use final_records and pair only seeds present in both recipes. Return seeds, candidate-minus-control validation differences, mean and standard_error. Sample SD/sqrt(n); one pair has SE=0; no pairs has mean/SE=null. Negative mean favors candidate."""
    rows=final_records(records,budget);base={r['seed']:r['validation'] for r in rows if r['recipe']==control};trial={r['seed']:r['validation'] for r in rows if r['recipe']==candidate};seeds=sorted(base.keys()&trial.keys());delta=[trial[s]-base[s] for s in seeds]
    mean=sum(delta)/len(delta) if delta else None
    se=math.sqrt(sum((x-mean)**2 for x in delta)/(len(delta)-1)/len(delta)) if len(delta)>1 else 0 if delta else None
    return dict(seeds=seeds,differences=delta,mean=mean,standard_error=se)


def select_recipe(records, control, budget):
    """Use seeds shared by ALL recipes with an exact-budget record. Rank recipes by mean validation over those seeds, then lexical recipe name. Return selected recipe, common seeds, selected mean test loss, and paired effects against control. If control absent or no common seeds, return selected=null,seeds=[],test=null,effects={}. Test values never influence selection."""
    rows=final_records(records,budget);recipes=sorted(set(r['recipe'] for r in rows));by={name:{r['seed']:r for r in rows if r['recipe']==name} for name in recipes}
    if control not in by:return dict(selected=None,seeds=[],test=None,effects={})
    common=sorted(set.intersection(*(set(m) for m in by.values())))
    if not common:return dict(selected=None,seeds=[],test=None,effects={})
    winner=min(recipes,key=lambda name:(sum(by[name][s]['validation'] for s in common)/len(common),name))
    common_rows=[r for r in rows if r['seed'] in common]
    return dict(selected=winner,seeds=common,test=sum(by[winner][s]['test'] for s in common)/len(common),effects={name:paired_effect(common_rows,control,name,budget) for name in recipes})

ROWS=[dict(recipe='base',seed=1,step=10,validation=3,test=1),dict(recipe='base',seed=2,step=10,validation=5,test=1),dict(recipe='new',seed=1,step=10,validation=2,test=9),dict(recipe='new',seed=2,step=10,validation=4,test=9)]
W=dict(records=ROWS,control='base',budget=10)
LABS.append(make('controlled-sweeps','Compare experiments without changing the question','Empirical deep learning',['alchemy','cs336'],
 [(final_records,'Align runs at equal training budget',['An early best checkpoint answers a different question.','Retries overwrite the same run key only at the specified step.']),
  (paired_effect,'Measure matched-seed effects',['Pair before subtracting.','Report uncertainty of the differences, not unrelated run variances.']),
  (select_recipe,'Keep selection separate from final evaluation',['Use the same seed intersection for every recipe.','A favorable test result must not choose the recipe.'])],
 [[([ROWS,10],ROWS,'Final records'),([ROWS,9],[],'Missing horizon'),([ROWS+[dict(ROWS[0],validation=7)],10],None,'Last retry wins'),([ROWS+[dict(ROWS[0],step=9,validation=0)],10],ROWS,'Early checkpoint ignored')],
  [([ROWS,'base','new',10],{'seeds':[1,2],'differences':[-1,-1],'mean':-1,'standard_error':0},'Paired improvement'),([ROWS,'base','missing',10],{'seeds':[],'differences':[],'mean':None,'standard_error':None},'No pairs'),([ROWS+[dict(ROWS[-1],validation=8)],'base','new',10],{'seeds':[1,2],'differences':[-1,3],'mean':1,'standard_error':2},'Sample standard error'),([ROWS[:1]+ROWS[2:3],'base','new',10],None,'One pair')],
  [([*W.values()],None,'Validation selection'),([[],'base',10],{'selected':None,'seeds':[],'test':None,'effects':{}},'No experiments'),([ROWS+[dict(recipe='third',seed=1,step=10,validation=1,test=10)],'base',10],None,'Global common seed'),([ROWS,'absent',10],{'selected':None,'seeds':[],'test':None,'effects':{}},'Missing control')]],
 [(W,None,'The validation winner has worse test loss: do not change the selection rule after seeing it.'),(dict(W,budget=9),None,'No exact-horizon records means no comparison.'),(dict(W,records=ROWS+[dict(ROWS[0],step=9,validation=0)]),None,'An attractive early checkpoint is not eligible.'),(dict(W,records=ROWS+[dict(recipe='third',seed=1,step=10,validation=1,test=10)]),None,'All candidates must share the comparison seeds.')],
 'A researcher has retried runs, unmatched seeds and checkpoints from different budgets. Build a comparison tool that answers a fixed experimental question without test-set selection.',
 ('Run records, a control recipe and an exact training horizon.','Matched effects, validation-selected recipe and its held-out test result.','Finite losses, nonnegative integer steps, integer seeds and recipe names. Records are observations, not proof of causality. No interpolation between horizons.'),
 ['Before running, predict whether a better validation recipe must improve test loss.','Add one unpaired seed with an extreme loss. Which comparison should change?'],
 [*alchemy('a1_basics','basics'),*alchemy('a6_generalization'),source('CS336 2025 · scaling and evaluation',CS336)],
 [('final_records.py',"row['step']==budget","row['step']<=budget"),('paired_effect.py','trial[s]-base[s]','base[s]-trial[s]'),('paired_effect.py','/(len(delta)-1)/len(delta)','/len(delta)/len(delta)'),('select_recipe.py',"by[name][s]['validation']","by[name][s]['test']")],['course-scaling-budget']))


def quadratic_state(point, hessian, linear):
    """Return loss=0.5*x^T H x+b^T x and gradient=Hx+b for a symmetric square H. H may be indefinite. Shapes agree and dimension>=1."""
    hx=[sum(h*x for h,x in zip(row,point)) for row in hessian]
    return dict(loss=.5*sum(x*h for x,h in zip(point,hx))+sum(x*b for x,b in zip(point,linear)),gradient=[h+b for h,b in zip(hx,linear)])


def momentum_path(point, hessian, linear, rates, noises, momentum, decay):
    """For each learning rate eta and supplied additive gradient-noise vector: v=momentum*v+gradient+noise; x=(1-eta*decay)*x-eta*v. Start v=0. Return points and exact noiseless losses INCLUDING the initial state. No bias correction or (1-momentum) scaling. 0<=momentum<1, eta/decay>=0."""
    point=point[:];velocity=[0.0]*len(point);points=[point[:]];losses=[quadratic_state(point,hessian,linear)['loss']]
    for eta,noise in zip(rates,noises):
        g=quadratic_state(point,hessian,linear)['gradient'];velocity=[momentum*v+a+n for v,a,n in zip(velocity,g,noise)]
        point=[(1-eta*decay)*x-eta*v for x,v in zip(point,velocity)];points.append(point[:]);losses.append(quadratic_state(point,hessian,linear)['loss'])
    return dict(points=points,losses=losses)


def landscape_study(point, hessian, linear, recipes, noises):
    """Run each {name,rates,momentum,decay} from the SAME initial point and supplied noises. Return runs mapping names to paths, and best by final noiseless loss then lexical name; no recipes gives best=null. Noise/rate lengths agree for every recipe. This finite-horizon toy model need not behave like a trained neural network."""
    runs={r['name']:momentum_path(point,hessian,linear,r['rates'],noises,r['momentum'],r['decay']) for r in recipes}
    best=min(runs,key=lambda n:(runs[n]['losses'][-1],n)) if runs else None
    return dict(runs=runs,best=best)

N=dict(point=[1,1],hessian=[[1,0],[0,4]],linear=[0,0],recipes=[dict(name='small',rates=[.1]*3,momentum=0,decay=0),dict(name='large',rates=[.6]*3,momentum=0,decay=0)],noises=[[0,0]]*3)
# Keep independent hand-computed oracles separate from the experiment sweep.
LABS.append(make('loss-landscape','Investigate curvature, momentum and learning-rate stability','Empirical deep learning',['alchemy','dlsys'],
 [(quadratic_state,'Measure a local loss landscape',['Include cross terms.','Different directions can have different curvature.']),
  (momentum_path,'Replay controlled trajectories',['Restart velocity at zero.','Record the initial point and every updated point.']),
  (landscape_study,'Compare fixed-horizon interventions',['Hold initialization and noise fixed.','Choose by final, not best intermediate, loss.'])],
 [[([[1,2],[[1,0],[0,2]],[0,0]],{'loss':4.5,'gradient':[1,4]},'Anisotropic bowl'),([[0],[[2]],[3]],{'loss':0,'gradient':[3]},'Linear term'),([[1,2],[[2,1],[1,2]],[0,0]],{'loss':7,'gradient':[4,5]},'Cross curvature'),([[1,1],[[1,0],[0,-1]],[0,0]],{'loss':0,'gradient':[1,-1]},'Saddle')],
  [([[1],[[2]],[0],[.5],[[0]],0,0],{'points':[[1],[0]],'losses':[1,0]},'One exact step'),([[1],[[2]],[0],[],[],0,0],{'points':[[1]],'losses':[1]},'Initial measurement'),([[1],[[2]],[0],[.1,.1],[[0],[0]],.5,0],{'points':[[1],[.8],[.54]],'losses':[1,.64,.2916]},'Momentum carries'),([[1],[[0]],[0],[.1],[[1]],0,.5],{'points':[[1],[.85]],'losses':[0,0]},'Noise and decay')],
  [([*N.values()],None,'Stability comparison'),([[1],[[1]],[0],[],[]],{'runs':{},'best':None},'No recipes'),([[1],[[2]],[0],[dict(name='a',rates=[0],momentum=0,decay=0),dict(name='b',rates=[0],momentum=0,decay=0)],[[0]]],None,'Tie by name'),([[1],[[1]],[0],[dict(name='run',rates=[.2,.2],momentum=.5,decay=.1)],[[1],[-1]]],None,'Paired noise')]],
 [(N,None,'A step safe for one direction can be unstable for a sharper direction.'),(dict(N,hessian=[[1,0],[0,1]]),None,'Changing only curvature changes the stability comparison.'),(dict(N,noises=[[1,-1],[-1,1],[0,0]]),None,'The same noise is replayed for both recipes.'),(dict(N,recipes=[dict(name='momentum',rates=[.1]*3,momentum=.8,decay=0)]),None,'Momentum changes the trajectory without changing the loss function.')],
 'A researcher needs a controlled test bed to distinguish curvature effects from stochasticity and optimizer state. Implement a replayable quadratic experiment runner.',
 ('A symmetric Hessian, linear term, initial point, optimizer recipes and deterministic noise.','Full trajectories, noiseless losses and the best final recipe.','All dimensions agree; learning-rate and noise sequences have equal lengths. Negative-curvature directions are allowed; all outputs remain finite.'),
 ['Predict stability when eta crosses 2/lambda_max in the positive-definite, noise-free, zero-momentum case.','Introduce momentum or a saddle direction. Which parts of that prediction no longer apply?'],
 [*alchemy('a2_hyperparameter_scaling','hyperparameter-scaling'),*alchemy('a3_optimization2','optimization-2')],
 [('quadratic_state.py','loss=.5*sum','loss=1.0*sum'),('momentum_path.py','momentum*v+a+n','a+n'),('momentum_path.py','(1-eta*decay)*x','x'),('landscape_study.py',"runs[n]['losses'][-1]","runs[n]['losses'][0]")],['course-optimizer']))


def rms_vector(vector, epsilon):
    """Return x/sqrt(mean(x^2)+epsilon), with epsilon>0 and nonempty x. This is RMS normalization without learned gain; do not subtract the mean."""
    denominator=math.sqrt(sum(x*x for x in vector)/len(vector)+epsilon)
    return [x/denominator for x in vector]


def residual_forward(vector, matrices, residual, normalize, branch_scale, epsilon):
    """Return states INCLUDING input for equal-width square matrices W. Each block: z=RMS(x) if normalize else x; branch=tanh(Wz); x_new=(x if residual else 0)+branch_scale*branch. No final normalization; no bias. All flags shared across depth."""
    states=[vector[:]]
    for matrix in matrices:
        x=states[-1];z=rms_vector(x,epsilon) if normalize else x
        branch=[math.tanh(sum(w*v for w,v in zip(row,z))) for row in matrix]
        states.append([(v if residual else 0)+branch_scale*b for v,b in zip(x,branch)])
    return states


def depth_probe(vector, matrices, residual, normalize, branch_scale, epsilon, direction, delta):
    """Return states, rms per state, directional_derivative of final output by centered finite differences along supplied direction, and sensitivity=norm(derivative)/norm(direction). delta>0, direction nonzero; derivative divides by 2*delta. This measures input sensitivity, not weight gradients or trained-model quality."""
    states=residual_forward(vector,matrices,residual,normalize,branch_scale,epsilon)
    plus=residual_forward([x+delta*d for x,d in zip(vector,direction)],matrices,residual,normalize,branch_scale,epsilon)[-1]
    minus=residual_forward([x-delta*d for x,d in zip(vector,direction)],matrices,residual,normalize,branch_scale,epsilon)[-1]
    derivative=[(a-b)/(2*delta) for a,b in zip(plus,minus)]
    return dict(states=states,rms=[math.sqrt(sum(v*v for v in s)/len(s)) for s in states],directional_derivative=derivative,sensitivity=math.sqrt(sum(v*v for v in derivative)/sum(d*d for d in direction)))

J=dict(vector=[.2,-.1],matrices=[[[.5,0],[0,.5]]]*4,residual=True,normalize=False,branch_scale=.5,epsilon=1e-6,direction=[1,0],delta=1e-5)
LABS.append(make('depth-signal','Measure signal and sensitivity through a residual stack','Empirical deep learning',['alchemy','dlsys'],
 [(rms_vector,'Normalize signal magnitude',['RMSNorm preserves a nonzero mean.','Epsilon is inside the square root.']),
  (residual_forward,'Compose architecture variants',['Normalize the branch input only.','The residual bypass uses the unnormalized state.']),
  (depth_probe,'Measure a controlled perturbation',['Perturb the initial input, not each intermediate.','Record the state convention consistently across variants.'])],
 [[([[3,4],3.5],[.75,1],'RMS scale'),([[0,0],1],[0,0],'Zero vector'),([[2,2],5],[2/3,2/3],'No mean subtraction'),([[-2],5],[-2/3],'Signed input')],
  [([[1],[],True,False,1,1],[[1]],'No blocks'),([[1],[[[0]]],True,False,1,1],[[1],[1]],'Identity bypass'),([[1],[[[0]]],False,False,1,1],[[1],[0]],'No bypass'),([[.5],[[[2]],[[1]]],True,True,.2,1],None,'Pre-normalized stack')],
  [([[1],[],True,False,1,1,[1],.01],{'states':[[1]],'rms':[1],'directional_derivative':[1],'sensitivity':1},'Identity sensitivity'),([[1],[[[0]]],False,False,1,1,[1],.01],{'states':[[1],[0]],'rms':[1,0],'directional_derivative':[0],'sensitivity':0},'Signal erased'),([*J.values()],None,'Residual depth'),([[1,2],[[[1,0],[0,1]]],True,True,0,1,[3,4],1e-4],None,'Zero branch scale')]],
 [(J,None,'Measure depth with an unchanged residual bypass.'),(dict(J,residual=False),None,'Remove only the bypass before comparing sensitivity.'),(dict(J,normalize=True),None,'Normalize the branch while keeping the bypass untouched.'),(dict(J,branch_scale=0),None,'A zero branch leaves an exact identity when residuals are enabled.')],
 'An architecture experiment needs comparable diagnostics across depth. Build a tiny residual-stack probe that distinguishes signal magnitude from input sensitivity.',
 ('A small vector, square block matrices, architecture flags and a perturbation direction.','Block states, RMS magnitudes and a directional sensitivity measurement.','Finite bounded inputs, width<=8, depth<=12. Use central differences exactly as specified; do not infer generalization from this untrained toy probe.'),
 ['Predict sensitivity when branch_scale=0, with and without the residual bypass.','Increase depth while holding matrices fixed. Compare normalized and unnormalized variants; explain the limits of this toy evidence.'],
 [*alchemy('a4_architecture1','stability')],
 [('rms_vector.py','/len(vector)+epsilon','+epsilon'),('residual_forward.py','(v if residual else 0)','v'),('residual_forward.py','branch_scale*b','b'),('depth_probe.py','/(2*delta)','/delta')],['course-normalization','paper-resnet']))


def low_rank_logits(hidden, left, right):
    """Compute H(LR): H is B×D, L is D×R, R is R×V; all dimensions positive. Return B×V logits without bias. This is a factorized output head, not a LoRA residual update."""
    middle=[[sum(x*w[j] for x,w in zip(row,left)) for j in range(len(right))] for row in hidden]
    return [[sum(x*w[j] for x,w in zip(row,right)) for j in range(len(right[0]))] for row in middle]


def head_metrics(hidden, left, right, labels):
    """Return logits, mean cross-entropy and head parameter count D*R+R*V. labels are valid class IDs. Use stable log-sum-exp. Count factor parameters, not the product matrix or hidden activations."""
    logits=low_rank_logits(hidden,left,right);loss=0.0
    for row,y in zip(logits,labels):
        m=max(row);loss+=math.log(sum(math.exp(x-m) for x in row))+m-row[y]
    return dict(logits=logits,loss=loss/len(labels),parameters=len(left)*len(right)+len(right)*len(right[0]))


def capacity_report(hidden, heads, labels, budget):
    """Evaluate candidate heads {name,left,right}; return results for all, and best feasible name by validation loss then parameter count then lexical name. Feasible means parameters<=budget; no fit gives null. Hidden features and validation examples are held fixed. This compares supplied checkpoints, not retrained capacity."""
    results={h['name']:head_metrics(hidden,h['left'],h['right'],labels) for h in heads}
    eligible=[name for name in results if results[name]['parameters']<=budget]
    best=min(eligible,key=lambda n:(results[n]['loss'],results[n]['parameters'],n)) if eligible else None
    return dict(results=results,best=best)

HEADS=[dict(name='rank1',left=[[1],[1]],right=[[1,-1]]),dict(name='rank2',left=[[1,0],[0,1]],right=[[1,-1],[-1,1]])]
L=dict(hidden=[[1,0],[0,1]],heads=HEADS,labels=[0,1],budget=8)
LABS.append(make('head-capacity','Compare factorized output heads under a parameter budget','Empirical deep learning',['alchemy','cs336'],
 [(low_rank_logits,'Compose the factorized projection',['The two factors have different roles and shapes.','A rank-one factorization can still produce many vocabulary logits.']),
  (head_metrics,'Measure likelihood and stored capacity',['Count the stored factors, not their dense product.','Keep numerical stabilization separate from model capacity.']),
  (capacity_report,'Make a constrained comparison',['Validation examples and hidden features stay fixed.','A more expressive head need not fit the budget or be better trained.'])],
 [[([[[1,2]],[[1],[2]],[[3,-1]]],[[15,-5]],'Rank-one projection'),([[[1,2]],[[1,0],[0,1]],[[1,0],[0,1]]],[[1,2]],'Identity factors'),([[[0,0]],[[1],[2]],[[3,4]]],[[0,0]],'Zero features'),([[[1,0],[0,1]],[[1],[1]],[[1,-1]]],[[1,-1],[1,-1]],'Shared bottleneck')],
  [([[[0]],[[1]],[[1,2]],[1]],{'logits':[[0,0]],'loss':math.log(2),'parameters':3},'Parameter accounting'),([[[1]],[[1]],[[0]],[0]],{'logits':[[0]],'loss':0,'parameters':2},'One class'),([[[1000]],[[1]],[[1,1]],[0]],None,'Large logits'),([L['hidden'],HEADS[1]['left'],HEADS[1]['right'],L['labels']],None,'Expressive head')],
  [([*L.values()],None,'Budgeted choice'),([L['hidden'],HEADS,L['labels'],0],None,'No feasible head'),([L['hidden'],[],L['labels'],10],{'results':{},'best':None},'Empty proposal set'),([L['hidden'],HEADS,L['labels'],4],None,'Rank-one-only budget')]],
 [(L,None,'The two factors determine both expressivity and storage.'),(dict(L,budget=4),None,'A tighter budget excludes one candidate.'),(dict(L,labels=[0,0]),None,'The task distribution changes which head performs well.'),(dict(L,budget=0),None,'No candidate fits a zero budget.')],
 'An architecture study asks whether a smaller output head can preserve held-out likelihood. Build the factorized projection and a fixed-feature comparison tool.',
 ('Held-out hidden vectors, named factorized heads, labels and a parameter budget.','Each head’s logits/loss/parameter count and the feasible winner.','No training or low-rank approximation is performed here. Factor shapes agree; candidate names are unique. Comparisons apply only to the supplied features and checkpoints.'),
 ['Predict whether lower factor rank necessarily means higher validation loss for supplied, unequally trained heads.','Construct hidden vectors whose label distinction is lost by a rank-one projection.'],
 [*alchemy('a5_capacity','capacity'),source('CS336 2025 · architectures and mixture of experts',CS336)],
 [('low_rank_logits.py','return [[sum(x*w[j]','return [[sum(-x*w[j]'),('head_metrics.py','len(left)*len(right)+len(right)*len(right[0])','len(left)*len(right[0])'),('capacity_report.py',"results[name]['parameters']<=budget","results[name]['parameters']<budget"),('capacity_report.py','best=min(eligible','best=max(eligible')],['paper-gpt2','lora-layer']))


def route_topk(scores, k):
    """For each token's finite expert-logit row select k highest scores (ties lower expert ID), then softmax only over the selected experts. Return [{experts,weights}] per token in descending-score order. 1<=k<=expert_count."""
    result=[]
    for row in scores:
        ids=sorted(range(len(row)),key=lambda i:(-row[i],i))[:k];m=max(row[i] for i in ids);p=[math.exp(row[i]-m) for i in ids];total=sum(p)
        result.append(dict(experts=ids,weights=[x/total for x in p]))
    return result


def capacity_dispatch(scores, k, capacity):
    """Route with route_topk, then process tokens in input order and each token's experts in selected order. Each expert admits at most capacity assignments. Return assignments [{token,expert,weight}], loads and dropped count. Dropped probability mass is NOT renormalized. capacity>=0."""
    routes=route_topk(scores,k);loads=[0]*len(scores[0]);assignments=[];dropped=0
    for token,route in enumerate(routes):
        for expert,weight in zip(route['experts'],route['weights']):
            if loads[expert]<capacity:assignments.append(dict(token=token,expert=expert,weight=weight));loads[expert]+=1
            else:dropped+=1
    return dict(assignments=assignments,loads=loads,dropped=dropped)


def expert_layer(inputs, scores, experts, k, capacity):
    """Each expert is a D×D linear matrix; inputs is T×D, scores T×E. Dispatch and sum weight*(expert@input) for admitted assignments only. Return outputs, loads, dropped and utilization=admitted/(E*capacity), or 0 for zero capacity. No residual, expert bias or post-drop renormalization."""
    dispatch=capacity_dispatch(scores,k,capacity);out=[[0.0]*len(inputs[0]) for _ in inputs]
    for a in dispatch['assignments']:
        for j,row in enumerate(experts[a['expert']]):out[a['token']][j]+=a['weight']*sum(w*x for w,x in zip(row,inputs[a['token']]))
    return dict(outputs=out,loads=dispatch['loads'],dropped=dispatch['dropped'],utilization=sum(dispatch['loads'])/(len(experts)*capacity) if capacity else 0.0)

M=dict(inputs=[[1],[2],[3]],scores=[[0,0],[0,0],[0,0]],experts=[[[1]],[[2]]],k=2,capacity=2)
LABS.append(make('expert-capacity','Dispatch sparse experts without losing capacity accounting','Empirical deep learning',['alchemy','cs336'],
 [(route_topk,'Select and normalize sparse routes',['Tie-breaking must be deterministic.','Normalize only the selected experts.']),
  (capacity_dispatch,'Enforce expert admission limits',['Capacity counts assignments, not unique token IDs globally.','A rejected assignment must not increment load.']),
  (expert_layer,'Combine admitted expert outputs',['Dropped mass is intentionally lost under this contract.','A token can receive zero output if every route was rejected.'])],
 [[([[[0,0]],1],[{'experts':[0],'weights':[1]}],'Tie selection'),([[[0,0]],2],[{'experts':[0,1],'weights':[.5,.5]}],'Equal routes'),([[[1,3,2]],2],None,'Selected normalization'),([[[1000,1000,-1000]],2],[{'experts':[0,1],'weights':[.5,.5]}],'Stable router')],
  [([[[0,0],[0,0]],1,1],{'assignments':[{'token':0,'expert':0,'weight':1}],'loads':[1,0],'dropped':1},'One expert fills'),([[[0,0]],2,0],{'assignments':[],'loads':[0,0],'dropped':2},'Zero capacity'),([[[1,0],[0,1]],2,1],None,'Partial token drops'),([[[0,1],[1,0]],1,2],None,'Independent capacities')],
  [([*M.values()],{'outputs':[[1.5],[3],[0]],'loads':[2,2],'dropped':2,'utilization':1},'Overflow token'),([[[2]],[[0,0]],[[[1]],[[3]]],2,1],{'outputs':[[4]],'loads':[1,1],'dropped':0,'utilization':1},'Weighted experts'),([[[2]],[[0,0]],[[[1]],[[3]]],2,0],{'outputs':[[0]],'loads':[0,0],'dropped':2,'utilization':0},'All dropped'),([[[1],[2]],[[1,0],[0,1]],[[[1]],[[3]]],2,1],None,'No mass renormalization')]],
 [(M,None,'The last token loses both assignments when capacity is exhausted.'),(dict(M,capacity=3),None,'More capacity preserves all routed mass.'),(dict(M,k=1),None,'Top-one ties can overload a single expert.'),(dict(M,capacity=0),None,'Zero capacity produces zero expert output.')],
 'A sparse model’s router is balanced on average but drops work in individual batches. Implement deterministic top-k routing, capacity admission and observable expert execution.',
 ('Token vectors, router scores, linear experts, top-k and per-expert capacity.','Expert outputs, admission counts, dropped assignments and utilization.','At least one token/expert; consistent positive width. This task models dispatch, not training a router or distributed all-to-all performance.'),
 ['Predict load and drops for tied router scores under top-one versus top-two routing.','Raise capacity without changing weights. Which token outputs can change, and why?'],
 [*alchemy('a5_capacity','capacity'),source('CS336 2025 · mixture-of-experts lecture',CS336)],
 [('route_topk.py','(-row[i],i)','(-row[i],-i)'),('capacity_dispatch.py','loads[expert]<capacity','loads[expert]<=capacity'),('expert_layer.py',"a['weight']*sum","1.0*sum"),('expert_layer.py','sum(dispatch[\'loads\'])/(len(experts)*capacity)','sum(dispatch[\'loads\'])/len(experts)')],['paper-deepseek-v3']))


def tiny_predict(inputs, weights):
    """One-hidden-layer regressor without biases: hidden=tanh(Wx), output=v·hidden. weights={hidden: H×D, readout:H}; inputs B×D. Return scalar predictions per row."""
    return [sum(v*math.tanh(sum(w*x for w,x in zip(row,features))) for row,v in zip(weights['hidden'],weights['readout'])) for features in inputs]


def tiny_gradient(inputs, targets, weights):
    """Return {loss,hidden,readout}: loss=mean((prediction-target)^2)/2 and its gradients. All examples equally weighted; B>0. Differentiate tanh and both layers; input weights must remain unchanged."""
    predictions=tiny_predict(inputs,weights);h=len(weights['readout']);d=len(inputs[0]);dw=[[0.0]*d for _ in range(h)];dv=[0.0]*h;loss=0.0
    for x,y,p in zip(inputs,targets,predictions):
        error=p-y;loss+=error**2/(2*len(inputs))
        for j in range(h):
            a=math.tanh(sum(w*z for w,z in zip(weights['hidden'][j],x)));dv[j]+=error*a/len(inputs)
            for k in range(d):dw[j][k]+=error*weights['readout'][j]*(1-a*a)*x[k]/len(inputs)
    return dict(loss=loss,hidden=dw,readout=dv)


def learning_curve(train_x, train_y, validation_x, validation_y, weights, lr, steps, decay, freeze_hidden):
    """Full-batch SGD on tiny_gradient; add decay*w to EACH trainable parameter's gradient. Freeze hidden weights completely when requested, including decay. Report initial and post-update train/validation HALF-MSE (excluding regularization), final weights, and unique_examples=len(train_x), exposures=len(train_x)*steps. Validation never affects updates. Positive-sized train/validation sets; lr/decay>=0, steps>=0."""
    weights=copy.deepcopy(weights);train=[];validation=[]
    for step in range(steps+1):
        g=tiny_gradient(train_x,train_y,weights);train.append(g['loss']);validation.append(tiny_gradient(validation_x,validation_y,weights)['loss'])
        if step==steps:break
        if not freeze_hidden:weights['hidden']=[[w-lr*(dw+decay*w) for w,dw in zip(row,grad)] for row,grad in zip(weights['hidden'],g['hidden'])]
        weights['readout']=[w-lr*(dw+decay*w) for w,dw in zip(weights['readout'],g['readout'])]
    return dict(weights=weights,train=train,validation=validation,unique_examples=len(train_x),exposures=len(train_x)*steps)

NN={'hidden':[[1]],'readout':[1]}
TRAIN=dict(train_x=[[-1],[1]],train_y=[-1,1],validation_x=[[-.5],[.5]],validation_y=[-.5,.5],weights=NN,lr=.2,steps=4,decay=0,freeze_hidden=False)
LABS.append(make('generalization','Run a tiny training and transfer-learning experiment','Empirical deep learning',['alchemy','dlsys'],
 [(tiny_predict,'Trace a nonlinear predictor',['Apply tanh to each hidden preactivation.','The output is linear in hidden activations.']),
  (tiny_gradient,'Differentiate both parameter families',['Use weights from before the update for every gradient.','The objective is half mean-squared error.']),
  (learning_curve,'Separate fitting, validation and frozen features',['Validation labels must never enter the training update.','A frozen family receives neither gradients nor weight decay.'])],
 [[([[[0]],NN],[0],'Zero input'),([[[1]],{'hidden':[[0]],'readout':[2]}],[0],'Zero hidden weight'),([[[1],[-1]],NN],[math.tanh(1),-math.tanh(1)],'Odd symmetry'),([[[1,2]],{'hidden':[[1,0],[0,1]],'readout':[2,-1]}],None,'Multiple features')],
  [([[[1]],[1],{'hidden':[[0]],'readout':[2]}],{'loss':.5,'hidden':[[-2]],'readout':[0]},'Hidden derivative at zero'),([[[0]],[1],NN],{'loss':.5,'hidden':[[0]],'readout':[0]},'No feature signal'),([[[1],[-1]],[1,-1],NN],None,'Mean batch derivative'),([[[1,2],[2,-1]],[.3,-.2],{'hidden':[[.1,.2],[-.3,.4]],'readout':[.5,-.6]}],None,'Cross-family derivative')],
  [([*TRAIN.values()],None,'Training curve'),([[[0]],[1],[[0]],[2],NN,.1,1,0,False],{'weights':NN,'train':[.5,.5],'validation':[2,2],'unique_examples':1,'exposures':1},'No trainable signal'),([TRAIN['train_x'],TRAIN['train_y'],TRAIN['validation_x'],TRAIN['validation_y'],NN,.2,0,0,False],None,'Zero updates'),([TRAIN['train_x'],TRAIN['train_y'],TRAIN['validation_x'],TRAIN['validation_y'],NN,.2,3,.5,True],None,'Frozen hidden family')]],
 [(TRAIN,None,'Training and validation losses are measured separately.'),(dict(TRAIN,freeze_hidden=True),None,'Only the readout learns from fixed features.'),(dict(TRAIN,decay=.5),None,'Weight decay changes the update, not the reported data loss.'),(dict(TRAIN,validation_y=[1,-1]),None,'Changing only validation labels changes diagnostics, never final weights.')],
 'A small-data study needs a complete, inspectable training loop to compare feature learning, frozen-feature transfer and regularization. Implement the model and its experimental harness.',
 ('Disjoint training/validation arrays, initial weights and an SGD recipe.','Learning curves, final weights and data-exposure accounting.','The toy regression task uses no biases or stochastic batching. Repeated passes increase exposure, not unique data. All numerical outputs remain finite; no claim that a toy trend transfers to language or DNA models.'),
 ['Predict whether changing only validation labels can change trained weights.','Compare frozen features, full fine-tuning and decay with identical initial weights. Then change the number of passes without adding unique examples.'],
 [*alchemy('a6_generalization'),*alchemy('a2_hyperparameter_scaling','hyperparameter-scaling')],
 [('tiny_predict.py','math.tanh(sum(w*x for w,x in zip(row,features)))','sum(w*x for w,x in zip(row,features))'),('tiny_gradient.py','(1-a*a)','1'),('learning_curve.py','if not freeze_hidden:','if True:'),('learning_curve.py','dw+decay*w','dw')],['course-classifier','course-optimizer']))


def discounted_returns(rewards, gamma):
    """Return reward-to-go per step, G[t]=r[t]+gamma*G[t+1], terminal G[T]=0. 0<=gamma<=1. Empty episodes return []."""
    result=[0.0]*len(rewards);total=0.0
    for i in range(len(rewards)-1,-1,-1):total=rewards[i]+gamma*total;result[i]=total
    return result


def audit_episode(events, gamma, time_penalty, coin_reward, goal_reward, repeat_coins):
    """Events are 'move', 'coin:<id>', or 'goal'. Stop at FIRST goal; ignore later events. Each processed event pays -time_penalty, plus coin_reward on the first visit to a coin (or every visit if repeat_coins), or goal_reward at goal. Return proxy_return (discounted from start), true_success (0/1), steps, unique_coins and returns. A time penalty also applies at the goal. No implicit terminal bonus without a goal."""
    seen=set();rewards=[];success=0
    for event in events:
        reward=-time_penalty
        if event.startswith('coin:'):
            coin=event[5:]
            if repeat_coins or coin not in seen:reward+=coin_reward
            seen.add(coin)
        if event=='goal':reward+=goal_reward;success=1
        rewards.append(reward)
        if success:break
    returns=discounted_returns(rewards,gamma)
    return dict(proxy_return=returns[0] if returns else 0.0,true_success=success,steps=len(rewards),unique_coins=len(seen),returns=returns)


def reward_audit(policies, gamma, time_penalty, coin_reward, goal_reward, repeat_coins):
    """policies maps names to nonempty lists of event episodes. Return mean proxy reward and success_rate per policy, selected name by MAX proxy (lexical name tie), and mismatch=true iff selected success rate is below the best observed success rate. No policies gives selected=null,mismatch=false. Evaluate fixed traces; this is not RL training."""
    results={}
    for name,episodes in policies.items():
        rows=[audit_episode(events,gamma,time_penalty,coin_reward,goal_reward,repeat_coins) for events in episodes]
        results[name]=dict(proxy=sum(r['proxy_return'] for r in rows)/len(rows),success_rate=sum(r['true_success'] for r in rows)/len(rows))
    chosen=min(results,key=lambda n:(-results[n]['proxy'],n)) if results else None
    mismatch=bool(chosen is not None and results[chosen]['success_rate']<max(r['success_rate'] for r in results.values()))
    return dict(policies=results,selected=chosen,mismatch=mismatch)

RA=dict(policies={'finish':[['move','goal']],'farm':[['coin:a']*4]},gamma=1,time_penalty=0,coin_reward=1,goal_reward=3,repeat_coins=True)
LABS.append(make('reward-audit','Detect a reward specification that favors the wrong behavior','Empirical deep learning',['alchemy','cs336'],
 [(discounted_returns,'Compute episodic credit',['Reward-to-go looks forward, not backward from the start.','Discount each additional future step once.']),
  (audit_episode,'Replay the reward contract',['Separate measured proxy reward from actual goal completion.','Termination and repeated pickups need explicit policies.']),
  (reward_audit,'Compare the proxy winner to task success',['Choose using the proxy, then audit the separate goal metric.','Report a mismatch instead of silently redefining the winner.'])],
 [[([[1,2,3],.5],[2.75,3.5,3],'Discounted future rewards'),([[],1],[],'Empty episode'),([[1,2],0],[1,2],'Immediate reward'),([[-1,3],1],[2,3],'Undiscounted return')],
  [([['move','goal'],1,1,2,5,False],{'proxy_return':3,'true_success':1,'steps':2,'unique_coins':0,'returns':[3,4]},'Terminal success'),([['coin:a','coin:a'],1,0,2,5,False],{'proxy_return':2,'true_success':0,'steps':2,'unique_coins':1,'returns':[2,0]},'Single pickup'),([['goal','coin:a'],1,0,2,5,True],{'proxy_return':5,'true_success':1,'steps':1,'unique_coins':0,'returns':[5]},'Stop at terminal'),([['coin:a','coin:a'],.5,0,2,5,True],{'proxy_return':3,'true_success':0,'steps':2,'unique_coins':1,'returns':[3,2]},'Repeatable loophole')],
  [([*RA.values()],{'policies':{'finish':{'proxy':3,'success_rate':1},'farm':{'proxy':4,'success_rate':0}},'selected':'farm','mismatch':True},'Proxy mismatch'),([{},1,0,1,3,True],{'policies':{},'selected':None,'mismatch':False},'No policies'),([RA['policies'],1,0,1,3,False],None,'One-time pickup repair'),([{'b':[['goal']],'a':[['goal']]},1,0,1,3,True],None,'Lexical tie')]],
 [(RA,None,'The proxy winner can fail the actual task.'),(dict(RA,repeat_coins=False),None,'A one-time pickup rule changes the incentive on the same traces.'),(dict(RA,gamma=.5),None,'Discounting changes the relative value of delayed rewards.'),(dict(RA,time_penalty=1),None,'A per-step cost changes long-trajectory incentives.')],
 'A policy evaluation dashboard reports better reward while actual task completion falls. Build a replay audit that exposes a repeatable-reward loophole.',
 ('Named policies with fixed event traces and an explicit reward/discount configuration.','Proxy scores, true success rates, selected policy and objective mismatch.','Allowed events are documented; IDs are nonempty strings. The exercise audits observations and incentives, not learned policies or causal effects of retraining.'),
 ['Predict the proxy winner before running, then check the separate success metric.','Change only the repeat-pickup rule. Does a repaired trace score prove a retrained agent cannot find another loophole?'],
 [*alchemy('a7_exotic_rl'),source('CS336 2025 · alignment and reward-based learning',CS336)],
 [('discounted_returns.py','gamma*total','total'),('audit_episode.py','repeat_coins or coin not in seen','True'),('audit_episode.py','if success:break','if False:break'),('reward_audit.py',"-results[n]['proxy']","results[n]['proxy']")],['course-policy-objective']))
