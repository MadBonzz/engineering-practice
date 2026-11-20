"""Original data, tokenization, resource allocation and policy-learning tasks."""
import math
import copy
import itertools
from collections import Counter
from common import make, source, A1, A4, A5, CS336, alchemy

LABS=[]


def merge_pair(tokens, pair, new_id):
    """Replace non-overlapping adjacent occurrences of pair by new_id, scanning left to right. Return a new token list; overlapping matches are not both consumed."""
    out=[];i=0
    while i<len(tokens):
        if i+1<len(tokens) and tokens[i:i+2]==pair:out.append(new_id);i+=2
        else:out.append(tokens[i]);i+=1
    return out


def learn_merges(documents, budget):
    """UTF-8 encode each document separately (initial IDs 0..255). Repeat up to budget times: count adjacent pairs across documents, choose highest frequency then lexicographically smallest pair of IDs, assign new ID 256+step and replace that pair in every document. Stop when no pair remains. Return [left,right,new_id] merges. Never merge across documents; empty documents allowed. This is byte-level BPE without regex pretokenization or special tokens."""
    sequences=[list(text.encode('utf-8')) for text in documents];merges=[]
    for step in range(budget):
        counts=Counter(pair for row in sequences for pair in zip(row,row[1:]))
        if not counts:break
        pair=min(counts,key=lambda p:(-counts[p],p));new=256+step
        sequences=[merge_pair(row,list(pair),new) for row in sequences];merges.append([*pair,new])
    return merges


def tokenize(documents, texts, budget):
    """Train merges on documents, apply them to each text in learned order, and decode through the learned byte vocabulary. Return merges, encoded token lists, decoded strings. Unseen Unicode must round-trip; text boundaries cannot merge. No normalization or special-token interpretation."""
    merges=learn_merges(documents,budget);vocab={i:bytes([i]) for i in range(256)}
    for a,b,new in merges:vocab[new]=vocab[a]+vocab[b]
    encoded=[]
    for text in texts:
        row=list(text.encode('utf-8'))
        for a,b,new in merges:row=merge_pair(row,[a,b],new)
        encoded.append(row)
    return dict(merges=merges,encoded=encoded,decoded=[b''.join(vocab[i] for i in row).decode('utf-8') for row in encoded])

U=dict(documents=['abab','ab'],texts=['abab','cab',''],budget=2)
LABS.append(make('byte-tokenizer','Build a reversible byte-pair tokenizer','Language-model pipeline',['cs336'],
 [(merge_pair,'Apply one ranked merge',['Consume two symbols only when a match is taken.','Overlapping candidates share input positions.']),
  (learn_merges,'Train deterministic merge ranks',['Count occurrences, not just distinct documents.','Document boundaries are hard merge barriers.']),
  (tokenize,'Encode and decode unseen text',['IDs expand to bytes, not individual Unicode characters.','Apply rules in training order, not by numeric frequency at inference.'])],
 [[([[1,1,1],[1,1],256],[256,1],'Overlapping pair'),([[],[1,2],256],[],'Empty input'),([[1,2,1,2],[1,2],256],[256,256],'Two matches'),([[1,2,3],[2,3],256],[1,256],'Tail match')],
  [([['ab'],1],[[97,98,256]],'One pair'),([['a','b'],2],[],'Boundary isolation'),([['abab','ab'],2],[[97,98,256],[256,256,257]],'Frequency and hierarchy'),([['bc','ab'],1],[[97,98,256]],'Lexicographic tie')],
  [([['ab'],['ab'],1],{'merges':[[97,98,256]],'encoded':[[256]],'decoded':['ab']},'Round trip'),([[],['é'],0],{'merges':[],'encoded':[[195,169]],'decoded':['é']},'Unseen UTF-8'),([['aaaa'],['aaa'],2],None,'Overlapping inference'),([['ab'],['a','b','🙂'],3],None,'Separate text boundaries')]],
 [(U,None,'A frequent pair is merged before a pair of merged tokens.'),(dict(documents=[],texts=['é','🙂',''],budget=0),None,'The byte vocabulary handles unseen Unicode.'),(dict(documents=['a','b'],texts=['ab'],budget=5),None,'Training never invents a cross-document merge.'),(dict(documents=['bc','ab'],texts=['abc'],budget=1),None,'Ties have a reproducible contract.')],
 'An ingestion service needs a tokenizer that can be trained on small shards and reproduce token IDs exactly. Implement merge training and lossless text transport.',
 ('Training documents, texts to encode, and a nonnegative merge budget.','Merge ranks, encoded sequences and reconstructed strings.','Each document is one pretokenized segment; no GPT-style regex, special-token policy or vocabulary pruning. Ties choose the smallest ID pair. Inputs are unchanged.'),
 ['Predict the tokenization of a repeated byte when merge candidates overlap.','Train on ASCII and encode an unseen emoji. Explain why decoding remains possible.'],
 [source('CS336 2025 · A1, byte-level BPE',A1)],
 [('merge_pair.py','i+=2','i+=1'),('learn_merges.py','-counts[p]','counts[p]'),('learn_merges.py',"sequences=[list(text.encode('utf-8')) for text in documents]","sequences=[list(''.join(documents).encode('utf-8'))]"),('tokenize.py','for a,b,new in merges:row=','for a,b,new in merges[::-1]:row=')]))


def canonical_words(text):
    """Normalize with Unicode casefold and split on whitespace. Punctuation is preserved; empty/whitespace-only text becomes []."""
    return text.casefold().split()


def shingles(text, width):
    """Return sorted unique word-window tuples as lists, using canonical_words. Positive width. A nonempty text shorter than width contributes its entire word tuple; empty text contributes no shingles."""
    words=canonical_words(text)
    if not words:return []
    return [list(x) for x in sorted(set(tuple(words[i:i+width]) for i in range(max(1,len(words)-width+1))))]


def curate(documents, holdout, width, threshold, min_words):
    """Process documents [{id,text}] in input order; IDs are unique. Reject too_short first (<min_words), then contaminated if Jaccard shingle similarity with ANY holdout >=threshold, then duplicate if similar to ANY previously KEPT document. Empty sets have similarity 0. Return kept IDs and rejected [{id,reason}]. 0<threshold<=1, min_words>=1. This exact small-corpus filter does not use MinHash or infer language/quality."""
    kept=[];rejected=[];seen=[];blocked=[set(map(tuple,shingles(t,width))) for t in holdout]
    def similar(a,b):return len(a&b)/len(a|b) if a|b else 0.0
    for row in documents:
        words=canonical_words(row['text']);s=set(map(tuple,shingles(row['text'],width)))
        reason='too_short' if len(words)<min_words else 'contaminated' if any(similar(s,b)>=threshold for b in blocked) else 'duplicate' if any(similar(s,b)>=threshold for b in seen) else None
        if reason:rejected.append(dict(id=row['id'],reason=reason))
        else:kept.append(row['id']);seen.append(s)
    return dict(kept=kept,rejected=rejected)

D=dict(documents=[{'id':'a','text':'Red blue green'},{'id':'b','text':'red BLUE green'},{'id':'c','text':'exam answer'},{'id':'d','text':' '}],holdout=['Exam answer'],width=2,threshold=1,min_words=2)
LABS.append(make('data-curation','Deduplicate training data without leaking evaluation text','Language-model pipeline',['cs336','alchemy'],
 [(canonical_words,'Canonicalize comparisons',['Case folding is stronger than ASCII lowercasing.','Preserve the specified punctuation behavior.']),
  (shingles,'Represent local overlap',['Use sets so repetition cannot inflate similarity.','Handle short and empty documents explicitly.']),
  (curate,'Enforce a deterministic admission policy',['Rejected documents must not become duplicate representatives.','Check contamination before ordinary deduplication.'])],
 [[([' A\tB  '],['a','b'],'Whitespace'),(['Straße'],['strasse'],'Unicode folding'),([''],[],'Empty document'),(['Hi, HI'],['hi,','hi'],'Punctuation preserved'),(['GROSSE Straße'],['grosse','strasse'],'Unicode held-out fold')],
  [(['a b c',2],[['a','b'],['b','c']],'Word windows'),(['a',3],[['a']],'Short document'),(['a a a',1],[['a']],'Set semantics'),(['',2],[],'Empty shingles'),(['solo',5],[['solo']],'Short held-out text')],
  [([*D.values()],{'kept':['a'],'rejected':[{'id':'b','reason':'duplicate'},{'id':'c','reason':'contaminated'},{'id':'d','reason':'too_short'}]},'Admission reasons'),([[],[],2,.5,1],{'kept':[],'rejected':[]},'Empty corpus'),([[{'id':'x','text':'a b'},{'id':'y','text':'b c'},{'id':'z','text':'c d'}],[],1,1/3,1],{'kept':['x','z'],'rejected':[{'id':'y','reason':'duplicate'}]},'Only kept representatives'),([[{'id':'x','text':'a b'}],['b c'],1,1/3,1],{'kept':[],'rejected':[{'id':'x','reason':'contaminated'}]},'Threshold equality')]],
 [(D,None,'Duplicates and evaluation contamination have different reasons.'),(dict(D,holdout=[]),None,'Removing the held-out corpus changes only contamination filtering.'),(dict(documents=[{'id':'x','text':'a b'},{'id':'y','text':'b c'},{'id':'z','text':'c d'}],holdout=[],width=1,threshold=1/3,min_words=1),None,'Duplicate suppression is not transitive through rejected records.'),(dict(D,min_words=10),None,'Length filtering runs first.')],
 'A training corpus contains duplicate records and fragments from evaluation material. Implement a small deterministic admission filter with auditable rejection reasons.',
 ('Ordered uniquely identified documents, held-out texts and filter parameters.','Kept IDs and ordered rejection reasons.','No data downloads. Similarity uses exact normalized shingle sets; no transitive clustering or semantic similarity. Rejected rows never become representatives.'),
 ['Predict whether input order can change the retained representative.','Vary the overlap threshold and explain the tradeoff between deduplication and retaining useful data.'],
 [source('CS336 2025 · A4, filtering and deduplication',A4),*alchemy('a6_generalization')],
 [('canonical_words.py','text.casefold()','text.lower()'),('shingles.py','max(1,len(words)-width+1)','max(0,len(words)-width+1)'),('curate.py','similar(s,b)>=threshold','similar(s,b)>threshold'),('curate.py',"else:kept.append(row['id']);seen.append(s)","else:kept.append(row['id'])")],['course-byte-tokenizer']))


def document_stream(documents, eos):
    """Concatenate token documents, appending eos to EACH document (including empty ones). Return {tokens,segments}; each token gets its source document index. EOS is reserved and absent from document contents."""
    tokens=[];segments=[]
    for i,doc in enumerate(documents):tokens.extend(doc+[eos]);segments.extend([i]*(len(doc)+1))
    return dict(tokens=tokens,segments=segments)


def pack_windows(documents, eos, pad, block):
    """Make nonoverlapping input blocks from document_stream, with targets shifted by one in the FULL stream. Pad the final block. mask[t]=1 iff a next token exists and belongs to the same document, otherwise 0. Return [{inputs,targets,mask}], retaining EOS-only blocks. block>0, eos/pad distinct reserved IDs."""
    stream=document_stream(documents,eos);tokens=stream['tokens'];segments=stream['segments'];out=[]
    for start in range(0,len(tokens),block):
        x=[];y=[];mask=[]
        for i in range(start,start+block):
            x.append(tokens[i] if i<len(tokens) else pad);y.append(tokens[i+1] if i+1<len(tokens) else pad)
            mask.append(int(i+1<len(tokens) and segments[i]==segments[i+1]))
        out.append(dict(inputs=x,targets=y,mask=mask))
    return out


def corpus_loss(documents, eos, pad, block, logits):
    """Logits have one block×V matrix per packed window; V includes all token IDs. Return windows, tokens (scored target count), nll (mean negative log probability) and perplexity=exp(nll). Ignore mask=0 positions; if no targets, nll/perplexity are null. Aggregate by scored tokens, not windows. Stable log-sum-exp required."""
    windows=pack_windows(documents,eos,pad,block);loss=0.0;count=0
    for window,scores in zip(windows,logits):
        for target,mask,row in zip(window['targets'],window['mask'],scores):
            if mask:
                m=max(row);loss+=math.log(sum(math.exp(x-m) for x in row))+m-row[target];count+=1
    return dict(windows=windows,tokens=count,nll=loss/count if count else None,perplexity=math.exp(loss/count) if count else None)

E=dict(documents=[[0,1],[1]],eos=2,pad=3,block=3,logits=[[[0]*4 for _ in range(3)] for _ in range(2)])
LABS.append(make('packed-evaluation','Pack language-model batches and score only valid targets','Language-model pipeline',['cs336','alchemy'],
 [(document_stream,'Preserve document boundaries',['EOS belongs to the document it terminates.','Empty documents still contribute their EOS.']),
  (pack_windows,'Shift labels before blocking',['A block boundary can occur inside a document.','A document boundary must not train a cross-document prediction.']),
  (corpus_loss,'Aggregate token-level likelihood',['Mask before accumulating the loss.','A mean of window means misweights partially filled blocks.'])],
 [[([[[0],[1]],2],{'tokens':[0,2,1,2],'segments':[0,0,1,1]},'Two documents'),([[[]],2],{'tokens':[2],'segments':[0]},'Empty document'),([[],2],{'tokens':[],'segments':[]},'No documents'),([[[0,1,0]],2],{'tokens':[0,1,0,2],'segments':[0,0,0,0]},'Single stream')],
  [([[[0]],2,3,2],[{'inputs':[0,2],'targets':[2,3],'mask':[1,0]}],'One document'),([[],2,3,2],[],'No windows'),([[[0,1,0]],2,3,2],[{'inputs':[0,1],'targets':[1,0],'mask':[1,1]},{'inputs':[0,2],'targets':[2,3],'mask':[1,0]}],'Cross-block next token'),([[[0],[1]],2,3,3],None,'Boundary inside block')],
  [([[[0]],2,3,2,[[[0]*4,[0]*4]]],{'windows':[{'inputs':[0,2],'targets':[2,3],'mask':[1,0]}],'tokens':1,'nll':math.log(4),'perplexity':4},'Uniform likelihood'),([[],2,3,2,[]],{'windows':[],'tokens':0,'nll':None,'perplexity':None},'No scored tokens'),([[[0,1,0]],2,3,2,[[[0]*4,[0]*4],[[0,0,math.log(3),0],[1000,0,0,0]]]],None,'Unequal scoring counts'),([[[0]],2,3,2,[[[1000]*4,[0]*4]]],None,'Large stable logits')]],
 [(E,None,'Uniform predictions have perplexity equal to vocabulary size.'),(dict(documents=[[]],eos=2,pad=3,block=2,logits=[[[0]*4,[0]*4]]),None,'An EOS-only document supplies no next-token target.'),(dict(documents=[[0,1,0]],eos=2,pad=3,block=2,logits=[[[0]*4,[0]*4],[[0,0,math.log(3),0],[1000,0,0,0]]]),None,'The partial final block contributes one target.'),(dict(E,logits=[[[1000]*4 for _ in range(3)] for _ in range(2)]),None,'A logit shift does not change the score.')],
 'An evaluator produces different perplexities when sequence length changes. Repair packing, boundary masks and token-weighted aggregation.',
 ('Tokenized documents, reserved EOS/padding IDs, block size and compatible logit blocks.','Packed batches, scored-token count, NLL and perplexity.','Do not score a prediction from one document into the next. EOS targets inside a document do count. Logits are finite and resulting perplexity is representable.'),
 ['Predict whether changing only block length changes the scored-token count.','Compare a token-weighted loss with a mean of per-window losses on an uneven final block.'],
 [source('CS336 2025 · A1, batching and evaluation',A1),*alchemy('a6_generalization')],
 [('document_stream.py','len(doc)+1','len(doc)'),('pack_windows.py','i+1<len(tokens) and segments[i]==segments[i+1]','i+1<len(tokens)'),('corpus_loss.py','if mask:','if True:'),('corpus_loss.py','loss/count if count','loss/len(windows) if count')],['course-byte-tokenizer','course-classifier']))


def fit_power(points):
    """Fit y=A*x^(-alpha) by ordinary least squares on natural logs. points are positive [x,y], at least two distinct x. Return {coefficient:A, exponent:alpha}. y is an already isolated positive loss component, not total loss with an unknown floor."""
    xs=[math.log(x) for x,y in points];ys=[math.log(y) for x,y in points];mx=sum(xs)/len(xs);my=sum(ys)/len(ys)
    slope=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs)
    return dict(coefficient=math.exp(my-slope*mx),exponent=-slope)


def predicted_loss(parameters, tokens, model, data, floor):
    """Return floor + A*parameters^(-alpha) + B*tokens^(-beta), using coefficient/exponent fits. parameters,tokens>0; floor>=0. This is a stated additive model, not an empirical guarantee."""
    return floor+model['coefficient']*parameters**(-model['exponent'])+data['coefficient']*tokens**(-data['exponent'])


def allocate_compute(model_points, data_points, floor, candidates, budget):
    """Fit model/data components, evaluate candidate {id,parameters,tokens} plans with compute=6*parameters*tokens<=budget. Return fits and best {id,loss,compute}; best is null if infeasible. Choose minimum loss, then smaller compute, then lexical ID. Never select from evaluation data. Distinct IDs, positive sizes."""
    model=fit_power(model_points);data=fit_power(data_points);plans=[]
    for row in candidates:
        compute=6*row['parameters']*row['tokens']
        if compute<=budget:plans.append((predicted_loss(row['parameters'],row['tokens'],model,data,floor),compute,row['id']))
    best=None
    if plans:
        loss,compute,key=min(plans);best=dict(id=key,loss=loss,compute=compute)
    return dict(model=model,data=data,best=best)

F=dict(coefficient=4,exponent=1)
K=dict(model_points=[[1,4],[2,2],[4,1]],data_points=[[1,4],[2,2],[4,1]],floor=1,candidates=[{'id':'wide','parameters':4,'tokens':1},{'id':'balanced','parameters':2,'tokens':2}],budget=24)
LABS.append(make('scaling-budget','Choose a training plan under a compute budget','Empirical deep learning',['cs336','alchemy'],
 [(fit_power,'Fit isolated scaling components',['Fit logs, but report the coefficient in original units.','A decreasing curve has a positive exponent in this convention.']),
  (predicted_loss,'Combine model and data constraints',['Model and token deficits contribute separately.','Keep the irreducible floor outside the power terms.']),
  (allocate_compute,'Compare feasible plans',['Check compute before ranking predicted losses.','State the extrapolation assumptions with the selected plan.'])],
 [[([[[1,4],[2,2],[4,1]]],F,'Inverse scaling'),([[[1,3],[2,3]]],{'coefficient':3,'exponent':0},'Flat component'),([[[1,8],[4,4],[16,2]]],{'coefficient':8,'exponent':.5},'Square-root scaling'),([[[2,1],[4,.5],[8,.25]]],{'coefficient':2,'exponent':1},'Nonunit x')],
  [([2,2,F,F,1],5,'Balanced resources'),([4,1,F,F,1],6,'Token-limited plan'),([1,4,F,F,0],5,'No floor'),([4,4,F,F,2],4,'Both resources grow')],
  [([*K.values()],None,'Budget choice'),([K['model_points'],K['data_points'],1,[],24],{'model':F,'data':F,'best':None},'No candidates'),([K['model_points'],K['data_points'],1,[{'id':'z','parameters':2,'tokens':2},{'id':'a','parameters':2,'tokens':2}],24],None,'Lexical tie'),([K['model_points'],K['data_points'],1,K['candidates'],1],{'model':F,'data':F,'best':None},'No feasible plan')]],
 [(K,None,'Balanced model and data can beat a larger model at equal compute.'),(dict(K,budget=1),None,'A budget can exclude every candidate.'),(dict(K,candidates=K['candidates']+[{'id':'large','parameters':8,'tokens':8}],budget=384),None,'A larger budget admits another regime.'),(dict(K,data_points=[[1,8],[2,4],[4,2]]),None,'A different data term changes the tradeoff.')],
 'A research scheduler must choose among proposed training runs using measured small-scale trends. Build a transparent fitting and budget-selection tool.',
 ('Isolated component measurements, a loss floor, candidate model/token sizes and a FLOP budget.','Fitted coefficients/exponents and the best feasible proposal.','The 6ND training cost and additive loss law are explicit approximations. This exercise does not claim to recover scaling laws from arbitrary total-loss data.'),
 ['Fit the same curve with and without its largest-scale point. What evidence would justify extrapolating?','Hold FLOPs fixed and compare model-heavy and data-heavy plans. Which assumption controls the winner?'],
 [source('CS336 2025 · A3 and scaling lectures',CS336),*alchemy('a1_basics','basics')],
 [('fit_power.py','exponent=-slope','exponent=slope'),('predicted_loss.py','return floor+','return 0+'),('allocate_compute.py','compute<=budget','compute<budget'),('allocate_compute.py','loss,compute,key=min(plans)','loss,compute,key=max(plans)')],['course-optimizer']))


def group_advantages(rewards, epsilon):
    """Each inner reward list is one prompt's nonempty rollout group. Return (r-group_mean)/(population_std+epsilon) per group. epsilon>0; equal rewards yield zeros. Groups may have different sizes."""
    out=[]
    for row in rewards:
        mean=sum(row)/len(row);std=math.sqrt(sum((r-mean)**2 for r in row)/len(row))
        out.append([(r-mean)/(std+epsilon) for r in row])
    return out


def clipped_objective(new_logp, old_logp, advantage, mask, clip):
    """For one rollout return {objective,clipped_fraction}. Per valid token use min(exp(new-old)*A, clamp(exp(new-old),1-clip,1+clip)*A), averaged over mask=1 tokens. clipped_fraction counts ratios outside the interval regardless of advantage. Empty valid set gives zeros. 0<clip<1; finite log-ratios in [-50,50]."""
    terms=[];clipped=0
    for new,old,keep in zip(new_logp,old_logp,mask):
        if keep:
            ratio=math.exp(new-old);bounded=max(1-clip,min(1+clip,ratio));terms.append(min(ratio*advantage,bounded*advantage));clipped+=int(ratio<1-clip or ratio>1+clip)
    return dict(objective=sum(terms)/len(terms) if terms else 0.0,clipped_fraction=clipped/len(terms) if terms else 0.0)


def policy_report(rewards, rollouts, epsilon, clip):
    """rollouts mirrors rewards' group structure; each rollout has new_logp,old_logp,mask. Compute group advantages and per-rollout objectives. Return advantages, details, loss=-mean(objective over ALL rollouts, including empty masks). This deliberately uses equal rollout weighting, not global-token weighting; no KL term."""
    advantages=group_advantages(rewards,epsilon);details=[]
    for group,values in zip(rollouts,advantages):
        details.append([clipped_objective(r['new_logp'],r['old_logp'],a,r['mask'],clip) for r,a in zip(group,values)])
    flat=[r['objective'] for group in details for r in group]
    return dict(advantages=advantages,details=details,loss=-sum(flat)/len(flat) if flat else 0.0)

V=dict(rewards=[[0,2]],rollouts=[[{'new_logp':[0],'old_logp':[0],'mask':[1]},{'new_logp':[math.log(2)],'old_logp':[0],'mask':[1]}]],epsilon=1,clip=.2)
LABS.append(make('policy-objective','Audit a group-relative policy update','Language-model pipeline',['cs336','alchemy'],
 [(group_advantages,'Normalize within each prompt',['Do not mix easy and hard prompt groups.','The contract uses population variance and epsilon outside the square root.']),
  (clipped_objective,'Apply sign-sensitive clipping',['A negative advantage changes which ratio is pessimistic.','Mask prompt and padding positions before reducing.']),
  (policy_report,'Aggregate rollout objectives',['Match each rollout to its own group advantage.','Use the specified rollout weighting even for unequal lengths.'])],
 [[([[[0,2]],1],[[-.5,.5]],'Centered group'),([[[3,3],[1]],1],[[0,0],[0]],'Equal rewards'),([[[0,2],[10,14]],1],[[-.5,.5],[-2/3,2/3]],'Separate prompts'),([[[1,2,3]],.1],None,'Three-way group')],
  [([[math.log(2)],[0],1,[1],.2],{'objective':1.2,'clipped_fraction':1},'Positive advantage'),([[math.log(2)],[0],-1,[1],.2],{'objective':-2,'clipped_fraction':1},'Negative advantage'),([[math.log(.5)],[0],-1,[1],.2],{'objective':-.8,'clipped_fraction':1},'Lower clipped bound'),([[0],[0],1,[0],.2],{'objective':0,'clipped_fraction':0},'Masked rollout')],
  [([*V.values()],None,'Integrated objective'),([[],[],1,.2],{'advantages':[],'details':[],'loss':0},'Empty batch'),([[[1,1]],V['rollouts'],1,.2],None,'No relative signal'),([[[0,2]],[[dict(V['rollouts'][0][0],mask=[0]),V['rollouts'][0][1]]],1,.2],None,'Masked first rollout')]],
 [(V,None,'A positive-advantage ratio can hit the clipping ceiling.'),(dict(V,rewards=[[2,0]]),None,'Reversing reward order reverses the learning signal.'),(dict(V,rewards=[[1,1]]),None,'Equal rewards supply no within-group signal.'),(dict(V,rollouts=[[dict(V['rollouts'][0][0],mask=[0]),V['rollouts'][0][1]]]),None,'Empty response masks contribute zero under this contract.')],
 'A reasoning-model trainer reuses sampled responses across updates. Implement an auditable group-relative objective with response masking and sign-correct clipping.',
 ('Reward groups, old/current token log probabilities, masks, normalization epsilon and clipping width.','Group advantages, rollout objective diagnostics and a scalar loss.','This is an objective calculation, not a model trainer. It excludes KL, reward parsing and optimizer updates; no claims about achieving reasoning improvements.'),
 ['Predict which clipping bound matters when advantage is negative.','Compare equal-rollout and equal-token reductions when response lengths differ. What behavior does each favor?'],
 [source('CS336 2025 · A5, GRPO and masked reductions',A5),*alchemy('a7_exotic_rl')],
 [('group_advantages.py','r-mean','r'),('clipped_objective.py','min(ratio*advantage,bounded*advantage)','max(ratio*advantage,bounded*advantage)'),('clipped_objective.py','if keep:','if True:'),('policy_report.py','loss=-sum(flat)','loss=sum(flat)')],['chat-sft','dpo-loss']))
