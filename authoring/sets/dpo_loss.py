import math


def completion(logps, mask):
    """logps contains per-token log probabilities already aligned to the predicted token. Sum only entries whose binary completion mask is 1. Prompt/padding positions have 0. Empty completion scores zero."""
    return sum(p for p,m in zip(logps,mask) if m)


def margin(pair):
    """pair has chosen and rejected records. Each record has policy, reference and mask arrays of equal length. Return (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected) using completion sums."""
    def advantage(item):
        return completion(item["policy"],item["mask"]) - completion(item["reference"],item["mask"])
    return advantage(pair["chosen"])-advantage(pair["rejected"])


def batch(pairs, beta):
    """For beta>0 return [mean_loss, gradients] where each example loss is -log(sigmoid(beta*margin)). gradients[i] is derivative of the mean loss with respect to that example's unscaled margin. Handle extreme margins without overflow. Empty batch returns [None, []]."""
    if not pairs:
        return [None,[]]
    losses,gradients = [],[]
    for pair in pairs:
        z = beta*margin(pair)
        losses.append(max(-z,0)+math.log1p(math.exp(-abs(z))))
        small = math.exp(-abs(z))
        sigmoid_neg = (small if z>=0 else 1)/(1+small)
        gradients.append(-beta*sigmoid_neg/len(pairs))
    return [sum(losses)/len(pairs),gradients]

# AUTHORING
TITLE = "DPO preference loss"
GROUP = "Training"
INTRO = "Train on preferred and rejected completions using scores from a policy and a frozen reference. Implement the masked objective and check its update signal at extreme margins."
PARTS = ["completion","margin","batch"]
LABELS = ["Score completion tokens","Compare policy preferences","Reduce a stable batch loss"]
HINTS = ["The log probabilities are already aligned.|Padding is not a completion.", "Compute each model's preference on the same pair.|An identical policy and reference have zero margin.", "Evaluate both signs of a large margin.|The derivative is for the mean, so batch size matters."]
def cases(n):
    def pair(k):
        return {"chosen":{"policy":[-8,-k,-2],"reference":[-1,-2,-3],"mask":[0,1,1]},
                "rejected":{"policy":[-7,-1,-3],"reference":[-2,-k,-5],"mask":[0,1,0]}}
    k = [1,3,1000,0,5,7][n%6]
    shared = {"policy":[-1],"reference":[-1],"mask":[1]}
    return [[[-2,-k,-5],[0,1,n%2]],[pair(k)],[[pair(k),{"chosen":shared,"rejected":shared},pair(1)] if n!=5 else [],.1+n%3]]
MUTANTS = [("if m)","if not m)"),('advantage(pair["chosen"])-advantage','advantage(pair["chosen"])+advantage'),("beta*margin(pair)","margin(pair)"),("sigmoid_neg/len(pairs)","sigmoid_neg")]

