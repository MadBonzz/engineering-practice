"""Course-lab authoring helpers. This directory is never served as a web asset."""
import inspect
import textwrap

DLSYS = 'https://dlsyscourse.org/lectures/'
ALCHEMY = 'https://deep-learning-alchemy.github.io/'
CS336 = 'https://cs336.stanford.edu/spring2025/'
A1 = 'https://github.com/stanford-cs336/assignment1-basics/blob/6191c1a18088ec6edce02bad4451e445193abb8b/cs336_spring2025_assignment1_basics.pdf'
A2 = 'https://github.com/stanford-cs336/assignment2-systems/blob/e495ed740080661ab084914674d3e10048778889/cs336_spring2025_assignment2_systems.pdf'
A4 = 'https://github.com/stanford-cs336/assignment4-data/blob/5a5f890cd9b72733e6273a596c40b696df3aa9df/cs336_spring2025_assignment4_data.pdf'
A5 = 'https://github.com/stanford-cs336/assignment5-alignment/blob/master/cs336_spring2025_assignment5_alignment.pdf'


def source(title, url):
    return dict(title=title, url=url)


def alchemy(unit, quiz=None):
    result = [source('Deep Learning Alchemy · assignment ' + unit, ALCHEMY+'assets/assignments/'+unit+'.pdf')]
    if quiz:
        result.append(source('Deep Learning Alchemy · prediction quiz', ALCHEMY+'quiz-'+quiz+'.html'))
    return result


def make(slug, title, group, courses, functions, cases, requests, mission, contract,
         questions, sources, mutants, prerequisites=(), scope=None, extra=''):
    """Each function occupies one editable module; dependencies point to earlier parts."""
    modules = {}
    parts = []
    for index, (fn, heading, hints) in enumerate(functions, 1):
        name = fn.__name__
        imports = '\n'.join('from '+prev.__name__+' import '+prev.__name__ for prev, _, _ in functions[:index-1])
        modules[name+'.py'] = 'import math\nimport copy\nimport itertools\nfrom collections import Counter\n'+imports+'\n\n'+textwrap.dedent(inspect.getsource(fn))
        parts.append(dict(title=heading, file=name+'.py', funcs=[name], requirement=inspect.getdoc(fn), hints=hints))
    final = functions[-1][0].__name__
    modules['gateway.py'] = f'''"""JSON request boundary. Trace this caller before editing the implementation modules."""
from {final} import {final}


def handle(request):
    """Evaluate an independent request; caller-owned input must remain unchanged."""
    return {final}(**request)
'''
    checks=[]
    for index, ((fn, _, _), entries) in enumerate(zip(functions,cases),1):
        for n, entry in enumerate(entries):
            args, expected, label = entry
            checks.append(dict(part=index,fn=fn.__name__+'.'+fn.__name__,args=args,title=label,public=n<2,**({'oracle':expected} if expected is not None else {})))
    requests = [(r, functions[-1][0](**r) if o is None else o, label) for r,o,label in requests]
    for n,(request,expected,label) in enumerate(requests):
        checks.append(dict(part=len(parts),fn='gateway.handle',args=[request],oracle=expected,title=label,public=False))
    return dict(slug='course-'+slug,title=title,group=group,level='Advanced' if group in ['Training systems','Empirical deep learning'] else 'Intermediate',
        duration='60–100 min',mission=mission,inputs=contract[0],outputs=contract[1],constraints=contract[2],
        scope=scope or 'A small, deterministic implementation using standard-library Python. No GPU, paid service or dataset download is needed. The workload checks behavior, not wall-clock speed.',
        courses=courses,source=sources,modules=modules,parts=parts,checks=checks,
        examples=[dict(input=r,output=o,why=why) for r,o,why in requests[:2]],
        questions=questions,prerequisites=list(prerequisites),mutants=mutants,
        experiments=[r for r,_,_ in requests],extra=extra)
