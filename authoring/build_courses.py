"""Build original course labs and validate references, exports, private grading and mutants.

Run: python -B authoring/build_courses.py
Generated artifacts are committed; this command needs only standard-library Python.
"""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from build_papers import build, dump_files, reference_outputs, write_zip
from build_sets import ROOT, PUBLIC, compact

sys.path.insert(0, str(ROOT/'authoring/courses'))
MODULES=['foundations','systems','language','experiments']


def load_labs():
    return [lab for name in MODULES for lab in importlib.import_module(name).LABS]


def prepare(lab, order):
    lab=copy.deepcopy(lab);lab['order']=order
    homework={"course-classifier":0,"course-autodiff":1,"course-tensor-views":3,"course-normalization":2,"course-optimizer":2,"course-convolution-backward":4,"course-tiled-matmul":3,"course-generalization":0}
    if lab["slug"] in homework:
        n=homework[lab["slug"]]
        lab["source"].append(dict(title=f"DL Systems · Homework {n} (original notebook)",url=f"https://github.com/dlsyscourse/hw{n}/blob/main/hw{n}.ipynb"))
    questions='\n'.join(f'{i}. {question}' for i,question in enumerate(lab['questions'],1))
    reflection=f'''# Prediction and experiment notebook

## Before running
{questions}

For each comparison, write a prediction, confidence, and the mechanism you expect.

## Protocol
State the one variable changed, controls held fixed, initial state/seeds, and metric.
Implement the three parts and pass the public tests before interpreting a run.
Run `python experiment.py` locally, or use the editable JSON examples in the bench.
Edit `fixtures/experiments.json` locally to add a controlled intervention.

## Observations
Record the configuration and actual numerical result. Include failed predictions.

## Explanation and next test
Explain the discrepancy, propose a counterexample to your current explanation,
and name one limit on transferring this small experiment to a trained large model.

This notebook is saved/exported with your attempt. Coding checks grade implementation
behavior; they do not automatically assess the quality of your scientific explanation.
'''
    lab['extra_files']={
        'REFLECTION.md':reflection,
        'fixtures/experiments.json':json.dumps(lab['experiments'],indent=2)+'\n',
        'experiment.py':'''"""Reproduce each controlled input; save JSON output for your experiment notebook."""
import copy
import json
from pathlib import Path
from gateway import handle


def main():
    requests=json.loads(Path(__file__).with_name("fixtures").joinpath("experiments.json").read_text())
    for index,request in enumerate(requests,1):
        try:
            result=handle(copy.deepcopy(request))
        except NotImplementedError as error:
            raise SystemExit("Complete the lab before running experiments: " + str(error))
        print(json.dumps({"experiment":index,"input":request,"output":result},allow_nan=False))


if __name__ == "__main__":
    main()
'''}
    lab['extra_readme']='\n\n## Prediction and experimentation\n\n'+questions+'''

Open REFLECTION.md before running a comparison. Save a prediction, the controlled
change, the measured result, and your explanation. `python experiment.py` runs the
provided small experiments after implementation. These are original practice tasks
inspired by the linked courses, not reproductions of their assignments or quizzes.

## Files and validation

Implement the named functions in the three part modules. You may also edit
`tests/test_mine.py` and `REFLECTION.md`. The gateway, workload protocol, supplied
fixtures and public tests define the integration boundary. Inputs must not be mutated.
Inputs satisfy the documented preconditions; unspecified malformed inputs are not graded.
Numerical comparisons allow relative error 1e-7 or absolute error 1e-8.

Run all public tests: `python -m unittest discover -s tests`
Run one part: `python -m unittest discover -s tests -p test_public.py -k Part1`
Run examples: `python run_lab.py fixtures/example_1.json`
Exporting the attempt includes your code, notebook and experiment fixtures.
Private grading also exercises boundaries and cross-part behavior. A passing score
checks implementation correctness, not a claim that the toy model predicts real training.
'''
    return lab


def check_mutations(lab, scenarios, exercise):
    """Run the actual hidden action bank against each authored wrong implementation."""
    cases=[dict(fn=c['actions'][0],args=c['actions'][1]) for c in scenarios]
    expected=[c['expected'] for c in scenarios]
    reference={**exercise['files'],**lab['modules']}
    report=[]
    # Actual action protocol must preserve caller-owned nested inputs.
    code='''import copy,json,sys
from workload import trial,equivalent
cases=json.load(sys.stdin)
for c in cases:
    original=copy.deepcopy(c['args'])
    result=trial(c['fn'],c['args'])
    assert c['args']==original, ('Input mutation',c['fn'])
print('Input immutability passed')
'''
    with tempfile.TemporaryDirectory() as tmp:
        base=Path(tmp);dump_files(base,reference)
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=base,input=json.dumps(cases),text=True,capture_output=True)
        if result.returncode:raise AssertionError(result.stderr)
        # Smoke-test the shipped experiment runner against the complete reference.
        result=subprocess.run([sys.executable,'-B','experiment.py'],cwd=base,text=True,capture_output=True)
        if result.returncode:raise AssertionError(result.stderr)
        for file,old,new in lab['mutants']:
            assert old in reference[file],(lab['slug'],file,old)
            mutant=reference[file].replace(old,new)
            compile(mutant,file,'exec')
            (base/file).write_text(mutant)
            try:
                observed=reference_outputs(base,cases)
                # Match the application's recursive numeric tolerance.
                def same(a,b):
                    if isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)) and not isinstance(b,bool):
                        return abs(a-b)<=max(1e-8,1e-7*max(abs(a),abs(b)))
                    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
                    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
                    return a==b
                killed=[scenarios[i]['title'] for i,(a,b) in enumerate(zip(observed,expected)) if not same(a,b)]
            except ValueError:
                killed=['Private workload raised an exception']
            finally:
                (base/file).write_text(reference[file])
            assert killed,(lab['slug'],'Surviving mutant',file,old,new)
            report.append(dict(file=file,change=old+' => '+new,killed_by=killed[0]))
    assert len(report)>=4
    return report


def main():
    catalog=[];bank={};report={};failures=[]
    labs=load_labs();assert len({x['slug'] for x in labs})==len(labs)
    for i,raw in enumerate(labs,1):
        lab=prepare(raw,i)
        try:
            meta,private=build(lab,course=True)
            exercise=json.loads((PUBLIC/'courses'/lab['slug']/'exercise.json').read_text())
            kills=check_mutations(lab,private,exercise)
        except Exception as exc:
            failures.append((lab['slug'],str(exc)));print('FAILED',lab['slug'],str(exc),flush=True);continue
        catalog.append(meta);bank[lab['slug']]=private
        report[lab['slug']]=dict(public_checks=sum(c.get('public',False) for c in lab['checks']),private_checks=len(private),mutants=kills,reference='passed',starter='fails with NotImplementedError',input_immutability='passed',experiment_runner='passed')
        print(lab['slug'],len(private),'private checks;',len(kills),'mutants rejected',flush=True)
    if failures:raise AssertionError(failures)
    from curriculum import write_curriculum
    reused=write_curriculum(ROOT,catalog)
    (PUBLIC/'course-new-catalog.json').write_text(compact(catalog))
    (ROOT/'worker/course-cases.json').write_text(compact(bank))
    (ROOT/'authoring/courses/validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Validated',len(catalog),'new labs.')


if __name__=='__main__':main()
