"""Build downloadable student workspaces and validate private grading workloads."""
import ast
import inspect
import json
from pathlib import Path
from pprint import pformat
import runpy
import sys
import tempfile
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
DEFAULT = "release-comparison"
ORDER = ["kv_cache","live_batching","paged_cache","shared_prefix","speculative_decode","block_attention",
         "quantized_linear","chat_sft","lora_layer","dpo_loss","data_parallel","paired_bootstrap","judge_validation","durable_replay"]

# Each stage can exercise several interfaces of the same engineering deliverable.
STAGES = {
    "shared_prefix": [("Select a reusable snapshot",[0,1]),("Branch and cancel requests",[2])],
    "speculative_decode": [("Verify greedy proposals",[0]),("Verify a sampled block",[1,2])],
    "block_attention": [("Accumulate stable block statistics",[0,1]),("Stream an attention result",[2])],
    "dpo_loss": [("Implement the preference objective",[0,1,2])],
    "paired_bootstrap": [("Estimate paired improvements",[0,1]),("Construct the interval",[2])],
    "judge_validation": [("Audit swapped verdicts",[0,1]),("Report agreement by slice",[2])],
    "durable_replay": [("Resume a single tool call",[0,1]),("Recover interleaved turns",[2])],
    "quantized_linear": [("Quantize and run the layer",[0,1]),("Export the deployment format",[2])],
}
DURATION = dict(dpo_loss="30–45 min",judge_validation="40–60 min",paired_bootstrap="45–65 min",
                kv_cache="65–90 min",live_batching="65–90 min",paged_cache="60–80 min",
                shared_prefix="45–65 min",speculative_decode="60–85 min",block_attention="50–75 min",
                quantized_linear="45–65 min",chat_sft="45–70 min",lora_layer="60–85 min",
                data_parallel="50–75 min",durable_replay="55–80 min")

WORKLOAD = '''import copy
import math
import solution

# The transport uses JSON-shaped inputs and outputs. Each trial starts fresh.
MUTABLE = {mutable}
PUBLIC_API = {api}

def trial(name, args):
    if name not in PUBLIC_API:
        raise ValueError("Unknown operation")
    original = copy.deepcopy(args)
    answer = getattr(solution, name)(*args)
    changed = MUTABLE.get(name, [])
    if any(args[i] != original[i] for i in range(len(args)) if i not in changed):
        raise AssertionError("An input documented as read-only changed.")
    result = {{"return": answer}}
    if changed:
        result["state"] = [args[i] for i in changed]
    return result

def equivalent(actual, expected):
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)):
        return type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-8)
    if isinstance(expected, list):
        return isinstance(actual, (list, tuple)) and len(actual)==len(expected) and all(equivalent(a,e) for a,e in zip(actual,expected))
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys()==expected.keys() and all(equivalent(actual[k],v) for k,v in expected.items())
    return actual == expected
'''
def compact(value):
    return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(",",":"))

def run(source, lab, name, args):
    import types
    module = types.ModuleType("solution")
    exec(source,module.__dict__)
    previous = sys.modules.get("solution")
    try:
        sys.modules["solution"] = module
        harness = {}
        exec(lab,harness)
        return harness["trial"](name,json.loads(compact(args)))
    finally:
        if previous is None:
            del sys.modules["solution"]
        else:
            sys.modules["solution"] = previous

def make_lab(path):
    m = runpy.run_path(str(path))
    slug = path.stem.replace("_","-")
    names = m["PARTS"]
    stages = STAGES.get(path.stem,[(label,[i]) for i,label in enumerate(m["LABELS"])])
    source = path.read_text().split("# AUTHORING",1)[0].strip()+"\n"
    tree = ast.parse(source)
    funcs = {node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
    readme = [f"# {m['TITLE']}",m["INTRO"],
              "Use solution.py for your implementation. workload.py is the read-only scenario driver. Inputs use ordinary Python lists, numbers, strings and dictionaries. Preserve input arguments unless a function contract explicitly calls for an in-place update. Numeric answers use relative tolerance 1e-7 and absolute tolerance 1e-8."]
    hints, private, files = [],[],{}
    student = source.splitlines()
    mutable = {"advance":[3,4],"decode":[1]} if path.stem=="kv_cache" else ({"resume":[0,1]} if path.stem=="durable_replay" else {})
    harness = WORKLOAD.format(mutable=repr(mutable),api=repr(names))
    for name in reversed(names):
        node = funcs[name]
        # Replace the docstring and body, retaining the callable signature.
        doclines = textwrap.wrap(ast.get_docstring(node), width=80)
        student[node.body[0].lineno-1:node.end_lineno] = ['    """'+doclines[0]]+['    '+line for line in doclines[1:]]+['    """',f'    raise NotImplementedError("Complete {name} in solution.py")']
    starter = "\n".join(student)+"\n"
    tests = ["import unittest","from workload import equivalent, trial",""]
    for part,(label,indices) in enumerate(stages,1):
        readme.append(f"## Part {part} · {label}")
        tests += ["",f"class Part{part}(unittest.TestCase):"]
        stage_hints = []
        for i in indices:
            name = names[i]
            sig = str(inspect.signature(m[name]))
            readme.append(f"{name}{sig} — {ast.get_docstring(funcs[name])}")
            stage_hints += m["HINTS"][i].split("|")
            seen = set()
            for n in range(22):
                args = m["cases"](n)[i]
                key = compact(args)
                if key in seen: continue
                seen.add(key)
                expected = run(source,harness,name,args)
                if n < 2:
                    tests += [f"    def test_{name}_example_{n+1}(self):",
                              f"        args = {pformat(args, width=76)}",
                              f"        observed = trial({name!r}, args)",
                              f"        expected = {pformat(expected, width=76)}",
                              "        self.assertTrue(equivalent(observed, expected), (observed, expected))",""]
                else:
                    private.append(dict(part=part,title=f"{label} · {name}",actions=[name,args],expected=expected))
        hints.append(stage_hints)
    readme += ["## Running the workspace","Choose a part from the test menu, or run the full public suite. Locally use python -m unittest discover -s tests. Add your own checks to tests/test_mine.py. Submit a snapshot to evaluate additional workloads."]
    if "two_process.py" in m.get("EXTRA",{}):
        readme.append("For a real two-process CPU check, download the starter and run python two_process.py after completing the reduction.")
    if "compare_prefix.py" in m.get("EXTRA",{}):
        readme.append("Use compare_prefix.py for an independent full-prefix baseline. Locally run python compare_prefix.py to check every token and compare CPU execution time after implementing the cache.")
    files = {"solution.py":starter,"workload.py":harness,"README.md":"\n\n".join(readme)+"\n",
             "tests/test_public.py":"\n".join(tests)+"\n",
             "tests/test_mine.py":'import unittest\nfrom solution import *\n\n\nclass MyTests(unittest.TestCase):\n    """Add your own test methods here."""\n'}
    files.update(m.get("EXTRA",{}))
    # Run the exact exported public test file against starter and reference.
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for file,body in files.items():
            out=base/file
            out.parent.mkdir(parents=True,exist_ok=True)
            out.write_text(body)
        cmd=[sys.executable,"-B","-m","unittest","discover","-s","tests"]
        check=subprocess.run(cmd,cwd=base,capture_output=True,text=True)
        assert check.returncode and "NotImplementedError" in check.stderr and "ImportError" not in check.stderr, (slug,check.stderr)
        (base/"solution.py").write_text(source)
        check=subprocess.run(cmd,cwd=base,capture_output=True,text=True)
        assert check.returncode==0,(slug,check.stderr)
        for script in ("two_process.py","compare_prefix.py"):
            if script in files:
                subprocess.run([sys.executable,"-B",script],cwd=base,check=True)
    checks = {}; exec(harness.replace("import solution","solution = None"),checks)
    mutants = []
    for old,new in m["MUTANTS"]:
        assert old in source,(slug,old)
        bad = source.replace(old,new)
        for case in private:
            op,args = case["actions"]
            try: different = not checks["equivalent"](run(bad,harness,op,args),case["expected"])
            except Exception: different=True
            if different:
                mutants.append([old,new,case["title"]])
                break
        else: raise ValueError(f"Surviving mutant in {slug}: {old}")
    assert len(mutants)>=4 and all(any(c["part"]==i for c in private) for i in range(1,len(stages)+1))
    info = dict(slug=slug,title=m["TITLE"],group=m["GROUP"],intro=m["INTRO"],duration=DURATION[path.stem],
                labels=[s[0] for s in stages],parts=len(stages),language="Python")
    folder=PUBLIC/"sets"/slug
    folder.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(folder/"starter.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for file,body in files.items():
            entry=zipfile.ZipInfo(file,(2026,1,1,0,0,0))
            entry.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(entry,body)
    (folder/"exercise.json").write_text(compact(dict(info,files=files,editable=["solution.py","tests/test_mine.py"],hints=hints)))
    print(f"{slug}: {len(stages)} parts, {len(private)} unique private workloads, {len(mutants)} mutants rejected")
    return info,private,mutants

def main():
    entries = [dict(slug=DEFAULT, title="Release comparison",group="Evaluation",duration="30–45 min",parts=3)]
    private,report = {},{}
    for name in ORDER:
        info,cases,kills=make_lab(ROOT/"authoring"/"sets"/(name+".py"))
        entries.append({k:info[k] for k in ("slug","title","group","duration","parts")})
        private[info["slug"]]=cases
        report[info["slug"]]=kills
    (PUBLIC/"catalog.json").write_text(compact(entries))
    (ROOT/"worker"/"practice-cases.json").write_text(compact(private))
    (ROOT/"authoring"/"mutation-results.json").write_text(json.dumps(report,indent=2))
if __name__=="__main__":
    main()

