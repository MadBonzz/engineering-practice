"""Build and verify the multi-file edition. Legacy workspaces remain readable."""
import ast
import copy
import inspect
import json
from pathlib import Path
from pprint import pformat
import runpy
import subprocess
import sys
import tempfile
import textwrap
import zipfile

from build_sets import ROOT, PUBLIC, ORDER, compact
from lab_designs import LABS, SESSION_POOL
import recovery_lab

CLI = '''"""Read one JSON problem input from a file or stdin and write the response."""
import json
import sys
from gateway import handle


def main():
    if len(sys.argv) == 2:
        with open(sys.argv[1], encoding="utf-8") as stream:
            request = json.load(stream)
    else:
        request = json.load(sys.stdin)
    print(json.dumps(handle(request), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
'''


def module_sources(source, groups, names, starter):
    """Resolve imports across domain modules while keeping the authored implementation intact."""
    parsed = ast.parse(source)
    owned = {fn: module for module, (_, funcs) in groups.items() for fn in funcs}
    funcs = {n.name: n for n in parsed.body if isinstance(n, ast.FunctionDef)}
    first = next(iter(groups))
    global_defs = {}
    for node in parsed.body:
        if isinstance(node, ast.Assign):
            global_defs[node.targets[0].id] = first
    owned.update(global_defs)
    result = {}
    for module, (description, symbols) in groups.items():
        nodes = [funcs[name] for name in symbols]
        uses = {n.id for node in nodes for n in ast.walk(node) if isinstance(n, ast.Name)}
        lines = ['"""' + description + '"""', '']
        for node in parsed.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)) and any((a.asname or a.name) in uses for a in node.names):
                lines.append(ast.get_source_segment(source, node))
        for other in groups:
            imports = sorted(symbol for symbol in uses if owned.get(symbol) == other and module != other)
            if imports:
                lines.append(f"from {other} import {', '.join(imports)}")
        if module == first:
            lines.extend(ast.get_source_segment(source,n) for n in parsed.body if isinstance(n, ast.Assign))
        for fn in nodes:
            body = ast.get_source_segment(source,fn)
            if starter and fn.name in names:
                doc = textwrap.fill(ast.get_docstring(fn),width=73, subsequent_indent="    ")
                body = (f"def {fn.name}({ast.unparse(fn.args)}):\n"
                        f'    """{doc}\n    """\n    raise NotImplementedError("{module}.{fn.name}")')
            lines.extend(["", "", body])
        result[module+".py"] = "\n".join(lines).strip()+"\n"
    return result


def sample_code(examples):
    return "import unittest\nfrom gateway import handle\nfrom workload import equivalent\n\n\nclass Examples(unittest.TestCase):\n"+ "\n".join(
        f"    def test_example_{i}(self):\n        request = {pformat(request,width=75)}\n        expected = {pformat(output,width=75)}\n        actual = handle(request)\n        self.assertTrue(equivalent(actual, expected), (actual, expected))\n"
        for i,(request,output,_) in enumerate(examples,1))


def request_for(slug, data, n):
    """Adversarial application traffic uses the same entry point as the examples."""
    if slug == "kv-cache":
        return dict(layers=1+n%3,events=[["a","push",[n]],["b","push",[]],["b","push",[3,n]],["a","push",[4]],["a","close",[]],["b","push",[0]],["a","push",[2]]])
    if slug == "chat-sft": return dict(zip(("chats","limit"),data[2]))
    if slug == "lora-layer": return dict(zip(("weight","a","b","x","target","alpha","lr"),data[1]))
    if slug == "data-parallel": return dict(zip(("weights","steps","lr"),data[2]))
    if slug == "block-attention": return dict(zip(("query","blocks","value_width"),data[2]))
    if slug == "paged-cache": return dict(zip(("blocks","block_size","events"),data[2]))
    if slug == "live-batching": return dict(zip(("requests","capacity"),data[2]))
    if slug == "dpo-loss": return dict(zip(("pairs","beta"),data[2]))
    if slug == "durable-replay":
        attempts = data[2][0]
        if n % 2 == 0:
            return dict(attempts=attempts)
        session, turn = attempts[0][:2]
        key = f"{len(session)}:{session}{turn}"
        checkpoint = {"journal": [[key,"intent","before restart"]],
                      "effects": {key:"historic result "+str(n)}}
        return dict(checkpoint=checkpoint, attempts=attempts)
    if slug == "judge-validation": return dict(rows=data[2][0])
    if slug == "paired-bootstrap": return dict(zip(("rows","mode","repeats","seed"),data[2]))
    if slug == "shared-prefix": return dict(events=data[2][0])
    if slug == "quantized-linear":
        row = data[0][0]
        return dict(weights=[row,[n*0.5]*len(row)],x=[1+j for j in range(len(row))],bias=[-1,n])
    if slug == "speculative-decode":
        if n%2: return dict(mode="greedy",draft=data[0][0],target=data[0][1])
        return dict(mode="sample",**dict(zip(("proposals","targets","drafts","uniforms","bonus_u"),data[2])))
    raise ValueError(slug)


def dump_files(base, files):
    for name,text in files.items():
        path = base/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text)


def build(entry, old_cases):
    slug = entry["slug"]
    orig = slug == "release-comparison"
    old = json.loads((PUBLIC / ("exercise.json" if orig else f"sets/{slug}/exercise.json")).read_text())
    d = LABS[slug]
    files = dict(old["files"])
    samples = [dict(input=i,output=o,why=w) for i,o,w in d.examples]
    modules = {name+".py":desc for name,(desc,_) in d.modules.items()}
    stages = []
    for section in files["README.md"].split("## Part ")[1:]:
        stages.append([line for line in section.split("## ")[0].splitlines()[1:] if line.strip()])
    if orig:
        refdir=ROOT/"assessments/release-comparison/private/reference_solution"
        reference = {p.name:p.read_text() for p in refdir.glob("*.py")}
        editable = old["editable"]
        files["workload.py"] = "import math\n\ndef equivalent(a,b):\n    return a == b\n"
        cases = []
    else:
        path = ROOT/"authoring/sets"/(slug.replace("-","_")+".py")
        meta = runpy.run_path(str(path))
        source = path.read_text().split("# AUTHORING",1)[0].strip()
        names = meta["PARTS"]
        reference = module_sources(source,d.modules,names,False)
        files.update(module_sources(source,d.modules,names,True))
        del files["solution.py"]
        editable = [module+".py" for module,(_,funcs) in d.modules.items() if any(x in names for x in funcs)]
        ownership = {fn:module for module,(_,fns) in d.modules.items() for fn in fns}
        imports = "\n".join(f"from {module} import {', '.join(fns)}" for module,(_,fns) in d.modules.items())
        driver = files["workload.py"].replace("import solution", imports+"\nfrom gateway import handle")
        driver = driver.replace(repr(names),"{"+", ".join(repr(n)+": "+n for n in names+["handle"])+"}").replace("getattr(solution, name)","PUBLIC_API[name]")
        files["workload.py"] = driver
        files["tests/test_mine.py"] = 'import unittest\nfrom gateway import handle\n\n\nclass MyTests(unittest.TestCase):\n    """Add your own regression tests here."""\n'
        if slug == "kv-cache":
            files["compare_prefix.py"] = files["compare_prefix.py"].replace("from solution import attend, project, serve","from model_ops import attend, project\nfrom stream_worker import serve")
            reference["session_pool.py"] = SESSION_POOL
            files["session_pool.py"] = SESSION_POOL.replace('        if request_id not in self.requests:\n            self.requests[request_id] = [[[], []] for _ in range(self.layers)]\n        return [decode(token, self.requests[request_id]) for token in tokens]', '        raise NotImplementedError("SessionPool.push")').replace("        self.requests.pop(request_id, None)", '        raise NotImplementedError("SessionPool.close")')
            modules["session_pool.py"]="Own an independent cache per client and free it on close."
            editable.append("session_pool.py")
            stages[-1].append("Complete SessionPool.push and close for interleaved requests. Each push must use the current state for that ID; close retires it. A fresh request may reuse the same ID.")
        if slug=="data-parallel":
            files["two_process.py"] = files["two_process.py"].replace("from solution import local, reduce_ranks","from rank_worker import local\nfrom collective import reduce_ranks")
        if slug == "durable-replay":
            files["journal.py"] = recovery_lab.JOURNAL
            files["score_client.py"] = recovery_lab.TOOL
            files["restart_demo.py"] = recovery_lab.DEMO
            files["tests/test_restart.py"] = recovery_lab.RESTART_TEST
            modules.update({"journal.py":"Read a turn's durable history and append a single event.",
                            "score_client.py":"Call the remote scorer with an idempotency key.",
                            "restart_demo.py":"Run a crash and a retry in two separate Python processes."})
            files["recovery.py"] = files["recovery.py"].replace("from effect_ids import effect_key", "from effect_ids import effect_key\nfrom journal import TurnJournal\nfrom score_client import ScoreClient")
            reference["recovery.py"] = recovery_lab.RECOVER
            files["turn_worker.py"] += recovery_lab.RESTORE_STUB
            reference["turn_worker.py"] += recovery_lab.RESTORE_REF
            stages[-1].append("turn_worker.py · continue_from(checkpoint, attempts) — Resume from copies of the supplied journal and effect store. Preserve recorded remote replies verbatim. Return only this invocation's replies and the full resulting stores. Run python restart_demo.py to exercise a restart in a second process.")
        editable.append("tests/test_mine.py")
        for lines in stages:
            for i,line in enumerate(lines):
                owner = ownership.get(line.split("(")[0])
                if owner: lines[i] = f"{owner}.py · {line}"
        cases = [dict(part=entry["parts"],title="End-to-end service request",actions=["handle",[request_for(slug,meta["cases"](i),i)]]) for i in range(8)]
    files["gateway.py"] = '"""Public service entry point. Each call receives one decoded JSON request."""\n'+d.gateway
    files["run_lab.py"] = CLI
    modules = {"gateway.py":"Start here. Follow the request from the public service into the components you own.",**modules}
    files["tests/test_examples.py"] = sample_code(d.examples)
    for i,s in enumerate(samples,1):
        files[f"fixtures/example_{i}.json"] = json.dumps(s["input"],indent=2,ensure_ascii=False)+"\n"
    readme = [f"# {entry['title']}",d.mission,"## Your task",d.inputs,d.outputs,"## Constraints",d.constraints,
              "## Codebase","Start at gateway.py. The command-line caller is run_lab.py. Standard-library Python only."]
    for name,desc in modules.items(): readme.append(f"- {name} ({'edit' if name in editable else 'provided'}) — {desc}")
    for i,s in enumerate(samples,1):
        readme += [f"## Example {i}","Input","\x60\x60\x60json\n"+json.dumps(s["input"],indent=2,ensure_ascii=False)+"\n\x60\x60\x60","Expected output","\x60\x60\x60json\n"+json.dumps(s["output"],indent=2,ensure_ascii=False)+"\n\x60\x60\x60",s["why"]]
    for i,lines in enumerate(stages,1):
        readme += [f"## Part {i} · {old['labels'][i-1]}",*lines]
    readme += ["## Run locally","python run_lab.py fixtures/example_1.json","python -m unittest discover -s tests","You can also pipe a JSON object to python run_lab.py. Use ordinary Python types behind the JSON interface. Numeric answers allow relative error 1e-7 or absolute error 1e-8."]
    files["README.md"] = "\n\n".join(readme)+"\n"
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        dump_files(base,files)
        cmd=[sys.executable,"-B","-m","unittest","discover","-s","tests"]
        starter = subprocess.run(cmd,cwd=base,capture_output=True,text=True)
        assert starter.returncode and "NotImplementedError" in starter.stderr and "ImportError" not in starter.stderr,(slug,starter.stderr)
        dump_files(base,reference)
        ref = subprocess.run(cmd,cwd=base,capture_output=True,text=True)
        assert ref.returncode==0,(slug,ref.stderr)
        if slug == "durable-replay":
            subprocess.run([sys.executable, "-B", "restart_demo.py"], cwd=base, capture_output=True, check=True)
        if not orig:
            code = """import json,sys
from workload import trial,equivalent
old,new=json.loads(sys.stdin.read())
for c in old:
    value=trial(*c["actions"])
    assert equivalent(value,c["expected"]),(c["actions"],value,c["expected"])
for c in new:
    c["expected"]=trial(*c["actions"])
print(json.dumps(new,allow_nan=False))
"""
            result=subprocess.run([sys.executable,"-B","-c",code],input=compact([old_cases,cases]),cwd=base,capture_output=True,text=True,check=True)
            cases=json.loads(result.stdout)
        for i,s in enumerate(samples,1):
            actual=json.loads(subprocess.check_output([sys.executable,"-B","run_lab.py",f"fixtures/example_{i}.json"],cwd=base))
            assert isinstance(actual,dict) and actual.keys() == s["output"].keys(), (slug,i,actual)
    details = dict(old,files=files,editable=editable,mission=d.mission,inputs=d.inputs,outputs=d.outputs,constraints=d.constraints,
                   modules=modules,examples=samples,stages=stages,level=d.level)
    folder=PUBLIC/"labs"/slug
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"exercise.json").write_text(compact(details))
    with zipfile.ZipFile(folder/"starter.zip","w") as archive:
        for name,content in files.items():
            info = zipfile.ZipInfo(name,(2026,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,content)
    return dict(**entry,level=d.level,summary=old["intro"],editable=editable), cases


if __name__=="__main__":
    entries = json.loads((PUBLIC/"catalog.json").read_text())
    old_cases=json.loads((ROOT/"worker/practice-cases.json").read_text())
    catalog, extra=[],{}
    for entry in entries:
        info,more=build(entry,old_cases.get(entry["slug"],[]))
        catalog.append(info)
        extra[entry["slug"]]=more
        print(info["title"], len(info["editable"])-1, "editable implementation modules")
    (PUBLIC/"laboratory.json").write_text(compact(catalog))
    (ROOT/"worker/lab-cases.json").write_text(compact(extra))

