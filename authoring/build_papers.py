"""Build paper workspaces and check their public examples, student starters, and grader."""
import ast
from collections import defaultdict
import json
from pathlib import Path
from pprint import pformat
import re
import runpy
import subprocess
import sys
import tempfile
import textwrap
import zipfile

from build_labs import CLI, dump_files, sample_code
from build_sets import PUBLIC, ROOT, WORKLOAD, compact

SOURCE = ROOT / "authoring" / "papers"
CAPSTONES = SOURCE / "capstones"
ORDER = ("transformer", "resnet", "word2vec", "bert", "gpt2", "vgg", "unet", "vae",
         "gan", "vit", "rnn", "lstm", "gru", "alexnet", "densenet", "ddpm", "llama",
         "gemma3", "deepseekv3", "trinity", "gptoss", "glm45", "kimi", "qwen", "inkling",
         "embeddinggemma", "llada", "tinyrecursive", "gateddelta")


def identifier(title):
    return re.sub("[^a-z0-9]", "", title.lower())


def model_name(paper):
    key = identifier(paper["title"])
    aliases = {"visiontransformer": "vit", "attentionisallyouneed": "transformer",
               "vanilla": "rnn", "arcee": "trinity", "originaltransformer": "transformer",
               "variationalautoencoder": "vae", "generativeadversarial": "gan"}
    for prefix, model in aliases.items():
        if key.startswith(prefix):
            return model
    return next((name for name in ORDER if key.startswith(name)), key)


def write_zip(path, files):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)


def stub(source, functions):
    lines = source.splitlines()
    nodes = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    for name in sorted(functions, key=lambda f: nodes[f].lineno, reverse=True):
        node = nodes[name]
        doc = ast.get_docstring(node, clean=False)
        if not doc:
            raise ValueError(f"Missing student-facing contract for {name}")
        node.body = [node.body[0], ast.Raise(exc=ast.Call(func=ast.Name(id="NotImplementedError",ctx=ast.Load()),args=[ast.Constant(name)],keywords=[]))]
        lines[node.lineno-1:node.end_lineno] = ast.unparse(node).splitlines()
    return "\n".join(lines)+"\n"


def driver(names, *, exact=False):
    # Case names remain fully qualified so modules can both define a forward().
    comparator = "def equivalent" + WORKLOAD.split("def equivalent", 1)[1]
    if exact:
        comparator = comparator.replace("math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-8)", "actual == expected")
    return ("import importlib\nimport math\n\n"
            f"ALLOWED = {pformat(sorted(set(names)))}\n\n"
            "def trial(name, args):\n"
            "    if name not in ALLOWED:\n        raise ValueError('Unknown interface')\n"
            "    module, function = name.rsplit('.', 1)\n"
            "    return getattr(importlib.import_module(module), function)(*args)\n\n"
            + comparator)


def article_readme(p, stages, systems=False):
    rows = ["# "+p["title"], p["mission"], "## Implementation scope",p["scope"],
            "## Input",p["inputs"],"## Output",p["outputs"],"## Constraints",p["constraints"]]
    for i,e in enumerate(p["examples"],1):
        rows += ["## Example "+str(i),"Input","~~~json\n"+json.dumps(e["input"],indent=2)+"\n~~~",
                 "Output","~~~json\n"+json.dumps(e["output"],indent=2)+"\n~~~",e["why"]]
    for i,(part,details) in enumerate(zip(p["parts"],stages),1):
        rows += [f"## Part {i} · {part['title']}",*details]
    rows += ["## Run locally","Standard-library Python 3.10 or newer. Start at gateway.py to trace the request." if systems else "Standard-library Python 3.10 or newer. Start at gateway.py to trace the model's entry point.",
             "~~~sh\npython run_lab.py fixtures/example_1.json\npython -m unittest discover -s tests\n~~~",
             ]
    if p.get("localRun"):
        rows += ["## Run the service locally", p["localRun"]]
    rows += ["## Reading"]+[f"- [{s['title']}]({s['url']})" for s in p["source"]]
    return "\n\n".join(rows)+"\n"


def reference_outputs(base, checks):
    code = ("import json,sys\nfrom workload import trial,equivalent\nresults=[]\n"
            "for case in json.load(sys.stdin):\n"
            "    value=trial(case['fn'], case['args'])\n"
            "    if 'oracle' in case: assert equivalent(value,case['oracle']),(case['title'],value,case['oracle'])\n"
            "    results.append(value)\n"
            "print(json.dumps(results,allow_nan=False))")
    proc=subprocess.run([sys.executable,"-B","-c",code],cwd=base,text=True,capture_output=True,
                        input=json.dumps(checks))
    if proc.returncode:
        raise ValueError(proc.stderr)
    return json.loads(proc.stdout)


def test_source(public, count):
    lines = ["import unittest","from workload import trial, equivalent",""]
    for part in range(1,count+1):
        lines += ["",f"class Part{part}(unittest.TestCase):"]
        for idx,(c,expected) in enumerate(public[part],1):
            name=re.sub(r"[^a-z0-9]+","_",c["title"].lower()).strip("_")[:80]
            lines += [f"    def test_{idx}_{name}(self):",f"        args = {pformat(c['args'],width=85)}",
                      f"        expected = {pformat(expected,width=85)}",f"        actual = trial({c['fn']!r}, args)",
                      "        self.assertTrue(equivalent(actual, expected), (actual, expected))",""]
    return "\n".join(lines)+"\n"


def run_suite(base, starter=False):
    result=subprocess.run([sys.executable,"-B","-m","unittest","discover","-s","tests"],
                          cwd=base,text=True,capture_output=True)
    if starter:
        if result.returncode == 0 or "NotImplementedError" not in result.stderr or "ImportError" in result.stderr:
            raise ValueError("Starter should fail on an unfinished method.\n"+result.stderr)
    elif result.returncode:
        raise ValueError(result.stderr)


def build(p, *, systems=False):
    name=p["slug"]
    prefix = "sys" if systems else "paper"
    assert re.fullmatch(prefix + r"-[a-z0-9-]+",name) and len(p["examples"])>=2 and p["source"],name
    parts=p["parts"]
    reference={name:textwrap.dedent(source).strip()+"\n" for name,source in p["modules"].items()}
    functions=defaultdict(list)
    stages=[]; descriptions={}
    for part in parts:
        module=part["file"]
        assert module in reference and part["hints"],(name,module)
        functions[module].extend(part["funcs"])
        docs={n.name:n for n in ast.parse(reference[module]).body if isinstance(n,ast.FunctionDef)}
        stages.append([part["requirement"]]+[f"{module} · {f}({ast.unparse(docs[f].args)}) — {ast.get_docstring(docs[f])}" for f in part["funcs"]])
        descriptions[module]=part["title"]
    assert len(functions)>=2,(name,"expected multiple editable modules")
    files=dict(reference)
    for path,names in functions.items():
        files[path]=stub(reference[path],names)
    for path,source in reference.items():
        descriptions.setdefault(path,ast.get_docstring(ast.parse(source)) or ("Run the workload." if systems else ("Run the complete example through the model." if path=="gateway.py" else "Provided model support.")))
    checks=p["checks"]
    files["workload.py"]=driver([c["fn"] for c in checks]+["gateway.handle"], exact=systems)
    files["run_lab.py"]=CLI
    files["tests/test_mine.py"]='import unittest\nfrom gateway import handle\n\n\nclass MyTests(unittest.TestCase):\n    """Add regression tests for your implementation here."""\n'
    files["tests/test_examples.py"]=sample_code([(e["input"],e["output"],e["why"]) for e in p["examples"]])
    for i,e in enumerate(p["examples"],1):
        files[f"fixtures/example_{i}.json"]=json.dumps(e["input"],indent=2)+"\n"
    files["README.md"]=article_readme(p,stages,systems)
    public=defaultdict(list);private=[]
    with tempfile.TemporaryDirectory() as tmp:
        base=Path(tmp)
        dump_files(base,{**files,**reference})
        expected=reference_outputs(base,checks)
        for check,answer in zip(checks,expected):
            part=check["part"]
            assert 1<=part<=len(parts),(name,check)
            if not public[part] and check["fn"]!="gateway.handle":
                public[part].append((check,answer))
            else:
                private.append(dict(part=part,title=check["title"],actions=[check["fn"],check["args"]],expected=answer))
        for part in range(1,len(parts)+1):
            assert public[part] and any(c["part"]==part for c in private),(name,part)
        assert sum(c["actions"][0]=="gateway.handle" for c in private)>=3,name
        files["tests/test_public.py"]=test_source(public,len(parts))
        dump_files(base,{"tests/test_public.py":files["tests/test_public.py"]})
        run_suite(base)
        for i in range(len(p["examples"])):
            subprocess.run([sys.executable,"-B","run_lab.py",f"fixtures/example_{i+1}.json"],cwd=base,capture_output=True,check=True)
        dump_files(base,files)
        run_suite(base,starter=True)
    base=PUBLIC/("systems" if systems else "papers")/name
    base.mkdir(parents=True,exist_ok=True)
    write_zip(base/"starter.zip",files)
    cap=CAPSTONES/model_name(p)
    training=not systems and cap.is_dir()
    if training:
        # Package the student starter only. Reference implementations stay in authoring.
        paths=["README.md","model.py","train.py","data.py"]+[n.name for n in sorted(cap.glob("test*.py"))]
        assert any(n.startswith("test") for n in paths)
        write_zip(base/"training.zip",{n:(cap/n).read_text() for n in paths})
    meta={k:p[k] for k in ("slug","title","group","level","duration")}
    meta.update(summary=p["mission"],parts=len(parts),editable=[*functions,"tests/test_mine.py"],training=training)
    exercise={**meta,"files":files,"labels":[s["title"] for s in parts],"stageFiles":[s["file"] for s in parts],"hints":[s["hints"] for s in parts],
              "stages":stages,"modules":{"gateway.py":descriptions["gateway.py"],**descriptions},
              **{k:p[k] for k in ("mission","inputs","outputs","constraints","scope","source","examples")}}
    if p.get("localRun"):
        exercise["localRun"] = p["localRun"]
    (base/"exercise.json").write_text(compact(exercise))
    return meta,private


if __name__=="__main__":
    sys.path.insert(0,str(SOURCE))
    papers=[]
    for path in sorted(SOURCE.glob("*.py")):
        papers.extend(runpy.run_path(str(path)).get("PAPERS",[]))
    assert len({p["slug"] for p in papers})==len(papers)
    papers.sort(key=lambda p:ORDER.index(model_name(p)) if model_name(p) in ORDER else len(ORDER))
    catalog=[];cases={}
    for paper in papers:
        try:
            meta, private=build(paper)
        except Exception as exc:
            raise RuntimeError(paper["slug"]) from exc
        catalog.append(meta);cases[paper["slug"]]=private
        print(paper["slug"],meta["parts"],len(private),flush=True)
    (PUBLIC/"paper-catalog.json").write_text(compact(catalog))
    (ROOT/"worker/paper-cases.json").write_text(compact(cases))
