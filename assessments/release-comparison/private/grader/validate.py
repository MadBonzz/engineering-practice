"""Build online fixtures with an independent oracle, then exercise real packages."""
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile
from contextlib import redirect_stdout

BASE = Path(__file__).resolve().parents[2]
APP = BASE.parents[1]
PRIVATE = BASE / "private"
STUDENT = BASE / "student"
REFERENCE = PRIVATE / "reference_solution"

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def expected(actions):
    history = []
    def read(key, model):
        rows = [r for r in history if r[0] == key and r[1] == model]
        return max(rows,key=lambda r:r[2])[3] if rows else None
    def pair(key):
        return [read(key,m) for m in ("baseline","candidate")]
    output = []
    for op,*args in actions:
        if op == "write":
            history.append(args)
            continue
        prompts = sorted({r[0] for r in history})
        if op == "get": value = read(*args)
        elif op == "pair":
            p = pair(*args)
            value = None if None in p else p
        elif op == "prompts": value = prompts
        else:
            pairs = [pair(k) for k in prompts if None not in pair(k)]
            w,l = pairs.count([False,True]),pairs.count([True,False])
            value = dict(paired=len(pairs),wins=w,losses=l,ties=len(pairs)-w-l,
                         decision="promote" if len(pairs)>=args[0] and w>l else "hold")
        output.append(value)
    return output

def evaluate(directory, cases):
    sys.path.insert(0,str(directory))
    try:
        replay = load(APP/"web/protocol.py","protocol").replay_scenario
        passed=[]
        for case in cases:
            try:
                value=replay(case["actions"])
                passed.append(json.dumps(value,sort_keys=True)==json.dumps(case["expected"],sort_keys=True))
            except Exception:
                passed.append(False)
        return passed
    finally:
        sys.path.pop(0)
        for mod in ("ledger","report","events"):
            sys.modules.pop(mod,None)

def main():
    suite=load(PRIVATE/"hidden_tests/cases.py","workloads")
    cases=[dict(part=part,title=title,actions=ops,expected=expected(ops)) for part,title,ops in suite.workloads()]
    (APP/"worker/cases.json").write_text(json.dumps(cases,separators=(",",":")))
    def public(directory):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/"submission"
            shutil.copytree(STUDENT,target)
            for f in directory.glob("*.py"): shutil.copy(f,target/f.name)
            result=subprocess.run([sys.executable,"-B","-m","unittest","discover","-s","tests"],cwd=target,capture_output=True,text=True)
            return result
    assert public(REFERENCE).returncode == 0
    assert public(STUDENT).returncode != 0
    assert all(evaluate(REFERENCE,cases))
    assert sum(evaluate(STUDENT,cases)) < len(cases)//3
    mutants=[
        ("Old delivery wins","ledger.py","event.revision > previous.revision","True"),
        ("Drops withdrawals","ledger.py","self._entries[key] = event","if event.correct is not None: self._entries[key] = event"),
        ("Treats false as missing","ledger.py","None in values","not all(values)"),
        ("Reverses model order","ledger.py","else values","else values[::-1]"),
        ("Strict sample minimum","report.py","paired >= min_pairs","paired > min_pairs"),
        ("Promotes tied changes","report.py","wins > losses","wins >= losses")]
    lines=[f"Reference passes all {len(cases)} private scenarios and public tests. The starter fails the graded public suite."]
    for name,filename,old,new in mutants:
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/"code"
            shutil.copytree(REFERENCE,target)
            f=target/filename
            s=f.read_text()
            assert old in s
            f.write_text(s.replace(old,new))
            kills=len(cases)-sum(evaluate(target,cases))
            assert kills>0,name
            lines.append(f"- {name}: rejected by {kills} private scenarios.")
    (PRIVATE/"mutation_report.md").write_text("\n".join(lines)+"\n")
    exclude={"__pycache__"}
    files={str(p.relative_to(STUDENT)):p.read_text() for p in sorted(STUDENT.rglob("*")) if p.is_file() and not exclude.intersection(p.parts)}
    manifest=json.loads((BASE/"manifest.json").read_text())
    bundle=dict(**manifest,files=files)
    (APP/"public/exercise.json").write_text(json.dumps(bundle))
    with zipfile.ZipFile(APP/"public/starter.zip","w",zipfile.ZIP_DEFLATED) as z:
        for path,data in files.items(): z.writestr(path,data)
    print("\n".join(lines))
if __name__=="__main__": main()
