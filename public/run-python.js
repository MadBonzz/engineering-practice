self.onmessage = async ({data}) => {
  try {
    const base = new URL("/runtime/", self.location.href).href;
    importScripts(base + "pyodide.js");
    const py = await loadPyodide({indexURL: base});
    self.postMessage({status:"running"});
    py.FS.mkdirTree("/attempt/tests");
    for (const [path, source] of Object.entries(data.files)) {
      if (path.endsWith(".py") || path.endsWith(".json")) {
        const relativeDir = path.includes("/") ? path.slice(0,path.lastIndexOf("/")) : "";
        if(relativeDir) py.FS.mkdirTree("/attempt/"+relativeDir);
        py.FS.writeFile("/attempt/"+path,source,{encoding:"utf8"});
      }
    }
    py.FS.chdir("/attempt");
    py.globals.set("scope", data.scope || "all");
    await py.runPythonAsync("import sys, os, json\nsys.path[:0] = ['/attempt', '/attempt/tests']\nsys.dont_write_bytecode = True");
    if (data.sampleRun) {
      py.globals.set("sample_json",JSON.stringify(data.sample));
      const out=py.runPython("from gateway import handle\njson.dumps(handle(json.loads(sample_json)), allow_nan=False)");
      self.postMessage({value:JSON.parse(out)});
    } else if (data.cases) {
      py.globals.set("cases_json", JSON.stringify(data.cases));
      const protocol = data.protocol === "function" ? "from workload import trial\ndef replay_scenario(case):\n    return trial(*case)" : await fetch("/replay.py").then(r => { if (!r.ok) throw Error("Could not load replay harness"); return r.text(); });
      py.runPython(protocol);
      const out = py.runPython(`
import contextlib, io
outputs = []
for actions in json.loads(cases_json):
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            value = replay_scenario(actions)
        json.dumps(value, allow_nan=False)
        outputs.append({"value":value})
    except Exception as exc:
        outputs.append({"error":type(exc).__name__ + ": " + str(exc)[:400]})
json.dumps(outputs, allow_nan=False)
`);
      self.postMessage({outputs:JSON.parse(out)});
    } else {
      const out = py.runPython(`
import io, unittest, time, contextlib
class RunRecord(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.records = []
    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append({"name":str(test), "ok":True})
    def addFailure(self,test,err):
        super().addFailure(test,err)
        self.records.append({"name":str(test), "ok":False, "detail":self.failures[-1][1][-3500:]})
    def addError(self,test,err):
        super().addError(test,err)
        self.records.append({"name":str(test), "ok":False, "detail":self.errors[-1][1][-3500:]})
    def addSkip(self,test,reason):
        super().addSkip(test,reason)
        self.records.append({"name":str(test),"ok":False,"detail":"Skipped: "+reason})
loader = unittest.TestLoader()
if scope == "all":
    suite = loader.discover("/attempt/tests")
elif scope == "mine":
    suite = loader.loadTestsFromName("test_mine")
else:
    suite = loader.loadTestsFromName("test_public.Part"+scope)
stream, console = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
    result = unittest.TextTestRunner(stream=stream,resultclass=RunRecord,verbosity=2).run(suite)
json.dumps({"records":result.records,"console":(console.getvalue()+"\\n"+stream.getvalue())[-16000:]})
`);
      self.postMessage({publicRun:JSON.parse(out)});
    }
  } catch (e) { self.postMessage({error:String(e.message||e).slice(-6500)}); }
};
