"use strict";
(async function () {
  const $=s=>document.querySelector(s), all=s=>[...document.querySelectorAll(s)];
  function make(tag,cls,txt){const n=document.createElement(tag);if(cls)n.className=cls;if(txt!==undefined)n.textContent=txt;return n;}
  function button(label,cls,handler){const n=make("button",cls,label);n.onclick=handler;return n;}
  function message(title,body){const box=make("div","empty");box.append(make("span","terminal-sign","›_"));const col=make("div");col.append(make("strong","",title),make("p","",body));box.append(col);return box;}
  const stored={get(k){try{return JSON.parse(localStorage.getItem(k));}catch{return null;}},set(k,v){try{localStorage.setItem(k,JSON.stringify(v));return true;}catch{return false;}}};
  const bench = window.BenchProgress;
  const session = bench.session;
  const params = new URL(location.href).searchParams;
  const slug = params.get("set") || "release-comparison";
  const paper = slug.startsWith("paper-");
  const systems = slug.startsWith("sys-");
  const course = slug.startsWith("course-");
  const structured = paper || systems || course;
  const labTrack = course ? "courses" : systems ? "systems" : "papers";
  const libraryUrl=new URL(bench.url());
  if(structured)libraryUrl.searchParams.set("track",labTrack);
  if(params.get("from")==="courses"){
    libraryUrl.searchParams.set("track","courses");
    if(params.has("course"))libraryUrl.searchParams.set("course",params.get("course"));
  }
  const catalog = await fetch(course?"/course-new-catalog.json":systems?"/systems-catalog.json":paper?"/paper-catalog.json":"/laboratory.json").then(r=>r.json());
  if (!catalog.some(row=>row.slug===slug)) { location.replace(libraryUrl.href); return; }
  const original = slug === "release-comparison";
  const earlier = !structured && !original && params.get("edition") === "1";
  const edition = earlier ? "earlier" : "current";
  const assetBase = (structured?"/"+labTrack+"/":earlier?"/sets/":"/labs/")+slug+"/";
  const prefix = "bench.v1."+session+"."+(original ? "" : (earlier ? "" : "labs.")+slug+".");
  $("#library-link").href=libraryUrl.href;
  if(structured)$("#library-link").setAttribute("aria-label",course?"Back to course tasks":systems?"Back to CS systems":"Back to paper implementations");
  function editionNote() {
    const box=$("#edition-note");
    const hasOld = !original && bench.localState(slug,"earlier");
    if(!earlier && !hasOld) return;
    box.hidden=false;
    box.replaceChildren(document.createTextNode(earlier?"You’re in the earlier workspace. ":"You have work in the earlier edition. "));
    const link=make("a","",earlier?"Open the multi-file lab":"Reopen your earlier attempt");
    link.href=bench.url(slug,earlier?"current":"earlier");
    box.append(link);
  }
  if(!structured)editionNote();
  if(!structured && !original && !earlier && !bench.localState(slug,"earlier")) bench.sync().then(editionNote).catch(()=>{});
  let activitySent=false;
  function markWork(solved=false) {
    bench.mark(slug,edition,solved);
    if(!activitySent) {
      activitySent=true;
      api("activity",{method:"POST"}).catch(()=>{activitySent=false;});
    }
  }
  const endpoint="/api/"+session+"/";
  const status=(t,ok=false)=>{const s=$("#save-state");s.textContent=t;s.dataset.saved=ok?"yes":"no";};
  async function api(url,opts={}){
    const r=await fetch(endpoint+url+"?set="+encodeURIComponent(slug)+"&edition="+(earlier?"1":"2"),opts);const data=await r.json();
    if(!r.ok){const e=Error(data?.error||"Request failed.");e.status=r.status;throw e;}return {data,tag:r.headers.get("etag")};
  }
  const ex=await fetch(assetBase+"exercise.json").then(r=>{if(!r.ok)throw Error("Could not load exercise.");return r.json();});
  document.title = ex.title + " · Practice Bench";
  $("#topic").textContent = ex.group;
  $("h1").replaceChildren(document.createTextNode(ex.title), make("span", "", "."));
  $("#duration").textContent = ex.duration;
  $("#part-count").textContent = ex.parts + (ex.parts === 1 ? " part" : " parts");
  $(".intro").textContent = ex.mission || ex.intro;
  $("#starter-link").href = assetBase + "starter.zip";
  if(structured) {
    if(course)$("#paper-sources").setAttribute("aria-label","Course assignments and references");
    if(systems)$("#paper-sources").setAttribute("aria-label","Technical references");
    if(ex.localRun){
      $("#local-service").hidden=false;
      $("#local-service-help").textContent=ex.localRun+" If you have already edited the lab in the browser, use Export attempt to run your version locally.";
      $("#local-service-link").href=assetBase+"starter.zip";
      $("#local-service-link").download=slug+"-starter.zip";
    }
    $("#paper-context").hidden=false;
    $("#paper-scope").textContent=ex.scope;
    for(const ref of ex.source) {
      const anchor=make("a","text-action",ref.title+" ↗");anchor.href=ref.url;
      anchor.target="_blank";anchor.rel="noopener noreferrer";$("#paper-sources").append(anchor);
    }
    if(ex.training){
      $("#paper-training").hidden=false;
      $("#training-link").href=assetBase+"training.zip";
      $("#training-link").download=slug+"-pytorch.zip";
    }
    ex.labels.forEach((label,i)=>{
      const action=button("Part "+(i+1)+" · "+label,"paper-step",()=>{
        openFile(ex.stageFiles[i]);$("#suite").value=String(i+1);$("#hint-part").value=String(i+1);renderHints();
        const part=all("#parts .part")[i];part.open=true;part.scrollIntoView({block:"nearest"});
      });
      $("#paper-steps").append(action);
    });
  }
  if(course){
    $("#course-context").hidden=false;
    for(const question of ex.questions){$("#course-questions").append(make("li","",question));}
    $("#reflection-open").onclick=()=>openFile("REFLECTION.md");
    for(const prerequisite of ex.prerequisites){
      const link=make("a","text-action",prerequisite.replace(/^course-|^paper-/,"").replaceAll("-"," "));
      const u=new URL(bench.url(prerequisite));u.searchParams.set("from","courses");
      if(params.has("course"))u.searchParams.set("course",params.get("course"));link.href=u.href;$("#course-prerequisites").append(link);
    }
    $("#course-prerequisites-wrap").hidden=!ex.prerequisites.length;
  }
  const local=stored.get(prefix+"draft");let files={...ex.files},etag=local?.etag||null;
  let conflict=false, pendingSync=null, dirty=!!local?.dirty, changed=0, debounce;
  function merge(values){for(const k of ex.editable) if(typeof values?.[k]==="string")files[k]=values[k];}
  if(local)merge(local.files);
  const edits=()=>Object.fromEntries(ex.editable.map(k=>[k,files[k]]));
  function persist(){return stored.set(prefix+"draft",{files:edits(),dirty,etag});}
  let cloudRead=true;
  try{
    const r=await api("draft");
    if(!dirty){if(r.data)merge(r.data.files);etag=r.tag;persist();}
    else if(etag!==r.tag){
      if(r.data && ex.editable.every(k=>files[k]===r.data.files[k])){etag=r.tag;dirty=false;persist();}
      else{conflict=true;status("Draft conflict");$("#load-cloud").hidden=false;}
    }
  }catch{cloudRead=false;status("Local draft · cloud offline");}
  function sync(){
    if (pendingSync) return pendingSync;
    if (!dirty || conflict) return;
    pendingSync = writeDraft().finally(() => { pendingSync=null; });
    return pendingSync;
  }
  async function writeDraft(){
    const version=changed;
    try{
      const headers={"Content-Type":"application/json"};
      if(etag)headers["If-Match"]=etag;
      const r=await api("draft",{method:"PUT",headers,body:JSON.stringify({files:edits()})});
      etag=r.tag;dirty=version!==changed;persist();status(dirty?"Saving…":"Saved",!dirty);
    }catch(e){if(e.status===409){conflict=true;status("Draft conflict");$("#load-cloud").hidden=false;}else status("Saved locally · cloud offline");}
    finally{if(dirty&&version!==changed&&!conflict)queueSync();}
  }
  function queueSync(){clearTimeout(debounce);debounce=setTimeout(sync,650);}
  if(!conflict&&cloudRead)status(dirty?"Saving…":"Saved",!dirty);
  if(dirty&&!conflict)queueSync();
  $("#load-cloud").onclick=async()=>{
    if(!window.confirm("Replace your local edits with the saved cloud draft? Export your local attempt first if you want a copy."))return;
    try{const r=await api("draft");if(!r.data)throw Error("No cloud draft available.");merge(r.data.files);etag=r.tag;dirty=false;persist();location.reload();}catch(e){status(e.message);}
  };
  window.addEventListener("online",sync);
  let active=ex.editable[0],display="tests",lock=false,stopWorker=null, testRun=null,log="",runNote="",review=null;
  let last=stored.get(prefix+"submission")||null;
  let history=[];
  async function loadHistory(){try{const response=await api("submissions");history=response.data;if(!last&&history.length)last=history[0].id;}catch{}}
  await loadHistory();
  function showHistory(out){
    if(!history.length)return;
    const wrap=make("details","rec");wrap.append(make("summary","","Submission history ("+history.length+")"));
    for(const item of history){const b=button(new Date(item.created).toLocaleString(),"filebtn",()=>{last=item.id;review=null;restore();});wrap.append(b);}
    out.append(wrap);
  }
  const parts = ex.labels;
  const partSelect = $("#hint-part");
  const suite = $("#suite");
  suite.replaceChildren(new Option("All public tests", "all"));
  partSelect.replaceChildren();
  parts.forEach((label, index) => {
    const n = index + 1;
    partSelect.append(new Option(n + " · " + label, n));
    suite.append(new Option("Part " + n + " only", n));
  });
  suite.append(new Option("My tests", "mine"));
  $("#library-link").addEventListener("click",async event=>{
    if(event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    clearTimeout(debounce);
    if (pendingSync) await pendingSync;
    if (dirty && !conflict) await sync();
    location.assign(libraryUrl.href);
  });
  const spec=files["README.md"], sections=spec.split(/^## /m);
  parts.forEach((title,i)=>{
    const d=make("details","part");if(i===0)d.open=true;
    const h=make("summary");h.append(make("span","pnum","0"+(i+1)),document.createTextNode(title));d.append(h);
    const s=sections.find(x=>x.startsWith("Part "+(i+1)))||"";
    for(const p of ex.stages?.[i] || s.split("\n").slice(1).filter(Boolean))d.append(make("p","",p));$("#parts").append(d);
  });
  const editor=CodeMirror.fromTextArea($("#code"),{lineNumbers:true,mode:"python",indentUnit:4,tabSize:4,indentWithTabs:false,extraKeys:{Tab:cm=>cm.execCommand(cm.somethingSelected()?"indentMore":"insertSoftTab"),"Ctrl-Enter":()=>runTests(),"Cmd-Enter":()=>runTests()}});
  editor.getInputField().setAttribute("aria-label","Python code editor");
  const views=new Map();let opening=false;
  const tabPaths=[...ex.editable];
  function openFile(path){
    if(!(path in files))return;
    active=path;if(!tabPaths.includes(path))tabPaths.splice(ex.editable.length,1,path);
    opening=true;
    if(!views.has(path))views.set(path,new CodeMirror.Doc(files[path],path.endsWith(".py")?"python":null));
    editor.swapDoc(views.get(path));editor.setOption("readOnly",!ex.editable.includes(path));
    opening=false;$("#active-path").textContent=path;$("#edit-mode").textContent=ex.editable.includes(path)?"Editable":"Read only";
    $("#tabs").replaceChildren();for(const p of tabPaths){const b=button("","",()=>openFile(p));b.setAttribute("role","tab");b.setAttribute("aria-selected",String(p===path));b.append(make("span","",p.endsWith(".py")?"◇":"≡"),document.createTextNode(p.split("/").pop()));$("#tabs").append(b);}
    for(const b of all(".filebtn"))b.classList.toggle("sel",b.dataset.file===path);
    editor.refresh();pos();
  }
  function pos(){const p=editor.getCursor();$("#position").textContent="Ln "+(p.line+1)+", Col "+(p.ch+1);}
  editor.on("cursorActivity",pos);
  editor.on("change",()=>{if(opening||!ex.editable.includes(active))return;files[active]=editor.getValue();markWork();changed++;dirty=true;const ok=persist();if(!conflict)status(ok?"Saving…":"Cloud save pending");queueSync();});
  const paths = [...new Set([...Object.keys(ex.modules || {}), ...Object.keys(files)])];
  for(const path of paths) {
    const b=button(path,"filebtn",()=>openFile(path));b.dataset.file=path;
    b.append(make("span","",ex.editable.includes(path)?"edit":"provided"));
    if(ex.modules?.[path]) b.append(make("small","file-description",ex.modules[path]));
    $("#filetree").append(b);
  }
  openFile(active);
  function showPane(name) {
    for (const panel of all(".pane")) panel.hidden = panel.id !== name;
    for (const tab of all("[data-pane]")) {
      const active = tab.dataset.pane === name;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-pressed", String(active));
    }
  }
  for (const tab of all("[data-pane]")) tab.onclick = () => showPane(tab.dataset.pane);
  showPane("brief");
  for (const link of all(".mobile-jump")) {
    link.onclick = event => {
      event.preventDefault();
      document.querySelector(link.hash).scrollIntoView({block: "start"});
    };
  }
  $("#view-codebase").onclick=()=>showPane("files");
  if (ex.examples) {
    $("#problem-contract").hidden=false;
    $("#input-contract").textContent=ex.inputs;
    $("#output-contract").textContent=ex.outputs;
    $("#limits").textContent=ex.constraints.replace("Numeric tolerances are specified below.","Floating-point answers allow relative error 1e-7 or absolute error 1e-8.");
    const start=$("#start-here");
    start.append(document.createTextNode("Open "));
    start.append(button("gateway.py","text-action",()=>openFile("gateway.py")));
    start.append(document.createTextNode(" to trace the caller. Implement the missing behavior in "+ex.editable.filter(p=>p.endsWith(".py")&&!p.startsWith("tests/")).join(", ")+"."));
    for(const [index,sample] of ex.examples.entries()) renderExample(index,sample);
  }
  function pretty(value, indent=0) {
    const short=JSON.stringify(value);
    if(short.length + indent <= 52 || value===null || typeof value!=="object") return short;
    const pad=" ".repeat(indent+2), end=" ".repeat(indent);
    const array=Array.isArray(value);
    const entries=array ? value.map(x=>pretty(x,indent+2)) : Object.entries(value).map(([k,v])=>JSON.stringify(k)+": "+pretty(v,indent+2));
    return (array?"[":"{")+"\n"+pad+entries.join(",\n"+pad)+"\n"+end+(array?"]":"}");
  }
  function renderExample(index,sample) {
    const card=make("details","example-card");card.open=index===0;
    card.append(make("summary","","Example "+(index+1)));
    const body=make("div","example-body");card.append(body);
    const input=make("textarea","sample-input");
    input.value=pretty(sample.input);input.rows=Math.min(12,Math.max(3,input.value.split("\n").length));input.spellcheck=false;
    input.setAttribute("aria-label","Example "+(index+1)+" JSON input");
    body.append(make("h3","","Input · editable JSON"),input,make("h3","","Expected output for the example"),make("pre","",pretty(sample.output)),make("p","",sample.why));
    const result=make("pre","sample-result");result.hidden=true;result.setAttribute("aria-live","polite");
    const run=button("Run this input","subtle example-run",async()=>{
      if(lock)return;
      result.hidden=false;
      let request;
      try {request=JSON.parse(input.value);}
      catch {result.textContent="Invalid JSON. Check the input and try again.";return;}
      busy(true);markWork();result.textContent="Running your code…";
      try {
        const res=await execute({files:{...files},sample:request,sampleRun:true});
        result.textContent="Your output\n"+pretty(res.value);
      } catch(e) {result.textContent="Your code raised an error\n"+e.message;}
      finally {busy(false);}
    });
    body.append(run,result);$("#worked-examples").append(card);
  }
  $("#fullspec").onclick=()=>openFile("README.md");
  const hints = ex.hints;
  const levels=stored.get(prefix+"hints")||parts.map(()=>0);
  function renderHints(){const i=Number($("#hint-part").value)-1;$("#hint-list").replaceChildren();for(let j=0;j<levels[i];j++)$("#hint-list").append(make("p","","Hint "+(j+1)+" · "+hints[i][j]));$("#next-hint").disabled=levels[i]>=hints[i].length;$("#next-hint").textContent=levels[i]>=hints[i].length?"All hints revealed":levels[i]===hints[i].length-1?"Request a strong hint":"Reveal hint "+(levels[i]+1);}
  $("#hint-part").onchange=renderHints;$("#next-hint").onclick=()=>{const i=Number($("#hint-part").value)-1;levels[i]=Math.min(hints[i].length,levels[i]+1);stored.set(prefix+"hints",levels);renderHints();};renderHints();
  $("#resume").onclick=async()=>{const u=new URL(location.href);u.searchParams.set("set",slug);u.hash="desk="+session;try{await navigator.clipboard.writeText(u.href);$("#resume-status").textContent="Link copied";}catch{window.prompt("Copy this resume link",u.href);}};
  function download(name,content,type){const a=make("a");const u=URL.createObjectURL(new Blob([content],{type}));a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),2000);}
  function exportFiles(name,source,result){
    const pkg={};for(const [k,v] of Object.entries(source))pkg[k]=fflate.strToU8(v);
    if(result)pkg["review.json"]=fflate.strToU8(JSON.stringify(result,null,2));
    download(name,fflate.zipSync(pkg),"application/zip");
  }
  $("#export").onclick=()=>exportFiles(slug+"-attempt.zip",files);
  function setDisplay(which) {
    display = which;
    showPane("feedback");
    for (const tab of all("[data-output]")) tab.classList.toggle("active", tab.dataset.output === which);
    render();
    document.querySelector(".side-nav").scrollIntoView({block: "nearest"});
  }
  for(const b of all("[data-output]"))b.onclick=()=>setDisplay(b.dataset.output);
  function render(){
    const out=$("#output");out.replaceChildren();
    if(display==="log"){out.append(log?make("pre","",log):message("Console is empty","Printed output and test logs appear here after a run."));return;}
    if(display==="review"){
      if(!review){out.append(message(last?"Submission saved":"No submission yet",last?"Reopen the saved result or retry grading.":"Submit a snapshot to see your private results."));if(last)out.append(button("Open last submission","subtle",restore));showHistory(out);return;}
      out.append(make("p","test-title",review.passed+" / "+review.total+" private scenarios passed"));
      review.parts.forEach(p=>out.append(make("div","rec "+(p.passed===p.total?"ok":"bad"),"Part "+p.part+" · "+parts[p.part-1]+" — "+p.passed+"/"+p.total)));
      for(const miss of review.misses){
        const d=make("details","rec");
        d.append(make("summary","","Inspect · "+miss.title),make("pre","",JSON.stringify({workload:miss.actions,expected:miss.expected,observed:miss.actual},null,2)));
        if(systems){
          const current=make("pre","sample-result");current.hidden=true;current.setAttribute("aria-live","polite");
          d.append(button("Replay against current code","subtle example-run",async()=>{
            if(lock)return;
            busy(true);current.hidden=false;current.textContent="Replaying…";
            try {
              const run=await execute({files:{...files},cases:[miss.actions]});
              const response=run.outputs[0];
              current.textContent=response.error || "Current output\n"+pretty(response.value);
            }catch(e){current.textContent=e.message;}
            finally{busy(false);}
          }),current);
        }
        out.append(d);
      }
      if (review.followups?.length) {
        const questions = make("details", "rec");
        questions.append(make("summary", "", "Interview follow-ups"));
        const list = make("ol");
        for (const question of review.followups) list.append(make("li", "", question));
        questions.append(list);
        out.append(questions);
      }
      out.append(button("Export graded snapshot","subtle",async()=>{try{const {data}=await api("submissions/"+last);exportFiles(slug+"-reviewed.zip",{...ex.files,...data.snapshot.files},data.result);}catch(e){notice(e.message);}}));showHistory(out);return;
    }
    if(runNote){out.append(message(lock?"Running":"Execution stopped",runNote));return;}
    if(!testRun){out.append($("#initial-output").content.cloneNode(true));return;}
    const rec=testRun.records;out.append(make("p","test-title",rec.filter(x=>x.ok).length+" / "+rec.length+" public tests passed"));
    for(const r of rec){const d=make("details","rec "+(r.ok?"ok":"bad"));const name=r.name.split(" (")[0];d.append(make("summary","",(r.ok?"✓ ":"× ")+name),make("pre","",r.detail||r.name));out.append(d);}
  }
  function notice(t){log=t;setDisplay("log");}
  function busy(val){lock=val;for(const b of all(".example-run"))b.disabled=val;$("#submit").disabled=val;$("#suite").disabled=val;$("#run").innerHTML=val?"Stop execution":"<span aria-hidden='true'>▷</span> Run tests";}
  function execute(data){
    return new Promise((resolve,reject)=>{
      const w=new Worker("/run-python.js?v=4");let timer;
      function end(error,value){clearTimeout(timer);w.terminate();stopWorker=null;if(error)reject(error);else resolve(value);}
      const timeout=()=>end(Error("Time limit reached. Your files are saved."));
      stopWorker=()=>end(Error("Execution stopped. Your files are saved."));
      timer=setTimeout(timeout,90000);
      w.onerror=e=>end(Error(e.message||"Python worker could not start."));
      w.onmessage=({data})=>{if(data.status){clearTimeout(timer);timer=setTimeout(timeout,18000);$("#run-state").textContent="Running";return;}end(data.error?Error(data.error):null,data);};
      w.postMessage({...data, protocol:original ? "release" : "function"});
    });
  }
  async function runTests(){
    if(lock){if(stopWorker)stopWorker();return;}markWork();busy(true);runNote="Starting an isolated Python session…";log="";setDisplay("tests");$("#run-state").textContent="Starting";
    try{const r=await execute({files:{...files},scope:$("#suite").value});testRun=r.publicRun;log=testRun.console;runNote="";$("#run-state").textContent="Finished";}
    catch(e){runNote=e.message;log=e.message;$("#run-state").textContent="Stopped";}finally{busy(false);render();}
  }
  $("#run").onclick=runTests;
  $("#submit").onclick=()=>$("#confirm").showModal();
  $("#confirm").addEventListener("close",()=>{if($("#confirm").returnValue==="submit")submitSnapshot();});
  async function grade(id,snapshot){
    const opts={method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({files:snapshot})};
    const {data}=await api("submissions/"+id,opts);
    last=id;stored.set(prefix+"submission",id);
    const {outputs}=await execute({files:{...ex.files,...snapshot},cases:data.cases});
    const graded=await api("submissions/"+id+"/grade",{...opts,body:JSON.stringify({outputs})});
    review=graded.data;markWork(review.total>0 && review.passed===review.total);await loadHistory();$("#run-state").textContent="Graded";setDisplay("review");
  }
  async function submitSnapshot(){
    if(lock)return;markWork();busy(true);review=null;runNote="Saving a snapshot and running the private workloads…";setDisplay("tests");$("#run-state").textContent="Submitting";
    try{await grade(crypto.randomUUID(),edits());}catch(e){notice(e.message);$("#run-state").textContent="Incomplete";}finally{runNote="";busy(false);}
  }
  async function restore(){
    if(lock||!last)return;busy(true);
    try{const {data}=await api("submissions/"+last);if(data.result){review=data.result;setDisplay("review");}else{await grade(last,data.snapshot.files);}}
    catch(e){notice(e.message);}finally{busy(false);}
  }
  render();
})().catch(e=>{document.querySelector("#save-state").textContent="Workspace could not open";document.querySelector("#output").textContent=e.message;});
