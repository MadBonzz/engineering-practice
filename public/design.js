import {openDraft} from "./design-draft.js";
import {drawConnections} from "./design-graph.js";

const $=id=>document.getElementById(id);
function el(tag,text,cls){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node;}
function reading(list,items) {
  for(const source of items){const li=el("li"),link=el("a",source.title);link.href=source.url;link.target="_blank";link.rel="noopener noreferrer";li.append(link);list.append(li);}
}
async function main() {
  const bench=window.BenchProgress,slug=new URLSearchParams(location.search).get("design");
  const response=await fetch("/design-cases.json");
  if(!response.ok)throw Error("The design library could not be loaded.");
  const catalog=await response.json(),problem=catalog.find(p=>p.slug===slug);
  const home=new URL(bench.url());home.searchParams.set("track","design");
  $("home").href=home.href;$("back").href=home.href;
  if(!problem)throw Error("This design was not found. Use the system design library above to choose a case.");
  document.title=problem.title+" · Practice Bench";
  $("title").textContent=problem.title;$("brief").textContent=problem.brief;
  $("meta").textContent=problem.family+" · "+problem.level+" · "+problem.minutes+" min";
  for(const field of ["inputs","outputs","why","walkthrough"])$(field).textContent=problem[field];
  $("sample-in").textContent=JSON.stringify(problem.sampleIn,null,2);$("sample-out").textContent=JSON.stringify(problem.sampleOut,null,2);
  problem.constraints.forEach(c=>$("constraints").append(el("li",c)));
  reading($("sources"),problem.sources);reading($("lessons"),[...problem.lessons,problem.series]);
  $("event").textContent=problem.twist.event;$("twist-prompt").textContent=problem.twist.prompt;
  $("math-prompt").textContent=problem.math.prompt;$("units").textContent="Answer in "+problem.math.unit;
  $("math-worked").textContent=problem.math.explanation;

  const defaults={answers:problem.tasks.map(()=>""),edges:[],twist:"",revealed:false,number:"",checked:false,review:problem.tasks.map(()=>false).concat(false),reviewed:false};
  let draft, editing=-1;
  const inputs=[],checks=[];
  for(const [i,task] of problem.tasks.entries()){
    const item=el("div",undefined,"task"),title=el("h3",task.title),label=el("label",task.prompt),field=el("textarea");
    field.rows=4;field.maxLength=20000;field.id="decision-"+i;label.htmlFor=field.id;
    field.addEventListener("input",()=>{draft.answers[i]=field.value;changed();});
    inputs.push(field);item.append(title,label,field);$("tasks").append(item);
  }
  for(const [i,criterion] of [...problem.tasks,{title:"Interviewer follow-up",check:problem.twist.check}].entries()){
    const label=el("label",undefined,"criterion"),field=el("input"),body=el("span");
    field.type="checkbox";body.append(el("b",criterion.title),document.createTextNode(criterion.check));
    field.addEventListener("change",()=>{draft.review[i]=field.checked;changed();});
    checks.push(field);label.append(field,body);$("rubric").append(label);
  }
  function showTab(name) {
    document.querySelectorAll("[data-tab]").forEach(b=>{const on=b.dataset.tab===name;b.setAttribute("aria-pressed",String(on));$(b.dataset.tab).hidden=!on;});
  }
  for(const b of document.querySelectorAll("[data-tab]"))b.onclick=()=>showTab(b.dataset.tab);
  function paintEdges(){
    drawConnections($("graph"),draft.edges);$("edges").replaceChildren();
    for(const [i,e] of draft.edges.entries()){
      const li=el("li"),span=el("span",(i+1)+". "+e.from+" → "+e.to+" · "+e.data),remove=el("button","Remove","link"), edit=el("button","Edit","link");
      edit.setAttribute("aria-label","Edit connection "+(i+1));remove.setAttribute("aria-label","Remove connection "+(i+1));
      edit.onclick=()=>{editing=i;$("from").value=e.from;$("to").value=e.to;$("payload").value=e.data;$("connection-form").querySelector("button").textContent="Save connection";$("from").focus();};
      remove.onclick=()=>{draft.edges.splice(i,1);resetForm();changed();paintEdges();};
      const buttons=el("div");buttons.style.whiteSpace="nowrap";buttons.append(edit,remove);li.append(span,buttons);$("edges").append(li);
    }
  }
  function resetForm(){editing=-1;$("connection-form").reset();$("connection-form").querySelector("button").textContent="Add connection";}
  function mathCorrect(){
    const numeric=Number(draft.number);
    return draft.number.trim()!=="" && Number.isFinite(numeric) && Math.abs(numeric-problem.math.answer)<=problem.math.tolerance+1e-10;
  }
  function paintMath(){
    $("math-result").textContent=!draft.checked?"":mathCorrect()?"Correct. "+problem.math.explanation:"Try again. Check the units, time window, and denominator in the question.";
  }
  function paint(record) {
    draft=record;inputs.forEach((t,i)=>t.value=draft.answers[i]||"");checks.forEach((f,i)=>f.checked=!!draft.review[i]);
    $("incident").open=!!draft.revealed;$("response").value=draft.twist;$("estimate").value=draft.number;resetForm();paintEdges();paintMath();
    if(draft.reviewed){bench.mark(slug,"current",true);$("finish").textContent="Design reviewed";$("finish-message").textContent="Recorded in your practice library.";}
  }
  function status(text,conflict=false){$("save").textContent=text;$("reload-cloud").hidden=!conflict;}
  const store=await openDraft(slug,defaults,paint,status);
  function changed(){store.changed();if(!draft.reviewed)$("finish-message").textContent="";}
  paint(store.state());$("studio").hidden=false;
  $("incident").ontoggle=()=>{if(draft.revealed!==$("incident").open){draft.revealed=$("incident").open;changed();}};
  $("response").oninput=()=>{draft.twist=$("response").value;changed();};
  $("connection-form").onsubmit=e=>{
    e.preventDefault();const edge={from:$("from").value.trim(),to:$("to").value.trim(),data:$("payload").value.trim()};
    if(!edge.from||!edge.to||!edge.data){$("graph-message").textContent="Give both components and the data a name.";return;}
    if(draft.edges.length>=24&&editing===-1){$("graph-message").textContent="This diagram supports up to 24 connections.";return;}
    if(editing===-1)draft.edges.push(edge);else draft.edges[editing]=edge;
    $("graph-message").textContent="";resetForm();paintEdges();changed();
  };
  $("estimate").oninput=()=>{draft.number=$("estimate").value;draft.checked=false;paintMath();changed();};
  $("math-form").onsubmit=e=>{e.preventDefault();draft.checked=true;changed();paintMath();};
  $("finish").onclick=async()=>{
    const missing=[];
    if(!draft.answers.every(a=>a.trim()))missing.push("write a response to each design decision");
    if(!draft.edges.length)missing.push("draw at least one architecture connection");
    if(!draft.twist.trim())missing.push("respond to the interviewer follow-up");
    if(!draft.checked||!mathCorrect())missing.push("pass the quantitative checkpoint");
    if(!draft.review.every(Boolean))missing.push("check each self-review criterion");
    if(missing.length){$("finish-message").textContent="To finish, "+missing.join(", ")+ ".";return;}
    draft.reviewed=true;changed();$("finish").textContent="Design reviewed";$("finish-message").textContent="Recorded in your practice library.";await store.sync();
  };
  for(const id of ["back","home"])$(id).onclick=async(e)=>{if(e.metaKey||e.ctrlKey||e.shiftKey||e.altKey)return;e.preventDefault();await store.sync();location.assign(home.href);};
  $("resume").onclick=async()=>{const url=new URL(location.href);url.hash="desk="+bench.session;try{await navigator.clipboard.writeText(url.href);$("tool-status").textContent="Resume link copied";}catch{window.prompt("Copy the workspace link",url.href);}};
  $("reload-cloud").onclick=async()=>{if(confirm("Load the saved cloud version? Export your current notes first if you want to keep both.")){try{await store.useCloud();}catch{status("Cloud draft unavailable",true);}}};
  $("export").onclick=()=>{
    const lines=["# "+problem.title,problem.brief,"## Operating constraints",...problem.constraints.map(s=>"- "+s),"## Design decisions"];
    problem.tasks.forEach((t,i)=>lines.push("### "+t.title,t.prompt,draft.answers[i]||"(unanswered)"));
    lines.push("## Architecture",...draft.edges.map(e=>"- "+e.from+" → "+e.to+" · "+e.data),"## Follow-up",problem.twist.event,problem.twist.prompt,draft.twist||"(unanswered)","## Quantitative checkpoint",problem.math.prompt,draft.number?draft.number+" "+problem.math.unit:"(unanswered)","## Self-review",...checks.map((_,i)=>(draft.review[i]?"[x] ":"[ ] ")+(problem.tasks[i]?.title||"Interviewer follow-up")),"## Reading",...problem.sources.map(s=>s.title+" — "+s.url));
    const url=URL.createObjectURL(new Blob([lines.join("\n\n")],{type:"text/markdown"})),a=el("a");a.href=url;a.download=slug+".md";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
}
main().catch(e=>{$("fatal").textContent=e.message;$("save").textContent="Unavailable";});
