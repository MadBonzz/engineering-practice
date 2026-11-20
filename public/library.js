"use strict";
(async function () {
  const $ = selector => document.querySelector(selector);
  const bench = window.BenchProgress;
  const track=new URLSearchParams(location.search).get("track");
  const design=track==="design", papers=track==="papers", systems=track==="systems", courses=track==="courses";
  let courseGuide=null;
  const doneLabel=design?"Reviewed":"Solved";
  const statuses = {fresh:"Not started",started:"In progress",solved:doneLabel};
  const symbols = {fresh:"○",started:"◐",solved:"✓"};
  const base=new URL(bench.url());
  $("#coding-track").href=base.href;base.searchParams.set("track","design");$("#design-track").href=base.href;
  base.searchParams.set("track","papers");$("#paper-track").href=base.href;
  base.searchParams.set("track","systems");$("#systems-track").href=base.href;
  base.searchParams.set("track","courses");$("#courses-track").href=base.href;
  $(courses?"#courses-track":design?"#design-track":papers?"#paper-track":systems?"#systems-track":"#coding-track").setAttribute("aria-current","page");
  if(courses){
    $("#track-intro").textContent="Build deep learning intuition through original coding labs and controlled experiments.";
    $("#course-filter").hidden=false;$("#course-guide").hidden=false;
    const requestedCourse=new URLSearchParams(location.search).get("course");
    if(["dlsys","alchemy","cs336"].includes(requestedCourse))$("#course").value=requestedCourse;
    const guide=await fetch("/course-map.json").then(r=>{if(!r.ok)throw Error("Course guide unavailable");return r.json();});
    courseGuide=guide;
    for(const course of guide.courses){
      const section=document.createElement("section"),heading=document.createElement("h3"),link=document.createElement("a");
      section.dataset.course=course.id;link.href=course.url;link.target="_blank";link.rel="noopener noreferrer";link.textContent=course.title;heading.append(link);section.append(heading);
      for(const text of [course.coverage,course.adaptations]){const p=document.createElement("p");p.textContent=text;section.append(p);}
      $("#course-coverage").append(section);
    }
  }
  if(systems){
    $("#track-intro").textContent="Build and debug distributed protocols, storage engines, operating systems, networks, and compilers.";
  }
  if(papers){
    $("#track-intro").textContent="Implement research papers and model components in small, testable Python projects.";
    $("#training-filter").hidden=false;
  }
  if(design){
    $("#track-intro").textContent="Practice ML system design with case briefs, architecture sketches, and quantitative checkpoints.";
    $("#state option[value=solved]").textContent=doneLabel;
    $("#retrieval-setting").hidden=false;$("#reading-guide").hidden=false;
    try{$("#retrieval").checked=localStorage.getItem("bench.includeRetrieval")==="true";}catch{}
  }
  const request = await fetch(courses?"/course-catalog.json":design?"/design-catalog.json":papers?"/paper-catalog.json":systems?"/systems-catalog.json":"/laboratory.json");
  if (!request.ok) throw Error("Could not load problems.");
  const catalog = await request.json();
  function visibleCatalog(){return catalog.filter(item=>(!design || !item.retrieval || $("#retrieval").checked) && (!courses || $("#course").value==="all" || item.courses.includes($("#course").value)));}
  function problemUrl(item,edition="current"){
    if(!design){
      const url=new URL(bench.url(item.slug,edition));
      if(courses){url.searchParams.set("from","courses");url.searchParams.set("course",$("#course").value);}
      return url.href;
    }
    const link=new URL(bench.url());link.searchParams.set("design",item.slug);return link.href;
  }
  if(courseGuide){
    const titles=new Map(catalog.map(item=>[item.slug,item.title]));
    for(const course of courseGuide.courses){
      const list=el("ol","course-units");
      for(const unit of course.units){
        const item=el("li"),heading=el("strong","",unit.title),links=el("div","course-unit-links");item.append(heading);
        for(const slug of unit.tasks){
          const link=el("a","",titles.get(slug));const url=new URL(bench.url(slug));url.searchParams.set("from","courses");url.searchParams.set("course",course.id);link.href=url.href;links.append(link);
        }
        item.append(links);list.append(item);
      }
      document.querySelector('[data-course="'+course.id+'"]').append(list);
    }
  }
  let topic = "All topics";
  const groups = [topic, ...new Set(catalog.map(item=>item.group))];
  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function progress(item) {
    const latest = bench.localState(item.slug,"current");
    const prior = design || papers || systems || item.kind==="course" || item.slug==="release-comparison" ? null : bench.localState(item.slug,"earlier");
    const track = latest || prior;
    return {state:track ? (track.solved ? "solved" : "started") : "fresh", priorOnly:!!prior && !latest,
            updated:track?.updated || 0};
  }
  for (const group of groups) {
    const b = el("button","",group);
    const count = catalog.filter(item=>group==="All topics" || item.group===group).length;
    b.append(el("span","",count));
    b.onclick=()=>{topic=group;render();};
    b.dataset.group=group;
    $("#topics").append(b);
  }
  function render() {
    const query=$("#search").value.trim().toLowerCase(),rows=$("#problems");
    rows.replaceChildren();
    const records=visibleCatalog().map(item=>({...item,...progress(item)}));
    for(const button of $("#topics").children) {
      const count=records.filter(item=>button.dataset.group==="All topics"||item.group===button.dataset.group).length;
      button.querySelector("span").textContent=count;button.hidden=count===0;
      const active=button.dataset.group===topic;
      button.classList.toggle("active",active);
      button.setAttribute("aria-pressed",String(active));
    }
    const filtered=records.filter(item=>(topic==="All topics"||topic===item.group) &&
      ($("#state").value==="all"||$("#state").value===item.state) &&
      ($("#difficulty").value==="all"||$("#difficulty").value===item.level) &&
      (!papers||!$("#training-only").checked||item.training) &&
      [item.title,item.group,item.summary,item.topic,item.family,item.model].join(" ").toLowerCase().includes(query));
    for (const item of filtered) {
      const row=el("a","row");row.href=problemUrl(item);
      const title=el("div");
      title.append(el("h3","",item.title),el("p","scenario",item.summary),el("p","meta",design?item.topic+" · "+item.duration+" · "+item.parts+" design decisions":item.group+" · "+item.duration+" · "+item.parts+(item.parts===1?" part":" parts")));
      if(courses)title.lastChild.append(document.createTextNode(" · "+(item.kind==="course"?"New course lab":"Existing lab")+" · "+item.courses.map(c=>({dlsys:"DL Systems",alchemy:"Alchemy",cs336:"CS336"})[c]).join(" / ")));
      if(papers && item.training)title.lastChild.append(document.createTextNode(" · PyTorch training project"));
      const state=el("span","state "+item.state);
      state.append(el("i","",symbols[item.state]),document.createTextNode(statuses[item.state]+(item.priorOnly?"¹":"")));
      if(item.priorOnly) state.title="Progress from the earlier edition. Open the lab to resume that workspace.";
      row.append(title,el("span","level",item.level),state);
      rows.append(row);
    }
    const solved=records.filter(x=>x.state==="solved").length;
    const started=records.filter(x=>x.state==="started").length;
    $("#bar").max=records.length;$("#bar").value=solved;
    $("#progress-count").textContent=solved+" / "+records.length+" "+doneLabel.toLowerCase();
    $("#working-count").textContent=started+" in progress";
    $("#untouched-count").textContent=(records.length-solved-started)+" not started";
    $("#result-count").textContent=filtered.length+" problem"+(filtered.length===1?"":"s");
    $("#no-results").hidden=filtered.length!==0;
    const resume=records.filter(x=>x.state==="started").sort((a,b)=>b.updated-a.updated)[0];
    $("#resume-last").hidden=!resume;
    if(resume) {$("#resume-last").href=problemUrl(resume,resume.priorOnly?"earlier":"current");$("#resume-last").textContent=design?"Resume last design ↗":"Resume last lab ↗";$("#resume-last").title=resume.title;}
    $("#legacy-note").hidden=!filtered.some(item=>item.priorOnly);
  }
  $("#search").addEventListener("input",render);
  $("#training-only").onchange=render;
  $("#course").onchange=()=>{topic=groups[0];const u=new URL(location.href);u.searchParams.set("course",$("#course").value);history.replaceState(null,"",u);render();};
  $("#retrieval").onchange=()=>{try{localStorage.setItem("bench.includeRetrieval",String($("#retrieval").checked));}catch{}topic=groups[0];render();};
  if(design){
    $("#reading-guide").addEventListener("toggle",async()=>{
      if(!$("#reading-guide").open || $("#reading-list").children.length)return;
      try {
        const response=await fetch("/design-reading.json");
        if(!response.ok)throw Error();
        for(const block of await response.json()){
          const group=el("details"),heading=el("summary","",block.section);
          group.append(heading);
          for(const s of block.links){const a=el("a","",s.title);a.href=s.url;a.target="_blank";a.rel="noopener noreferrer";group.append(a);}
          $("#reading-list").append(group);
        }
      }catch{$("#reading-list").textContent="Study links could not be loaded. Reopen to retry.";}
    });
  }
  for(const id of ["state","difficulty"])$("#"+id).addEventListener("change",render);
  $("#clear").onclick=()=>{topic=groups[0];$("#search").value="";$("#state").value="all";$("#difficulty").value="all";$("#training-only").checked=false;render();};
  const foot=el("p","warning","¹ Progress from the earlier edition. You can reopen that attempt from the lab.");
  foot.id="legacy-note";$("#problems").after(foot);
  async function refresh(){
    render();
    try {await bench.sync();$("#sync-warning").hidden=true;}
    catch {$("#sync-warning").hidden=false;$("#sync-warning").textContent="Showing progress saved on this browser. Cloud progress could not be loaded.";}
    render();
  }
  window.addEventListener("storage",render);
  window.addEventListener("pageshow",e=>{if(e.persisted)refresh();});
  await refresh();
})().catch(e=>{document.querySelector("#result-count").textContent=e.message;});

