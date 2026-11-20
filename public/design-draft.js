const bench=window.BenchProgress;
export async function openDraft(slug, initial, onState, onMessage) {
  const key="bench.design.v1."+bench.session+"."+slug, endpoint="/api/"+bench.session+"/design?set="+encodeURIComponent(slug);
  let cache={draft:initial,etag:null,dirty:false}, revision=0, saving=null, conflict=false, offline=false, timer;
  try {cache=JSON.parse(localStorage.getItem(key)) || cache;} catch {}
  function remember(){try {localStorage.setItem(key,JSON.stringify(cache));return true;}catch{return false;}}
  const equivalent=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
  async function read(force=false) {
    const res=await fetch(endpoint,{cache:"no-store"});
    if(!res.ok)throw Error("Could not read cloud draft");
    const cloud=await res.json(), tag=res.headers.get("ETag");
    offline=false;
    if(force || !cache.dirty || (cloud && equivalent(cache.draft,cloud))) {
      cache={draft:cloud || initial,dirty:false,etag:tag};conflict=false;remember();
    } else if(cache.etag!==tag) conflict=true;
  }
  try {await read();} catch {offline=true;}
  function state(){return cache.draft;}
  function display(){
    if(conflict)onMessage("Newer cloud draft available",true);
    else if(offline)onMessage("Device draft · cloud unavailable");
    else if(cache.dirty)onMessage("Saved on this device");
    else onMessage(cache.etag?"Saved to workspace":"Ready");
  }
  async function post() {
    const sending=revision;
    try {
      const headers={"Content-Type":"application/json"};
      if(cache.etag)headers["If-Match"]=cache.etag;
      const res=await fetch(endpoint,{method:"PUT",headers,body:JSON.stringify(cache.draft)});
      if(res.status===409){await read();display();return;}
      if(!res.ok)throw Error("Could not sync");
      offline=false;cache.etag=res.headers.get("ETag");cache.dirty=sending!==revision;
      remember();display();
    } catch {offline=true;onMessage("Device draft · cloud unavailable");}
  }
  async function sync(){
    clearTimeout(timer);
    if(saving){await saving;if(cache.dirty&&!conflict&&!offline)return sync();return;}
    if(conflict||!cache.dirty)return;
    saving=post().finally(()=>{saving=null;});await saving;
  }
  function changed(){
    revision++;cache.dirty=true;const saved=remember();
    bench.mark(slug,"current",!!cache.draft.reviewed);
    onMessage(conflict?"Newer cloud draft available":saved?"Saved on this device":"Cloud save pending",conflict);
    clearTimeout(timer);timer=setTimeout(sync,500);
  }
  async function useCloud(){if(saving)await saving;await read(true);onState(state());display();}
  window.addEventListener("online",sync);
  window.addEventListener("pagehide",()=>{if(cache.dirty)remember();});
  display();if(cache.dirty&&!conflict)sync();return {state,changed,sync,useCloud};
}
