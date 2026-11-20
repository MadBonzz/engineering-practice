"use strict";
window.BenchProgress = (() => {
  function get(key) {try {return JSON.parse(localStorage.getItem(key));} catch {return null;}}
  function set(key, value) {try {localStorage.setItem(key, JSON.stringify(value));} catch {}}
  const pattern = /^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i;
  const requested = new URLSearchParams(location.hash.slice(1)).get("desk");
  const session = [requested, get("bench.active.v1")].find(v => typeof v==="string" && pattern.test(v)) || crypto.randomUUID();
  set("bench.active.v1",session);
  const root = "bench.activity."+session+".";
  function itemKey(slug, edition) {return root+edition+"."+slug;}
  function getActivity(slug, edition) {return get(itemKey(slug,edition));}
  function mark(slug, edition="current", solved=false, updated=Date.now()) {
    const old = getActivity(slug,edition);
    set(itemKey(slug,edition),{solved:solved || !!old?.solved, updated:Math.max(updated,old?.updated||0)});
  }
  function localState(slug,edition) {
    const original = slug==="release-comparison";
    const prefix="bench.v1."+session+"."+(original?"":(edition==="earlier"?"":"labs.")+slug+".");
    const draft=get(prefix+"draft");
    return getActivity(slug,edition) || (draft?.dirty || draft?.etag || get(prefix+"submission") ? {updated:0,solved:false} : null);
  }
  function url(slug, edition="current") {
    const u = new URL("/",location.href);
    if(slug) u.searchParams.set("set",slug);
    if(slug && edition==="earlier") u.searchParams.set("edition","1");
    u.hash="desk="+session;
    return u.href;
  }
  async function sync() {
    let cursor=null;
    do {
      const suffix=cursor?"?cursor="+encodeURIComponent(cursor):"";
      const response=await fetch("/api/"+session+"/progress"+suffix,{cache:"no-store"});
      if(!response.ok)throw Error("Cloud progress is unavailable.");
      const page=await response.json();
      for(const row of page.items) mark(row.slug,row.edition,row.solved,row.updated);
      cursor=page.next;
    } while(cursor);
  }
  return {session,url,mark,localState,sync};
})();

