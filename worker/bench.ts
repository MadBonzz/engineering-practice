import {progressPage} from "./progress";
import {designDraft,designSlugs} from "./design";
import labCases from "./lab-cases.json";
import catalog from "../public/laboratory.json";
import paperCatalog from "../public/paper-catalog.json";
import paperScenarios from "./paper-cases.json";
import systemsCatalog from "../public/systems-catalog.json";
import systemsScenarios from "./systems-cases.json";
import courseCatalog from "../public/course-new-catalog.json";
import courseScenarios from "./course-cases.json";
import releaseCases from "./cases.json";
import practiceCases from "./practice-cases.json";

import followups from "./practice-followups.json";

type Scenario = { part: number; title: string; actions: any[]; expected: any };
const sets: Record<string, Scenario[]> = practiceCases;
const projectScenarios: Record<string, Scenario[]> = {...paperScenarios, ...systemsScenarios, ...courseScenarios};
const projectFiles=new Map([...paperCatalog, ...systemsCatalog, ...courseCatalog].map(p=>[p.slug,p.editable]));
const allSlugs=new Set([...catalog.map(row=>row.slug),...projectFiles.keys(),...designSlugs]);

const uuid = /^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$/i;
const releaseFiles = ["ledger.py", "report.py", "tests/test_mine.py"];
const json = (value: unknown, status = 200, headers = {}) => Response.json(value, {status, headers: {"Cache-Control":"no-store", ...headers}});
const error = (message: string,status=400) => json({error:message},status);
function validFiles(files: Record<string,string> | undefined, filesAllowed: string[]) {
  return files && Object.keys(files).length === filesAllowed.length && filesAllowed.every(p => typeof files[p]==="string" && files[p].length <= 90000);
}
function same(a:any,b:any, numeric = false):boolean {
  if (numeric && typeof a === "number" && typeof b === "number") {
    return Number.isFinite(a) && Number.isFinite(b) && Math.abs(a-b) <= Math.max(1e-8, 1e-7*Math.max(Math.abs(a),Math.abs(b)));
  }
  if(a===b) return true;
  if(!a || !b || typeof a!=="object" || typeof b!=="object" || Array.isArray(a)!==Array.isArray(b)) return false;
  const keys=Object.keys(a);
  return keys.length===Object.keys(b).length && keys.every(k=>Object.hasOwn(b,k) && same(a[k],b[k],numeric));
}
export async function benchAPI(req:Request,bucket:R2Bucket) {
  const url=new URL(req.url);
  if(req.method !== "GET" && req.headers.get("Origin") && req.headers.get("Origin")!==url.origin) return error("Origin mismatch",403);
  const [, ,session,action,attempt,operation]=url.pathname.split("/");
  if(!uuid.test(session||"") || (attempt && !uuid.test(attempt))) return error("Unknown workspace",404);
  if(!bucket) return error("Cloud storage is temporarily unavailable.",503);
  if(action==="progress" && req.method==="GET") {
    return json(await progressPage(bucket,session,url.searchParams.get("cursor"),allSlugs));
  }
  const slug = url.searchParams.get("set") || "release-comparison";
  if(action==="design" && !attempt)return designDraft(req,bucket,session,slug);
  const original = slug === "release-comparison";
  const editableProject=projectFiles.get(slug);
  if (!original && !editableProject && !Object.hasOwn(sets,slug)) return error("Unknown practice set",404);
  if (editableProject && url.searchParams.get("edition")!=="2") return error("Unknown lab edition",404);
  const lab = !original && url.searchParams.get("edition")==="2";
  const cases: Scenario[] = editableProject ? projectScenarios[slug] : original ? releaseCases : lab ? [...sets[slug], ...labCases[slug as keyof typeof labCases]] : sets[slug];
  const approximateNumbers = !original && !Object.hasOwn(systemsScenarios, slug);
  const filesAllowed = editableProject ?? (original ? releaseFiles : lab ? catalog.find(item=>item.slug===slug)!.editable : ["solution.py", "tests/test_mine.py"]);
  const root = "bench/v1/" + session + "/" + (original ? "" : (lab ? "labs/" : "sets/") + slug + "/");
  const key=root+"draft.json";
  const recordKey=root+"submissions/"+attempt+".json";
  const resultKey=root+"results/"+attempt+".json";
  async function body(){
    const text=await req.text();
    if(text.length > 600000) throw Error("Request exceeds the size limit.");
    return JSON.parse(text);
  }
  if(action==="activity" && req.method==="POST"){
    await bucket.put(root+"activity.json","{}",{onlyIf:{etagDoesNotMatch:"*"}});
    return json({started:true});
  }
  if(action==="draft" && req.method==="GET"){
    const stored=await bucket.get(key);
    return stored ? new Response(stored.body,{headers:{"Content-Type":"application/json","Cache-Control":"no-store","ETag":stored.httpEtag}}) : json(null);
  }
  if(action==="draft" && req.method==="PUT"){
    const data=await body();
    if(!validFiles(data.files,filesAllowed)) return error("Invalid draft files.");
    const previous=req.headers.get("If-Match");
    const saved=await bucket.put(key,JSON.stringify({files:data.files,updated:Date.now()}),{onlyIf:previous?{etagMatches:previous.replace(/^"|"$/g,"")}:{etagDoesNotMatch:"*"}});
    return saved?json({saved:true},200,{ETag:saved.httpEtag}):error("A newer draft exists in another tab. Export your files before reloading.",409);
  }
  if(action==="submissions" && !attempt && req.method==="GET"){
    const list=await bucket.list({prefix:root+"submissions/",limit:1000});
    const items=list.objects.map(x=>({id:x.key.split("/").pop()!.replace(".json",""),created:x.uploaded.getTime()})).sort((a,b)=>b.created-a.created);
    return json(items);
  }
  if(action==="submissions" && attempt && req.method==="POST" && !operation){
    const data=await body();
    if(!validFiles(data.files,filesAllowed)) return error("Invalid submission files.");
    const allowed={files:data.files,created:Date.now(),exercise:slug};
    const saved=await bucket.put(recordKey,JSON.stringify(allowed),{onlyIf:{etagDoesNotMatch:"*"}});
    if(!saved){
      const existing=await bucket.get(recordKey);
      if(!existing || !same((await existing.json<any>()).files,data.files)) return error("Submission ID is already in use.",409);
    }
    return json({cases:cases.map(c=>c.actions)});
  }
  if(action==="submissions" && attempt && req.method==="GET"){
    const record=await bucket.get(recordKey);
    if(!record)return error("Submission unavailable.",404);
    const result=await bucket.get(resultKey);
    return json({snapshot:await record.json(), result:result?await result.json():null});
  }
  if(action==="submissions" && attempt && operation==="grade" && req.method==="POST"){
    if(!await bucket.head(recordKey)) return error("Submit a snapshot first.",404);
    const existing=await bucket.get(resultKey);
    if(existing) return json(await existing.json());
    const {outputs}=await body();
    if(!Array.isArray(outputs)||outputs.length!==cases.length) return error("Incomplete execution results.");
    const parts=Array.from({length: Math.max(...cases.map(c=>c.part))},(_,i)=>({part:i+1,passed:0,total:0}));
    const misses:any[]=[];
    cases.forEach((c,i)=>{
      const part=parts[c.part-1];part.total++;
      if(Object.hasOwn(outputs[i] ?? {}, "value") && same(outputs[i].value,c.expected,approximateNumbers)) {part.passed++;return;}
      if(!misses.some(m=>m.part===c.part)){
        const got=outputs[i];
        if (!original) {
          misses.push({part:c.part,title:c.title,actions:c.actions,expected:c.expected,actual:got?.error ?? got?.value ?? null});
          return;
        }
        const mismatch=c.expected.findIndex((v:any,j:number)=>!same(v,got?.value?.[j]));
        let reads=0;
        const end=c.actions.findIndex(a=>a[0]!=="write" && reads++===Math.max(0,mismatch));
        misses.push({part:c.part,title:c.title,actions:c.actions.slice(0,end<0?c.actions.length:end+1),expected:c.expected[Math.max(0,mismatch)],actual:got?.error??got?.value?.[Math.max(0,mismatch)]??null});
      }
    });
    const result={parts,misses,followups:original ? [] : followups[slug as keyof typeof followups],passed:parts.reduce((s,p)=>s+p.passed,0),total:cases.length,graded:Date.now()};
    const saved=await bucket.put(resultKey,JSON.stringify(result),{onlyIf:{etagDoesNotMatch:"*"},customMetadata:{solved: result.passed===result.total ? "yes":"no"}});
    if(saved)return json(result);
    return json(await (await bucket.get(resultKey))!.json());
  }
  return error("Unknown endpoint.",404);
}
