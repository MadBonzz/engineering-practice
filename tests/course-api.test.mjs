import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFileSync} from 'node:fs';
import {resolve,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {stripTypeScriptTypes} from 'node:module';

const root=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const cache=new Map();
function loadModule(file){
  if(cache.has(file))return cache.get(file);
  if(file.endsWith('.json'))return JSON.parse(readFileSync(file,'utf8'));
  let code=readFileSync(file,'utf8');const names=[],values=[];
  code=code.replace(/import\s+(\{[^}]+\}|\w+)\s+from\s+["']([^"']+)["'];?/g,(_,binding,path)=>{
    const target=resolve(dirname(file),path.endsWith('.json')?path:path+'.ts');
    const imported=loadModule(target);
    if(binding.startsWith('{'))for(const name of binding.slice(1,-1).split(',').map(x=>x.trim())){names.push(name);values.push(imported[name]);}
    else {names.push(binding);values.push(imported);}
    return '';
  });
  const exported=[...code.matchAll(/export\s+(?:async\s+)?(?:function|const)\s+(\w+)/g)].map(x=>x[1]);
  code=stripTypeScriptTypes(code,{mode:'strip'}).replace(/\bexport\s+/g,'');
  const result=new Function(...names,code+'\nreturn {'+exported.join(',')+'};')(...values);cache.set(file,result);return result;
}
const {benchAPI}=loadModule(resolve(root,'worker/bench.ts'));
const catalog=loadModule(resolve(root,'public/course-new-catalog.json'));
const bank=loadModule(resolve(root,'worker/course-cases.json'));
const session='12345678-1234-4234-8234-123456789abc';
const attempt='12345678-1234-4234-8234-123456789def';
class Bucket {
  rows=new Map();serial=0;
  async get(key){const row=this.rows.get(key);return row?{...row,body:row.text,json:async()=>JSON.parse(row.text)}:null;}
  async head(key){return this.get(key);}
  async put(key,text,options={}){
    const old=this.rows.get(key),only=options.onlyIf;
    if(only?.etagDoesNotMatch==='*'&&old)return null;
    if(only?.etagMatches&&old?.etag!==only.etagMatches)return null;
    const etag=String(++this.serial),row={text,etag,httpEtag:'"'+etag+'"',uploaded:new Date(),customMetadata:options.customMetadata??{}};
    this.rows.set(key,row);return row;
  }
  async list({prefix}){return {objects:[...this.rows].filter(([key])=>key.startsWith(prefix)).map(([key,row])=>({key,...row})),truncated:false};}
}
function call(bucket,slug,path,method='GET',body,headers={}){
  return benchAPI(new Request('https://bench.test/api/'+session+'/'+path+'?set='+slug+'&edition=2',{method,headers:{'Content-Type':'application/json',...headers},...(body?{body:JSON.stringify(body)}:{})}),bucket);
}
function edits(slug){const ex=loadModule(resolve(root,'public/courses',slug,'exercise.json'));return Object.fromEntries(ex.editable.map(path=>[path,ex.files[path]]));}

test('all course starters, including notebooks, save through the real draft validator',async()=>{
  const bucket=new Bucket();
  for(const row of catalog){
    const files=edits(row.slug);files['REFLECTION.md']='Prediction before running: test notebook';
    const save=await call(bucket,row.slug,'draft','PUT',{files});assert.equal(save.status,200,row.slug);
    const read=await call(bucket,row.slug,'draft');assert.deepEqual((await read.json()).files,files);
  }
});
test('course drafts preserve conflict detection and prevent undeclared files',async()=>{
  const bucket=new Bucket(),slug=catalog[0].slug,files=edits(slug);
  const first=await call(bucket,slug,'draft','PUT',{files});const etag=first.headers.get('ETag');
  assert.equal((await call(bucket,slug,'draft','PUT',{files})).status,409);
  files['REFLECTION.md']='New observation';
  assert.equal((await call(bucket,slug,'draft','PUT',{files},{'If-Match':etag})).status,200);
  assert.equal((await call(bucket,slug,'draft','PUT',{files},{'If-Match':etag})).status,409);
  assert.equal((await call(bucket,slug,'draft','PUT',{files:{...files,'arbitrary.py':''}})).status,400);
});
test('course snapshots, scoring, immutable results, history and progress integrate',async()=>{
  for(const row of catalog){
    const bucket=new Bucket(),slug=row.slug,files=edits(slug);
    const submit=await call(bucket,slug,'submissions/'+attempt,'POST',{files});assert.equal(submit.status,200);
    const workloads=await submit.json();assert.equal(workloads.cases.length,bank[slug].length);
    assert.equal(JSON.stringify(workloads).includes('expected'),false);
    const outputs=bank[slug].map(x=>({value:x.expected}));
    const grade=await call(bucket,slug,'submissions/'+attempt+'/grade','POST',{outputs});const result=await grade.json();
    assert.equal(result.passed,result.total,slug);assert.equal(result.parts.length,3);
    const repeat=await call(bucket,slug,'submissions/'+attempt+'/grade','POST',{outputs:[]});assert.deepEqual(await repeat.json(),result);
    const changed={...files,'REFLECTION.md':'different snapshot'};
    assert.equal((await call(bucket,slug,'submissions/'+attempt,'POST',{files:changed})).status,409);
    const record=await (await call(bucket,slug,'submissions/'+attempt)).json();assert.deepEqual(record.snapshot.files,files);
    const progress=await (await call(bucket,slug,'progress')).json();assert(progress.items.some(x=>x.slug===slug&&x.solved));
  }
});
test('incorrect course outputs receive misses and incomplete outputs are rejected',async()=>{
  const bucket=new Bucket(),slug=catalog[0].slug;
  await call(bucket,slug,'submissions/'+attempt,'POST',{files:edits(slug)});
  assert.equal((await call(bucket,slug,'submissions/'+attempt+'/grade','POST',{outputs:[]})).status,400);
  const response=await call(bucket,slug,'submissions/'+attempt+'/grade','POST',{outputs:bank[slug].map(()=>({error:'NotImplementedError'}))});
  const result=await response.json();assert.equal(result.passed,0);assert.equal(result.misses.length,3);
});
test('existing paper and legacy sets still resolve after adding course projects',async()=>{
  for(const [slug,path] of [['paper-transformer','papers'],['kv-cache','labs']]){
    const bucket=new Bucket();const ex=loadModule(resolve(root,'public',path,slug,'exercise.json'));
    const files=Object.fromEntries(ex.editable.map(name=>[name,ex.files[name]]));
    assert.equal((await call(bucket,slug,'draft','PUT',{files})).status,200);
  }
});
