import cases from "../public/design-cases.json";

export const designSlugs = cases.map(item=>item.slug);
const bySlug=new Map(cases.map(item=>[item.slug,item]));
type Draft = {
  answers:string[];edges:{from:string;to:string;data:string}[];twist:string;
  revealed:boolean;number:string;checked:boolean;review:boolean[];reviewed:boolean;
};
function textField(x:unknown,max:number):x is string {return typeof x==="string" && x.length<=max;}
function valid(value:any,count:number): value is Draft {
  return value && Array.isArray(value.answers) && value.answers.length===count && value.answers.every((x:unknown)=>textField(x,20000)) &&
    Array.isArray(value.review) && value.review.length===count+1 && value.review.every((x:unknown)=>typeof x==="boolean") &&
    Array.isArray(value.edges) && value.edges.length<=24 && value.edges.every((e:any)=>e && textField(e.from,48) && !!e.from.trim() && textField(e.to,48) && !!e.to.trim() && textField(e.data,80) && !!e.data.trim()) &&
    textField(value.twist,20000) && textField(value.number,80) && ["revealed","reviewed","checked"].every(k=>typeof value[k]==="boolean");
}
export async function designDraft(req:Request,bucket:R2Bucket,session:string,slug:string) {
  const problem=bySlug.get(slug);
  const json=(value:unknown,status=200,headers={})=>Response.json(value,{status,headers:{"Cache-Control":"no-store",...headers}});
  if(!problem)return json({error:"Unknown design case"},404);
  const key="bench/v1/"+session+"/designs/"+slug+"/draft.json";
  if(req.method==="GET"){
    const object=await bucket.get(key);
    return object ? new Response(object.body,{headers:{"Content-Type":"application/json","Cache-Control":"no-store","ETag":object.httpEtag}}) : json(null);
  }
  if(req.method!=="PUT")return json({error:"Method not allowed"},405);
  const raw=await req.text();
  if(raw.length>300000)return json({error:"Design exceeds the size limit"},413);
  const value=JSON.parse(raw);
  if(!valid(value,problem.tasks.length))return json({error:"Invalid draft"},400);
  const match=req.headers.get("If-Match");
  // A design may be revised after completing its first self-review.
  let priorReviewed=false;
  if(value.reviewed){
    const n=Number(value.number);
    const complete=value.answers.every(a=>a.trim()) && value.edges.length>0 && !!value.twist.trim() &&
      value.review.every(Boolean) && value.checked && value.number.trim()!=="" && Number.isFinite(n) &&
      Math.abs(n-problem.math.answer)<=problem.math.tolerance+1e-10;
    if(!complete){
      const prior=await bucket.head(key);
      priorReviewed=prior?.customMetadata?.solved==="yes";
      if(!priorReviewed)return json({error:"Complete this design's checkpoint and self-review first."},400);
    }
  }
  const saved=await bucket.put(key,JSON.stringify(value),{
    onlyIf:match?{etagMatches:match.replace(/^"|"$/g,"")}:{etagDoesNotMatch:"*"},
    customMetadata:{solved:value.reviewed?"yes":"no"}
  });
  return saved ? json({saved:true},200,{ETag:saved.httpEtag}) : json({error:"A newer design draft exists."},409);
}
