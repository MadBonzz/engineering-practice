type Activity = {slug:string; edition:"current"|"earlier"; solved:boolean; updated:number};
const release = "release-comparison";

function identify(path: string, slugs: Set<string>) {
  const bits = path.split("/");
  if (["sets","labs","designs"].includes(bits[0])) {
    const [group, slug, ...rest] = bits;
    if (!slugs.has(slug)) return;
    return {slug, edition: group === "sets" ? "earlier" as const : "current" as const, tail: rest.join("/")};
  }
  return {slug:release, edition:"current" as const, tail:path};
}

export async function progressPage(bucket:R2Bucket, session:string, cursor:string|null, slugs:Set<string>) {
  const prefix = "bench/v1/"+session+"/";
  const options = {prefix, limit:40, ...(cursor ? {cursor} : {}), include:["customMetadata"] as ("customMetadata")[]};
  const listing = await bucket.list(options);
  const items: Activity[] = [];
  // An older result predates the score metadata. Read its stored grade once per listing.
  for(const object of listing.objects) {
    const target = identify(object.key.slice(prefix.length), slugs);
    if (!target || !/^(draft\.json|activity\.json|(?:submissions|results)\/[\w-]+\.json)$/.test(target.tail)) continue;
    let solved = object.key.startsWith(prefix+"designs/") && object.customMetadata?.solved==="yes";
    if(target.tail.startsWith("results/")) {
      if (object.customMetadata?.solved !== undefined) solved = object.customMetadata.solved === "yes";
      else {
        const saved = await bucket.get(object.key);
        const grade = await saved?.json<{passed:number;total:number}>();
        solved = !!grade && grade.total > 0 && grade.passed === grade.total;
      }
    }
    items.push({slug:target.slug,edition:target.edition,solved,updated:object.uploaded.getTime()});
  }
  return {items, next:listing.truncated ? listing.cursor : null};
}

