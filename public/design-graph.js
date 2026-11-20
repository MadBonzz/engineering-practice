const NS="http://www.w3.org/2000/svg";
function shape(name,attributes,label) {
  const node=document.createElementNS(NS,name);
  for(const [key,value] of Object.entries(attributes)) node.setAttribute(key,value);
  if(label!==undefined)node.textContent=label;
  return node;
}

export function drawConnections(container, edges) {
  container.replaceChildren();
  if(!edges.length) {
    const p=document.createElement("p");p.textContent="Add the first connection to draw your system.";container.append(p);return;
  }
  const names=[...new Set(edges.flatMap(e=>[e.from,e.to]))];
  const width=548, height=names.length*60+28;
  const svg=shape("svg",{viewBox:"0 0 "+width+" "+height,width:"100%",height,role:"img","aria-label":names.length+" components and "+edges.length+" directed connection"+(edges.length===1?"":"s")+". Details follow below."});
  svg.style.minWidth="315px";svg.style.maxWidth="548px";
  const defs=shape("defs",{}), arrow=shape("marker",{id:"edge-tip",markerWidth:7,markerHeight:7,refX:6,refY:3,orient:"auto"});
  arrow.append(shape("path",{d:"M0,0 L6,3 L0,6",fill:"none",stroke:"#575388"}));defs.append(arrow);svg.append(defs);
  const y=name=>30+names.indexOf(name)*60;
  for(const [i,e] of edges.entries()) {
    const downward=y(e.from)<=y(e.to), side=downward?1:-1, boundary=downward?413:135;
    const travel=30+((i*17)%83), lane=boundary+side*travel, start=y(e.from)+8, end=y(e.to)-8;
    svg.append(shape("path",{d:"M "+boundary+" "+start+" H "+lane+" V "+end+" H "+(boundary+side*2),fill:"none",stroke:"#77749e","stroke-width":"1.7","marker-end":"url(#edge-tip)"}));
    svg.append(shape("text",{x:lane+side*4,y:(start+end)/2+3,"text-anchor":downward?"start":"end","font-size":11,fill:"#504c79"},String(i+1)));
  }
  for(const name of names) {
    const center=y(name);
    svg.append(shape("rect",{x:135,y:center-18,width:278,height:36,rx:7,fill:"#eeecf7",stroke:"#c4bfd8"}));
    const label=shape("text",{x:274,y:center+4,"text-anchor":"middle","font-size":13,fill:"#302e43"},name.length>34?name.slice(0,32)+"…":name);
    label.append(shape("title",{},name));svg.append(label);
  }
  container.append(svg);
}
