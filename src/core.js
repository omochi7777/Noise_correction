// ===== mottle-fix core (runs inside a Web Worker) =====
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}

function makeWS(w,h){
  const n=w*h, f=()=>new Float32Array(n);
  return {m:f(),c:f(),a:f(),b:f(),t:f(),sq:f(),rp:new Float64Array(w+1),cp:new Float64Array(h+1)};
}

// box mean with clipped window
function box(src,w,h,r,out,W){
  const tmp=W.t, rp=W.rp, cp=W.cp;
  for(let y=0;y<h;y++){
    const o=y*w; rp[0]=0;
    for(let x=0;x<w;x++) rp[x+1]=rp[x]+src[o+x];
    for(let x=0;x<w;x++){const a=x-r<0?0:x-r, b=x+r+1>w?w:x+r+1; tmp[o+x]=(rp[b]-rp[a])/(b-a);}
  }
  for(let x=0;x<w;x++){
    cp[0]=0;
    for(let y=0;y<h;y++) cp[y+1]=cp[y]+tmp[y*w+x];
    for(let y=0;y<h;y++){const a=y-r<0?0:y-r, b=y+r+1>h?h:y+r+1; out[y*w+x]=(cp[b]-cp[a])/(b-a);}
  }
}

// self-guided filter (He et al.), in place
function guided(I,w,h,r,eps,W){
  const n=w*h,{m,c,a,b,sq}=W;
  for(let i=0;i<n;i++) sq[i]=I[i]*I[i];
  box(I,w,h,r,m,W); box(sq,w,h,r,c,W);
  for(let i=0;i<n;i++){const v=c[i]-m[i]*m[i]; const ai=v/(v+eps); a[i]=ai; b[i]=m[i]-ai*m[i];}
  box(a,w,h,r,m,W); box(b,w,h,r,c,W);
  for(let i=0;i<n;i++) I[i]=m[i]*I[i]+c[i];
}

function median3(P,w,h){
  const out=new Float32Array(w*h), v=new Float32Array(9);
  for(let y=0;y<h;y++){
    const y0=y>0?y-1:0, y2=y<h-1?y+1:h-1;
    for(let x=0;x<w;x++){
      const x0=x>0?x-1:0, x2=x<w-1?x+1:w-1;
      let k=0;
      for(const yy of [y0,y,y2]){const o=yy*w; v[k++]=P[o+x0]; v[k++]=P[o+x]; v[k++]=P[o+x2];}
      for(let i=1;i<9;i++){const t=v[i]; let j=i-1; while(j>=0&&v[j]>t){v[j+1]=v[j];j--;} v[j+1]=t;}
      out[y*w+x]=v[4];
    }
  }
  return out;
}

function lum(R,G,B,n,scale){const L=new Float32Array(n); for(let i=0;i<n;i++) L[i]=(0.299*R[i]+0.587*G[i]+0.114*B[i])*scale; return L;}

// Immerkær noise estimate on flat areas (L in 0..255)
function noiseSigma(L,w,h){
  if(w<3||h<3) return 0;
  const g=new Float32Array((w-2)*(h-2)), cv=new Float32Array((w-2)*(h-2));
  let k=0;
  for(let y=1;y<h-1;y++) for(let x=1;x<w-1;x++){
    const i=y*w+x;
    const tl=L[i-w-1],t=L[i-w],tr=L[i-w+1],l=L[i-1],c=L[i],r=L[i+1],bl=L[i+w-1],b=L[i+w],br=L[i+w+1];
    cv[k]=Math.abs(tl-2*t+tr-2*l+4*c-2*r+bl-2*b+br);
    g[k]=Math.abs((tr+2*r+br)-(tl+2*l+bl))+Math.abs((bl+2*b+br)-(tl+2*t+tr));
    k++;
  }
  // exclude the top 30% gradients (edges, lines, text)
  const hist=new Uint32Array(1024); let mx=0; for(let i=0;i<k;i++) if(g[i]>mx) mx=g[i];
  const sc=1023/(mx||1); for(let i=0;i<k;i++) hist[(g[i]*sc)|0]++;
  let acc=0, thr=mx; for(let i=0;i<1024;i++){acc+=hist[i]; if(acc>=k*0.7){thr=(i+1)/sc; break;}}
  let s=0,cnt=0; for(let i=0;i<k;i++) if(g[i]<=thr){s+=cv[i];cnt++;}
  return cnt? Math.sqrt(Math.PI/2)*s/(6*cnt):0;
}

// mottle (low-frequency blotch) estimate: std of (box r) minus (box 3r) residual in flat areas, 0..255
function mottleLevel(L,w,h,r){
  const n=w*h,W=makeWS(w,h),a=new Float32Array(n),b=new Float32Array(n);
  box(L,w,h,r,a,W); box(L,w,h,r*4,b,W);
  const d=new Float32Array(n); for(let i=0;i<n;i++) d[i]=Math.abs(a[i]-b[i]);
  const sorted=Float32Array.from(d).sort();
  // median of absolute band-pass response; edges sit in the tail
  return sorted[(n*0.5)|0]*1.4826;
}

function kmeans(R,G,B,n,k,seed){
  const rnd=mulberry32(seed), N=Math.min(n,40000);
  const sr=new Float32Array(N),sg=new Float32Array(N),sb=new Float32Array(N);
  for(let i=0;i<N;i++){const j=(rnd()*n)|0; sr[i]=R[j];sg[i]=G[j];sb[i]=B[j];}
  const wr=0.3,wg=0.45,wb=0.25;
  const C=new Float32Array(k*3), D=new Float32Array(N).fill(1e9);
  let f=(rnd()*N)|0; C[0]=sr[f];C[1]=sg[f];C[2]=sb[f];
  for(let c=1;c<k;c++){
    let tot=0; const pr=C[(c-1)*3],pg=C[(c-1)*3+1],pb=C[(c-1)*3+2];
    for(let i=0;i<N;i++){const dr=sr[i]-pr,dg=sg[i]-pg,db=sb[i]-pb; const d=wr*dr*dr+wg*dg*dg+wb*db*db; if(d<D[i])D[i]=d; tot+=D[i];}
    let t=rnd()*tot, j=0; for(;j<N-1;j++){t-=D[j]; if(t<=0)break;}
    C[c*3]=sr[j];C[c*3+1]=sg[j];C[c*3+2]=sb[j];
  }
  const lab=new Int32Array(N), sum=new Float64Array(k*3), cnt=new Uint32Array(k);
  for(let it=0;it<12;it++){
    sum.fill(0);cnt.fill(0);
    for(let i=0;i<N;i++){
      let best=0,bd=1e9;
      for(let c=0;c<k;c++){const dr=sr[i]-C[c*3],dg=sg[i]-C[c*3+1],db=sb[i]-C[c*3+2]; const d=wr*dr*dr+wg*dg*dg+wb*db*db; if(d<bd){bd=d;best=c;}}
      lab[i]=best; cnt[best]++; sum[best*3]+=sr[i];sum[best*3+1]+=sg[i];sum[best*3+2]+=sb[i];
    }
    for(let c=0;c<k;c++){ if(cnt[c]){C[c*3]=sum[c*3]/cnt[c];C[c*3+1]=sum[c*3+1]/cnt[c];C[c*3+2]=sum[c*3+2]/cnt[c];}
      else {const j=(rnd()*N)|0; C[c*3]=sr[j];C[c*3+1]=sg[j];C[c*3+2]=sb[j];} }
  }
  return C;
}

function smoothstep(a,b,x){const t=Math.min(1,Math.max(0,(x-a)/(b-a)));return t*t*(3-2*t);}

// p: {strength 0-100, passes 1-3, despeckle bool, colors 0|k, flatten 0-100, protect 0-100}
function processImage(rgba,w,h,p,cache){
  const t0=Date.now(), n=w*h;
  cache=cache||{};
  const key=JSON.stringify([p.strength,p.passes,p.despeckle,p.colors,p.flatten,p.protect]);
  let core=cache.core&&cache.core.key===key?cache.core:null;
  if(!core){ core=coreStage(rgba,w,h,p,cache); core.key=key; cache.core=core; }
  return finishStage(rgba,w,h,p,core,t0);
}
function coreStage(rgba,w,h,p,cache){
  const n=w*h;
  let R=new Float32Array(n),G=new Float32Array(n),B=new Float32Array(n);
  for(let i=0;i<n;i++){R[i]=rgba[i*4]/255;G[i]=rgba[i*4+1]/255;B[i]=rgba[i*4+2]/255;}
  const scale=Math.max(0.6,Math.max(w,h)/768);
  if(!cache.before){ cache.before={sigma:noiseSigma(lum(R,G,B,n,255),w,h), mottle:mottleLevel(lum(R,G,B,n,255),w,h,Math.max(1,Math.round(2*scale)))}; }

  let bR=R,bG=G,bB=B;
  if(p.despeckle){bR=median3(R,w,h);bG=median3(G,w,h);bB=median3(B,w,h);}

  const sR=Float32Array.from(bR),sG=Float32Array.from(bG),sB=Float32Array.from(bB);
  const W=makeWS(w,h);
  const s=p.strength/100;
  if(s>0){
    const r=Math.max(1,Math.round((1+s*8)*scale));
    const e=0.006+s*0.11, eps=e*e;
    for(let k=0;k<p.passes;k++){guided(sR,w,h,r,eps,W);guided(sG,w,h,r,eps,W);guided(sB,w,h,r,eps,W);}
  }

  // palette flattening
  let palette=null;
  const fR=sR,fG=sG,fB=sB;
  if(p.colors>0 && p.flatten>0){
    const k=p.colors, C=kmeans(sR,sG,sB,n,k,1234+k);
    const lab=new Uint8Array(n), qR=new Float32Array(n),qG=new Float32Array(n),qB=new Float32Array(n), cnt=new Uint32Array(k);
    const wr=0.3,wg=0.45,wb=0.25;
    for(let i=0;i<n;i++){
      let best=0,bd=1e9;
      for(let c=0;c<k;c++){const dr=sR[i]-C[c*3],dg=sG[i]-C[c*3+1],db=sB[i]-C[c*3+2]; const d=wr*dr*dr+wg*dg*dg+wb*db*db; if(d<bd){bd=d;best=c;}}
      lab[i]=best; cnt[best]++; qR[i]=C[best*3];qG[i]=C[best*3+1];qB[i]=C[best*3+2];
    }
    // anti-alias label boundaries
    const aR=Float32Array.from(qR),aG=Float32Array.from(qG),aB=Float32Array.from(qB);
    for(let y=1;y<h-1;y++) for(let x=1;x<w-1;x++){
      const i=y*w+x, l=lab[i];
      if(lab[i-1]!==l||lab[i+1]!==l||lab[i-w]!==l||lab[i+w]!==l){
        let r=0,g=0,b=0;
        for(let dy=-1;dy<=1;dy++) for(let dx=-1;dx<=1;dx++){const j=i+dy*w+dx; r+=qR[j];g+=qG[j];b+=qB[j];}
        aR[i]=r/9;aG[i]=g/9;aB[i]=b/9;
      }
    }
    const f=p.flatten/100;
    for(let i=0;i<n;i++){fR[i]+=(aR[i]-fR[i])*f;fG[i]+=(aG[i]-fG[i])*f;fB[i]+=(aB[i]-fB[i])*f;}
    palette=[];
    for(let c=0;c<k;c++) palette.push({r:Math.round(C[c*3]*255),g:Math.round(C[c*3+1]*255),b:Math.round(C[c*3+2]*255),share:cnt[c]/n});
    palette.sort((a,b)=>b.share-a.share);
  }

  // edge / text protection
  if(p.protect>0){
    const L=lum(bR,bG,bB,n,1), Lb=new Float32Array(n);
    box(L,w,h,Math.max(1,Math.round(scale)),Lb,W);
    const pr=p.protect/100;
    for(let y=1;y<h-1;y++) for(let x=1;x<w-1;x++){
      const i=y*w+x;
      const gx=(Lb[i-w+1]+2*Lb[i+1]+Lb[i+w+1])-(Lb[i-w-1]+2*Lb[i-1]+Lb[i+w-1]);
      const gy=(Lb[i+w-1]+2*Lb[i+w]+Lb[i+w+1])-(Lb[i-w-1]+2*Lb[i-w]+Lb[i-w+1]);
      const g=(Math.abs(gx)+Math.abs(gy))/4;
      const m=smoothstep(0.05,0.16,g)*pr;
      if(m>0){fR[i]+=(bR[i]-fR[i])*m;fG[i]+=(bG[i]-fG[i])*m;fB[i]+=(bB[i]-fB[i])*m;}
    }
  }

  return {fR,fG,fB,palette,scale,before:cache.before};
}

function finishStage(rgba,w,h,p,core,t0){
  const n=w*h, W=makeWS(w,h);
  const fR=Float32Array.from(core.fR),fG=Float32Array.from(core.fG),fB=Float32Array.from(core.fB);
  if(p.sharpen>0){
    const amt=p.sharpen/100*1.2, rad=Math.max(1,Math.round((p.sharpRadius||1)*Math.max(1,core.scale*0.75))), bl=new Float32Array(n);
    for(const P of [fR,fG,fB]){box(P,w,h,rad,bl,W); for(let i=0;i<n;i++) P[i]+=(P[i]-bl[i])*amt;}
  }
  const c=p.contrast||0, br=p.brightness||0, sa=p.saturation||0;
  if(c||br||sa){
    const k=c>=0?1+c/50:1+c/100, off=br/200, sk=Math.max(0,1+sa/50);
    for(let i=0;i<n;i++){
      let r=fR[i],g=fG[i],b=fB[i];
      if(sa){const L=0.299*r+0.587*g+0.114*b; r=L+(r-L)*sk; g=L+(g-L)*sk; b=L+(b-L)*sk;}
      fR[i]=(r-0.5)*k+0.5+off; fG[i]=(g-0.5)*k+0.5+off; fB[i]=(b-0.5)*k+0.5+off;
    }
  }
  const out=new Uint8ClampedArray(n*4);
  for(let i=0;i<n;i++){out[i*4]=fR[i]*255+0.5;out[i*4+1]=fG[i]*255+0.5;out[i*4+2]=fB[i]*255+0.5;out[i*4+3]=rgba[i*4+3];}
  const Lo=new Float32Array(n); for(let i=0;i<n;i++) Lo[i]=0.299*out[i*4]+0.587*out[i*4+1]+0.114*out[i*4+2];
  const sigmaAfter=noiseSigma(Lo,w,h);
  const mrAfter=mottleLevel(Lo,w,h,Math.max(1,Math.round(2*core.scale)));
  return {out,stats:{sigmaBefore:core.before.sigma,sigmaAfter,mottleBefore:core.before.mottle,mottleAfter:mrAfter,palette:core.palette,ms:Date.now()-t0}};
}

// suggest settings from measured noise
function suggest(sigma,mottle){
  const lvl=Math.max(sigma*0.8,mottle);
  const strength=Math.round(Math.min(90,Math.max(20,lvl*9)));
  return {strength,passes:lvl>5?2:1,despeckle:sigma>1.2,colors:0,flatten:0,protect:55,sharpen:20};
}
if(typeof module!=='undefined') module.exports={processImage,suggest};
