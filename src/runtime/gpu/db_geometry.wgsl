@group(0) @binding(0) var<storage,read> prediction:array<$INPUT>;
@group(0) @binding(1) var<storage,read> metadata:array<u32>;
@group(0) @binding(2) var<storage,read> rows:array<u32>;
@group(0) @binding(3) var<storage,read_write> scratch:array<vec2<f32>>;
@group(0) @binding(4) var<storage,read_write> boxes:array<u32>;
@group(0) @binding(5) var<uniform> params:Params;
fn p(i:u32)->u32{return params.words[i/4u][i%4u];}
fn cross(o:vec2<f32>,a:vec2<f32>,b:vec2<f32>)->f32{let u=a-o;let v=b-o;return sf(mf(u.x,v.y),mf(u.y,v.x));}
fn flength(d:vec2<f32>)->f32{return dvalue(dq(ds(af(mf(d.x,d.x),mf(d.y,d.y)))));}
fn divide(a:f32,b:f32)->f32{return dvalue(dd(ds(a),ds(b)));}
fn project(a:vec2<f32>,b:vec2<f32>)->f32{return af(mf(a.x,b.x),mf(a.y,b.y));}
fn from_axes(u:vec2<f32>,v:vec2<f32>,a:f32,b:f32)->vec2<f32>{return vec2<f32>(af(mf(u.x,a),mf(v.x,b)),af(mf(u.y,a),mf(v.y,b)));}
// Axial edges must remain exactly axial: reciprocal-sqrt normalization can
// otherwise move integer corners below a crop-size floor boundary.
fn direction(delta:vec2<f32>)->vec2<f32>{
    if delta.x==0.0{return vec2<f32>(0.0,sign(delta.y));}
    if delta.y==0.0{return vec2<f32>(sign(delta.x),0.0);}
    let len=flength(delta);return vec2<f32>(divide(delta.x,len),divide(delta.y,len));
}
fn ordered(input:array<vec2<f32>,4>)->array<vec2<f32>,4>{
    var lo=0u;var hi=0u;
    for(var i=1u;i<4u;i++){if input[i].x+input[i].y<input[lo].x+input[lo].y{lo=i;}if input[i].x+input[i].y>input[hi].x+input[hi].y{hi=i;}}
    var remain:array<u32,2>;var n=0u;
    for(var i=0u;i<4u;i++){if i!=lo && i!=hi && n<2u{remain[n]=i;n++;}}
    var result=input;
    if n==2u && lo!=hi{
        let a=input[remain[0]];let b=input[remain[1]];
        result=array<vec2<f32>,4>(input[lo],select(b,a,a.y-a.x<b.y-b.x),input[hi],select(a,b,a.y-a.x<b.y-b.x));
        var good=true;for(var i=0u;i<4u;i++){good=good && cross(result[i],result[(i+1u)%4u],result[(i+2u)%4u])>0.0;}if good{return result;}
    }
    let center=(input[0]+input[1]+input[2]+input[3])*0.25;
    var angles:array<f32,4>;var order=array<u32,4>(0u,1u,2u,3u);
    for(var i=0u;i<4u;i++){angles[i]=atan2(input[i].y-center.y,input[i].x-center.x);}
    for(var i=1u;i<4u;i++){let k=order[i];var j=i;while j>0u && angles[order[j-1u]]>angles[k]{order[j]=order[j-1u];j--;}order[j]=k;}
    var first=0u;
    for(var i=1u;i<4u;i++){let a=input[order[i]];let b=input[order[first]];if a.x+a.y<b.x+b.y || (a.x+a.y==b.x+b.y && (a.y<b.y || (a.y==b.y && a.x<b.x))){first=i;}}
    for(var i=0u;i<4u;i++){result[i]=input[order[(first+i)%4u]];}return result;
}
struct Rect {points:array<vec2<f32>,4>,side:vec2<f32>,valid:u32};
fn rectangle(base:u32,count:u32)->Rect{
    var result:Rect;result.valid=0u;if count<3u{return result;}
    var first=0u;
    for(var i=1u;i<count;i++){let a=scratch[base+i];let b=scratch[base+first];if a.x<b.x || (a.x==b.x && a.y<b.y){first=i;}}
    var best=3.402823e38;
    for(var e=0u;e<count;e++){
        let edge=(first+e)%count;let delta=scratch[base+(edge+1u)%count]-scratch[base+edge];let len=flength(delta);if len<=0.0{continue;}
        let u=direction(delta);let v=vec2<f32>(-u.y,u.x);var low=vec2<f32>(3.402823e38);var high=vec2<f32>(-3.402823e38);
        for(var i=0u;i<count;i++){let q=scratch[base+i];let projected=vec2<f32>(project(q,u),project(q,v));low=min(low,projected);high=max(high,projected);}
        let side=high-low;let area=mf(side.x,side.y);
        if area<best{best=area;result.side=side;result.points=array<vec2<f32>,4>(from_axes(u,v,low.x,low.y),from_axes(u,v,high.x,low.y),from_axes(u,v,high.x,high.y),from_axes(u,v,low.x,high.y));result.valid=u32(area>0.0);}
    }
    result.points=ordered(result.points);return result;
}
fn push_hull(base:u32,count:u32,lower:u32,point:vec2<f32>)->u32{
    var n=count;while n>=lower+2u && cross(scratch[base+n-2u],scratch[base+n-1u],point)<=0.0{n--;}
    scratch[base+n]=point;return n+1u;
}
fn hull_rows(base:u32,rowbase:u32,ybase:u32,height:u32)->u32{
    var count=0u;
    for(var y=0u;y<height;y++){let lo=rows[(rowbase+y)*2u];let hi=rows[(rowbase+y)*2u+1u];if lo>hi{continue;}count=push_hull(base,count,0u,vec2<f32>(f32(lo),f32(ybase+y)));if hi!=lo{count=push_hull(base,count,0u,vec2<f32>(f32(hi),f32(ybase+y)));}}
    if count<2u{return 0u;}let lower=count-1u;
    var skipped=false;
    for(var yy=height;yy>0u;yy--){let y=yy-1u;let lo=rows[(rowbase+y)*2u];let hi=rows[(rowbase+y)*2u+1u];if lo>hi{continue;}
        if skipped{count=push_hull(base,count,lower,vec2<f32>(f32(hi),f32(ybase+y)));}else{skipped=true;}
        if hi!=lo{count=push_hull(base,count,lower,vec2<f32>(f32(lo),f32(ybase+y)));}
    }return count-1u;
}
fn hull_points(base:u32,n:u32)->u32{
    for(var i=1u;i<n;i++){let q=scratch[base+i];var j=i;while j>0u{let prev=scratch[base+j-1u];if prev.x<q.x || (prev.x==q.x && prev.y<=q.y){break;}scratch[base+j]=prev;j--;}scratch[base+j]=q;}
    var unique=0u;
    for(var i=0u;i<n;i++){let q=scratch[base+i];if unique==0u || any(q!=scratch[base+unique-1u]){scratch[base+unique]=q;unique++;}}
    if unique<3u{return 0u;}
    var count=0u;let out=base+512u;
    for(var i=0u;i<unique;i++){count=push_hull(out,count,0u,scratch[base+i]);}let lower=count-1u;
    for(var j=unique-1u;j>0u;j--){count=push_hull(out,count,lower,scratch[base+j-1u]);}
    return count-1u;
}
fn score(points:array<vec2<f32>,4>)->f32{
    var low=points[0];var high=low;for(var i=1u;i<4u;i++){low=min(low,points[i]);high=max(high,points[i]);}
    let left=max(i32(floor(low.x)),0);let right=min(i32(ceil(high.x)),i32(p(0u))-1);let top=max(i32(floor(low.y)),0);let bottom=min(i32(ceil(high.y)),i32(p(1u))-1);
    var q:array<vec2<i32>,4>;for(var i=0u;i<4u;i++){q[i]=vec2<i32>(points[i])-vec2<i32>(left,top);}
    var sum=0.0;var correction=0.0;var count=0u;
    for(var y=top;y<=bottom;y++){
        let cy=y-top;var thresholds:array<f32,4>;var lows:array<i32,4>;var highs:array<i32,4>;var nt=0u;var nh=0u;
        for(var i=0u;i<4u;i++){let a=q[i];let b=q[(i+3u)%4u];let d=b-a;
            if (a.y>cy)!=(b.y>cy){thresholds[nt]=af(divide(mf(f32(d.x),f32(cy-a.y)),f32(d.y)),f32(a.x));nt++;}
            if cy<min(a.y,b.y) || cy>max(a.y,b.y){continue;}
            if d.y==0{lows[nh]=min(a.x,b.x);highs[nh]=max(a.x,b.x);nh++;}
            else{let num=(cy-a.y)*d.x;if num%d.y==0{let x=a.x+num/d.y;if x>=min(a.x,b.x) && x<=max(a.x,b.x){lows[nh]=x;highs[nh]=x;nh++;}}}
        }
        for(var i=1u;i<nt;i++){let v=thresholds[i];var j=i;while j>0u && thresholds[j-1u]>v{thresholds[j]=thresholds[j-1u];j--;}thresholds[j]=v;}
        var t=0u;
        for(var x=left;x<=right;x++){let cx=x-left;while t<nt && thresholds[t]<=f32(cx){t++;}var inside=((nt-t)&1u)!=0u;
            for(var k=0u;k<nh;k++){inside=inside || (cx>=lows[k] && cx<=highs[k]);}
            // Keep each rounding observable: DXIL fast-math otherwise erases
            // Kahan's correction and large polygons accumulate visible drift.
            if inside{let value=sf(f32(prediction[u32(y)*p(0u)+u32(x)]),correction);let next=af(sum,value);correction=sf(sf(next,sum),value);sum=next;count++;}
        }
    }return select(0.0,sum/f32(max(count,1u)),count>0u);
}
fn rounded(q:vec2<f32>)->vec2<f32>{return sign(q)*floor(abs(q)+vec2<f32>(0.5));}
fn offset_point(q:vec2<f32>,normal:vec4<f32>,radius:vec2<f32>)->vec2<f32>{
    return vec2<f32>(daway(da(ds(q.x),dm(normal.xy,radius))),daway(da(ds(q.y),dm(normal.zw,radius))));
}
fn unclip(base:u32,points:array<vec2<f32>,4>)->u32{
    var area=ds(0.0);var perimeter=ds(0.0);
    for(var i=0u;i<4u;i++){let a=points[i];let b=points[(i+1u)%4u];area=da(area,da(dm(ds(a.x),ds(b.y)),-dm(ds(b.x),ds(a.y))));let d=b-a;perimeter=da(perimeter,ds(flength(d)));}
    if dvalue(area)<0.0{area=-area;}
    let delta=dd(dm(dm(area,ds(0.5)),ds(bitcast<f32>(p(3u)))),perimeter);if !(dvalue(delta)>0.0 && dvalue(delta)<3.402823e38){return 0u;}
    var pts:array<vec2<f32>,4>;var n=0u;
    for(var i=0u;i<4u;i++){let q=round(points[i]);if n>0u && all(pts[n-1u]==q){continue;}pts[n]=q;n++;}
    if n>=2u && all(pts[0]==pts[n-1u]){n--;}if n<3u{return 0u;}
    var winding=ds(0.0);for(var i=0u;i<n;i++){let a=pts[i];let b=pts[(i+1u)%n];winding=da(winding,da(dm(ds(a.x),ds(b.y)),-dm(ds(b.x),ds(a.y))));}
    if dvalue(winding)<0.0{for(var i=0u;i<n/2u;i++){let q=pts[i];pts[i]=pts[n-1u-i];pts[n-1u-i]=q;}}
    var normals:array<vec4<f32>,4>;for(var i=0u;i<n;i++){let d=pts[(i+1u)%n]-pts[i];let len=dq(da(dm(ds(d.x),ds(d.x)),dm(ds(d.y),ds(d.y))));normals[i]=vec4<f32>(dd(ds(d.y),len),dd(ds(-d.x),len));}
    var count=0u;var previous=n-1u;
    for(var i=0u;i<n;i++){
        let q=pts[i];let incoming=normals[previous];let outgoing=normals[i];scratch[base+count]=offset_point(q,incoming,delta);count++;
        let turn=da(dm(incoming.xy,outgoing.zw),-dm(incoming.zw,outgoing.xy));
        if dvalue(turn)>=0.0{
            if dvalue(da(dm(incoming.xy,outgoing.xy),dm(incoming.zw,outgoing.zw)))<0.985{
                let start=datan2(incoming.zw,incoming.xy);var end=datan2(outgoing.zw,outgoing.xy);if dvalue(end)<dvalue(start){end=da(end,dm(dpi(),ds(2.0)));}
                let angle=da(end,-start);let fraction=dd(angle,dm(dpi(),ds(2.0)));let chord=ds(min(0.25,dvalue(delta)));
                var steps=max(2u,u32(dvalue(dd(dm(fraction,dpi()),dacos(da(ds(1.0),-dd(chord,delta)))))));steps=min(steps,max(2u,u32(dvalue(dm(ds(222.0),fraction)))));
                let step=dd(angle,ds(f32(steps)));let sc=dsincos(step);let initial=dsincos(start);var vx=initial.zw;var vy=initial.xy;
                for(var j=0u;j<=steps;j++){scratch[base+count]=offset_point(q,vec4<f32>(vx,vy),delta);count++;let next=da(dm(vx,sc.zw),-dm(vy,sc.xy));vy=da(dm(vy,sc.zw),dm(vx,sc.xy));vx=next;}
            }
        }else{scratch[base+count]=q;count++;}
        scratch[base+count]=offset_point(q,outgoing,delta);count++;previous=i;
    }return count;
}
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
    let index=id.x;if index>=metadata[1]{return;}let m=16u+index*8u;let height=metadata[m+4u]-metadata[m+2u]+1u;let base=metadata[m+5u]*4u+index*1024u;
    let out=index*16u;boxes[out+9u]=0u;
    let n=hull_rows(base,metadata[m+5u],metadata[m+2u],height);var r=rectangle(base,n);
    if r.valid==0u || min(r.side.x,r.side.y)<3.0{return;}
    let confidence=score(r.points);if !(confidence>=bitcast<f32>(p(2u))){return;}
    let expanded=unclip(base,r.points);let hull=hull_points(base,expanded);r=rectangle(base+512u,hull);
    if r.valid==0u || min(r.side.x,r.side.y)<5.0{return;}
    for(var i=0u;i<4u;i++){scratch[base+i]=clamp(round(vec2<f32>(mf(divide(r.points[i].x,f32(p(0u))),f32(p(6u))),mf(divide(r.points[i].y,f32(p(1u))),f32(p(7u))))),vec2<f32>(0.0),vec2<f32>(f32(p(4u)),f32(p(5u))));}
    let mapped=hull_points(base,4u);r=rectangle(base+512u,mapped);
    if r.valid==0u || distance(r.points[0],r.points[1])<=4.0 || distance(r.points[0],r.points[3])<=4.0{return;}
    for(var i=0u;i<4u;i++){boxes[out+i*2u]=bitcast<u32>(r.points[i].x);boxes[out+i*2u+1u]=bitcast<u32>(r.points[i].y);}
    boxes[out+8u]=bitcast<u32>(confidence);boxes[out+9u]=1u;
}
