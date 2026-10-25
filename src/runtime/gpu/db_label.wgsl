@group(0) @binding(0) var<storage,read> prediction:array<$INPUT>;
@group(0) @binding(1) var<storage,read_write> mask:array<u32>;
@group(0) @binding(2) var<storage,read_write> parents:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> counts:array<atomic<u32>>;
@group(0) @binding(4) var<storage,read_write> ranks:array<u32>;
@group(0) @binding(5) var<storage,read_write> blocks:array<u32>;
@group(0) @binding(6) var<storage,read_write> metadata:array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params:Params;
fn p(i:u32)->u32{return params.words[i/4u][i%4u];}
fn root(value:u32)->u32{var v=value;loop{let next=atomicLoad(&parents[v]);if next==v{break;}v=next;}return v;}
fn join(a:u32,b:u32){
    var left=root(a);var right=root(b);
    loop{
        if left==right{break;}
        let hi=max(left,right);let lo=min(left,right);
        let previous=atomicMin(&parents[hi],lo);
        if previous==hi{break;}
        left=root(previous);right=root(lo);
    }
}
fn is_foreground(i:u32)->bool{return f32(prediction[i])>bitcast<f32>(p(2u));}
fn qualifier(i:u32)->u32{
    if atomicLoad(&parents[i])!=i{return 0u;}
    let count=atomicLoad(&counts[i]);if count<3u || (count&0x80000000u)!=0u{return 0u;}
    return select(2u,1u,mask[i]!=0u);
}
var<workgroup> tile_parents:array<atomic<u32>,64>;
fn tile_root(i:u32)->u32{var r=i;loop{let next=atomicLoad(&tile_parents[r]);if next==r{break;}r=next;}return r;}
fn tile_join(a:u32,b:u32){var x=tile_root(a);var y=tile_root(b);loop{if x==y{break;}let hi=max(x,y);let lo=min(x,y);let previous=atomicMin(&tile_parents[hi],lo);if previous==hi{break;}x=tile_root(previous);y=tile_root(lo);}}
var<workgroup> scan_fg:array<u32,64>;
var<workgroup> scan_bg:array<u32,64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>,@builtin(num_workgroups) grid:vec3<u32>){
    let block=group.x+group.y*grid.x;let i=block*64u+lane;let w=p(0u);let h=p(1u);let n=w*h;let mode=p(4u);
    if mode==1u{
        let columns=(w+7u)/8u;let bx=(block%columns)*8u;let by=(block/columns)*8u;
        atomicStore(&tile_parents[lane],lane);workgroupBarrier();
        let x=bx+lane%8u;let y=by+lane/8u;
        if x<w && y<h {
            let pixel=y*w+x;let fg=mask[pixel];
            if lane%8u<7u && x+1u<w && mask[pixel+1u]==fg{tile_join(lane,lane+1u);}
            if lane/8u<7u && y+1u<h {
                if mask[pixel+w]==fg{tile_join(lane,lane+8u);}
                if fg!=0u {
                    if lane%8u>0u && mask[pixel+w-1u]!=0u{tile_join(lane,lane+7u);}
                    if lane%8u<7u && x+1u<w && mask[pixel+w+1u]!=0u{tile_join(lane,lane+9u);}
                }
            }
        }workgroupBarrier();
        if x<w && y<h{let r=tile_root(lane);atomicStore(&parents[y*w+x],(by+r/8u)*w+bx+r%8u);}return;
    }
    if mode==4u{
        if block>=(n+63u)/64u{return;}
        var kind=0u;if i<n{kind=qualifier(i);}
        scan_fg[lane]=u32(kind==1u);scan_bg[lane]=u32(kind==2u);workgroupBarrier();
        for(var d=1u;d<64u;d*=2u){var a=0u;var b=0u;if lane>=d{a=scan_fg[lane-d];b=scan_bg[lane-d];}workgroupBarrier();scan_fg[lane]+=a;scan_bg[lane]+=b;workgroupBarrier();}
        if i<n{ranks[i]=select(scan_bg[lane],scan_fg[lane],kind==1u);}
        if lane==63u{blocks[block*4u]=scan_fg[63];blocks[block*4u+1u]=scan_bg[63];}
        return;
    }
    // Scan 64 pixel-block totals per workgroup. The final serial scan then
    // visits only one total per 4096 pixels, preserving canonical root order.
    if mode==9u{
        let block_count=(n+63u)/64u;
        if block>=(block_count+63u)/64u{return;}
        var fg=0u;var bg=0u;
        if i<block_count{fg=blocks[i*4u];bg=blocks[i*4u+1u];}
        scan_fg[lane]=fg;scan_bg[lane]=bg;workgroupBarrier();
        for(var d=1u;d<64u;d*=2u){var a=0u;var b=0u;if lane>=d{a=scan_fg[lane-d];b=scan_bg[lane-d];}workgroupBarrier();scan_fg[lane]+=a;scan_bg[lane]+=b;workgroupBarrier();}
        if i<block_count{blocks[i*4u+2u]=scan_fg[lane]-fg;blocks[i*4u+3u]=scan_bg[lane]-bg;}
        if lane==63u{let base=(block_count+block)*4u;blocks[base]=scan_fg[63];blocks[base+1u]=scan_bg[63];}
        return;
    }
    if mode==5u{
        if i!=0u{return;}
        var fg=0u;var bg=0u;
        let block_count=(n+63u)/64u;
        for(var b=0u;b<(block_count+63u)/64u;b++){let base=(block_count+b)*4u;blocks[base+2u]=fg;blocks[base+3u]=bg;fg+=blocks[base];bg+=blocks[base+1u];}
        atomicStore(&metadata[0],min(fg,p(3u)));atomicStore(&metadata[1],min(fg+bg,p(3u)));
        for(var c=0u;c<min(fg+bg,p(3u));c++){let base=16u+c*8u;atomicStore(&metadata[base+1u],w);atomicStore(&metadata[base+2u],h);}
        return;
    }
    if mode==8u{
        if i!=0u{return;}
        var rows=0u;
        for(var c=0u;c<atomicLoad(&metadata[1]);c++){let base=16u+c*8u;atomicStore(&metadata[base+5u],rows);rows+=atomicLoad(&metadata[base+4u])-atomicLoad(&metadata[base+2u])+1u;}
        atomicStore(&metadata[3],rows);return;
    }
    if i>=n{return;}
    let x=i%w;let y=i/w;
    if mode==0u{
        let value=f32(prediction[i]);if !(value<=3.402823e38 && value>=-3.402823e38){atomicOr(&metadata[2],1u);}
        var yes=is_foreground(i);
        if p(5u)!=0u{if x+1u<w{yes=yes||is_foreground(i+1u);}if y+1u<h{yes=yes||is_foreground(i+w);if x+1u<w{yes=yes||is_foreground(i+w+1u);}}}
        mask[i]=u32(yes);atomicStore(&parents[i],i);atomicStore(&counts[i],0u);return;
    }
    if mode==2u{
        let fg=mask[i];
        if (p(6u)==0u || x%8u==7u) && x+1u<w && mask[i+1u]==fg{join(i,i+1u);}
        if y+1u<h{
            if (p(6u)==0u || y%8u==7u) && mask[i+w]==fg{join(i,i+w);}
            if fg!=0u{if (p(6u)==0u || y%8u==7u || x%8u==0u) && x>0u && mask[i+w-1u]!=0u{join(i,i+w-1u);}if (p(6u)==0u || y%8u==7u || x%8u==7u) && x+1u<w && mask[i+w+1u]!=0u{join(i,i+w+1u);}}
        }return;
    }
    if mode==3u{
        let r=root(i);atomicStore(&parents[i],r);var boundary=0u;
        if mask[i]!=0u{
            var edge=x==0u||y==0u||x+1u==w||y+1u==h;
            for(var dy=-1;dy<=1;dy++){for(var dx=-1;dx<=1;dx++){let nx=i32(x)+dx;let ny=i32(y)+dy;if nx>=0 && ny>=0 && nx<i32(w) && ny<i32(h){edge=edge||mask[u32(ny)*w+u32(nx)]==0u;}}}
            boundary=u32(edge);
        }else{
            if x==0u||y==0u||x+1u==w||y+1u==h{atomicOr(&counts[r],0x80000000u);}
            if x>0u{boundary+=mask[i-1u];}if x+1u<w{boundary+=mask[i+1u];}if y>0u{boundary+=mask[i-w];}if y+1u<h{boundary+=mask[i+w];}
        }
        if boundary!=0u{atomicAdd(&counts[r],boundary);}return;
    }
    if mode==6u{
        let kind=qualifier(i);if kind==0u{ranks[i]=0xffffffffu;return;}
        let upper=((n+63u)/64u+block/64u)*4u;
        let offset=select(blocks[block*4u+3u]+blocks[upper+3u]+atomicLoad(&metadata[0]),blocks[block*4u+2u]+blocks[upper+2u],kind==1u);
        let rank=ranks[i]+offset-1u;ranks[i]=select(0xffffffffu,rank,rank<p(3u));
        if rank<p(3u){let base=16u+rank*8u;atomicStore(&metadata[base],i);atomicStore(&metadata[base+6u],kind);}
        return;
    }
    if mode==7u{
        let rank=ranks[atomicLoad(&parents[i])];if rank==0xffffffffu{return;}
        let base=16u+rank*8u;atomicMin(&metadata[base+1u],x);atomicMin(&metadata[base+2u],y);atomicMax(&metadata[base+3u],x);atomicMax(&metadata[base+4u],y);
    }
}


