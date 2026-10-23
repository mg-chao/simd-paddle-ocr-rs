@group(0) @binding(0) var<storage,read> boxes:array<u32>;
@group(0) @binding(1) var<storage,read_write> output:array<u32>;
@group(0) @binding(2) var<uniform> params:Params;
fn p(i:u32)->u32{return params.words[i/4u][i%4u];}
fn swap(a:u32,b:u32){for(var k=0u;k<16u;k++){let v=output[(a+1u)*16u+k];output[(a+1u)*16u+k]=output[(b+1u)*16u+k];output[(b+1u)*16u+k]=v;}}
fn point(i:u32)->vec2<f32>{return vec2<f32>(bitcast<f32>(output[(i+1u)*16u]),bitcast<f32>(output[(i+1u)*16u+1u]));}
fn less(a:vec2<f32>,b:vec2<f32>)->bool{return a.y<b.y || (a.y==b.y && a.x<b.x);}
fn swap_greater(a:u32,b:u32){if less(point(b),point(a)){swap(a,b);}}
fn down_heap(base:u32,start:u32,n:u32){
    var i=start;
    loop{if i>n/2u{break;}var child=2u*i;if child<n && less(point(base+child-1u),point(base+child)){child++;}
        if !less(point(base+i-1u),point(base+child-1u)){break;}swap(base+i-1u,base+child-1u);i=child;
    }
}
fn canonical_sort(count:u32){
    if count<2u{return;}
    var stack:array<vec3<u32>,64>;var top=1u;stack[0]=vec3<u32>(0u,count,2u*(32u-countLeadingZeros(count)));
    loop{
        if top==0u{break;}top--;let entry=stack[top];let base=entry.x;let n=entry.y;let depth=entry.z;
        if n<2u{continue;}
        if n<=16u{
            if n==2u{swap_greater(base,base+1u);}else if n==3u{swap_greater(base,base+1u);swap_greater(base,base+2u);swap_greater(base+1u,base+2u);}
            else{for(var i=1u;i<n;i++){var j=i;while j>0u && less(point(base+j),point(base+j-1u)){swap(base+j,base+j-1u);j--;}}}continue;
        }
        if depth==0u{
            for(var i=n/2u;i>0u;i--){down_heap(base,i,n);}for(var i=n;i>1u;i--){swap(base,base+i-1u);down_heap(base,1u,i-1u);}continue;
        }
        let hi=n-1u;let middle=hi>>1u;
        swap_greater(base,base+middle);swap_greater(base,base+hi);swap_greater(base+middle,base+hi);
        let pivot=point(base+middle);swap(base+middle,base+hi-1u);var left=0u;var right=hi-1u;
        loop{left++;while less(point(base+left),pivot){left++;}right--;while less(pivot,point(base+right)){right--;}
            if left>=right{break;}swap(base+left,base+right);
        }
        if left!=hi-1u{swap(base+left,base+hi-1u);}
        stack[top]=vec3<u32>(base,left,depth-1u);top++;stack[top]=vec3<u32>(base+left+1u,n-left-1u,depth-1u);top++;
    }
}
@compute @workgroup_size(1)
fn main(){
    var count=0u;
    for(var i=0u;i<p(0u);i++){if boxes[i*16u+9u]==0u{continue;}for(var k=0u;k<16u;k++){output[(count+1u)*16u+k]=boxes[i*16u+k];}count++;}
    canonical_sort(count);
    for(var i=1u;i<count;i++){var j=i;while j>0u{let a=point(j-1u);let b=point(j);if abs(b.y-a.y)>=10.0 || b.x>=a.x{break;}swap(j,j-1u);j--;}}
    output[0]=count;
}
