//! Metal-only WGSL sources and its existing Naga compilation path.
use super::*;
pub(super) const PARAMETERS: &str = "struct Params { words: array<vec4<u32>,64> };\n";
pub(super) fn source(key: KernelKey) -> RuntimeResult<String> {
    let half = key.precision != 0;
    let scalar = if half { "f16" } else { "f32" };
    let enable = if half { "enable f16;\n" } else { "" };
    let body = match key.id {
        #[cfg(test)]
        KernelId::CounterProbe => return Ok("@group(0) @binding(0) var<storage,read_write> value:array<u32>; @compute @workgroup_size(1) fn main(){value[0]+=1u;}".into()),
        #[cfg(test)]
        KernelId::RoundHalfProbe => format!("{}\n{}",include_str!("round_f16.wgsl"),"@group(0) @binding(0) var<storage,read> input:array<f32>; @group(0) @binding(1) var<storage,read_write> output:array<f32>; @group(0) @binding(2) var<uniform> params:Params; @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){if id.x<params.words[0].x {output[id.x+params.words[0].y]=round_to_half(input[id.x]);}}"),
        #[cfg(test)]
        KernelId::SqueezeExcite | KernelId::SmallProjection | KernelId::HalfStoreProbe => {
            let kind=match key.id {KernelId::SqueezeExcite=>operators::ShaderKind::SqueezeExcite,KernelId::SmallProjection=>operators::ShaderKind::SmallProjection,_=>operators::ShaderKind::General};
            let body=if key.id==KernelId::HalfStoreProbe{"@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){if id.x<p(0u){store_output(id.x,load_a(id.x));}}".to_owned()}else{kind.source()};
            let mut source=format!("{}\n{body}",kind.common_source()).replace("$ENABLE",enable).replace("$HALF_OUTPUT",if key.precision&32!=0{"true"}else{"false"});
            for (i,token) in ["$A","$B","$C","$D","$E","$O"].iter().enumerate(){source=source.replace(token,if key.precision&(1<<i)!=0{"f16"}else{"f32"});}
            return Ok(source);
        }
        KernelId::Crop => format!(
            "{}{}",
            include_str!("precise.wgsl"),
            include_str!("crop.wgsl")
        ),
        KernelId::CropMeta => format!(
            "{}{}",
            include_str!("precise.wgsl"),
            include_str!("crop_meta.wgsl")
        ),
        KernelId::PackBgr => include_str!("pack_bgr.wgsl").into(),
        KernelId::Rotate => include_str!("rotate.wgsl").into(),
        KernelId::DbRows => include_str!("db_rows.wgsl").into(),
        KernelId::DbSort => include_str!("db_sort.wgsl").into(),
        KernelId::HeadReduce => include_str!("head_reduce.wgsl").into(),
        KernelId::HeadReduceParallel => return Ok(include_str!("head_reduce_parallel.wgsl").into()),
        KernelId::Ctc => include_str!("ctc.wgsl").into(),
        KernelId::DbLabel => include_str!("db_label.wgsl").replace("$INPUT", scalar),
        KernelId::DbGeometry | KernelId::DbGeometryFast => format!(
            "{}{}",
            if key.id == KernelId::DbGeometryFast {
                include_str!("precise_native_trig.wgsl")
            } else {
                include_str!("precise.wgsl")
            },
            include_str!("db_geometry.wgsl").replace("$INPUT", scalar)
        ),
        KernelId::Heads => include_str!("heads.wgsl").replace("$INPUT", scalar),
        KernelId::Heads4 => include_str!("heads4.wgsl").replace("$INPUT", scalar),
        KernelId::HeadsFeaturesF32 => include_str!("heads_features_f32.wgsl").into(),
        KernelId::Image => format!(
            "{}\nfn normalize_store(value:f32)->{scalar}{{return {};}}\n{}",
            include_str!("round_f16.wgsl"),
            if half {
                "f16(round_to_half(value))"
            } else {
                "value"
            },
            include_str!("image.wgsl").replace("$OUT", scalar)
        ),
        _ => {
            return Err(RuntimeError::with_kind(
                ErrorKind::Unsupported,
                "invalid resident Metal kernel",
            ));
        }
    };
    Ok(format!("{enable}{PARAMETERS}{body}"))
}
