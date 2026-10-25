use super::*;
use operators::ShaderKind;
impl ShaderKind {
    pub fn source(self) -> String {
        match self {
            Self::Attention | Self::AttentionViews => include_str!("attention.wgsl").replace(
                "$VIEWS",
                if self == Self::AttentionViews {
                    "true"
                } else {
                    "false"
                },
            ),
            Self::SqueezeExcite => include_str!("se.wgsl").into(),
            Self::SmallProjection => include_str!("small_projection.wgsl").into(),
            Self::Transposed => include_str!("transposed.wgsl").into(),
            Self::ReducePartial
            | Self::ReduceFinish
            | Self::SoftmaxPartial
            | Self::SoftmaxFinish => include_str!("reduce_two_stage.wgsl")
                .replace(
                    "$SOFTMAX",
                    if matches!(self, Self::SoftmaxPartial | Self::SoftmaxFinish) {
                        "true"
                    } else {
                        "false"
                    },
                )
                .replace(
                    "$FINISH",
                    if matches!(self, Self::ReduceFinish | Self::SoftmaxFinish) {
                        "true"
                    } else {
                        "false"
                    },
                ),
            Self::ConvMpp2Staged
            | Self::ConvMpp3s2Staged
            | Self::ConvMpp2s1
            | Self::ConvMpp3s1
            | Self::ConvMpp3s2
            | Self::ConvMpp3s21
            | Self::ConvMpp3s12
            | Self::ConvMpp7s1
            | Self::ConvMpp7s2 => {
                unreachable!("native MPP pipelines require Metal passthrough compilation")
            }
            Self::PointwiseF16
            | Self::LinearNative
            | Self::LinearMpp
            | Self::TailProjection
            | Self::PointwiseNative64
            | Self::DepthwiseNative3
            | Self::DepthwiseNative5
            | Self::DepthwiseNative7
            | Self::DepthwiseNative9
            | Self::DepthwiseNative1x7 => {
                unreachable!("native pipelines require Metal passthrough compilation")
            }
            Self::PackNhwc => include_str!("pack_nhwc.wgsl").into(),
            Self::Convolution => include_str!("conv.wgsl").into(),
            Self::General => include_str!("operators.wgsl").into(),
            Self::MatMul16 | Self::MatMul32 | Self::Conv16 | Self::Conv32 => self.tiled_source(),
            Self::Depthwise3 | Self::Depthwise5 => include_str!("depthwise.wgsl")
                .replace("$KERNEL", if self == Self::Depthwise3 { "3" } else { "5" }),
            Self::Reduce | Self::Softmax | Self::LayerNorm => include_str!("reduce.wgsl").replace(
                "$MODE",
                match self {
                    Self::Reduce => "0",
                    Self::Softmax => "1",
                    _ => "2",
                },
            ),
        }
    }
    pub fn common_source(self) -> String {
        let source = self.metal_common_source();
        if cfg!(target_os = "macos") {
            source
        } else {
            // Vulkan's default F32-to-F16 conversion need not use ties-to-even.
            // Round before conversion while retaining the qualified Metal source.
            source.replace("$O(value)", "$O(output_round(value))")
        }
    }
    pub(super) fn metal_common_source(self) -> String {
        let rounding = include_str!("round_f16.wgsl").replace(
            "fn round_to_half(value:f32)->f32 {",
            "fn output_round(value:f32)->f32 {\n    if !$HALF_OUTPUT {return value;}",
        );
        include_str!("common.wgsl").replace("$ROUND_F16", &rounding)
    }
    fn tiled_source(self) -> String {
        let tile = if matches!(self, Self::MatMul32 | Self::Conv32) {
            32
        } else {
            16
        };
        let mut init = String::new();
        let mut accumulate = String::new();
        let mut store = String::new();
        for register in 0..tile * tile / 64 {
            let row = register * 64 / tile;
            init.push_str(&format!(
                        "var sum{register}=0.0; if CONV && p(19u)!=0u && mr+local_row+{row}u<rows {{ sum{register}=load_c(mr+local_row+{row}u); }}\n"
                    ));
            accumulate.push_str(&format!(
                "sum{register}=sum{register}+left[(local_row+{row}u)*16u+k]*right_value;\n"
            ));
            store.push_str(&format!(
                        "{{ let row=mr+local_row+{row}u; let col=nc+local_column; if row<rows && col<columns {{ let index=(batch*rows+row)*columns+col; store_output(index,epilogue(sum{register},index)); }} }}\n"
                    ));
        }
        include_str!("tiled.wgsl")
            .replace("$TILE", &tile.to_string())
            .replace("$SHARED", &(tile * 16).to_string())
            .replace("$INIT", &init)
            .replace("$ACCUMULATE", &accumulate)
            .replace("$STORE", &store)
            .replace(
                "$CONV",
                if matches!(self, Self::Conv16 | Self::Conv32) {
                    "true"
                } else {
                    "false"
                },
            )
    }
}
