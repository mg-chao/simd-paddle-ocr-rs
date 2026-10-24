use super::*;
use crate::{
    runtime::bound::{BatchedMatMul, BinaryOp, Broadcast, Conv},
    tensor::MAX_RANK,
};

const PACKET_WORDS: usize = 128;
const EMPTY_PACKET: [u32; PACKET_WORDS] = [0; PACKET_WORDS];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum ShaderKind {
    PointwiseNhwc,
    Attention,
    AttentionViews,
    SeChain,
    SqueezeExcite,
    Transposed,
    Convolution,
    General,
    CooperativeMatMul,
    MatMul16,
    MatMul32,
    ConvPrescaled16,
    ConvPrescaled32,
    Conv16,
    Conv32,
    PointwiseNchw16,
    PointwiseNchw32,
    PointwiseBlocked32,
    PointwisePrescaled16,
    PointwisePrescaled32,
    Depthwise3,
    Depthwise5,
    Reduce,
    Softmax,
    LayerNorm,
    ReducePartial,
    ReduceFinish,
    SoftmaxPartial,
    SoftmaxFinish,
}
impl ShaderKind {
    pub const ALL: &'static [Self] = &[
        Self::PointwiseNhwc,
        Self::Attention,
        Self::AttentionViews,
        Self::SeChain,
        Self::SqueezeExcite,
        Self::Transposed,
        Self::Convolution,
        Self::General,
        Self::MatMul16,
        Self::MatMul32,
        Self::ConvPrescaled16,
        Self::ConvPrescaled32,
        Self::Conv16,
        Self::Conv32,
        Self::PointwiseNchw16,
        Self::PointwiseNchw32,
        Self::PointwisePrescaled16,
        Self::PointwisePrescaled32,
        Self::Depthwise3,
        Self::Depthwise5,
        Self::Reduce,
        Self::Softmax,
        Self::LayerNorm,
        Self::ReducePartial,
        Self::ReduceFinish,
        Self::SoftmaxPartial,
        Self::SoftmaxFinish,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::PointwiseNhwc => "pointwise_nhwc16",
            Self::Attention => "attention_online",
            Self::AttentionViews => "attention_qkv_views",
            Self::SeChain => "se_projection_chain",
            Self::SqueezeExcite => "squeeze_excitation",
            Self::Transposed => "conv_transpose_tile8",
            Self::Convolution => "conv_generic",
            Self::General => "general",
            Self::CooperativeMatMul => "experimental_cooperative_matmul8",
            Self::MatMul16 => "matmul_tile16",
            Self::MatMul32 => "matmul_tile32",
            Self::ConvPrescaled16 => "conv_prescaled16",
            Self::ConvPrescaled32 => "conv_prescaled32",
            Self::Conv16 => "conv_implicit16",
            Self::Conv32 => "conv_implicit32",
            Self::PointwiseNchw16 => "pointwise_nchw16",
            Self::PointwiseNchw32 => "pointwise_nchw32",
            Self::PointwiseBlocked32 => "pointwise_blocked32",
            Self::PointwisePrescaled16 => "pointwise_prescaled16",
            Self::PointwisePrescaled32 => "pointwise_prescaled32",
            Self::Depthwise3 => "depthwise3",
            Self::Depthwise5 => "depthwise5",
            Self::Reduce => "reduce_contiguous",
            Self::Softmax => "softmax_rows",
            Self::LayerNorm => "layer_norm",
            Self::ReducePartial => "reduce_partial",
            Self::ReduceFinish => "reduce_finish",
            Self::SoftmaxPartial => "softmax_partial",
            Self::SoftmaxFinish => "softmax_finish",
        }
    }
    pub fn source(self) -> String {
        match self {
            Self::PointwiseBlocked32 => include_str!("pointwise_blocked.wgsl").into(),
            Self::CooperativeMatMul => include_str!("cooperative.wgsl").into(),
            Self::PointwiseNhwc => include_str!("pointwise.wgsl").into(),
            Self::Attention | Self::AttentionViews => include_str!("attention.wgsl").replace(
                "$VIEWS",
                if self == Self::AttentionViews {
                    "true"
                } else {
                    "false"
                },
            ),
            Self::SeChain => include_str!("se_chain.wgsl").into(),
            Self::SqueezeExcite => include_str!("se.wgsl").into(),
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
            Self::Convolution => include_str!("conv.wgsl").into(),
            Self::General => include_str!("operators.wgsl").into(),
            Self::MatMul16
            | Self::MatMul32
            | Self::Conv16
            | Self::Conv32
            | Self::ConvPrescaled16
            | Self::ConvPrescaled32 => self.tiled_source(),
            Self::PointwiseNchw16
            | Self::PointwiseNchw32
            | Self::PointwisePrescaled16
            | Self::PointwisePrescaled32 => self.tiled_source(),
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
    fn tiled_source(self) -> String {
        let tile = if matches!(
            self,
            Self::MatMul32
                | Self::Conv32
                | Self::ConvPrescaled32
                | Self::PointwiseNchw32
                | Self::PointwisePrescaled32
        ) {
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
            .replace(
                "$PRESCALE",
                if matches!(
                    self,
                    Self::ConvPrescaled16
                        | Self::ConvPrescaled32
                        | Self::PointwisePrescaled16
                        | Self::PointwisePrescaled32
                ) {
                    "true"
                } else {
                    "false"
                },
            )
            .replace("$TILE", &tile.to_string())
            .replace("$SHARED", &(tile * 16).to_string())
            .replace("$INIT", &init)
            .replace("$ACCUMULATE", &accumulate)
            .replace("$STORE", &store)
            .replace(
                "$POINTWISE",
                if matches!(
                    self,
                    Self::PointwiseNchw16
                        | Self::PointwiseNchw32
                        | Self::PointwisePrescaled16
                        | Self::PointwisePrescaled32
                ) {
                    "true"
                } else {
                    "false"
                },
            )
            .replace(
                "$CONV",
                if matches!(
                    self,
                    Self::Conv16
                        | Self::Conv32
                        | Self::ConvPrescaled16
                        | Self::ConvPrescaled32
                        | Self::PointwiseNchw16
                        | Self::PointwiseNchw32
                        | Self::PointwisePrescaled16
                        | Self::PointwisePrescaled32
                ) {
                    "true"
                } else {
                    "false"
                },
            )
    }
}
#[derive(Clone)]
pub(super) enum RecipeKind {
    Compute(ShaderKind),
    Copy {
        source_offset: u64,
        destination_offset: u64,
        bytes: u64,
    },
}
#[derive(Clone)]
pub(super) struct Recipe {
    pub node: usize,
    pub nodes: Vec<usize>,
    pub kind: RecipeKind,
    pub inputs: Vec<TensorId>,
    pub output: TensorId,
    pub words: [u32; PACKET_WORDS],
    pub x: u32,
    pub y: u32,
}

struct DispatchParameters {
    output_count: usize,
    row_stride: usize,
}

struct ConvolutionParameters<'a> {
    geometry: &'a Conv,
    has_bias: bool,
}

struct BinaryParameters<'a> {
    operation: BinaryOp,
    broadcast: &'a Broadcast,
}

enum UnaryParameters {
    Square,
    Sqrt,
    Relu,
    Erf,
    Sigmoid,
    HardSigmoid { alpha: f32, beta: f32 },
}

enum ElementwiseParameters<'a> {
    Binary(BinaryParameters<'a>),
    Unary(UnaryParameters),
}

struct MatrixParameters<'a> {
    rows: usize,
    inner: usize,
    columns: usize,
    batch: Option<&'a BatchedMatMul>,
}

struct ReductionParameters<'a> {
    input: Shape,
    reduction_count: usize,
    output_strides: &'a [usize; MAX_RANK],
}

struct NormalizationParameters {
    channels: usize,
    channel_plane: usize,
    epsilon: f32,
}

struct SpatialExtent {
    height: usize,
    width: usize,
}

enum SpatialParameters {
    ConvTranspose {
        input_channels: usize,
        output_channels: usize,
        input: SpatialExtent,
        has_bias: bool,
    },
    MaxPool {
        input: SpatialExtent,
    },
    AveragePool {
        input: SpatialExtent,
        output: SpatialExtent,
    },
    Resize {
        input: SpatialExtent,
        factor: usize,
    },
}

struct IndexingParameters<'a> {
    rank: usize,
    base_offset: usize,
    output_dimensions: &'a [usize; MAX_RANK],
    input_strides: &'a [usize; MAX_RANK],
}

enum KernelParameters<'a> {
    Convolution(ConvolutionParameters<'a>),
    Elementwise(ElementwiseParameters<'a>),
    Matrix(MatrixParameters<'a>),
    Reduction(ReductionParameters<'a>),
    Normalization(NormalizationParameters),
    Spatial(SpatialParameters),
    Indexing(IndexingParameters<'a>),
    Softmax { columns: usize },
}

/// The only boundary between typed kernel parameters and the WGSL packet ABI.
/// All integer narrowing, rank bounds, packet bounds and field overlap checks
/// happen here. Cached recipes retain the resulting packet rather than borrows.
impl KernelParameters<'_> {
    fn serialize(&self, dispatch: DispatchParameters) -> RuntimeResult<[u32; PACKET_WORDS]> {
        let mut packet = PacketWriter::new();
        packet.unsigned(0, dispatch.output_count)?;
        packet.unsigned(1, dispatch.row_stride)?;
        match self {
            Self::Convolution(parameters) => {
                // Conv shares the shape compiler's checked NCHW/OIHW geometry.
                let geometry = parameters.geometry;
                packet.unsigned(2, geometry.batch)?;
                packet.unsigned(3, geometry.cin)?;
                packet.unsigned(4, geometry.cout)?;
                packet.unsigned(5, geometry.height)?;
                packet.unsigned(6, geometry.width)?;
                packet.unsigned(7, geometry.oh)?;
                packet.unsigned(8, geometry.ow)?;
                packet.unsigned(9, geometry.cpg)?;
                packet.unsigned(10, geometry.opg)?;
                packet.unsigned(11, geometry.kh)?;
                packet.unsigned(12, geometry.kw)?;
                packet.unsigned(13, geometry.sh)?;
                packet.unsigned(14, geometry.sw)?;
                packet.unsigned(15, geometry.dh)?;
                packet.unsigned(16, geometry.dw)?;
                packet.unsigned(17, geometry.pt)?;
                packet.unsigned(18, geometry.pl)?;
                packet.flag(19, parameters.has_bias)?;
            }
            Self::Elementwise(ElementwiseParameters::Binary(parameters)) => {
                packet.word(
                    2,
                    match parameters.operation {
                        BinaryOp::Add => 0,
                        BinaryOp::Mul => 1,
                        BinaryOp::Div => 2,
                        BinaryOp::Sub => 3,
                    },
                )?;
                let broadcast = parameters.broadcast;
                packet.unsigned(3, broadcast.rank)?;
                packet.axes(4, broadcast.rank, &broadcast.dimensions)?;
                packet.axes(12, broadcast.rank, &broadcast.a_strides)?;
                packet.axes(20, broadcast.rank, &broadcast.b_strides)?;
            }
            Self::Elementwise(ElementwiseParameters::Unary(parameters)) => {
                let operation = match parameters {
                    UnaryParameters::Square => 4,
                    UnaryParameters::Sqrt => 5,
                    UnaryParameters::Relu => 6,
                    UnaryParameters::Erf => 7,
                    UnaryParameters::Sigmoid => 8,
                    UnaryParameters::HardSigmoid { alpha, beta } => {
                        packet.float(3, *alpha)?;
                        packet.float(4, *beta)?;
                        9
                    }
                };
                packet.word(2, operation)?;
            }
            Self::Matrix(parameters) => {
                packet.word(2, 20)?;
                packet.unsigned(3, parameters.rows)?;
                packet.unsigned(4, parameters.inner)?;
                packet.unsigned(5, parameters.columns)?;
                let rank = parameters.batch.map_or(0, |batch| batch.batch_rank);
                if rank > MAX_RANK - 2 {
                    return Err(resource(
                        "GPU matrix batch rank exceeds the tensor rank limit",
                    ));
                }
                packet.unsigned(6, rank)?;
                if let Some(batch) = parameters.batch {
                    packet.axes(8, rank, &batch.dimensions)?;
                    packet.axes(16, rank, &batch.a_strides)?;
                    packet.axes(24, rank, &batch.b_strides)?;
                }
            }
            Self::Reduction(parameters) => {
                let rank = parameters.input.rank();
                packet.word(2, 18)?;
                packet.unsigned(3, rank)?;
                packet.unsigned(4, parameters.reduction_count)?;
                packet.dimensions(8, parameters.input.dimensions())?;
                packet.axes(16, rank, parameters.output_strides)?;
            }
            Self::Normalization(parameters) => {
                packet.word(2, 14)?;
                packet.unsigned(3, parameters.channels)?;
                packet.unsigned(4, parameters.channel_plane)?;
                packet.float(5, parameters.epsilon)?;
            }
            Self::Spatial(parameters) => match parameters {
                SpatialParameters::ConvTranspose {
                    input_channels,
                    output_channels,
                    input,
                    has_bias,
                } => {
                    packet.word(2, 11)?;
                    packet.unsigned(3, *input_channels)?;
                    packet.unsigned(4, *output_channels)?;
                    packet.unsigned(5, input.height)?;
                    packet.unsigned(6, input.width)?;
                    packet.flag(7, *has_bias)?;
                }
                SpatialParameters::MaxPool { input } => {
                    packet.word(2, 12)?;
                    packet.unsigned(3, input.height)?;
                    packet.unsigned(4, input.width)?;
                }
                SpatialParameters::AveragePool { input, output } => {
                    packet.word(2, 13)?;
                    packet.unsigned(3, input.height)?;
                    packet.unsigned(4, input.width)?;
                    packet.unsigned(5, output.height)?;
                    packet.unsigned(6, output.width)?;
                }
                SpatialParameters::Resize { input, factor } => {
                    packet.word(2, 17)?;
                    packet.unsigned(3, input.height)?;
                    packet.unsigned(4, input.width)?;
                    packet.unsigned(5, *factor)?;
                }
            },
            Self::Indexing(parameters) => {
                packet.word(2, 15)?;
                packet.unsigned(3, parameters.rank)?;
                packet.unsigned(4, parameters.base_offset)?;
                packet.axes(8, parameters.rank, parameters.output_dimensions)?;
                packet.axes(16, parameters.rank, parameters.input_strides)?;
            }
            Self::Softmax { columns } => {
                packet.word(2, 21)?;
                packet.unsigned(3, *columns)?;
            }
        }
        Ok(packet.words)
    }
}

struct PacketWriter {
    words: [u32; PACKET_WORDS],
    written: u128,
}

impl PacketWriter {
    fn new() -> Self {
        Self {
            words: EMPTY_PACKET,
            written: 0,
        }
    }

    fn word(&mut self, offset: usize, value: u32) -> RuntimeResult<()> {
        let destination = self
            .words
            .get_mut(offset)
            .ok_or_else(|| resource("GPU parameter field exceeds packet bounds"))?;
        let mask = 1u128 << offset;
        if self.written & mask != 0 {
            return Err(resource("GPU parameter fields overlap"));
        }
        *destination = value;
        self.written |= mask;
        Ok(())
    }

    fn unsigned(&mut self, offset: usize, value: usize) -> RuntimeResult<()> {
        let value =
            u32::try_from(value).map_err(|_| resource("GPU operator parameter exceeds U32"))?;
        self.word(offset, value)
    }

    fn flag(&mut self, offset: usize, value: bool) -> RuntimeResult<()> {
        self.word(offset, u32::from(value))
    }

    fn float(&mut self, offset: usize, value: f32) -> RuntimeResult<()> {
        if !value.is_finite() {
            return Err(resource("GPU operator float parameter is nonfinite"));
        }
        self.word(offset, value.to_bits())
    }

    fn axes(
        &mut self,
        offset: usize,
        rank: usize,
        values: &[usize; MAX_RANK],
    ) -> RuntimeResult<()> {
        let values = values
            .get(..rank)
            .ok_or_else(|| resource("GPU operator rank exceeds the tensor rank limit"))?;
        self.dimensions(offset, values)
    }

    fn dimensions(&mut self, offset: usize, values: &[usize]) -> RuntimeResult<()> {
        if values.len() > MAX_RANK {
            return Err(resource("GPU operator rank exceeds the tensor rank limit"));
        }
        let end = offset
            .checked_add(values.len())
            .ok_or_else(|| resource("GPU parameter field offset overflow"))?;
        if end > PACKET_WORDS {
            return Err(resource("GPU parameter field exceeds packet bounds"));
        }
        for (axis, &value) in values.iter().enumerate() {
            self.unsigned(offset + axis, value)?;
        }
        Ok(())
    }
}

pub(super) fn recipes(
    program: &Program,
    plan: &ShapePlan,
    limits: &wgpu::Limits,
) -> RuntimeResult<Vec<Recipe>> {
    let mut result = Vec::new();
    for (node, step) in plan.steps.iter().enumerate() {
        let instruction = &program.instructions[node];
        let mut count = plan.shapes[step.output.0].element_count();
        let inputs = instruction.inputs.to_vec();
        let parameters = match &step.kind {
            BoundKind::Conv(geometry) => KernelParameters::Convolution(ConvolutionParameters {
                geometry,
                has_bias: inputs.len() == 3,
            }),
            BoundKind::Binary(broadcast, operation) => {
                KernelParameters::Elementwise(ElementwiseParameters::Binary(BinaryParameters {
                    operation: *operation,
                    broadcast,
                }))
            }
            BoundKind::Square => {
                KernelParameters::Elementwise(ElementwiseParameters::Unary(UnaryParameters::Square))
            }
            BoundKind::Sqrt => {
                KernelParameters::Elementwise(ElementwiseParameters::Unary(UnaryParameters::Sqrt))
            }
            BoundKind::Relu => {
                KernelParameters::Elementwise(ElementwiseParameters::Unary(UnaryParameters::Relu))
            }
            BoundKind::Erf => {
                KernelParameters::Elementwise(ElementwiseParameters::Unary(UnaryParameters::Erf))
            }
            BoundKind::Sigmoid => KernelParameters::Elementwise(ElementwiseParameters::Unary(
                UnaryParameters::Sigmoid,
            )),
            BoundKind::HardSigmoid(alpha, beta) => KernelParameters::Elementwise(
                ElementwiseParameters::Unary(UnaryParameters::HardSigmoid {
                    alpha: *alpha,
                    beta: *beta,
                }),
            ),
            BoundKind::ConvTranspose(geometry) => {
                KernelParameters::Spatial(SpatialParameters::ConvTranspose {
                    input_channels: geometry.cin,
                    output_channels: geometry.cout,
                    input: SpatialExtent {
                        height: geometry.height,
                        width: geometry.width,
                    },
                    has_bias: inputs.len() == 3,
                })
            }
            BoundKind::MaxPool { height, width } => {
                KernelParameters::Spatial(SpatialParameters::MaxPool {
                    input: SpatialExtent {
                        height: *height,
                        width: *width,
                    },
                })
            }
            BoundKind::AveragePool {
                height,
                width,
                output_height,
                output_width,
            } => KernelParameters::Spatial(SpatialParameters::AveragePool {
                input: SpatialExtent {
                    height: *height,
                    width: *width,
                },
                output: SpatialExtent {
                    height: *output_height,
                    width: *output_width,
                },
            }),
            BoundKind::BatchNormalization {
                channels,
                plane,
                epsilon,
            } => KernelParameters::Normalization(NormalizationParameters {
                channels: *channels,
                channel_plane: *plane,
                epsilon: *epsilon,
            }),
            BoundKind::Transpose(mapping) => KernelParameters::Indexing(IndexingParameters {
                rank: mapping.rank,
                base_offset: mapping.base,
                output_dimensions: &mapping.dimensions,
                input_strides: &mapping.strides,
            }),
            BoundKind::Concat(concat) => {
                let mut offset = 0;
                result
                    .try_reserve(
                        concat
                            .outer
                            .checked_mul(inputs.len())
                            .ok_or_else(|| resource("GPU concat command count overflow"))?,
                    )
                    .map_err(|e| {
                        RuntimeError::caused_by(ErrorKind::Allocation, "GPU concat commands", e)
                    })?;
                for &input in &inputs {
                    let count = plan.shapes[input.0].element_count();
                    let chunk = count / concat.outer;
                    for outer in 0..concat.outer {
                        result.push(Recipe {
                            node,
                            nodes: vec![node],
                            kind: RecipeKind::Copy {
                                source_offset: (outer * chunk * 4) as u64,
                                destination_offset: ((outer * concat.output_chunk + offset) * 4)
                                    as u64,
                                bytes: (chunk * 4) as u64,
                            },
                            inputs: vec![input],
                            output: step.output,
                            words: EMPTY_PACKET,
                            x: 0,
                            y: 0,
                        });
                    }
                    offset += chunk;
                }
                continue;
            }
            BoundKind::Resize {
                height,
                width,
                factor,
            } => KernelParameters::Spatial(SpatialParameters::Resize {
                input: SpatialExtent {
                    height: *height,
                    width: *width,
                },
                factor: *factor,
            }),
            BoundKind::Reduce(reduction) => KernelParameters::Reduction(ReductionParameters {
                input: reduction.input,
                reduction_count: reduction.reduction,
                output_strides: &reduction.output_strides,
            }),
            BoundKind::Copy => {
                result.push(Recipe {
                    node,
                    nodes: vec![node],
                    kind: RecipeKind::Copy {
                        source_offset: 0,
                        destination_offset: 0,
                        bytes: (count * 4) as u64,
                    },
                    inputs,
                    output: step.output,
                    words: EMPTY_PACKET,
                    x: 0,
                    y: 0,
                });
                continue;
            }
            BoundKind::MatMul {
                rows,
                inner,
                columns,
            } => KernelParameters::Matrix(MatrixParameters {
                rows: *rows,
                inner: *inner,
                columns: *columns,
                batch: None,
            }),
            BoundKind::BatchedMatMul(batch) => KernelParameters::Matrix(MatrixParameters {
                rows: batch.rows,
                inner: batch.inner,
                columns: batch.columns,
                batch: Some(batch),
            }),
            BoundKind::Softmax { rows, columns } => {
                count = *rows;
                KernelParameters::Softmax { columns: *columns }
            }
        };
        let kind =
            RecipeKind::Compute(if matches!(&parameters, KernelParameters::Convolution(_)) {
                ShaderKind::Convolution
            } else {
                ShaderKind::General
            });
        let (x, y) = dispatch_grid(count, limits)?;
        let row_stride = (x as usize)
            .checked_mul(WORKGROUP_SIZE)
            .ok_or_else(|| resource("GPU dispatch stride overflow"))?;
        let words = parameters.serialize(DispatchParameters {
            output_count: count,
            row_stride,
        })?;
        result.push(Recipe {
            node,
            nodes: vec![node],
            kind,
            inputs,
            output: step.output,
            words,
            x,
            y,
        });
    }
    Ok(result)
}

pub(super) fn specialize(
    recipes: &mut [Recipe],
    plan: &ShapePlan,
    limits: &wgpu::Limits,
    precision: GpuPrecision,
    backend: wgpu::Backend,
) -> RuntimeResult<()> {
    let metal_f32 = backend == wgpu::Backend::Metal && precision == GpuPrecision::F32;
    for r in recipes {
        if !matches!(
            r.kind,
            RecipeKind::Compute(ShaderKind::General | ShaderKind::Convolution)
        ) {
            continue;
        }
        let mut matrix = None;
        let mut prefer_spatial32 = false;
        let mut narrow_pointwise = false;
        match &plan.steps[r.nodes[0]].kind {
            BoundKind::Conv(g) if g.dh == 1 && g.dw == 1 && g.sh <= 2 && g.sw <= 2 => {
                let rgb_stem = metal_f32
                    && g.cin == 3
                    && g.kh == 3
                    && g.kw == 3
                    && g.sh == 2
                    && g.sw == 2
                    && g.cout >= 16;
                if g.cpg == 1
                    && g.opg == 1
                    && g.cin == g.cout
                    && g.kh == g.kw
                    && matches!(g.kh, 3 | 5)
                {
                    r.kind = RecipeKind::Compute(if g.kh == 3 {
                        ShaderKind::Depthwise3
                    } else {
                        ShaderKind::Depthwise5
                    });
                    (r.x, r.y) = dispatch_grid(r.words[0] as usize, limits)?;
                    r.words[1] = r.x * 64;
                } else if g.cpg == g.cin
                    && g.kh == g.kw
                    && (matches!(g.kh, 1..=3) || (metal_f32 && matches!(g.kh, 5 | 7)))
                    && g.cout >= 8
                    && (g.cpg >= 8 || rgb_stem)
                {
                    matrix = Some((g.cout, g.oh * g.ow, g.batch, true));
                    // Short recognition outputs benefit from direct 16x16
                    // pointwise reuse before the usual 128-group threshold.
                    // Keep the smaller maps and all other operators unchanged.
                    narrow_pointwise = metal_f32
                        && g.kh == 1
                        && g.sh == 1
                        && g.sw == 1
                        && g.pt == 0
                        && g.pl == 0
                        && g.height == 1
                        && g.oh == 1
                        && g.ow == g.width
                        && g.cin >= 128
                        && g.cout >= 64
                        && g.ow >= 64;
                    // Large spatial maps provide enough groups even at 32/64
                    // output channels. Measured Metal 3x3, 5x5 and 7x7 layers
                    // benefit from the larger tile's operand reuse. Require
                    // complete output-channel tiles and retain occupancy guards.
                    prefer_spatial32 = rgb_stem
                        || (metal_f32
                            && matches!(g.kh, 3 | 5 | 7)
                            && g.cin >= 32
                            && g.cout.is_multiple_of(32)
                            && g.oh * g.ow >= 1024);
                }
            }
            BoundKind::MatMul {
                rows,
                inner,
                columns,
            } if *inner >= 16 && *rows >= 4 && *columns >= 16 => {
                matrix = Some((*rows, *columns, 1, false))
            }
            BoundKind::BatchedMatMul(m) if m.inner >= 16 && m.rows >= 4 && m.columns >= 16 => {
                let batches = plan.shapes[r.output.0].element_count() / (m.rows * m.columns);
                matrix = Some((m.rows, m.columns, batches, false));
            }
            BoundKind::ConvTranspose(g) if g.cin >= 8 && g.cout >= 8 => {
                let groups = g.batch * g.cout.div_ceil(8) * (g.height * g.width).div_ceil(8);
                r.words[32] = g.cout.div_ceil(8) as u32;
                r.words[33] = (g.height * g.width).div_ceil(8) as u32;
                r.words[34] = groups as u32;
                (r.x, r.y) = dispatch_grid(groups * 64, limits)?;
                r.words[35] = r.x;
                r.kind = RecipeKind::Compute(ShaderKind::Transposed);
            }
            BoundKind::Reduce(m)
                if r.nodes.len() == 1 && m.contiguous == Some(m.reduction) && m.reduction >= 64 =>
            {
                r.kind = RecipeKind::Compute(ShaderKind::Reduce);
                r.words[3] = m.reduction as u32;
                (r.x, r.y) = dispatch_grid(r.words[0] as usize * 64, limits)?;
                r.words[1] = r.x;
            }
            BoundKind::Softmax { rows, columns } if *columns >= 64 => {
                r.kind = RecipeKind::Compute(ShaderKind::Softmax);
                (r.x, r.y) = dispatch_grid(rows * 64, limits)?;
                r.words[1] = r.x;
            }
            _ => {}
        }
        if let Some((rows, columns, batches, conv)) = matrix {
            // Avoid leaving most compute units idle on a few large tiles.
            // Other narrow/small matrices retain the generic kernel; intermediate
            // shapes use more 16x16 groups before moving to 32x32 tiles.
            let groups16 = rows.div_ceil(16) * columns.div_ceil(16) * batches;
            let groups32 = rows.div_ceil(32) * columns.div_ceil(32) * batches;
            if groups16 < 128 && !(narrow_pointwise && groups16 >= 64) {
                continue;
            }
            let tile = if (rows >= 128 || prefer_spatial32) && columns >= 64 && groups32 >= 64 {
                32usize
            } else {
                16
            };
            r.kind = RecipeKind::Compute(match (conv, tile) {
                (true, 32) => ShaderKind::Conv32,
                (true, _) => ShaderKind::Conv16,
                (false, 32) => ShaderKind::MatMul32,
                _ => ShaderKind::MatMul16,
            });
            r.words[32] = rows.div_ceil(tile) as u32;
            r.words[33] = columns.div_ceil(tile) as u32;
            r.words[35] = batches as u32;
            let groups = rows.div_ceil(tile) * columns.div_ceil(tile) * batches;
            (r.x, r.y) = dispatch_grid(groups * 64, limits)?;
            r.words[34] = r.x;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_writer_rejects_overlap_bounds_and_integer_narrowing() {
        let mut packet = PacketWriter::new();
        packet.word(7, 123).unwrap();
        assert!(packet.word(7, 456).is_err());
        assert_eq!(packet.words[7], 123);
        assert!(packet.word(PACKET_WORDS, 1).is_err());
        assert!(packet.word(usize::MAX, 1).is_err());
        packet.unsigned(0, u32::MAX as usize).unwrap();
        assert_eq!(packet.words[0], u32::MAX);
        if let Some(too_large) = (u32::MAX as usize).checked_add(1) {
            assert!(packet.unsigned(1, too_large).is_err());
            assert_eq!(packet.words[1], 0);
        }
    }

    #[test]
    fn packet_writer_checks_rank_extent_and_preserves_float_bits() {
        let mut packet = PacketWriter::new();
        let dimensions = [3; MAX_RANK];
        assert!(packet.axes(8, MAX_RANK + 1, &dimensions).is_err());
        assert!(packet.dimensions(PACKET_WORDS - 1, &[1, 2]).is_err());
        assert!(packet.dimensions(usize::MAX, &[1]).is_err());
        assert_eq!(packet.written, 0);
        packet.axes(8, 3, &dimensions).unwrap();
        assert_eq!(&packet.words[8..12], &[3, 3, 3, 0]);
        assert_eq!(packet.written.count_ones(), 3);
        packet.float(3, -0.0).unwrap();
        assert_eq!(packet.words[3], (-0.0f32).to_bits());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(packet.float(4, value).is_err());
        }
        assert_eq!(packet.words[4], 0);
    }
}
