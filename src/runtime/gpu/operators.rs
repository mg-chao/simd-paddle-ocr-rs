use super::*;
use crate::{
    runtime::kernels::{BatchedMatMul, BinaryOp, Broadcast, Conv},
    tensor::MAX_RANK,
};

const PACKET_WORDS: usize = 128;
const EMPTY_PACKET: [u32; PACKET_WORDS] = [0; PACKET_WORDS];

#[derive(Clone)]
pub(super) enum ShaderKind {
    Convolution,
    General,
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
