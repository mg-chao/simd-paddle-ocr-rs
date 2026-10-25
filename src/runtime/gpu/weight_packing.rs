//! Shared immutable weight layouts. Every format retains its normal tracked
//! reservation; bindings and pending batches hold the exact resolved buffer.
use super::*;
use operators::{Recipe, RecipeKind, ShaderKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum WeightPacking {
    SpatialF16,
    PointwiseF16,
}

impl WeightPacking {
    fn for_recipe(recipe: &Recipe) -> Option<Self> {
        match recipe.kind {
            RecipeKind::Compute(
                ShaderKind::ConvMpp2Staged
                | ShaderKind::ConvMpp3s2Staged
                | ShaderKind::ConvMpp2s1
                | ShaderKind::ConvMpp3s1
                | ShaderKind::ConvMpp3s2
                | ShaderKind::ConvMpp3s21
                | ShaderKind::ConvMpp3s12
                | ShaderKind::ConvMpp7s1
                | ShaderKind::ConvMpp7s2,
            ) => Some(Self::SpatialF16),
            RecipeKind::Compute(ShaderKind::PointwiseF16) => Some(Self::PointwiseF16),
            _ => None,
        }
    }

    fn precision(self) -> GpuPrecision {
        match self {
            Self::SpatialF16 | Self::PointwiseF16 => GpuPrecision::MixedF16,
        }
    }

    fn padded_rows(self, rows: usize) -> RuntimeResult<usize> {
        if matches!(self, Self::SpatialF16) {
            return conv_weight_pack::padded_output_channels(rows);
        }
        let alignment = 64;
        rows.checked_add(alignment - 1)
            .map(|n| n / alignment * alignment)
            .filter(|&n| rows != 0 && n != 0)
            .ok_or_else(|| resource("packed pointwise output channel padding overflow"))
    }
}

impl GpuCompiledModel {
    /// Resolve binding 1 to the actual immutable operand representation.
    pub(super) fn specialized_weight(&self, recipe: &Recipe) -> RuntimeResult<Option<GpuBuffer>> {
        let Some(packing) = WeightPacking::for_recipe(recipe) else {
            return Ok(None);
        };
        let &id = recipe
            .inputs
            .get(1)
            .ok_or_else(|| RuntimeError::new("specialized GPU kernel has no weight operand"))?;
        {
            let values = self.original_weight(id)?;
            let shape = values.shape();
            let dimensions = shape.dimensions();
            if dimensions.len() != 4
                || recipe.words[36] as usize != packing.padded_rows(dimensions[0])?
            {
                return Err(RuntimeError::new(
                    "specialized GPU weight stride does not match its packing",
                ));
            }
        }
        self.packed_weight(id, packing).map(Some)
    }

    pub(super) fn packed_operand_precision(&self, recipe: &Recipe) -> Option<GpuPrecision> {
        WeightPacking::for_recipe(recipe).map(WeightPacking::precision)
    }

    fn original_weight(&self, id: TensorId) -> RuntimeResult<&crate::tensor::Tensor<f32>> {
        if self.precision != GpuPrecision::F32 && self.matrix_precision != GpuMatrixPrecision::F16 {
            return Err(RuntimeError::new(
                "immutable GPU weight packing requires the original FP32 model",
            ));
        }
        match self
            .program
            .model
            .graph()
            .tensors
            .get(id.0)
            .and_then(|tensor| tensor.constant.as_ref())
        {
            Some(TensorData::F32(values)) => Ok(values),
            _ => Err(RuntimeError::new(
                "packed GPU weights must be FP32 constants",
            )),
        }
    }

    fn packed_weight(&self, id: TensorId, packing: WeightPacking) -> RuntimeResult<GpuBuffer> {
        let mut cache = self
            .weights
            .packed
            .lock()
            .map_err(|_| RuntimeError::new("GPU immutable weight cache poisoned"))?;
        if let Some(buffer) = cache.get(&(id, packing)) {
            return Ok(buffer.clone());
        }
        let values = self.original_weight(id)?;
        let shape = values.shape();
        let packed = pack_values(
            values.data(),
            shape.dimensions(),
            packing,
            &self.context.limits,
        )?;
        let encoded = packing.precision().encode(&packed);
        let buffer = self.context.checked("GPU immutable weight packing", || {
            self.context.create_buffer_init(
                "GPU packed immutable weights",
                &encoded,
                BufferUsage::STORAGE,
            )
        })?;
        // Failed allocation/upload never publishes an entry. Holding the shared
        // cache lock also prevents duplicate reservations across model requests.
        cache.insert((id, packing), buffer.clone());
        Ok(buffer)
    }
}

fn pack_values(
    source: &[f32],
    dimensions: &[usize],
    packing: WeightPacking,
    limits: &crate::runtime::gpu::GpuLimits,
) -> RuntimeResult<Vec<f32>> {
    let [rows, inner, kh, kw] = dimensions else {
        return Err(RuntimeError::new(
            "packed GPU weights must have OIHW rank four",
        ));
    };
    if dimensions.contains(&0)
        || dimensions
            .iter()
            .copied()
            .try_fold(1usize, usize::checked_mul)
            != Some(source.len())
        || (matches!(packing, WeightPacking::PointwiseF16) && (*kh != 1 || *kw != 1))
    {
        return Err(RuntimeError::new("invalid packed GPU weight geometry"));
    }
    let padded = packing.padded_rows(*rows)?;
    let elements = padded
        .checked_mul(*inner)
        .and_then(|n| n.checked_mul(*kh))
        .and_then(|n| n.checked_mul(*kw))
        .ok_or_else(|| resource("packed GPU weight size overflow"))?;
    // Use the conservative FP32 storage limit before allocating the host
    // permutation, including formats whose final device representation is half.
    checked_storage_bytes(elements, limits)?;
    if matches!(packing, WeightPacking::SpatialF16) {
        return conv_weight_pack::pack_hwio(source, *rows, *inner, *kh, *kw);
    }
    let mut packed = allocate_f32(elements)?;
    for k in 0..*inner {
        for row in 0..*rows {
            packed[k * padded + row] = source[row * inner + k];
        }
    }
    Ok(packed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointwise_formats_preserve_k_major_values_and_zero_output_tails() {
        let dimensions = [5, 3, 1, 1];
        let source: Vec<_> = (0..15).map(|i| (i as f32 - 7.) / 13.).collect();
        for (packing, padded) in [(WeightPacking::PointwiseF16, 64)] {
            let packed = pack_values(
                &source,
                &dimensions,
                packing,
                &crate::runtime::gpu::GpuLimits::default(),
            )
            .unwrap();
            assert_eq!(packed.len(), 3 * padded);
            for k in 0..3 {
                for row in 0..padded {
                    let expected = if row < 5 { source[row * 3 + k] } else { 0. };
                    assert_eq!(packed[k * padded + row].to_bits(), expected.to_bits());
                }
            }
            let precision = packing.precision();
            let encoded = precision.encode(&packed);
            let mut decoded = vec![0.; packed.len()];
            precision.decode(&encoded, &mut decoded).unwrap();
            for (&actual, &original) in decoded.iter().zip(&packed) {
                let expected = if precision == GpuPrecision::F32 {
                    original
                } else {
                    half::f16::from_f32(original).to_f32()
                };
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn spatial_formats_share_exact_hwio_permutation_before_storage_encoding() {
        let dimensions = [3, 2, 2, 3];
        let source: Vec<_> = (0..36).map(|i| (i as f32 - 17.) / 19.).collect();
        for packing in [WeightPacking::SpatialF16] {
            let packed = pack_values(
                &source,
                &dimensions,
                packing,
                &crate::runtime::gpu::GpuLimits::default(),
            )
            .unwrap();
            assert_eq!(packed.len(), 2 * 3 * 2 * 32);
            for y in 0..2 {
                for x in 0..3 {
                    for input in 0..2 {
                        for output in 0..32 {
                            let expected = if output < 3 {
                                source[((output * 2 + input) * 2 + y) * 3 + x]
                            } else {
                                0.
                            };
                            let actual = packed[((y * 3 + x) * 2 + input) * 32 + output];
                            assert_eq!(actual.to_bits(), expected.to_bits());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn packing_rejects_inconsistent_geometry_and_binding_overflow_before_allocation() {
        let limits = crate::runtime::gpu::GpuLimits::default();
        assert!(pack_values(&[1.; 6], &[2, 3, 1], WeightPacking::PointwiseF16, &limits).is_err());
        assert!(
            pack_values(
                &[1.; 6],
                &[2, 3, 1, 2],
                WeightPacking::PointwiseF16,
                &limits
            )
            .is_err()
        );
        assert!(pack_values(&[1.; 6], &[2, 3, 1, 2], WeightPacking::SpatialF16, &limits).is_err());
        let small = crate::runtime::gpu::GpuLimits {
            max_buffer_size: 128,
            max_storage_buffer_binding_size: 128,
            ..limits
        };
        assert!(
            pack_values(
                &[1.; 33],
                &[33, 1, 1, 1],
                WeightPacking::PointwiseF16,
                &small
            )
            .is_err()
        );
        for packing in [WeightPacking::PointwiseF16, WeightPacking::SpatialF16] {
            assert!(packing.padded_rows(0).is_err());
            assert!(packing.padded_rows(usize::MAX).is_err());
        }
    }
}
