//! Immutable, architecture-independent F32 weight panels.
//!
//! A panel contains sixteen output columns. Four-, eight-, and sixteen-lane
//! kernels can consume the same allocation; tail columns are initialized to
//! zero. The normalized model remains the owner of the original coefficients.

use crate::model::TensorId;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

/// The same source tensor needs distinct panels when its storage order differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MatrixPacking {
    RowMajor,
    ConvOutputMajor,
}

/// Owned by an explicit compiler, shared across compilation policies.
/// Entries are populated only for formats selected by a compiled consumer.
#[derive(Debug, Default)]
pub(crate) struct WeightCache {
    matrices: Mutex<HashMap<(TensorId, MatrixPacking), Arc<PackedMatrix>>>,
    transposes: Mutex<HashMap<TensorId, Arc<[PackedMatrix; 4]>>>,
}
impl WeightCache {
    pub(crate) fn matrix<E>(
        &self,
        id: TensorId,
        kind: MatrixPacking,
        build: impl FnOnce() -> Result<PackedMatrix, E>,
    ) -> Result<Arc<PackedMatrix>, E> {
        let key = (id, kind);
        if let Some(panel) = self
            .matrices
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&key)
        {
            return Ok(panel.clone());
        }
        // Allocation and packing never run under the cache lock. If concurrent
        // compilers race, retain one complete immutable panel and discard the other.
        let panel = Arc::new(build()?);
        let mut cache = self
            .matrices
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Ok(cache.entry(key).or_insert(panel).clone())
    }
    pub(crate) fn transpose<E>(
        &self,
        id: TensorId,
        build: impl FnOnce() -> Result<[PackedMatrix; 4], E>,
    ) -> Result<Arc<[PackedMatrix; 4]>, E> {
        if let Some(panel) = self
            .transposes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&id)
        {
            return Ok(panel.clone());
        }
        let panel = Arc::new(build()?);
        let mut cache = self
            .transposes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Ok(cache.entry(id).or_insert(panel).clone())
    }
    pub(crate) fn bytes(&self) -> usize {
        let matrices = self
            .matrices
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .map(|panel| panel.bytes())
            .sum::<usize>();
        matrices
            + self
                .transposes
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .values()
                .map(|panels| panels.iter().map(PackedMatrix::bytes).sum::<usize>())
                .sum::<usize>()
    }
}

pub(crate) const PACKED_MATRIX_VERSION: u32 = 1;
pub(crate) const MATRIX_BLOCK_COLUMNS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PackedMatrixInfo {
    pub version: u32,
    pub inner: usize,
    pub columns: usize,
    pub padded_columns: usize,
    pub bytes: usize,
}

#[derive(Debug)]
pub(crate) struct PackedMatrix {
    info: PackedMatrixInfo,
    values: Box<[f32]>,
}

impl PackedMatrix {
    /// Pack row-major `[inner][columns]` as `[column_block][inner][16]`.
    pub(crate) fn new(weights: &[f32], inner: usize, columns: usize) -> Result<Self, &'static str> {
        if inner == 0 || columns == 0 {
            return Err("packed matrix dimensions must be positive");
        }
        let original = inner
            .checked_mul(columns)
            .ok_or("packed matrix size overflow")?;
        if weights.len() != original {
            return Err("packed matrix coefficient length differs from dimensions");
        }
        let blocks = columns.div_ceil(MATRIX_BLOCK_COLUMNS);
        let padded_columns = blocks
            .checked_mul(MATRIX_BLOCK_COLUMNS)
            .ok_or("packed matrix column padding overflow")?;
        let length = inner
            .checked_mul(padded_columns)
            .ok_or("packed matrix size overflow")?;
        let bytes = length
            .checked_mul(size_of::<f32>())
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or("packed matrix byte size overflow")?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(length)
            .map_err(|_| "packed matrix allocation failed")?;
        values.resize(length, 0.0);
        for block in 0..blocks {
            let first = block * MATRIX_BLOCK_COLUMNS;
            let count = (columns - first).min(MATRIX_BLOCK_COLUMNS);
            for k in 0..inner {
                let destination = (block * inner + k) * MATRIX_BLOCK_COLUMNS;
                values[destination..destination + count]
                    .copy_from_slice(&weights[k * columns + first..k * columns + first + count]);
            }
        }
        Ok(Self {
            info: PackedMatrixInfo {
                version: PACKED_MATRIX_VERSION,
                inner,
                columns,
                padded_columns,
                bytes,
            },
            values: values.into_boxed_slice(),
        })
    }

    pub(crate) fn data(&self) -> &[f32] {
        &self.values
    }

    pub(crate) fn byte_len(&self) -> usize {
        self.info.bytes
    }

    pub(crate) fn bytes(&self) -> usize {
        self.byte_len()
    }

    pub(crate) fn matches(&self, inner: usize, columns: usize) -> bool {
        self.info.version == PACKED_MATRIX_VERSION
            && self.info.inner == inner
            && self.info.columns == columns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_failures_are_not_published_and_formats_are_distinct() {
        let cache = WeightCache::default();
        let id = TensorId(2);
        let failure = cache.matrix::<&str>(id, MatrixPacking::RowMajor, || Err("allocation probe"));
        assert_eq!(failure.unwrap_err(), "allocation probe");
        assert_eq!(cache.bytes(), 0);
        let make = || PackedMatrix::new(&[1., 2., 3., 4.], 2, 2);
        let first = cache.matrix(id, MatrixPacking::RowMajor, make).unwrap();
        let same = cache
            .matrix::<&str>(id, MatrixPacking::RowMajor, || {
                panic!("cached factory must not run")
            })
            .unwrap();
        assert!(Arc::ptr_eq(&first, &same));
        let transposed = cache
            .matrix(id, MatrixPacking::ConvOutputMajor, make)
            .unwrap();
        assert!(!Arc::ptr_eq(&first, &transposed));
        assert_eq!(cache.bytes(), first.bytes() + transposed.bytes());
        assert!(
            cache
                .transpose::<&str>(id, || Err("allocation probe"))
                .is_err()
        );
        let taps = cache
            .transpose(id, || Ok::<_, &str>([make()?, make()?, make()?, make()?]))
            .unwrap();
        let same = cache
            .transpose::<&str>(id, || panic!("cached tap factory must not run"))
            .unwrap();
        assert!(Arc::ptr_eq(&taps, &same));
        assert_eq!(cache.bytes(), 6 * first.bytes());
    }

    #[test]
    fn block_panels_preserve_bits_and_zero_pad_column_tails() {
        for columns in [1, 3, 15, 16, 17, 31, 32, 33] {
            let inner = 7;
            let weights = (0..inner * columns)
                .map(|i| f32::from_bits(0x3f00_0000 + i as u32))
                .collect::<Vec<_>>();
            let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
            assert!(packed.matches(inner, columns));
            assert_eq!(packed.byte_len(), inner * columns.div_ceil(16) * 16 * 4);
            for block in 0..columns.div_ceil(16) {
                for k in 0..inner {
                    for lane in 0..16 {
                        let column = block * 16 + lane;
                        let actual = packed.data()[(block * inner + k) * 16 + lane];
                        let expected = if column < columns {
                            weights[k * columns + column]
                        } else {
                            0.0
                        };
                        assert_eq!(actual.to_bits(), expected.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn malformed_shapes_and_padding_overflow_fail_before_allocation() {
        assert!(PackedMatrix::new(&[], 0, 3).is_err());
        assert!(PackedMatrix::new(&[], 3, 0).is_err());
        assert!(PackedMatrix::new(&[1.0], 2, 2).is_err());
        assert!(PackedMatrix::new(&[], usize::MAX, 16).is_err());
    }
}
