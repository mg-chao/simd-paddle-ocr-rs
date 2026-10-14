//! Checked, contiguous tensor storage. No inference or layout conversion lives here.

use serde::{Deserialize, Serialize};
use std::{fmt, mem::size_of};

pub const MAX_RANK: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorError(String);

impl TensorError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}
impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for TensorError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    F32,
    I32,
    I64,
    U8,
}

impl DType {
    pub fn element_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::I64 => 8,
            Self::U8 => 1,
        }
    }
    pub fn from_onnx(value: i64) -> Result<Self, TensorError> {
        match value {
            1 => Ok(Self::F32),
            6 => Ok(Self::I32),
            7 => Ok(Self::I64),
            2 => Ok(Self::U8),
            _ => Err(TensorError::new(format!(
                "unsupported ONNX tensor dtype {value}"
            ))),
        }
    }
}

/// Serialized metadata, kept distinct from a bound runtime shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dim {
    Known(usize),
    Symbol(String),
    Unknown,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeSpec {
    UnknownRank,
    Ranked(Vec<Dim>),
}

/// Ranks 0 through 8 require no allocation. Scalar [] has one element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    dimensions: [usize; MAX_RANK],
    rank: u8,
    elements: usize,
}
impl Shape {
    pub fn new(dimensions: &[usize]) -> Result<Self, TensorError> {
        if dimensions.len() > MAX_RANK {
            return Err(TensorError::new(format!(
                "rank {} exceeds {MAX_RANK}",
                dimensions.len()
            )));
        }
        let elements = if dimensions.contains(&0) {
            0
        } else {
            dimensions
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| TensorError::new("shape element count overflow"))?
        };
        let mut inline = [0; MAX_RANK];
        inline[..dimensions.len()].copy_from_slice(dimensions);
        Ok(Self {
            dimensions: inline,
            rank: dimensions.len() as u8,
            elements,
        })
    }
    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions[..self.rank()]
    }
    pub fn rank(&self) -> usize {
        usize::from(self.rank)
    }
    pub fn element_count(&self) -> usize {
        self.elements
    }
    pub fn byte_count(&self, dtype: DType) -> Result<usize, TensorError> {
        self.bytes_for(dtype.element_bytes())
    }
    fn bytes_for(&self, element_bytes: usize) -> Result<usize, TensorError> {
        let bytes = self
            .elements
            .checked_mul(element_bytes)
            .ok_or_else(|| TensorError::new("shape byte count overflow"))?;
        if bytes > isize::MAX as usize {
            return Err(TensorError::new("tensor allocation exceeds isize::MAX"));
        }
        Ok(bytes)
    }
    pub fn broadcast(self, other: Self) -> Result<Self, TensorError> {
        let rank = self.rank().max(other.rank());
        let mut dimensions = [1; MAX_RANK];
        for offset in 0..rank {
            let a = if offset < self.rank() {
                self.dimensions[self.rank() - 1 - offset]
            } else {
                1
            };
            let b = if offset < other.rank() {
                other.dimensions[other.rank() - 1 - offset]
            } else {
                1
            };
            dimensions[rank - 1 - offset] = if a == b {
                a
            } else if a == 1 {
                b
            } else if b == 1 {
                a
            } else {
                return Err(TensorError::new(format!(
                    "cannot broadcast {:?} with {:?}",
                    self.dimensions(),
                    other.dimensions()
                )));
            };
        }
        Self::new(&dimensions[..rank])
    }
}

#[derive(Debug, Clone)]
pub struct Tensor<T> {
    shape: Shape,
    data: Vec<T>,
}
impl<T> Tensor<T> {
    pub fn from_vec(shape: Shape, data: Vec<T>) -> Result<Self, TensorError> {
        shape.bytes_for(size_of::<T>())?;
        if data.len() != shape.element_count() {
            return Err(TensorError::new(format!(
                "shape {:?} needs {} elements, got {}",
                shape.dimensions(),
                shape.element_count(),
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }
    pub fn shape(&self) -> Shape {
        self.shape
    }
    pub fn dimensions(&self) -> &[usize] {
        self.shape.dimensions()
    }
    pub fn data(&self) -> &[T] {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
    pub fn view(&self) -> TensorView<'_, T> {
        TensorView {
            shape: self.shape,
            data: &self.data,
        }
    }
    pub fn view_mut(&mut self) -> TensorViewMut<'_, T> {
        TensorViewMut {
            shape: self.shape,
            data: &mut self.data,
        }
    }
    pub fn reshape(&self, shape: Shape) -> Result<TensorView<'_, T>, TensorError> {
        self.view().reshape(shape)
    }
}
impl<T: Default + Clone> Tensor<T> {
    pub fn zeros(shape: Shape) -> Result<Self, TensorError> {
        shape.bytes_for(size_of::<T>())?;
        let mut data = Vec::new();
        data.try_reserve_exact(shape.element_count())
            .map_err(|error| TensorError::new(format!("tensor allocation failed: {error}")))?;
        data.resize(shape.element_count(), T::default());
        Ok(Self { shape, data })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a, T> {
    shape: Shape,
    data: &'a [T],
}
impl<'a, T> TensorView<'a, T> {
    /// Borrow checked contiguous storage without allocating or taking ownership.
    pub fn from_slice(shape: Shape, data: &'a [T]) -> Result<Self, TensorError> {
        shape.bytes_for(size_of::<T>())?;
        if shape.element_count() != data.len() {
            return Err(TensorError::new("view shape changes element count"));
        }
        Ok(Self { shape, data })
    }
    pub fn shape(&self) -> Shape {
        self.shape
    }
    pub fn data(&self) -> &'a [T] {
        self.data
    }
    pub fn reshape(self, shape: Shape) -> Result<Self, TensorError> {
        if shape.element_count() != self.data.len() {
            return Err(TensorError::new("reshape changes element count"));
        }
        Ok(Self {
            shape,
            data: self.data,
        })
    }
}
#[derive(Debug)]
pub struct TensorViewMut<'a, T> {
    shape: Shape,
    data: &'a mut [T],
}
impl<'a, T> TensorViewMut<'a, T> {
    pub fn shape(&self) -> Shape {
        self.shape
    }
    pub fn data(&self) -> &[T] {
        self.data
    }
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data
    }
    pub fn reshape(self, shape: Shape) -> Result<Self, TensorError> {
        if shape.element_count() != self.data.len() {
            return Err(TensorError::new("reshape changes element count"));
        }
        Ok(Self {
            shape,
            data: self.data,
        })
    }
}

#[derive(Debug, Clone)]
pub enum TensorData {
    F32(Tensor<f32>),
    I32(Tensor<i32>),
    I64(Tensor<i64>),
    U8(Tensor<u8>),
}
impl TensorData {
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::U8(_) => DType::U8,
        }
    }
    pub fn shape(&self) -> Shape {
        match self {
            Self::F32(t) => t.shape(),
            Self::I32(t) => t.shape(),
            Self::I64(t) => t.shape(),
            Self::U8(t) => t.shape(),
        }
    }
    pub fn dimensions(&self) -> &[usize] {
        match self {
            Self::F32(t) => t.dimensions(),
            Self::I32(t) => t.dimensions(),
            Self::I64(t) => t.dimensions(),
            Self::U8(t) => t.dimensions(),
        }
    }
    pub fn len(&self) -> usize {
        self.shape().element_count()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn as_f32(&self) -> Result<&[f32], TensorError> {
        match self {
            Self::F32(t) => Ok(t.data()),
            _ => Err(TensorError::new("expected f32 tensor")),
        }
    }
    pub fn as_i32(&self) -> Result<&[i32], TensorError> {
        match self {
            Self::I32(t) => Ok(t.data()),
            _ => Err(TensorError::new("expected i32 tensor")),
        }
    }
    pub fn as_i64(&self) -> Result<&[i64], TensorError> {
        match self {
            Self::I64(t) => Ok(t.data()),
            _ => Err(TensorError::new("expected i64 tensor")),
        }
    }
    pub fn as_u8(&self) -> Result<&[u8], TensorError> {
        match self {
            Self::U8(t) => Ok(t.data()),
            _ => Err(TensorError::new("expected u8 tensor")),
        }
    }
    pub fn from_raw(dtype: DType, shape: Shape, bytes: Vec<u8>) -> Result<Self, TensorError> {
        let expected = shape.byte_count(dtype)?;
        if bytes.len() != expected {
            return Err(TensorError::new(format!(
                "tensor payload has {} bytes; shape requires {expected}",
                bytes.len()
            )));
        }
        fn decode<T, const N: usize>(
            bytes: &[u8],
            convert: impl Fn([u8; N]) -> T,
        ) -> Result<Vec<T>, TensorError> {
            let mut values = Vec::new();
            values
                .try_reserve_exact(bytes.len() / N)
                .map_err(|error| TensorError::new(format!("tensor allocation failed: {error}")))?;
            for chunk in bytes.chunks_exact(N) {
                let mut fixed = [0; N];
                fixed.copy_from_slice(chunk);
                values.push(convert(fixed));
            }
            Ok(values)
        }
        Ok(match dtype {
            DType::F32 => Self::F32(Tensor::from_vec(
                shape,
                decode(&bytes, f32::from_le_bytes)?,
            )?),
            DType::I32 => Self::I32(Tensor::from_vec(
                shape,
                decode(&bytes, i32::from_le_bytes)?,
            )?),
            DType::I64 => Self::I64(Tensor::from_vec(
                shape,
                decode(&bytes, i64::from_le_bytes)?,
            )?),
            DType::U8 => Self::U8(Tensor::from_vec(shape, bytes)?),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scalar_empty_and_zero_before_overflow() {
        assert_eq!(Shape::new(&[]).unwrap().element_count(), 1);
        assert_eq!(Shape::new(&[0]).unwrap().element_count(), 0);
        assert_eq!(
            Shape::new(&[usize::MAX, usize::MAX, 0])
                .unwrap()
                .element_count(),
            0
        );
        assert!(Shape::new(&[usize::MAX, 2]).is_err());
        assert!(
            Shape::new(&[usize::MAX])
                .unwrap()
                .byte_count(DType::I64)
                .is_err()
        );
        assert!(Shape::new(&[1; 9]).is_err());
        assert_eq!(Shape::new(&[1; 5]).unwrap().rank(), 5);
    }
    #[test]
    fn broadcasting_is_checked_and_preserves_zero() {
        let pair =
            |a: &[usize], b: &[usize]| Shape::new(a).unwrap().broadcast(Shape::new(b).unwrap());
        assert_eq!(pair(&[0], &[1]).unwrap().dimensions(), &[0]);
        assert_eq!(pair(&[], &[2, 3]).unwrap().dimensions(), &[2, 3]);
        assert_eq!(pair(&[2, 3], &[3]).unwrap().dimensions(), &[2, 3]);
        assert_eq!(pair(&[2, 3], &[2, 1]).unwrap().dimensions(), &[2, 3]);
        assert_eq!(
            pair(&[2, 3, 4, 5], &[1, 3, 1, 1]).unwrap().dimensions(),
            &[2, 3, 4, 5]
        );
        assert!(pair(&[0], &[2]).is_err());
        assert!(pair(&[2, 3], &[2]).is_err());
    }
    #[test]
    fn owned_tensor_and_exclusive_view() {
        let shape = Shape::new(&[2, 3]).unwrap();
        assert!(Tensor::from_vec(shape, vec![0f32; 5]).is_err());
        let mut tensor = Tensor::<f32>::zeros(shape).unwrap();
        tensor
            .view_mut()
            .reshape(Shape::new(&[6]).unwrap())
            .unwrap()
            .data_mut()[4] = 7.;
        assert_eq!(
            tensor.reshape(Shape::new(&[3, 2]).unwrap()).unwrap().data()[4],
            7.
        );
        assert!(tensor.reshape(Shape::new(&[5]).unwrap()).is_err());
    }
    #[test]
    fn raw_decoding_preserves_bits_and_checks_types() {
        let bits = 0x7fc01234u32;
        let data = TensorData::from_raw(
            DType::F32,
            Shape::new(&[]).unwrap(),
            bits.to_le_bytes().to_vec(),
        )
        .unwrap();
        assert_eq!(data.as_f32().unwrap()[0].to_bits(), bits);
        assert!(data.as_i64().is_err());
        assert!(TensorData::from_raw(DType::I64, Shape::new(&[]).unwrap(), vec![0; 7]).is_err());
        assert!(
            TensorData::from_raw(DType::F32, Shape::new(&[0]).unwrap(), vec![])
                .unwrap()
                .is_empty()
        );
    }
}
