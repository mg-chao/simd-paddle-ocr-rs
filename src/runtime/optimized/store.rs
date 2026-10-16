//! Bind common ordered convolution epilogues once, before entering the kernel.
//! Operand slices borrow the current disjoint arena views and never outlive a
//! native call. Longer/unusual programs keep the original generic interpreter.
use super::{Activation, Epilogue, EpilogueStore, Layout, activation_op, simd};
use simd::StoreTransform;
use std::simd::Simd;

pub(super) struct PreparedStore<'data, 'program, F> {
    common: Option<Recipe<'data>>,
    fallback: EpilogueStore<'program, F>,
}

impl<'data, 'program, F> PreparedStore<'data, 'program, F>
where
    F: Fn(usize) -> &'data [f32] + Sync,
{
    pub(super) fn new(fallback: EpilogueStore<'program, F>) -> Self {
        let common = Recipe::bind(&fallback);
        Self { common, fallback }
    }
}

impl<'data, F> StoreTransform for PreparedStore<'data, '_, F>
where
    F: Fn(usize) -> &'data [f32] + Sync,
{
    #[inline(always)]
    fn scalar(&self, value: f32, index: usize) -> f32 {
        if let Some(recipe) = &self.common {
            recipe.apply(
                value,
                |operand, value| value + operand.scalar(index),
                |activation, value| {
                    simd::unary_value_with_math(
                        activation_op(&activation),
                        value,
                        self.fallback.math,
                    )
                },
            )
        } else {
            self.fallback.scalar(value, index)
        }
    }

    #[inline(always)]
    fn vector<const LANES: usize>(
        &self,
        value: Simd<f32, LANES>,
        index: usize,
    ) -> Simd<f32, LANES> {
        if let Some(recipe) = &self.common {
            recipe.vector(value, index, None, self.fallback.math)
        } else {
            self.fallback.vector(value, index)
        }
    }

    #[inline(always)]
    fn vector_channel<const LANES: usize>(
        &self,
        value: Simd<f32, LANES>,
        index: usize,
        channel: usize,
    ) -> Simd<f32, LANES> {
        if let Some(recipe) = &self.common {
            recipe.vector(value, index, Some(channel), self.fallback.math)
        } else {
            self.fallback.vector(value, index)
        }
    }
}

#[derive(Clone, Copy)]
enum Operand<'data> {
    Scalar(f32),
    NhwcChannel(&'data [f32]),
    CanonicalChannel { values: &'data [f32], plane: usize },
    Full(&'data [f32]),
}

impl<'data> Operand<'data> {
    fn bind(values: &'data [f32], channels: usize, plane: usize, layout: Layout) -> Self {
        if values.len() == 1 {
            Self::Scalar(values[0])
        } else if values.len() == channels {
            if layout == Layout::Nhwc {
                Self::NhwcChannel(values)
            } else {
                Self::CanonicalChannel { values, plane }
            }
        } else {
            Self::Full(values)
        }
    }

    #[inline(always)]
    fn scalar(self, index: usize) -> f32 {
        match self {
            Self::Scalar(value) => value,
            Self::NhwcChannel(values) => values[index % values.len()],
            Self::CanonicalChannel { values, plane } => values[index / plane % values.len()],
            Self::Full(values) => values[index],
        }
    }

    #[inline(always)]
    fn vector_at<const LANES: usize>(
        self,
        index: usize,
        channel: Option<usize>,
    ) -> Simd<f32, LANES> {
        if let (Self::NhwcChannel(values), Some(channel)) = (self, channel) {
            load(values, channel)
        } else {
            self.vector(index)
        }
    }

    #[inline(always)]
    fn vector<const LANES: usize>(self, index: usize) -> Simd<f32, LANES> {
        match self {
            Self::Scalar(value) => Simd::splat(value),
            Self::NhwcChannel(values) => load(values, index % values.len()),
            Self::CanonicalChannel { values, plane } => {
                let first = index / plane;
                if first == (index + LANES - 1) / plane {
                    Simd::splat(values[first % values.len()])
                } else {
                    // A spatial vector may cross channels or batches when the
                    // plane is short; preserve each lane's original indexing.
                    Simd::from_array(std::array::from_fn(|lane| {
                        values[(index + lane) / plane % values.len()]
                    }))
                }
            }
            Self::Full(values) => load(values, index),
        }
    }
}

#[inline(always)]
fn load<const LANES: usize>(values: &[f32], index: usize) -> Simd<f32, LANES> {
    let count = (values.len() - index).min(LANES);
    if count == LANES {
        Simd::from_slice(&values[index..index + LANES])
    } else {
        Simd::load_or_default(&values[index..])
    }
}

/// Keep the arithmetic order explicit. In particular, DET has activation
/// followed by an Add; grouping all Adds ahead of activation would be wrong.
enum Recipe<'data> {
    Identity,
    Activation(Activation),
    Add(Operand<'data>),
    AddActivation(Operand<'data>, Activation),
    ActivationAdd(Activation, Operand<'data>),
    AddAdd(Operand<'data>, Operand<'data>),
    AddAddActivation(Operand<'data>, Operand<'data>, Activation),
    AddActivationAdd(Operand<'data>, Activation, Operand<'data>),
    ActivationAddAdd(Activation, Operand<'data>, Operand<'data>),
}

impl<'data> Recipe<'data> {
    // Call the SIMD operations directly: a closure passed through the generic
    // scalar recipe evaluator can become an outlined baseline-ISA function,
    // losing the surrounding kernel's AVX feature context at every store.
    // Share one activation call site. Duplicating the complete unary program
    // in six recipe arms makes unoptimized native family dispatch reserve
    // multi-megabyte frames even when the selected recipe is very small.
    // Release inlining preserves the calling kernel's native ISA context.
    // Debug keeps a boundary so nested family dispatch does not reserve all
    // unary temporaries in its own unoptimized stack frames.
    #[cfg_attr(debug_assertions, inline(never))]
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn vector<const LANES: usize>(
        &self,
        value: Simd<f32, LANES>,
        index: usize,
        channel: Option<usize>,
        math: simd::UnaryMath,
    ) -> Simd<f32, LANES> {
        #[inline(always)]
        fn add<const LANES: usize>(
            operand: Operand<'_>,
            value: Simd<f32, LANES>,
            index: usize,
            channel: Option<usize>,
        ) -> Simd<f32, LANES> {
            value + operand.vector_at(index, channel)
        }
        let (value, activation, first, second) = match *self {
            Self::Identity => return value,
            Self::Add(a) => return add(a, value, index, channel),
            Self::AddAdd(a, b) => return add(b, add(a, value, index, channel), index, channel),
            Self::Activation(act) => (value, act, None, None),
            Self::AddActivation(a, act) => (add(a, value, index, channel), act, None, None),
            Self::ActivationAdd(act, a) => (value, act, Some(a), None),
            Self::AddAddActivation(a, b, act) => (
                add(b, add(a, value, index, channel), index, channel),
                act,
                None,
                None,
            ),
            Self::AddActivationAdd(a, act, b) => {
                (add(a, value, index, channel), act, Some(b), None)
            }
            Self::ActivationAddAdd(act, a, b) => (value, act, Some(a), Some(b)),
        };
        let value = simd::unary_vector_with_math(activation_op(&activation), value, math);
        let value = if let Some(a) = first {
            add(a, value, index, channel)
        } else {
            value
        };
        if let Some(b) = second {
            add(b, value, index, channel)
        } else {
            value
        }
    }

    fn bind<F>(store: &EpilogueStore<'_, F>) -> Option<Self>
    where
        F: Fn(usize) -> &'data [f32] + Sync,
    {
        let operand = |slot: usize| {
            Operand::bind(
                (store.data)(store.inputs[slot]),
                store.channels,
                store.plane,
                store.layout,
            )
        };
        Some(match store.epilogues {
            [] => Self::Identity,
            [Epilogue::Activation(act)] => Self::Activation(*act),
            [Epilogue::Add(a)] => Self::Add(operand(*a)),
            [Epilogue::Add(a), Epilogue::Activation(act)] => Self::AddActivation(operand(*a), *act),
            [Epilogue::Activation(act), Epilogue::Add(a)] => Self::ActivationAdd(*act, operand(*a)),
            [Epilogue::Add(a), Epilogue::Add(b)] => Self::AddAdd(operand(*a), operand(*b)),
            [
                Epilogue::Add(a),
                Epilogue::Add(b),
                Epilogue::Activation(act),
            ] => Self::AddAddActivation(operand(*a), operand(*b), *act),
            [
                Epilogue::Add(a),
                Epilogue::Activation(act),
                Epilogue::Add(b),
            ] => Self::AddActivationAdd(operand(*a), *act, operand(*b)),
            [
                Epilogue::Activation(act),
                Epilogue::Add(a),
                Epilogue::Add(b),
            ] => Self::ActivationAddAdd(*act, operand(*a), operand(*b)),
            _ => return None,
        })
    }

    #[inline(always)]
    fn apply<T: Copy>(
        &self,
        value: T,
        add: impl Fn(Operand<'data>, T) -> T,
        activate: impl Fn(Activation, T) -> T,
    ) -> T {
        match *self {
            Self::Identity => value,
            Self::Activation(act) => activate(act, value),
            Self::Add(a) => add(a, value),
            Self::AddActivation(a, act) => activate(act, add(a, value)),
            Self::ActivationAdd(act, a) => add(a, activate(act, value)),
            Self::AddAdd(a, b) => add(b, add(a, value)),
            Self::AddAddActivation(a, b, act) => activate(act, add(b, add(a, value))),
            Self::AddActivationAdd(a, act, b) => add(b, activate(act, add(a, value))),
            Self::ActivationAddAdd(act, a, b) => add(b, add(a, activate(act, value))),
        }
    }
}

#[cfg(test)]
mod tests;
