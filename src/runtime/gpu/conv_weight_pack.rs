//! Exact FP32 OIHW -> HWIO permutation, padding output channels to 32.
//! The shared weight owner caches this immutable permutation by tensor ID and
//! packing kind. Padding keeps every 32-channel matrix filter read valid.
use super::*;

pub(super) fn padded_output_channels(output_channels: usize) -> RuntimeResult<usize> {
    output_channels
        .checked_add(31)
        .map(|n| n / 32 * 32)
        .filter(|&n| output_channels > 0 && n > 0)
        .ok_or_else(|| resource("MPP HWIO output channel padding overflow"))
}

pub(super) fn pack_hwio(
    source: &[f32],
    output_channels: usize,
    input_channels_per_group: usize,
    kernel_height: usize,
    kernel_width: usize,
) -> RuntimeResult<Vec<f32>> {
    let taps = kernel_height.checked_mul(kernel_width);
    let count = taps
        .and_then(|n| n.checked_mul(input_channels_per_group))
        .and_then(|n| n.checked_mul(output_channels));
    let Some(count) = count.filter(|&n| n > 0 && n == source.len()) else {
        return Err(resource(
            "MPP HWIO weight permutation has inconsistent dimensions",
        ));
    };
    let _ = count;
    let taps = taps.unwrap();
    let output_stride = padded_output_channels(output_channels)?;
    let padded_count = taps
        .checked_mul(input_channels_per_group)
        .and_then(|n| n.checked_mul(output_stride))
        .ok_or_else(|| resource("MPP padded HWIO weight size overflow"))?;
    let mut packed = allocate_f32(padded_count)?;
    for tap in 0..taps {
        for input in 0..input_channels_per_group {
            for output in 0..output_channels {
                packed[(tap * input_channels_per_group + input) * output_stride + output] =
                    source[(output * input_channels_per_group + input) * taps + tap];
            }
        }
    }
    Ok(packed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_pack_obeys_hwio_index_contract_without_rounding() {
        let source = (0..12).map(|n| n as f32).collect::<Vec<_>>();
        let packed = pack_hwio(&source, 2, 2, 1, 3).unwrap();
        assert_eq!(packed.len(), 6 * 32);
        for (row, expected) in [[0., 6.], [3., 9.], [1., 7.], [4., 10.], [2., 8.], [5., 11.]]
            .into_iter()
            .enumerate()
        {
            assert_eq!(&packed[row * 32..row * 32 + 2], &expected);
            assert!(
                packed[row * 32 + 2..(row + 1) * 32]
                    .iter()
                    .all(|v| v.to_bits() == 0)
            );
        }
        let values = [-0.0f32, f32::from_bits(1), 0.1f32, -0.125f32];
        assert_eq!(
            pack_hwio(&values, 1, 1, 1, 4)
                .unwrap()
                .as_chunks::<32>()
                .0
                .iter()
                .map(|chunk| {
                    assert!(chunk[1..].iter().all(|v| v.to_bits() == 0));
                    chunk[0]
                })
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert!(pack_hwio(&source, 3, 2, 1, 3).is_err());
        assert!(pack_hwio(&source, usize::MAX, 2, 1, 3).is_err());
        assert_eq!(padded_output_channels(32).unwrap(), 32);
        assert_eq!(padded_output_channels(33).unwrap(), 64);
        assert_eq!(padded_output_channels(48).unwrap(), 64);
        assert!(padded_output_channels(0).is_err());
        assert!(padded_output_channels(usize::MAX).is_err());
    }
}
