//! Channel concatenation without leaving a channels-last graph segment.
use super::*;

pub(super) fn execute<'a>(
    inputs: usize,
    data: impl Fn(usize) -> &'a [f32] + Sync,
    pixels: usize,
    output: &mut [f32],
    cpu: &CpuContext,
    threads: usize,
) {
    if pixels == 0 {
        assert!(output.is_empty());
        return;
    }
    assert_eq!(output.len() % pixels, 0);
    let channels = output.len() / pixels;
    let workers = if output.len() >= 262_144 { threads } else { 1 };
    super::super::parallel::aligned_bands(
        output,
        channels,
        pixels,
        workers,
        1,
        cpu,
        |band, first| {
            let mut offset = 0;
            for slot in 0..inputs {
                let source = data(slot);
                assert_eq!(source.len() % pixels, 0);
                let count = source.len() / pixels;
                for (row, destination) in band.chunks_exact_mut(channels).enumerate() {
                    let start = (first + row) * count;
                    destination[offset..offset + count]
                        .copy_from_slice(&source[start..start + count]);
                }
                offset += count;
            }
            assert_eq!(offset, channels);
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_unequal_channels_batches_duplicate_sources_and_bits() {
        #[cfg(feature = "parallel")]
        let cpu = CpuContext::new(4).unwrap();
        #[cfg(not(feature = "parallel"))]
        let cpu = CpuContext::new(1).unwrap();
        for pixels in [1, 2 * 7 * 11, 2 * 97 * 53] {
            let bits = [0, 0x8000_0000, 0x7fc0_0001, 0x7f80_0000, 0xff80_0000];
            let sources: Vec<Vec<f32>> = [3, 17, 32]
                .into_iter()
                .map(|channels| {
                    (0..pixels * channels)
                        .map(|i| f32::from_bits(bits[i % bits.len()]))
                        .collect()
                })
                .collect();
            let slots = [0, 1, 0, 2];
            let mut actual = vec![f32::NAN; pixels * 55];
            execute(
                slots.len(),
                |i| &sources[slots[i]],
                pixels,
                &mut actual,
                &cpu,
                4,
            );
            let mut expected = Vec::with_capacity(actual.len());
            for pixel in 0..pixels {
                for slot in slots {
                    let channels = sources[slot].len() / pixels;
                    expected.extend_from_slice(
                        &sources[slot][pixel * channels..(pixel + 1) * channels],
                    );
                }
            }
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }
}
