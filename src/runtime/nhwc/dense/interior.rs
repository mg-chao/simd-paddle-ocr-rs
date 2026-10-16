//! Prevalidated receptive fields for the native six-row dense tile.
//!
//! Interior tiles bind six input spans and tap offsets. Border tiles bind
//! checked channel-zero indices with explicit padding sentinels. K advances
//! only a channel/tap cursor; private descriptors prove every scalar load.
use super::*;

const MAX_TAPS: usize = 25;
static PADDING_ZERO: f32 = 0.0;

pub(super) struct Taps {
    offsets: [usize; MAX_TAPS],
    count: usize,
    channels: usize,
    span: usize,
    y_extent: usize,
    x_extent: usize,
}

#[derive(Clone, Copy)]
pub(super) struct Inputs<'a>([&'a [f32]; ROWS]);

/// Each six-row u32 table occupies 600 bytes; eight grouped tiles use 4,800
/// table bytes, plus descriptors and enum tags. Larger borrowed inputs retain
/// the general gather instead of truncating indices.
pub(super) struct BorderInputs<'a> {
    input: &'a [f32],
    bases: [[u32; MAX_TAPS]; ROWS],
}

// Both variants stay on the stack: boxing would allocate per six-row tile.
#[allow(clippy::large_enum_variant)]
pub(super) enum TileInputs<'a> {
    Interior(Inputs<'a>),
    Border(BorderInputs<'a>),
}

fn encode_base(base: usize) -> Option<u32> {
    u32::try_from(base).ok().filter(|&base| base != u32::MAX)
}

trait Reader {
    /// ROW < ROWS, tap < taps.count and channel < taps.channels. The private
    /// descriptor must have been constructed from these same Taps/Conv inputs.
    unsafe fn read<const ROW: usize>(&self, taps: &Taps, tap: usize, channel: usize) -> f32;
}

impl Reader for Inputs<'_> {
    #[inline(always)]
    unsafe fn read<const ROW: usize>(&self, taps: &Taps, tap: usize, channel: usize) -> f32 {
        // SAFETY: inputs() binds each span through max tap offset + channels.
        // The tile checks its K interval and maintains the tap/channel cursor.
        let offset = unsafe { *taps.offsets.get_unchecked(tap) } + channel;
        unsafe { *self.0.get_unchecked(ROW).get_unchecked(offset) }
    }
}

impl Reader for BorderInputs<'_> {
    #[inline(always)]
    unsafe fn read<const ROW: usize>(&self, _: &Taps, tap: usize, channel: usize) -> f32 {
        // SAFETY: border_inputs() fills every active tap and checks the entire
        // channel span for each non-sentinel base. Select a valid scalar address
        // before loading: this also lets native kernels broadcast directly from
        // memory rather than keeping six guarded scalar values live at once.
        let base = unsafe { *self.bases.get_unchecked(ROW).get_unchecked(tap) };
        let pointer = if base == u32::MAX {
            std::ptr::addr_of!(PADDING_ZERO)
        } else {
            self.input.as_ptr().wrapping_add(base as usize + channel)
        };
        // Only the selected address is dereferenced. It is either the static
        // positive zero or a coefficient inside the descriptor's checked span.
        unsafe { *pointer }
    }
}

impl Taps {
    pub(super) fn new(p: &Conv) -> Option<Self> {
        let count = p.kh.checked_mul(p.kw)?;
        if p.cin == 0 || count == 0 || count > MAX_TAPS {
            return None;
        }
        p.cin.checked_mul(count)?;
        let mut offsets = [0; MAX_TAPS];
        for ky in 0..p.kh {
            for kx in 0..p.kw {
                offsets[ky * p.kw + kx] = ky
                    .checked_mul(p.dh)?
                    .checked_mul(p.width)?
                    .checked_add(kx.checked_mul(p.dw)?)?
                    .checked_mul(p.cin)?;
            }
        }
        Some(Self {
            span: offsets[count - 1].checked_add(p.cin)?,
            offsets,
            count,
            channels: p.cin,
            y_extent: (p.kh - 1).checked_mul(p.dh)?,
            x_extent: (p.kw - 1).checked_mul(p.dw)?,
        })
    }

    pub(super) fn inputs<'a>(
        &self,
        p: &Conv,
        input: &'a [f32],
        origins: &[(usize, usize, usize)],
    ) -> Option<TileInputs<'a>> {
        self.interior_inputs(p, input, origins)
            .map(TileInputs::Interior)
            .or_else(|| {
                self.border_inputs(p, input, origins)
                    .map(TileInputs::Border)
            })
    }

    fn interior_inputs<'a>(
        &self,
        p: &Conv,
        input: &'a [f32],
        origins: &[(usize, usize, usize)],
    ) -> Option<Inputs<'a>> {
        if origins.len() != ROWS {
            return None;
        }
        let mut rows = [&[][..]; ROWS];
        for (dst, &(n, y, x)) in rows.iter_mut().zip(origins) {
            if n >= p.batch || y >= p.oh || x >= p.ow {
                return None;
            }
            let sy = y.checked_mul(p.sh)?.checked_sub(p.pt)?;
            let sx = x.checked_mul(p.sw)?.checked_sub(p.pl)?;
            if sy.checked_add(self.y_extent)? >= p.height
                || sx.checked_add(self.x_extent)? >= p.width
            {
                return None;
            }
            let start = n
                .checked_mul(p.height)?
                .checked_add(sy)?
                .checked_mul(p.width)?
                .checked_add(sx)?
                .checked_mul(p.cin)?;
            *dst = input.get(start..start.checked_add(self.span)?)?;
        }
        Some(Inputs(rows))
    }

    fn border_inputs<'a>(
        &self,
        p: &Conv,
        input: &'a [f32],
        origins: &[(usize, usize, usize)],
    ) -> Option<BorderInputs<'a>> {
        let expected = p
            .batch
            .checked_mul(p.height)?
            .checked_mul(p.width)?
            .checked_mul(p.cin)?;
        if origins.len() != ROWS
            || input.len() >= u32::MAX as usize
            || input.len() != expected
            || p.cin != self.channels
            || p.kh.checked_mul(p.kw)? != self.count
        {
            return None;
        }
        let mut bases = [[u32::MAX; MAX_TAPS]; ROWS];
        for (row, &(n, y, x)) in bases.iter_mut().zip(origins) {
            if n >= p.batch || y >= p.oh || x >= p.ow {
                return None;
            }
            for ky in 0..p.kh {
                for kx in 0..p.kw {
                    let sy = y.checked_mul(p.sh)?.checked_add(ky.checked_mul(p.dh)?)?;
                    let sx = x.checked_mul(p.sw)?.checked_add(kx.checked_mul(p.dw)?)?;
                    let Some(sy) = sy.checked_sub(p.pt).filter(|&sy| sy < p.height) else {
                        continue;
                    };
                    let Some(sx) = sx.checked_sub(p.pl).filter(|&sx| sx < p.width) else {
                        continue;
                    };
                    let base = n
                        .checked_mul(p.height)?
                        .checked_add(sy)?
                        .checked_mul(p.width)?
                        .checked_add(sx)?
                        .checked_mul(p.cin)?;
                    input.get(base..base.checked_add(self.channels)?)?;
                    row[ky * p.kw + kx] = encode_base(base)?;
                }
            }
        }
        Some(BorderInputs { input, bases })
    }
}

#[inline(always)]
fn multiply_add<const LANES: usize, const FUSED: bool>(
    value: Simd<f32, LANES>,
    weights: Simd<f32, LANES>,
    sum: Simd<f32, LANES>,
) -> Simd<f32, LANES> {
    if FUSED {
        value.mul_add(weights, sum)
    } else {
        value * weights + sum
    }
}

/// Called only from the guarded eight-lane AVX dense entry points.
#[inline(always)]
pub(super) unsafe fn accumulate<const LANES: usize, const VECTORS: usize, const FUSED: bool>(
    inputs: &TileInputs<'_>,
    taps: &Taps,
    panel: &[f32],
    k0: usize,
    seeds: [[Simd<f32, LANES>; VECTORS]; ROWS],
) -> [[Simd<f32, LANES>; VECTORS]; ROWS] {
    // SAFETY: the caller has entered AVX, and FUSED selects an FMA-guarded
    // entry. The private descriptors establish the scalar-load bounds below.
    unsafe {
        match inputs {
            TileInputs::Interior(inputs) => {
                if FUSED {
                    tile_fma(inputs, taps, panel, k0, seeds)
                } else {
                    tile_avx(inputs, taps, panel, k0, seeds)
                }
            }
            TileInputs::Border(inputs) => {
                if FUSED {
                    tile_fma(inputs, taps, panel, k0, seeds)
                } else {
                    tile_avx(inputs, taps, panel, k0, seeds)
                }
            }
        }
    }
}

struct PortableBound<'a, R> {
    inputs: &'a R,
    taps: &'a Taps,
}

impl<R: Reader> super::super::pointwise::PortableReader for PortableBound<'_, R> {
    #[inline(always)]
    unsafe fn read<const ROW: usize>(&self, tap: usize, channel: usize) -> f32 {
        // SAFETY: construction uses the same validated descriptor as the
        // native reader. The paired tile proves its tap/channel K interval.
        unsafe { self.inputs.read::<ROW>(self.taps, tap, channel) }
    }
}

struct PortableGeneral<'a> {
    p: &'a Conv,
    input: &'a [f32],
    origins: &'a [(usize, usize, usize)],
}

impl super::super::pointwise::PortableReader for PortableGeneral<'_> {
    #[inline(always)]
    unsafe fn read<const ROW: usize>(&self, tap: usize, channel: usize) -> f32 {
        let p = self.p;
        let (n, y, x) = self.origins[ROW];
        let (ky, kx) = (tap / p.kw, tap % p.kw);
        let sy = y * p.sh + ky * p.dh;
        let sx = x * p.sw + kx * p.dw;
        let pointer = if sy < p.pt || sy - p.pt >= p.height || sx < p.pl || sx - p.pl >= p.width {
            std::ptr::addr_of!(PADDING_ZERO)
        } else {
            let index = ((n * p.height + sy - p.pt) * p.width + sx - p.pl) * p.cin + channel;
            &self.input[index] as *const f32
        };
        // SAFETY: select a valid scalar address before loading. The source
        // branch is bounds checked; the padding branch names static +0.0.
        unsafe { *pointer }
    }
}

/// Portable4 shares the same six-row, two-vector tile for validated interior
/// spans, checked border tables, and the general geometry fallback.
/// SAFETY: optional descriptors must bind this exact Conv/input/origin group.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn accumulate_portable<const LANES: usize, const OFFSET: usize>(
    p: &Conv,
    input: &[f32],
    origins: &[(usize, usize, usize)],
    taps: Option<&Taps>,
    inputs: Option<&TileInputs<'_>>,
    panel: &[f32],
    k0: usize,
    seeds: [[Simd<f32, LANES>; 2]; ROWS],
) -> [[Simd<f32, LANES>; 2]; ROWS] {
    use super::super::pointwise::portable_tile;
    assert_eq!(origins.len(), ROWS);
    // SAFETY: all three readers cover the same checked K interval, and their
    // scalar-load contracts are maintained by the private paired microkernel.
    unsafe {
        match (taps, inputs) {
            (Some(taps), Some(TileInputs::Interior(inputs))) => portable_tile::<LANES, OFFSET>(
                &PortableBound { inputs, taps },
                taps.count,
                taps.channels,
                panel,
                k0,
                seeds,
            ),
            (Some(taps), Some(TileInputs::Border(inputs))) => portable_tile::<LANES, OFFSET>(
                &PortableBound { inputs, taps },
                taps.count,
                taps.channels,
                panel,
                k0,
                seeds,
            ),
            _ => portable_tile::<LANES, OFFSET>(
                &PortableGeneral { p, input, origins },
                p.kh * p.kw,
                p.cin,
                panel,
                k0,
                seeds,
            ),
        }
    }
}

macro_rules! tile {
    ($name:ident, $features:literal, $fused:literal) => {
        #[target_feature(enable = $features)]
        #[inline(never)]
        unsafe fn $name<const LANES: usize, const VECTORS: usize, R: Reader>(
            inputs: &R,
            taps: &Taps,
            panel: &[f32],
            k0: usize,
            seeds: [[Simd<f32, LANES>; VECTORS]; ROWS],
        ) -> [[Simd<f32, LANES>; VECTORS]; ROWS] {
            assert_eq!(LANES, 8);
            assert_eq!(VECTORS, 2);
            assert_eq!(panel.len() % 16, 0);
            assert!(
                k0.checked_add(panel.len() / 16)
                    .is_some_and(|end| end <= taps.channels * taps.count)
            );
            let [s0, s1, s2, s3, s4, s5] = seeds;
            let (mut a0, mut b0) = (s0[0], s0[1]);
            let (mut a1, mut b1) = (s1[0], s1[1]);
            let (mut a2, mut b2) = (s2[0], s2[1]);
            let (mut a3, mut b3) = (s3[0], s3[1]);
            let (mut a4, mut b4) = (s4[0], s4[1]);
            let (mut a5, mut b5) = (s5[0], s5[1]);
            let (mut channel, mut tap) = (k0 / taps.count, k0 % taps.count);
            for weights in panel.as_chunks::<16>().0 {
                let w0 = Simd::from_slice(&weights[..LANES]);
                let w1 = Simd::from_slice(&weights[LANES..]);
                macro_rules! update {
                    ($row:literal, $low:ident, $high:ident) => {
                        // SAFETY: the checked K interval keeps channel/tap in
                        // range. Each reader prevalidates its scalar spans.
                        let value = Simd::splat(unsafe { inputs.read::<$row>(taps, tap, channel) });
                        $low = multiply_add::<LANES, $fused>(value, w0, $low);
                        $high = multiply_add::<LANES, $fused>(value, w1, $high);
                    };
                }
                update!(0, a0, b0);
                update!(1, a1, b1);
                update!(2, a2, b2);
                update!(3, a3, b3);
                update!(4, a4, b4);
                update!(5, a5, b5);
                tap += 1;
                if tap == taps.count {
                    tap = 0;
                    channel += 1;
                }
            }
            let low = [a0, a1, a2, a3, a4, a5];
            let high = [b0, b1, b2, b3, b4, b5];
            std::array::from_fn(|row| {
                std::array::from_fn(|v| if v == 0 { low[row] } else { high[row] })
            })
        }
    };
}
tile!(tile_avx, "avx", false);
tile!(tile_fma, "avx,fma", true);

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> Conv {
        Conv {
            batch: 1,
            cin: 3,
            cout: 17,
            height: 3,
            width: 8,
            oh: 3,
            ow: 8,
            cpg: 3,
            opg: 17,
            kh: 3,
            kw: 3,
            sh: 1,
            sw: 1,
            dh: 1,
            dw: 1,
            pt: 1,
            pl: 1,
        }
    }

    #[test]
    fn border_reader_validates_guarded_spans_and_preserves_scalar_bits() {
        let p = plan();
        let length = p.height * p.width * p.cin;
        let mut guarded = vec![f32::from_bits(0x7fc0_1234); length + 2];
        for (i, value) in guarded[1..=length].iter_mut().enumerate() {
            *value = match i % 4 {
                0 => -0.0,
                1 => 0.0,
                2 => f32::from_bits(0x7fc0_4321),
                _ => i as f32,
            };
        }
        let input = &guarded[1..=length];
        let taps = Taps::new(&p).unwrap();
        for y in 0..p.oh {
            let origins = std::array::from_fn::<_, ROWS, _>(|r| (0, y, r));
            assert!(taps.interior_inputs(&p, input, &origins).is_none());
            let border = taps.border_inputs(&p, input, &origins).unwrap();
            for tap in 0..taps.count {
                for channel in [0, p.cin - 1] {
                    let expected = std::array::from_fn::<_, ROWS, _>(|row| {
                        let (_, y, x) = origins[row];
                        let sy = y + tap / p.kw;
                        let sx = x + tap % p.kw;
                        if sy < p.pt || sy - p.pt >= p.height || sx < p.pl || sx - p.pl >= p.width {
                            0.0
                        } else {
                            input[((sy - p.pt) * p.width + sx - p.pl) * p.cin + channel]
                        }
                    });
                    // SAFETY: each explicit row is below ROWS and the loops
                    // use only validated active taps and channel endpoints.
                    let actual = unsafe {
                        [
                            border.read::<0>(&taps, tap, channel),
                            border.read::<1>(&taps, tap, channel),
                            border.read::<2>(&taps, tap, channel),
                            border.read::<3>(&taps, tap, channel),
                            border.read::<4>(&taps, tap, channel),
                            border.read::<5>(&taps, tap, channel),
                        ]
                    };
                    assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
                }
            }
        }
        let origins = [(0, 0, 0); ROWS];
        assert!(
            taps.border_inputs(&p, &input[..length - 1], &origins)
                .is_none()
        );
        assert!(taps.border_inputs(&p, input, &[(1, 0, 0); ROWS]).is_none());
        assert!(
            taps.border_inputs(&p, input, &[(0, p.oh, 0); ROWS])
                .is_none()
        );
        assert!(
            taps.border_inputs(&p, input, &[(0, 0, p.ow); ROWS])
                .is_none()
        );
    }

    #[test]
    fn border_reader_all_padding_and_index_fallback_are_explicit() {
        let mut p = plan();
        p.pt = 10;
        p.pl = 10;
        p.oh = 21;
        p.ow = 26;
        let input = vec![f32::NAN; p.height * p.width * p.cin];
        let taps = Taps::new(&p).unwrap();
        let border = taps.border_inputs(&p, &input, &[(0, 0, 0); ROWS]).unwrap();
        assert!(border.bases.iter().flatten().all(|&base| base == u32::MAX));
        for tap in 0..taps.count {
            // SAFETY: this descriptor is all-padding; tap/channel are valid.
            assert_eq!(
                unsafe { border.read::<5>(&taps, tap, p.cin - 1) }.to_bits(),
                0
            );
        }
        assert_eq!(encode_base(0), Some(0));
        assert_eq!(encode_base(u32::MAX as usize - 1), Some(u32::MAX - 1));
        assert_eq!(encode_base(u32::MAX as usize), None);
        assert_eq!(encode_base(u32::MAX as usize + 1), None);
        p.height = usize::MAX;
        assert!(taps.border_inputs(&p, &input, &[(0, 0, 0); ROWS]).is_none());
    }
}
