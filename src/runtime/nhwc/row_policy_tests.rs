//! Fixed dispatch boundaries and independent row-band/store contracts.
use super::*;
fn plan(depthwise: bool, rows: usize, width: usize) -> Conv {
    Conv {
        batch: 1,
        cin: if depthwise { 8 } else { 1 },
        cout: if depthwise { 8 } else { 16 },
        height: rows,
        width,
        oh: rows,
        ow: width,
        cpg: 1,
        opg: if depthwise { 1 } else { 16 },
        kh: 1,
        kw: 1,
        sh: 1,
        sw: 1,
        dh: 1,
        dw: 1,
        pt: 0,
        pl: 0,
    }
}
#[test]
fn row_worker_counts_match_fixed_dense_and_depthwise_thresholds_and_row_caps() {
    // Dense16 columns *2rows *62500positions =2M; depthwise8 *2 *62500 =1M.
    for depthwise in [false, true] {
        assert_eq!(conv_row_workers(&plan(depthwise, 2, 62499), 8), 1);
        assert_eq!(conv_row_workers(&plan(depthwise, 2, 62500), 8), 2);
        assert_eq!(conv_row_workers(&plan(depthwise, 2, 62501), 8), 2);
        assert_eq!(conv_row_workers(&plan(depthwise, 2, 62500), 0), 1);
        assert_eq!(conv_row_workers(&plan(depthwise, 2, 62500), 1), 1);
        assert_eq!(conv_row_workers(&plan(depthwise, 2, 62500), 2), 2);
    }
    let mut batched = plan(false, 3, 62500);
    batched.batch = 2;
    assert_eq!(conv_row_workers(&batched, 8), 6);
    assert_eq!(conv_row_workers(&batched, 4), 4);
    assert_eq!(conv_row_workers(&plan(false, 1, 125000), 8), 1);
    let mut group_one = plan(false, 2, 500000);
    group_one.cout = 1;
    group_one.opg = 1;
    assert_eq!(conv_row_workers(&group_one, 8), 1); //1M is below dense threshold.
    group_one.width = 1000000;
    group_one.ow = 1000000;
    assert_eq!(conv_row_workers(&group_one, 8), 2);
}
#[test]
fn whole_row_bands_cover_batches_and_tails_without_rebasing_global_store_indices() {
    let cpu = CpuContext::new(if cfg!(feature = "parallel") { 8 } else { 1 }).unwrap();
    for (batch, rows, width, channels) in [(1, 3, 7, 8), (2, 5, 17, 16), (2, 3, 9, 17)] {
        for budget in [1, 2, 4, 8] {
            let row_length = width * channels;
            let mut output = vec![f32::NAN; batch * rows * row_length];
            crate::runtime::parallel::aligned_bands(
                &mut output,
                row_length,
                batch * rows,
                budget,
                1,
                &cpu,
                |band, first_row| {
                    assert_eq!(band.len() % row_length, 0);
                    for (offset, value) in band.iter_mut().enumerate() {
                        *value = (first_row * row_length + offset) as f32;
                    }
                },
            );
            for (index, value) in output.iter().enumerate() {
                assert_eq!(value.to_bits(), (index as f32).to_bits());
            }
        }
    }
}

struct IndexedStore;
impl StoreTransform for IndexedStore {
    fn scalar(&self, value: f32, index: usize) -> f32 {
        value + (index % 13) as f32 / 64.
    }
    fn vector<const N: usize>(&self, value: Simd<f32, N>, index: usize) -> Simd<f32, N> {
        value
            + Simd::from_array(std::array::from_fn(|lane| {
                ((index + lane) % 13) as f32 / 64.
            }))
    }
}
#[test]
fn dense_and_depthwise_row_bands_keep_independent_reduction_bits_and_global_stores() {
    let cpu = CpuContext::new(if cfg!(feature = "parallel") { 8 } else { 1 }).unwrap();
    let mut backends = vec![Backend::Scalar, Backend::Portable4];
    for (available, backend) in [
        (simd::avx_available(), Backend::Avx),
        (simd::avx_fma_available(), Backend::AvxFma),
        (simd::avx2_fma_available(), Backend::Avx2Fma),
        (simd::avx512_available(), Backend::Avx512),
    ] {
        if available {
            backends.push(backend);
        }
    }
    for (batch, cin, cout, channels_per_group, kernel, height, width) in [
        (1, 3, 16, 3, 3, 5, 17),
        (2, 3, 17, 3, 3, 3, 9),
        (2, 8, 8, 1, 5, 7, 17),
        (1, 16, 16, 1, 7, 9, 19),
    ] {
        let p = Conv {
            batch,
            cin,
            cout,
            height,
            width,
            oh: height,
            ow: width,
            cpg: channels_per_group,
            opg: if channels_per_group == 1 { 1 } else { cout },
            kh: kernel,
            kw: kernel,
            sh: 1,
            sw: 1,
            dh: 1,
            dw: 1,
            pt: kernel / 2,
            pl: kernel / 2,
        };
        let inner = p.cpg * kernel * kernel;
        let input: Vec<_> = (0..batch * cin * height * width)
            .map(|i| ((i % 17) as f32 - 8.) / 32.)
            .collect();
        let weights: Vec<_> = (0..cout * inner)
            .map(|i| ((i % 7) as f32 - 3.) / 64.)
            .collect();
        let bias: Vec<_> = (0..cout).map(|i| (i as f32 + 1.) / 32.).collect();
        let mut transposed = vec![0.; inner * cout];
        for co in 0..cout {
            for k in 0..inner {
                transposed[k * cout + co] = weights[co * inner + k];
            }
        }
        let packed = PackedMatrix::new(&transposed, inner, cout).unwrap();
        let mut expected = vec![0.; batch * height * width * cout];
        for (pixel, row) in expected.chunks_exact_mut(cout).enumerate() {
            let n = pixel / (height * width);
            let y = pixel / width % height;
            let x = pixel % width;
            for (co, value) in row.iter_mut().enumerate() {
                let mut sum = bias[co];
                let group = co / p.opg;
                for ci in 0..p.cpg {
                    for ky in 0..kernel {
                        for kx in 0..kernel {
                            let iy = y as isize + ky as isize - p.pt as isize;
                            let ix = x as isize + kx as isize - p.pl as isize;
                            if iy >= 0 && iy < height as isize && ix >= 0 && ix < width as isize {
                                sum += input[((n * height + iy as usize) * width + ix as usize)
                                    * cin
                                    + group * p.cpg
                                    + ci]
                                    * weights[((co * p.cpg + ci) * kernel + ky) * kernel + kx];
                            }
                        }
                    }
                }
                *value = IndexedStore.scalar(sum, pixel * cout + co);
            }
        }
        for backend in &backends {
            for budget in [1, 2, 4, 8] {
                let mut actual = vec![f32::NAN; expected.len()];
                crate::runtime::parallel::aligned_bands(
                    &mut actual,
                    width * cout,
                    batch * height,
                    budget,
                    1,
                    &cpu,
                    |output, first_row| {
                        conv_dispatch_store(
                            *backend,
                            &p,
                            &input,
                            &weights,
                            &bias,
                            Some(&packed),
                            output,
                            first_row * width,
                            &mut [],
                            &IndexedStore,
                        )
                    },
                );
                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "{backend:?} budget={budget} cout={cout} kernel={kernel}"
                );
            }
        }
    }
}

#[test]
fn scalar_single_spatial_row_pointwise_keeps_reference_six_pixel_workers_and_math() {
    let (rows, inner, columns) = (96, 585, 128);
    let p = Conv {
        batch: 1,
        cin: inner,
        cout: columns,
        height: 1,
        width: rows,
        oh: 1,
        ow: rows,
        cpg: inner,
        opg: columns,
        kh: 1,
        kw: 1,
        sh: 1,
        sw: 1,
        dh: 1,
        dw: 1,
        pt: 0,
        pl: 0,
    };
    assert!(packed_pointwise_eligible(Backend::Scalar, &p, true));
    assert_eq!(PointwiseBands::new(rows, inner, columns, 8).workers, 8);
    // The general row policy has only one row here; eligible pointwise must
    // bypass it and retain six-pixel shards, as production execute now does.
    assert_eq!(conv_row_workers(&p, 8), 1);
    let dyadic_input: Vec<_> = (0..rows * inner)
        .map(|i| ((i % 17) as f32 - 8.) / 32.)
        .collect();
    let dyadic_weights: Vec<_> = (0..inner * columns)
        .map(|i| ((i % 7) as f32 - 3.) / 64.)
        .collect();
    let mut dyadic_expected = vec![0.; rows * columns];
    for row in 0..rows {
        for col in 0..columns {
            let mut sum = 0.;
            for k in 0..inner {
                sum += dyadic_input[row * inner + k] * dyadic_weights[k * columns + col];
            }
            dyadic_expected[row * columns + col] = sum;
        }
    }
    // Hand oracle across the512 boundary: bias5 -> +1e20 atK511
    // loses5, -1e20 atK512 cancels, and +0.5 atK513 survives.
    // Bias-after gives5.5; an independently summed second512-K block gives0.
    let ordered_input = vec![1.; rows * inner];
    let mut ordered_weights = vec![0.; inner * columns];
    ordered_weights[511 * columns..512 * columns].fill(1e20);
    ordered_weights[512 * columns..513 * columns].fill(-1e20);
    ordered_weights[513 * columns..514 * columns].fill(0.5);
    let cpu = CpuContext::new(if cfg!(feature = "parallel") { 8 } else { 1 }).unwrap();
    for (name, input, weights, bias, expected) in [
        (
            "dyadic",
            dyadic_input,
            dyadic_weights,
            vec![0.; columns],
            dyadic_expected,
        ),
        (
            "ordered-bias",
            ordered_input,
            ordered_weights,
            vec![5.; columns],
            vec![0.5f32; rows * columns],
        ),
    ] {
        let packed = PackedMatrix::new(&weights, inner, columns).unwrap();
        for budget in [1, 2, 4, 8] {
            let mut actual = vec![f32::NAN; expected.len()];
            pointwise_execute(
                Backend::Scalar,
                &input,
                &bias,
                &packed,
                &mut actual,
                inner,
                columns,
                &cpu,
                budget,
                &simd::IdentityStore,
            );
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "{name} budget={budget}"
            );
        }
    }
}
