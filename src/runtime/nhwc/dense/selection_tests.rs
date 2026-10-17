use super::*;

#[test]
fn spatial_sme_crossover_requires_native_backend_dense_three_and_bounded_rows() {
    let p = Conv {
        batch: 1,
        cin: 32,
        cout: 16,
        height: 4,
        width: 8,
        oh: 4,
        ow: 8,
        cpg: 32,
        opg: 16,
        kh: 3,
        kw: 3,
        sh: 1,
        sw: 1,
        dh: 1,
        dw: 1,
        pt: 1,
        pl: 1,
    };
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    let available = super::super::pointwise::sme::available();
    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    let available = false;
    for backend in [
        Backend::Scalar,
        Backend::Portable4,
        Backend::NeonFma,
        Backend::Avx,
        Backend::AvxFma,
        Backend::Avx2Fma,
        Backend::Avx512,
    ] {
        assert_eq!(
            sme_scratch_elements(backend, &p),
            if available && backend == Backend::NeonFma {
                64 * 32 * 9
            } else {
                0
            }
        );
    }
    for invalid in [
        Conv {
            cin: 31,
            cpg: 31,
            ..p.clone()
        },
        Conv {
            cout: 15,
            opg: 15,
            ..p.clone()
        },
        Conv {
            cpg: 16,
            opg: 8,
            ..p.clone()
        },
        Conv {
            kh: 1,
            kw: 1,
            ..p.clone()
        },
        Conv {
            kh: 2,
            kw: 2,
            ..p.clone()
        },
        Conv {
            oh: 1,
            ow: 31,
            ..p.clone()
        },
    ] {
        assert_eq!(sme_scratch_elements(Backend::NeonFma, &invalid), 0);
    }
}
