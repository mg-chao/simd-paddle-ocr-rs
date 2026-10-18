use super::*;

#[test]
fn horizontal_interior_matches_each_taps_independent_bounds() {
    for (width, ow, kernel, stride, dilation, pad) in [
        (240, 240, 7, 1, 1, 3),
        (33, 17, 7, 2, 1, 3),
        (35, 35, 7, 1, 2, 7),
        (17, 17, 6, 1, 1, 3),
        (5, 5, 7, 1, 1, 3),
        (1, 9, 9, 1, 2, 12),
        (33, 7, 7, 3, 2, 0),
    ] {
        let p = Conv {
            batch: 1,
            cin: 32,
            cout: 32,
            height: 1,
            width,
            oh: 1,
            ow,
            cpg: 1,
            opg: 1,
            kh: 7,
            kw: kernel,
            sh: 1,
            sw: stride,
            dh: 1,
            dw: dilation,
            pt: 3,
            pl: pad,
        };
        let interior = horizontal_interior(&p);
        for x in 0..ow {
            let all_taps_valid = (0..kernel).all(|tap| {
                (x * stride + tap * dilation)
                    .checked_sub(pad)
                    .is_some_and(|source| source < width)
            });
            assert_eq!(
                interior.contains(&x),
                all_taps_valid,
                "width={width} ow={ow} kernel={kernel} stride={stride} dilation={dilation} pad={pad} x={x}"
            );
        }
        if width == 240 {
            assert_eq!(interior, 3..237);
        }
    }
}
