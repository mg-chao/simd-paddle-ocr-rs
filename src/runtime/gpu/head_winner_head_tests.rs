// Uses the existing Half-head fixture and independent Half-operand oracle.
#[test]
#[ignore = "requires physical Metal; whole Half head selection, score modes, profiling, strict and shape fallback"]
fn gpu_parallel_head_winner_whole_head_policy_and_profile() {
    let [rows, inner, classes] = [33, 17, 4099];
    let features = dense(rows * inner, 13, 63, 128.);
    let weights = dense(inner * classes, 17, 127, 256.);
    let mut bias = dense(classes, 7, 31, 256.);
    bias[classes - 1] = 10.;
    let expected = oracle([rows, inner, classes], &features, &weights, &bias);
    let context = context();
    let (_, head, features) = fixture(
        &context,
        [rows, inner, classes],
        GpuPrecision::MixedF16,
        GpuMatrixPrecision::F16,
        [&features, &weights, &bias],
    );
    assert!(head.half_weights.is_some());
    for (probability, profiling) in [(false, false), (true, false), (false, true)] {
        let mut batch = Batch::new(context.clone()).unwrap();
        batch.profiling = profiling;
        let winners = head.project(&mut batch, &features, probability).unwrap();
        assert!(
            batch
                .statistics
                .kernels
                .iter()
                .any(|name| name == NATIVE_PROJECTION_F16_NAME)
        );
        assert_eq!(
            batch
                .statistics
                .kernels
                .iter()
                .any(|name| name == head_winner::NAME),
            !probability,
        );
        assert_eq!(
            batch
                .statistics
                .kernels
                .iter()
                .any(|name| name == "projection_winners"),
            probability
        );
        let (actual, statistics) = batch.read(&winners, rows * 16, false).unwrap();
        assert_oracle(&actual, &expected, probability);
        assert_eq!(statistics.dispatches, 2);
        assert_eq!(statistics.diagnostic_readback_bytes, 0);
        assert_eq!(statistics.native_resident_dispatches, 2);
        if profiling {
            let profile = statistics.profile.unwrap();
            assert_eq!(profile.kernels.len(), 2);
            assert!(
                profile
                    .kernels
                    .iter()
                    .any(|kernel| kernel.kernel == head_winner::NAME)
            );
        } else {
            assert!(statistics.profile.is_none());
        }
    }
    // A strict head on the same capable device must keep its original reducer.
    let strict_features = dense(rows * inner, 13, 63, 128.);
    let (_, strict, features) = fixture(
        &context,
        [rows, inner, classes],
        GpuPrecision::F32,
        GpuMatrixPrecision::F32,
        [&strict_features, &weights, &bias],
    );
    assert!(strict.half_weights.is_none());
    let mut batch = Batch::new(context).unwrap();
    let winners = strict.project(&mut batch, &features, false).unwrap();
    assert!(
        !batch
            .statistics
            .kernels
            .iter()
            .any(|name| name == head_winner::NAME)
    );
    assert!(
        batch
            .statistics
            .kernels
            .iter()
            .any(|name| name == "projection_winners")
    );
    let (actual, _) = batch.read(&winners, rows * 16, false).unwrap();
    assert_oracle(&actual, &expected, false);
}
