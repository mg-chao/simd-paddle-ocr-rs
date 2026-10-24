// FP32 pointwise GEMM: weights[Cout, Cin] * input[Cin, HW]. Direct
// fragment loads require all three dimensions to be positive multiples of 8.
// Each subgroup owns a 32x32 tile. Scratch is only for bias and the epilogue.
alias MA = coop_mat8x8<f32, A>;
alias MB = coop_mat8x8<f32, B>;
alias MC = coop_mat8x8<f32, C>;
// 64 floats per subgroup, up to 16 subgroups (64 threads / minimum width 4).
var<workgroup> result: array<f32, 1024>;
var<workgroup> subgroup_count: u32;

fn finish_fragment(dest: u32, lane: u32, width: u32, batch: u32, row: u32, col: u32) {
    let rows = p(4u);
    let columns = p(7u) * p(8u);
    for (var i = lane; i < 64u; i += width) {
        let r = row + i / 8u;
        let c = col + i % 8u;
        if batch < p(35u) && r < rows && c < columns {
            let index = (batch * rows + r) * columns + c;
            store_output(index, epilogue(result[dest + i], index));
        }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) workgroup_lane: u32,
        @builtin(subgroup_invocation_id) lane: u32,
        @builtin(subgroup_size) width: u32,
        @builtin(subgroup_id) sg: u32,
        @builtin(num_subgroups) groups: u32) {
    if workgroup_lane == 0u { subgroup_count = groups; }
    // Establish uniform control flow for matrix operations and barriers.
    let count = workgroupUniformLoad(&subgroup_count);
    let block = group.x + group.y * p(34u);
    // Two logical tiles per workgroup, independent of the subgroup width.
    // One subgroup takes two phases; extra subgroups mask outputs. All matrix
    // loads remain in bounds, including the unused tile in an odd-sized grid.
    for (var phase = 0u; phase < (2u + count - 1u) / count; phase++) {
        let local_tile = phase * count + sg;
        let tile = block * 2u + local_tile;
        let batch = select(p(35u), tile / (p(32u) * p(33u)), local_tile < 2u);
        let mr = (tile / p(33u)) % p(32u) * $TILEu;
        let nc = tile % p(33u) * $TILEu;
        let rows = p(4u);
        let inner = p(3u);
        let columns = p(7u) * p(8u);
        let dest = sg * 64u;
        $INIT
        let ai = min(batch, p(35u) - 1u) * inner * columns;
        for (var base = 0u; base < inner; base += 8u) {
            $STEP
        }
        $FINISH
    }
}
