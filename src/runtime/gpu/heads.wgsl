// Compact public heads. Each class keeps bias-first, ascending-K accumulation.
struct Parameters {
    rows: u32,
    classes: u32,
    inner: u32,
    tiles: u32,
    has_bias: u32,
    project_groups_x: u32,
    reduce_stride: u32,
    padding: u32,
}
struct Record { index: u32, score: f32, valid: u32, padding: u32, }
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> partial: array<Record>;
@group(0) @binding(4) var<storage, read_write> output: array<Record>;
@group(0) @binding(5) var<uniform> params: Parameters;

var<workgroup> tile_scores: array<f32, 64>;
var<workgroup> tile_indices: array<u32, 64>;
var<workgroup> tile_valid: array<u32, 64>;
const ABSENT: u32 = 0xffffffffu;

fn finite(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}
fn wins(index: u32, score: f32, best_index: u32, best_score: f32) -> bool {
    return index != ABSENT && (best_index == ABSENT || score > best_score ||
        (score == best_score && index < best_index));
}
fn finish_tile(lane: u32, row: u32, tile: u32) {
    workgroupBarrier();
    var width = 32u;
    loop {
        if width == 0u { break; }
        if lane < width {
            let other = lane + width;
            if wins(tile_indices[other], tile_scores[other], tile_indices[lane], tile_scores[lane]) {
                tile_scores[lane] = tile_scores[other];
                tile_indices[lane] = tile_indices[other];
            }
            tile_valid[lane] = tile_valid[lane] & tile_valid[other];
        }
        workgroupBarrier();
        width = width / 2u;
    }
    if lane == 0u {
        partial[row * params.tiles + tile] = Record(tile_indices[0], tile_scores[0], tile_valid[0], 0u);
    }
}

@compute @workgroup_size(64)
fn project_tiles(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let flat_group = group.x + group.y * params.project_groups_x;
    let row = flat_group / params.tiles;
    if row >= params.rows { return; }
    let tile = flat_group % params.tiles;
    let column = tile * 64u + lane;
    var value = 0.0;
    var index = ABSENT;
    var valid = 1u;
    if column < params.classes {
        if params.has_bias != 0u { value = bias[column]; }
        for (var k = 0u; k < params.inner; k++) {
            value = value + source[row * params.inner + k] * weights[k * params.classes + column];
        }
        if finite(value) { index = column; }
        else { valid = 0u; }
    }
    tile_scores[lane] = value;
    tile_indices[lane] = index;
    tile_valid[lane] = valid;
    finish_tile(lane, row, tile);
}

@compute @workgroup_size(64)
fn probability_tiles(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let flat_group = group.x + group.y * params.project_groups_x;
    let row = flat_group / params.tiles;
    if row >= params.rows { return; }
    let tile = flat_group % params.tiles;
    let column = tile * 64u + lane;
    var value = 0.0;
    var index = ABSENT;
    var valid = 1u;
    if column < params.classes {
        value = source[row * params.classes + column];
        if finite(value) && value >= 0.0 && value <= 1.0 { index = column; }
        else { valid = 0u; }
    }
    tile_scores[lane] = value;
    tile_indices[lane] = index;
    tile_valid[lane] = valid;
    finish_tile(lane, row, tile);
}

@compute @workgroup_size(64)
fn reduce_tiles(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x + id.y * params.reduce_stride;
    if row >= params.rows { return; }
    var best = Record(ABSENT, 0.0, 1u, 0u);
    for (var tile = 0u; tile < params.tiles; tile++) {
        let candidate = partial[row * params.tiles + tile];
        best.valid = best.valid & candidate.valid;
        if wins(candidate.index, candidate.score, best.index, best.score) {
            best.index = candidate.index;
            best.score = candidate.score;
        }
    }
    output[row] = best;
}

@compute @workgroup_size(64)
fn classify(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x + id.y * params.reduce_stride;
    if row >= params.rows { return; }
    var first = 0.0;
    var second = 0.0;
    if params.has_bias != 0u { first = bias[0]; second = bias[1]; }
    for (var k = 0u; k < params.inner; k++) {
        let value = source[row * params.inner + k];
        first = first + value * weights[k * 2u];
        second = second + value * weights[k * 2u + 1u];
    }
    if !finite(first) || !finite(second) {
        output[row] = Record(0u, 0.0, 0u, 0u);
        return;
    }
    let label = select(0u, 1u, second > first);
    let maximum = max(first, second);
    let e0 = exp(first - maximum);
    let e1 = exp(second - maximum);
    var sum = 0.0;
    sum = sum + e0;
    sum = sum + e1;
    let score = select(e0, e1, label == 1u) / sum;
    output[row] = Record(label, score, select(0u, 1u, finite(score)), 0u);
}
