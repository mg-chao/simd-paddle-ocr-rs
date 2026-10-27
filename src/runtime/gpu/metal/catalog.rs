//! Static shader interface catalog, shared by the build script and native executor.
pub struct Entry {
    pub id: &'static str,
    pub body: &'static str,
    pub entry: &'static str,
    pub defines: &'static str,
    pub mask: u8,
    pub threads: u32,
    pub reads: &'static [bool],
    pub neural: bool,
    pub test: bool,
}
macro_rules! neural {
    ($id:ident, $body:literal, $mask:expr, $defines:literal) => {
        Entry {
            id: stringify!($id),
            body: include_str!($body),
            entry: "neural",
            defines: $defines,
            mask: $mask,
            threads: 64,
            reads: &[true, true, true, true, true, false],
            neural: true,
            test: false,
        }
    };
}
macro_rules! resident {
    ($id:ident, $body:literal, $entry:literal, $mask:expr, $threads:expr, $reads:expr) => {
        Entry {
            id: stringify!($id),
            body: include_str!($body),
            entry: $entry,
            defines: "",
            mask: $mask,
            threads: $threads,
            reads: $reads,
            neural: false,
            test: false,
        }
    };
}
pub const ENTRIES: &[Entry] = &[
    neural!(
        Attention,
        "shaders/attention.metal",
        47,
        "#define VIEWS 0\n"
    ),
    neural!(
        AttentionViews,
        "shaders/attention.metal",
        47,
        "#define VIEWS 1\n"
    ),
    neural!(
        SqueezeExcite,
        "shaders/excitation.metal",
        47,
        "#define SMALL_PROJECTION 0\n"
    ),
    neural!(
        SmallProjection,
        "shaders/excitation.metal",
        63,
        "#define SMALL_PROJECTION 1\n"
    ),
    neural!(Transposed, "shaders/transposed.metal", 63, ""),
    neural!(Convolution, "shaders/convolution.metal", 63, ""),
    neural!(General, "shaders/general.metal", 63, ""),
    neural!(Depthwise3, "shaders/depthwise.metal", 63, "#define KH 3\n"),
    neural!(Depthwise5, "shaders/depthwise.metal", 63, "#define KH 5\n"),
    neural!(Depthwise7, "shaders/depthwise.metal", 63, "#define KH 7\n"),
    neural!(Reduce, "shaders/reduce.metal", 39, "#define MODE 0\n"),
    neural!(Softmax, "shaders/reduce.metal", 39, "#define MODE 1\n"),
    neural!(LayerNorm, "shaders/reduce.metal", 39, "#define MODE 2\n"),
    neural!(
        ReducePartial,
        "shaders/reduce_staged.metal",
        35,
        "#define SOFTMAX 0\n#define FINISH 0\n"
    ),
    neural!(
        ReduceFinish,
        "shaders/reduce_staged.metal",
        35,
        "#define SOFTMAX 0\n#define FINISH 1\n"
    ),
    neural!(
        SoftmaxPartial,
        "shaders/reduce_staged.metal",
        35,
        "#define SOFTMAX 1\n#define FINISH 0\n"
    ),
    neural!(
        SoftmaxFinish,
        "shaders/reduce_staged.metal",
        35,
        "#define SOFTMAX 1\n#define FINISH 1\n"
    ),
    neural!(
        Matmul16,
        "shaders/tiled.metal",
        63,
        "#define TILE 16\n#define IS_CONV 0\n"
    ),
    neural!(
        Matmul32,
        "shaders/tiled.metal",
        63,
        "#define TILE 32\n#define IS_CONV 0\n"
    ),
    neural!(
        Conv16,
        "shaders/tiled.metal",
        63,
        "#define TILE 16\n#define IS_CONV 1\n"
    ),
    neural!(
        Conv32,
        "shaders/tiled.metal",
        63,
        "#define TILE 32\n#define IS_CONV 1\n"
    ),
    Entry {
        threads: 256,
        ..neural!(
            Conv64,
            "shaders/tiled.metal",
            63,
            "#define TILE 64\n#define IS_CONV 1\n"
        )
    },
    Entry {
        threads: 256,
        ..neural!(
            Matmul64,
            "shaders/tiled.metal",
            63,
            "#define TILE 64\n#define IS_CONV 0\n"
        )
    },
    resident!(
        Image,
        "shaders/image.metal",
        "image",
        4,
        64,
        &[true, true, false]
    ),
    resident!(
        Crop,
        "shaders/crop.metal",
        "crop",
        0,
        64,
        &[true, true, false]
    ),
    resident!(
        CropMeta,
        "shaders/crop_meta.metal",
        "crop_meta",
        0,
        64,
        &[true, false]
    ),
    resident!(
        PackBgr,
        "shaders/pack_bgr.metal",
        "pack_bgr",
        0,
        64,
        &[true, false]
    ),
    resident!(
        Rotate,
        "shaders/rotate.metal",
        "rotate",
        0,
        64,
        &[true, true, false]
    ),
    resident!(
        DbLabel,
        "shaders/db_label.metal",
        "db_label",
        1,
        64,
        &[true, false, false, false, false, false, false]
    ),
    resident!(
        DbRows,
        "shaders/db_rows.metal",
        "db_rows",
        0,
        64,
        &[true, true, true, false]
    ),
    resident!(
        DbGeometry,
        "shaders/db_geometry.metal",
        "db_geometry",
        1,
        1,
        &[true, true, true, false, false]
    ),
    resident!(
        DbGeometryFast,
        "shaders/db_geometry.metal",
        "db_geometry",
        1,
        1,
        &[true, true, true, false, false]
    ),
    resident!(
        DbSort,
        "shaders/db_sort.metal",
        "db_sort",
        0,
        1,
        &[true, false]
    ),
    resident!(
        Heads,
        "shaders/heads.metal",
        "heads",
        1,
        64,
        &[true, true, true, false]
    ),
    resident!(
        Heads4,
        "shaders/heads4.metal",
        "heads4",
        1,
        64,
        &[true, true, true, false]
    ),
    resident!(
        Heads4Subgroup,
        "shaders/heads4.metal",
        "heads4",
        1,
        64,
        &[true, true, true, false]
    ),
    resident!(
        HeadsFeaturesF32,
        "shaders/heads_features_f32.metal",
        "heads_features_f32",
        1,
        64,
        &[true, false]
    ),
    resident!(
        HeadReduce,
        "shaders/head_reduce.metal",
        "head_reduce",
        0,
        64,
        &[true, false]
    ),
    resident!(
        HeadReduceParallel,
        "shaders/head_reduce_parallel.metal",
        "head_reduce_parallel",
        0,
        64,
        &[true, false]
    ),
    resident!(Ctc, "shaders/ctc.metal", "ctc", 0, 64, &[true, true, false]),
    Entry {
        id: "ProjectionMatrixF32",
        body: include_str!("shaders/projection.metal"),
        entry: "projection",
        defines: "#define MATRIX float\n#define MATRIX8 simdgroup_float8x8\n",
        mask: 0,
        threads: 128,
        reads: &[true, true, true, false],
        neural: false,
        test: false,
    },
    Entry {
        id: "ProjectionMatrixF16",
        body: include_str!("shaders/projection.metal"),
        entry: "projection",
        defines: "#define MATRIX half\n#define MATRIX8 simdgroup_half8x8\n",
        mask: 0,
        threads: 128,
        reads: &[true, true, true, false],
        neural: false,
        test: false,
    },
    Entry {
        id: "CounterProbe",
        body: "kernel void counter(device uint* value [[buffer(0)]],constant Params& p [[buffer(1)]]){value[0]+=1u;}",
        entry: "counter",
        defines: "",
        mask: 0,
        threads: 1,
        reads: &[false],
        neural: false,
        test: true,
    },
    Entry {
        id: "RoundHalfProbe",
        body: "kernel void round_half(device const float* input [[buffer(0)]],device float* output [[buffer(1)]],constant Params& p [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<p.words[0].x)output[i+p.words[0].y]=round_to_half(input[i]);}",
        entry: "round_half",
        defines: "",
        mask: 0,
        threads: 64,
        reads: &[true, false],
        neural: false,
        test: true,
    },
    Entry {
        id: "HalfStoreProbe",
        body: "void run(uint3 id,uint3,uint,threadgroup float*)const{if(id.x<p(0))store_output(id.x,load_a(id.x));}\n",
        entry: "neural",
        defines: "",
        mask: 32,
        threads: 64,
        reads: &[true, true, true, true, true, false],
        neural: true,
        test: true,
    },
];

pub fn source(entry: &Entry) -> String {
    let mut result = String::from("#include <metal_stdlib>\nusing namespace metal;\n");
    for precision in 0..64u8 {
        if precision & !entry.mask != 0 {
            continue;
        }
        let name = format!("ocr_{}_{}", entry.id, precision);
        result.push_str(entry.defines);
        if matches!(
            entry.id,
            "Matmul16" | "Matmul32" | "Matmul64" | "Conv16" | "Conv32" | "Conv64"
        ) {
            result.push_str(if matches!(precision & 3, 0 | 3) {
                "#define DIRECT_MATRIX 1\n"
            } else {
                "#define DIRECT_MATRIX 0\n"
            });
            result.push_str(if precision & 3 == 3 {
                "#define MATRIX half\n#define MATRIX8 simdgroup_half8x8\n"
            } else {
                "#define MATRIX float\n#define MATRIX8 simdgroup_float8x8\n"
            });
        }
        result.push_str(&format!("namespace impl_{name} {{\n"));
        let half = if entry.neural {
            precision & 32 != 0
        } else {
            precision != 0
        };
        result.push_str(&format!("#define HALF_OUTPUT {}\n", u8::from(half)));
        for (index, slot) in ["A", "B", "C", "D", "E", "O"].iter().enumerate() {
            result.push_str(&format!(
                "#define STORAGE_{slot} {}\n",
                if precision & (1 << index) != 0 {
                    "half"
                } else {
                    "float"
                }
            ));
        }
        let scalar = if half { "half" } else { "float" };
        result.push_str(&format!("#define INPUT {scalar}\n#define OUT {scalar}\n"));
        result.push_str(include_str!("shaders/half.metal"));
        if entry.neural {
            result.push_str(&include_str!("shaders/neural.metal").replace(
                "constant uint round_intermediate [[function_constant(0)]];",
                "constant uint round_intermediate = HALF_OUTPUT;",
            ));
            result.push_str(entry.body);
            result.push_str(&include_str!("shaders/entry.metal").replace(
                "kernel void neural",
                &format!("[[host_name(\"{name}\")]] kernel void neural"),
            ));
        } else {
            result.push_str(include_str!("shaders/resident.metal"));
            result.push_str(if entry.id == "DbGeometryFast" {
                include_str!("shaders/precise_fast.metal")
            } else {
                include_str!("shaders/precise.metal")
            });
            result.push_str(&entry.body.replace(
                &format!("kernel void {}", entry.entry),
                &format!("[[host_name(\"{name}\")]] kernel void {}", entry.entry),
            ));
        }
        result.push_str("\n}\n");
        for variable in [
            "KH",
            "MATRIX",
            "MATRIX8",
            "DIRECT_MATRIX",
            "TILE_THREADS",
            "INNER_TILE",
            "STAGING_FLOATS",
            "SG_ROWS",
            "SG_COLS",
            "HALF_OUTPUT",
            "STORAGE_A",
            "STORAGE_B",
            "STORAGE_C",
            "STORAGE_D",
            "STORAGE_E",
            "STORAGE_O",
            "INPUT",
            "OUT",
            "SCRATCH_FLOATS",
            "VIEWS",
            "MODE",
            "SOFTMAX",
            "FINISH",
            "TILE",
            "IS_CONV",
            "SMALL_PROJECTION",
        ] {
            result.push_str(&format!("#undef {variable}\n"));
        }
    }
    result
}
