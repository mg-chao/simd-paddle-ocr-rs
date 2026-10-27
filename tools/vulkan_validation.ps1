param(
    [Parameter(Mandatory = $true)][string]$LayerDirectory,
    [switch]$Numerics
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$layers = (Resolve-Path -LiteralPath $LayerDirectory).Path
if (-not (Test-Path -LiteralPath (Join-Path $layers 'VkLayer_khronos_validation.json'))) {
    throw 'Khronos validation layer manifest is missing; validation cannot run.'
}
Push-Location -LiteralPath $repo
$oldLayers = $env:VK_LAYER_PATH
$oldValidation = $env:SIMD_OCR_VULKAN_VALIDATION
try {
    $env:VK_LAYER_PATH = $layers
    $env:SIMD_OCR_VULKAN_VALIDATION = '1'
    cargo test --release --locked --features gpu,profiling --lib --no-run
    if ($LASTEXITCODE -ne 0) { throw 'Release build failed.' }
    $tree = cargo tree --locked --features gpu --target x86_64-pc-windows-msvc -e normal,build --prefix none
    if ($LASTEXITCODE -ne 0) { throw 'Dependency inspection failed.' }
    if ($tree -match '^(wgpu|wgpu-core|wgpu-hal|pollster|naga) v') {
        throw 'Windows dependency graph contains a removed backend dependency.'
    }
    $filters = @('vulkan_')
    if ($Numerics) {
        $filters += @(
            'gpu_round_f16_',
            'gpu_hardware_convolution_matches_scalar',
            'gpu_hardware_specialized_tiles_tails_and_long_reductions',
            'gpu_hardware_attention_se_resize_placement_and_rank_five',
            'gpu_resident_preprocess_matches_byte_domain_cpu'
        )
    }
    foreach ($filter in $filters) {
        cargo test --release --locked --features gpu,profiling --lib $filter -- --ignored --nocapture --test-threads=1
        if ($LASTEXITCODE -ne 0) { throw "Physical Vulkan check failed: $filter" }
    }
} finally {
    $env:VK_LAYER_PATH = $oldLayers
    $env:SIMD_OCR_VULKAN_VALIDATION = $oldValidation
    Pop-Location
}
