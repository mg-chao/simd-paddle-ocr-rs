# Vulkan GPU 与 RapidOCR 的逐图对照

2026-11-24，在 Windows 物理机 AMD Ryzen 9 5950X、Radeon RX 6750 XT（Windows 驱动 32.0.21045.29003）上验收。默认 GPU 精度 F32、矩阵 F32、8 个 CPU 线程。Rust 公开入口使用 `ExecutionTarget::Gpu`，运行记录确认检测、分类、识别均在 GPU 上执行。每个预设另用 `ocr --execution gpu --statistics --json` 核对一张，输出按 Rust 字段的 F32 精度与公开 API 采集的完整 JSON 相等。

## 整图与行级结果

每行各有 128 张原图。整图指标要求行数、行序和每行原始文本完全相同，CER 计入空格、大小写、标点和换行。框匹配沿用冻结的四边形 IoU ≥ 0.5 一对一比较。箭头左侧是本轮改动前的 Rust GPU release，右侧是最终 GPU release。

| 预设 | 整图相同 /128 | 编辑数 | CER | 漏行 / 多行 | 顺序逆序对 | 执行失败 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny | 20 → 128 | 11,761 → 0 | 8.74190% → 0 | 121 / 126 → 0 / 0 | 1,397 → 0 | 0 → 0 |
| small | 22 → 128 | 10,537 → 0 | 7.66210% → 0 | 68 / 57 → 0 / 0 | 1,614 → 0 | 0 → 0 |
| medium | 21 → 128 | 14,097 → 0 | 8.82403% → 0 | 111 / 82 → 0 / 0 | 2,351 → 0 | 4 → 0 |

| 预设 | 匹配行 | 平均 IoU | 平均角点偏差（原图像素） | 行识别置信度平均绝对误差 | 最终最大分数差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny | 5,380 → 5,501 | 0.914412 → 1 | 1.29809 → 0 | 0.02582262 → 0.00000078 | 0.00002003 |
| small | 5,224 → 5,292 | 0.915320 → 1 | 1.29148 → 0 | 0.01382411 → 0.00000188 | 0.00012003 |
| medium | 6,113 → 6,533 | 0.904954 → 1 | 1.30889 → 0 | 0.01597514 → 0.00000300 | 0.00008999 |

tiny、small、medium 的最终参考字符数分别为 134,536、137,521、164,089。medium 旧版的 CER 与行级统计只含成功执行的 124 张，共 159,757 个参考字符。4 张失败单列。按独立预设复查，以前相同而现在不同的图片数均为零。成功空图分别为 11、10、0 张，与 Python 一致。

## 固定参考和定位证据

参考是本机 `E:\RapidOCR\python` 提交 `28490093dfb75dc4cd0f5d4f1d2259aeb44f5351` 的正常 RapidOCR 入口，ONNX Runtime CPU 1.24.4、OpenCV 4.12.0、NumPy 2.2.6、Python 3.12.10。沿用独立的 tiny、small、medium 输出，未变更已有参考或比较脚本。模型和字典取自 `..\SimdPaddleOCR\models`。每个预设的 `reference/<preset>/config.json` 含完整生效配置与文件 SHA-256。采集脚本重验输入与权重哈希。76 张 PNG、49 张 JPG 保留原图，3 张 WebP 两端读取由同一解码生成的无损 PNG。

- **检测后处理。** 同一检测图分别进入 Python DB 和原生 GPU DB，可单独复现膨胀锚点、孔轮廓、候选筛选、倾斜框和阅读顺序的差异。修正了 2×2 膨胀方向、轮廓候选的截断顺序、掩码评分、最小外接矩形、整数外扩与坐标映射。small 索引 39 的初始矩形为 15×9，两端得到相同矩形，外扩半径应为 4.5 像素。tiny 39 与 small 8、19 的倾斜框触发另一个边界：卡尺行列式为 `1 - 2^-24`，GPU 的倒数曾舍入为 1。通过相邻浮点数的乘积残差选择商后，物理 GPU 上的三个最小外接矩形和最终四角均与 OpenCV 逐值相同。
- **裁剪与归一化。** 对齐 OpenCV 透视系数、1/32 像素插值相位、15 位定点三次插值权重、边界及缩放坐标舍入。支持 Vulkan shaderFloat64 的设备使用双精度变换系数，其他 Vulkan 设备保留双单精度实现。例如 113 像素缩放到 319 像素时，目的列 220 的水平系数应为 1245。分开取整坐标的整数与小数部分曾得到 1246，从而改变一个像素值。用原始分子比较两个相邻 F32 坐标后，该例及宽坐标样本均得到 CPU 系数。高层 GPU 行分数按每个保留字符五位小数汇总。
- **模型算子。** 固定同一输入激活重放空间 ReduceMean，原有并行树改变了相加顺序。默认 Vulkan F32 路径按行起始地址重现 ORT 的四元素分组，每个线程沿固定顺序提前读取 16 组元素，区分全局池化的累加顺序，并修正最终除法。诊断程序只在测试时独立加载 ONNX Runtime。
- **峰值显存。** medium 索引 22、28、57、96 原报激活和输出需要 1,101,000,704 至 1,289,355,264 字节。物理计划按生命周期复用缓冲区，在原有时间顺序方案和大张量优先方案中选较小者。四张图均在原 1 GiB 配额下成功完成。

## 仍可测到的数值差异

对 tiny 39、small 8、medium 59，用同一份归一化检测张量运行两个执行器。F32 概率图的最大绝对差分别为 0.00003016、0.00000724、0.00002063。三张图在 0.3 阈值上的像素翻转均为零。交换概率图后再分别运行两套 DB，106、95、95 个框的顺序和坐标逐项一致。独立重放各自的第一层空间均值时，32、48、128 个输出均与 ORT 位级相同。

对识别分数差较大的行，记录 RapidOCR 在正常调用中实际送入识别模型的输入，将同一张量送入原生 GPU 识别图，再用同一个 Python CTC 解码器解码。tiny 13 的 `4` 为 0.50496 / 0.50498，small 42 的 `RJR` 为 0.92357 / 0.92369，medium 95 的 `000` 为 0.85673 / 0.85678。各行所有时间步的胜出类别相同。单独检查 small 的首个 RGB 卷积，输出最大差为 0.00000381。再将首个 ReLU 后的 ORT 激活交给 GPU 后续层，行分数从 0.92369 变成 0.92361，仍与参考 0.92357 有差距。切图会影响算子融合，以上数值用于定位。最终表始终使用完整公开入口。

**显式半精度。** `--gpu-precision mixed-f16 --gpu-matrix f16` 使用半精度存储、半精度矩阵操作数及单精度累加。各取一张经 CLI 复测，tiny 2 的编辑数 0 → 0，small 76 为 19 → 1，medium 59 为 891 → 110。最终 medium 漏 1 行。将同一 F32 输入交给混合精度检测模型，small 和 medium 分别有 68、202 个概率像素跨越 0.3 阈值。用两端 DB 处理同一张混合精度概率图，分别得到了完全相同的 29 和 95 个框。只将混合精度检测图代入完整 Python 后续流程，small 的一个字符编辑被完整复现，medium 相对标准参考已有 58 个编辑。medium 的完整混合精度文本又与这个诊断结果有 54 个编辑。编辑距离受插入、删除和对齐影响，不能相加。全量 384 个结论对应 Vulkan 默认 F32。这里的半精度结论限于上述样本。Metal 和其他 GPU 型号未在本机验证。

## 性能

计时样本为 `009b147315000d75c16a87c0bc2814b1.png`，764 × 950，三个预设各返回 6 行。使用同一张物理 GPU、8 个 CPU 线程、F32，模型先加载，计时从 `request.run` 开始到 `try_to_owned` 完成。图像解码、模型加载、JSON 序列化均在区间外。各版本总共预热 1 次，给 tiny，每个预设各正式测 1 次。small 与 medium 都是模型加载后的首个请求。

| 预设 | 修改前 ms | 最终 ms | 单次变化 |
| --- | ---: | ---: | ---: |
| tiny | 24.653 | 27.409 | +11.18% |
| small | 86.151 | 87.432 | +1.49% |
| medium | 249.224 | 251.932 | +1.09% |

tiny 单次增加 2.756 ms，本次样本上仍有性能代价。单次测量无法估计运行间波动，也无法据此外推整个图库的延迟分布。各版本性能执行均在 60 秒预算内，构建与全量正确性采集单列。

tiny 第一版对齐后的端到端耗时为 27.917 ms，比原版增加 3.264 ms。为定位这个变化，固定该图的 960 × 768 检测图及其 8 个连通区域，用 Vulkan 设备时间戳单独测 DB 几何着色器。旧版和对齐版均接受 6 个框，各单次调度为 12.958 和 6.218 ms。对同图首个空间均值的 `[1,32,240,192]` 激活，旧两阶段并行树为 0.279 ms，最大数值误差 0.00006461。初次逐位对齐的实现为 2.005 ms。对保持累加顺序的等价实现分别测一次，最终提前读取方案为 0.578 ms，32 个均值逐位一致。独立着色器计时只覆盖该算子，不能相加推算整图延迟。

最终版本通过 409 项常规 release 库测试，并在本机单独运行和通过了 20 项物理 GPU 测试，覆盖裁剪逐字节比较、检测几何、读序、空间均值、混合精度几何与后端策略。

## 复跑

在仓库根目录用 PowerShell 运行。复跑目录应使用新的名称。采集会校验模型、源图与已存在采集的可执行文件哈希，不同构建不能混用同一目录。

```powershell
$env:PYTHONUTF8 = '1'
cargo build --locked --offline --release --features gpu --example ocr --example rapidocr_gpu_capture --example rapidocr_tensor_probe
$exe = (Resolve-Path target/release/examples/rapidocr_gpu_capture.exe).Path
foreach ($p in @('tiny','small','medium')) {
    python tools/rapidocr_parity/gpu.py gpu-rerun $p --exe $exe --indices 2 59 76
}
foreach ($p in @('tiny','small','medium')) {
    python tools/rapidocr_parity/gpu.py gpu-rerun $p --exe $exe
}
python tools/rapidocr_parity/report.py gpu-before gpu-rerun
foreach ($p in @('tiny','small','medium')) {
    python tools/rapidocr_parity/gpu.py gpu-time-rerun $p --exe $exe --indices 2 --timing
}
python tools/rapidocr_parity/gpu_cli.py target/release/examples/ocr.exe gpu-rerun reports/rapidocr-parity/check-cli
python tools/rapidocr_parity/gpu_cli.py target/release/examples/ocr.exe gpu-rerun reports/rapidocr-parity/check-mixed --mixed
python tools/rapidocr_parity/gpu_replay.py small 8 reports/rapidocr-parity/check-map
python tools/rapidocr_parity/gpu_replay.py medium 59 reports/rapidocr-parity/check-map-half --mixed
python tools/rapidocr_parity/gpu_score.py small 42 5 reports/rapidocr-parity/check-score --capture gpu-rerun
```

模型、字典、输入和配置的原始记录在 `reports/rapidocr-parity`。修改前的采集目录为 `gpu-before`，修改后为 `gpu-aligned`。对照入口 `gpu-before-to-gpu-aligned.json` 同时列出原先完全一致的图片发生退化的情况。两次采集用的可执行文件 SHA-256 分别是 `56b7337ef493abee8f54f80d035b3c7f0db1a1f0d14fd09b0c506c6f1622acb7` 和 `2ec0619c9508d45787ee2678154bda39a592b8858945279bc1709188047f5e1f`。

旧版 release 的采集器已单独留在 `reports/rapidocr-parity/gpu-work/before-capture.exe`。要重新采集旧版，把上面两处 `gpu-rerun` 采集目录改为新的目录名并将 `$exe` 指向该文件，再用 `report.py` 比较两个新目录。

本机验收包 `reports/rapidocr-parity/gpu-acceptance.zip` 含三套独立 Python 输出、两版 GPU 的逐图输出、执行配置、聚合对照、硬件测试日志、关键诊断和两版采集程序。原图和无损 WebP 替代输入沿用原验收包。
