C# 与 Rust 的历史本机性能对比 — 2026-10-24

本页保留单一原生 Metal 实现重构之前的历史测量，尤其保留 CPU 对比证据。
下述 GPU 数据属于已移除的旧实现，不代表当前性能或当前 API；当前说明见
[Metal 性能证据](gpu-performance.md)。

这次历史实测中，父目录 C# 的原生 Metal 路径比当时 Rust Full GPU 快
**3.20 / 4.17 / 3.67 倍**，分别对应 Tiny / Small / Medium。
CPU 路径则是 Rust 更快，C# 耗时分别为 Rust 的 **1.07 / 1.61 / 2.40 倍**。
这是当时实现和默认精度策略的对比，不能解释成语言本身的性能差距。

全部运行在物理 Mac mini：Apple M4、10 核 GPU、16 GiB 统一内存、macOS 27。
没有使用 CI、远程执行或父库文档中的虚拟机成绩。
C# 来源为父目录 `SimdPaddleOCR`，版本
`33ff8a81672452041d110e8ba52dcd2639e54228`，源代码工作区保持干净。
Rust 为当时优化后的工作区，包含共享 pointwise 权重打包和选型改进。

两边使用相同 ONNX 模型、字典和解码后的 RGB 字节：仓库样本的
500×500 图片，共识别出 16 行。配置均为单请求、识别 batch=2、输出 CTC spans；
line workers=4、detector threads=8、preprocess workers=16。
Rust GPU 为 Metal、Full 模式、一个 GPU slot、FP32；C# 为显式
`OcrBackend.Metal`，使用原生混合 FP16 路径。两边 CPU 均为 FP32。

Rust 使用锁定依赖的 release/all-features 构建，固定 nightly-2026-09-08。
C# 使用 release 构建、父工作区固定的 .NET SDK 10.0.303 和 ARM64
.NET Runtime 10.0.11，设置 `DOTNET_TieredCompilation=0`。
本机原先没有可用 SDK 可执行文件，因此将对应的官方 macOS ARM64 SDK
解压到了用户缓存并核验 SHA512；没有修改系统安装或全局 PATH。
下载来源为 [Microsoft .NET 10 官方页面](https://dotnet.microsoft.com/en-us/download/dotnet/10.0)，
具体下载地址、校验值保存在本地 `runtime-metadata.json`。

下表为三次普通热运行的中位耗时，单位 ms，越低越好：

| 模型 | C# CPU | Rust CPU | C# Metal | Rust Full GPU | Rust GPU / C# Metal |
|---|---:|---:|---:|---:|---:|
| Tiny | 53.330 | 49.911 | 19.967 | 63.925 | 3.20× |
| Small | 195.350 | 121.024 | 43.492 | 181.491 | 4.17× |
| Medium | 873.212 | 364.060 | 162.961 | 598.185 | 3.67× |

| 模型/实现 | 三次热运行 ms |
|---|---|
| Tiny C# CPU | 53.330, 53.146, 53.777 |
| Tiny Rust CPU | 51.279, 49.785, 49.911 |
| Tiny C# Metal | 23.310, 19.967, 18.776 |
| Tiny Rust GPU | 63.953, 63.925, 63.911 |
| Small C# CPU | 193.863, 196.738, 195.350 |
| Small Rust CPU | 121.024, 119.733, 127.032 |
| Small C# Metal | 48.136, 43.492, 43.018 |
| Small Rust GPU | 181.895, 180.087, 181.491 |
| Medium C# CPU | 902.441, 863.404, 873.212 |
| Medium Rust CPU | 364.060, 354.286, 408.124 |
| Medium C# Metal | 162.961, 162.377, 165.736 |
| Medium Rust GPU | 598.185, 602.398, 595.088 |

共有 12 个配置，每个配置一次冷运行兼预热、三次普通计时，之后另做一次
不计入热运行样本的诊断 profile。总性能执行时间 **20.997 秒**，包括 harness、
进程启动、构建后的初始化、预热和 profile；不包括下载 SDK、构建和此前的正确性预检。
执行前设定了累计 59 秒上限，没有自动增加采样，也没有重跑相同配置挑选更好成绩。
Tiny C# Metal 的样本范围约 4.5 ms，Medium Rust CPU 的范围约 54 ms，
这些比值应视为该样本的小规模测量结果。

计时从启动外层请求工作线程，到最后一个 OCR 调用返回为止。
图片解码、模型/请求构建、输出校验、序列化和独立 profile 不在普通计时区间。
C# 的公开 OCR 调用返回拥有所有权的结果，相关数组/字符串构建在调用内；
Rust 返回借用视图，额外的 owned copy 在计时后执行。这是两边现有 API 的差异，
本次没有人为改写推理流程来统一它们。

输出检查需要与速度一起看：

- 三档模型的四条路径都给出相同的完整文字和 16 行检测结果。
  C# 与 Rust 的检测框坐标最大差约 `1.45e-5` 像素，低于已有 0.5 像素门限。
- C# CPU 和 C# Metal 的第八行均报告类别 1、旋转 180°；Rust 报告类别 0、旋转 0°。
  该行文字都是 `ODM OEM`。方向差异在 C# CPU 上也存在，不能全部归因于 GPU 半精度。
- Small 和 Medium 的该行分类置信度，在 C# CPU 与 Rust CPU 之间超出已有 FP32 分数门限。
- Small C# Metal 在第十一行存在一个 CTC span 结束列差异；它与 C# CPU、Rust CPU
  都不完全一致。Tiny/Medium 的 C# Metal 对 C# CPU 则通过既有混合精度检查。
- C# 所有 GPU session 都确认为 `Backends.Metal.MetalSession` 且 `GpuAlive=true`，
  adapter 为 `Apple M4`；冷运行和每次热运行都检查了回退状态。
- 12 个配置都通过各自固定输出参照、精确重复输出和 profile 输出一致性检查。
  **C# 没有通过与 Rust 的完整输出契约一致性门限**，上述差异没有被忽略或标为通过。

预检首先使用 Rust CPU 输出作为公共 oracle，保存了全部差异。由于方向元数据
已经不同，正式 C# 耗时测量改用其各自预检生成的固定输出验证稳定性，另行检查跨实现差异。
`reference_scope` 明确记录这种参照范围；C# timing JSON 的 `qualified=true`
只证明该范围内的稳定性，不能读成与 Rust 完全等价。
Rust 的正式测量仍使用 Rust CPU oracle。

C# Metal 的 [GEMM shader](../../SimdPaddleOCR/src/Sdcb.SimdPaddleOCR/Backends/Metal/Shaders/gemm.metal)
包含 `simdgroup_half8x8` 与 `simdgroup_multiply_accumulate` 矩阵路径；
其 [Metal session](../../SimdPaddleOCR/src/Sdcb.SimdPaddleOCR/Backends/Metal/MetalSession.cs)
仍有 CPU staging/CTC 等工作。当时 Rust 使用 wgpu WGSL 的 FP32 普通计算内核，
Full 模式把图像处理、投影和 CTC 保留在 GPU。精度、矩阵内核、布局、调度及 CPU/GPU
分工都不同。本次不是逐项消融实验，不能把全部速度差归给其中某一项。
结果说明，若继续缩小 Rust GPU 差距，原生矩阵路径与经过准确率验证的混合精度
比继续微调现有 FP32 内核更值得单独评估；本次没有实施新的推理优化。

为完成测试，仅修改了 Rust 仓库内的比较工具：
[OcrBench.cs](../tools/reference/OcrBench.cs) 增加 Metal 选择和实际后端检查；
[gpu_performance.py](../tools/gpu_performance.py) 增加 Metal C# 比较、明确的独立参照选项，
并把 `.metal` 资源和上层构建配置纳入源码哈希/归档。
默认公共 oracle 行为保留。父目录 C# 生产代码和 Rust 推理代码均未因这次对比而修改。
Release 构建与 Python 语法检查通过，全部 12 个正式配置的运行状态和语义字段已检查。

历史数据记录位于忽略目录 `reports/csharp-comparison-2026-10-24/`。
旧实现的冻结可执行文件和源码归档属于清理范围，不再承诺保留；JSON 中的
旧路径和哈希仅表示当时的来源，不是当前可执行文件或可用下载链接。

- `performance/`：模型/字典/输入哈希、精确命令、36 个普通热运行样本和
  12 个历史独立 profile 的数据记录。
- `qualification/`：C# 对 Rust 公共 oracle 的初始预检结果及原始错误列表。
- `csharp-references/`：用于稳定性验证的各实现/后端固定输出。
- `summary.json`：耗时和三个方向的输出比较，包含 C# GPU 对 C# CPU。
- `validation.json`：当时 164 个 Rust 构建输入、234 个 C# 构建输入、二进制、
  模型/字典和源码归档的历史核对结果，不表示旧实现文件仍被保留；当时父库
  Git 状态保持干净。
- `performance-budget.json`：累计执行时间和复现命令。

本次只测一张图片、一个并发配置和这台物理 M4。没有验证其他图片、其他 GPU、
持续吞吐、大批量请求或等精度的 C#/Rust GPU 对比；没有可比口径的 C# 设备内存统计。
