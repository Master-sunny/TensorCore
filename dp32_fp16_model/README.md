# FP16 DP32 -> FP32 点积行为模型

本工程实现 32 项点积加累加值：

```text
d = dot32(a_fp16[32], b_fp16[32]) + c_fp32
```

输出为 IEEE binary32 的原始 32 位编码。`c` 可省略，默认 +0。

来源：Brian Hickmann 等，*Intel Nervana Neural Network Processor-T (NNP-T)
Fused Floating Point Many-Term Dot Product*，IEEE ARITH 2020，pp.133-136，
DOI: 10.1109/ARITH48897.2020.00029。
公开原文：https://www.acsel-lab.com/arithmetic/arith27/pdfs/ARITH2020-38ozUqVB9oR8ZyQdVxXcK/712000a133/712000a133.pdf

**定位：这是按论文数据通路建立、把 BF16 输入替换为 FP16 的可复现模型。
不是 Intel 芯片的位精确复刻，也不宣称复现 NVIDIA Tensor Core。**
论文没有公开 37 位内部位分配、所有截断边界和特殊值行为，必须补充建模约定。
源码中的这些约定均固定且可测试。

## 1. 快速运行

需要支持 C++17 的编译器。模型和 C++ testbench 仅依赖标准库，
不需要 SystemC、Boost、GPU、Python 或 FP16 硬件指令。

Linux / WSL / Git Bash：

```bash
bash run_tests.sh
```

脚本编译 demo 和 testbench，并运行包内已有的参考向量。

也可以手动编译：

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Iinclude src/dp32.cpp src/demo.cpp -o demo
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Iinclude src/dp32.cpp tests/testbench.cpp -o testbench
./demo
./testbench tests/golden_vectors.txt
```

Windows MinGW 可使用同样编译命令，然后运行 `demo.exe`、
`testbench.exe tests/golden_vectors.txt`。需在工程根目录运行。

附带 CMake 配置（若电脑已安装 CMake）：

```bash
cmake -S . -B build-cmake
cmake --build build-cmake --config Release
ctest --test-dir build-cmake -C Release --output-on-failure
```

## 2. 文件

| 文件 | 用途 |
|---|---|
| `include/dp32.hpp` | 模型接口、配置、结果标志和逐阶段 trace |
| `src/dp32.cpp` | 整数方式实现的点积与最终 RNE |
| `src/demo.cpp` | 32 个 `1*2` 加 3，期望 67，并输出内部信息 |
| `tests/testbench.cpp` | 定向测试、FP16 全编码检查、参考向量及流水线检查 |
| `tests/golden_vectors.txt` | 独立 Python Fraction 参考程序生成的输入和预期结果 |
| `tools/generate_vectors.py` | 精确有理数参考程序，只在重新生成向量时需要 Python |
| `run_tests.sh` | 一键编译与测试 |
| `TEST_REPORT.txt` | 交付时实际运行记录 |

## 3. 论文过程与代码对应

| 数据通路 | 本工程实现 |
|---|---|
| 输入译码 | 直接解码 FP16 sign/exponent/fraction |
| 32 路乘法 | 11 位有效数相乘，保留完整整数乘积，不先舍入成 FP16 |
| Find Max Exponent | 对每个非零乘积计算规格化指数，再求最大值 |
| Align / Truncate | 所有乘积对齐到同一个二进制网格，按绝对值截断 |
| to 2's Complement | 截断后转换为 37 位补码 |
| 32:2 Carry-Save Compressor | 实际执行 3:2 压缩层，32 项归约到 2 项 |
| Carry Propagate Adder | 两个剩余向量相加，保留 37 位，恢复有符号值 |
| to Sign Magnitude / Normalize | 分离符号，左移规格化，保留全部剩余信息，不舍入 |
| Modified FP32 Adder | 宽中间值与 FP32 C 相加，不把 C 放进 32 项归约 |
| RND | 最后执行 RNE，生成 FP32 编码 |

论文的 completion adder、LZA 和时钟门控属于电路/时序优化，本行为模型
直接计算同样的数值结果，不模拟推测电路、门延迟或功耗。

### 3.1 明确的 37 位约定

对每个有限非零乘积 `p[k]`：

```text
E[k] = floor(log2(abs(p[k])))
Emax = max(E[k])
grid_exp = Emax - 30
Q[k] = sign(p[k]) * floor(abs(p[k]) / 2^grid_exp)
S = sum(Q[k])
dot_value = S * 2^grid_exp
result = RNE_FP32(dot_value + exact_value(c))
```

37 位补码中分配：1 位符号、5 位求和增长空间、31 位乘积有效数
（含领先位，因此小数精度为 30 位）。每项幅值严格小于 `2^31`，
32 项总幅值严格小于 `2^36`，可放入 37 位有符号数。

这一分配是**公开、保守的模型选择**，不是原文披露的布局。
这里的 37 位不是独立定义的 FP37 存储格式，也不是 FP16 的 block-scale 编码。

截断顺序也是模型约定：先对**绝对值向零截断**，再转补码。
这与对负补码直接算术右移可能不同；不得随意交换步骤。
源码中的宽度、有效数精度和位掩码相互约束，不应只改一个常量。

CSA 压缩层的项数为 `32 -> 22 -> 15 -> 10 -> 7 -> 5 -> 4 -> 3 -> 2`。
每层均保留模 `2^37` 的和，最终实际带符号结果不会溢出。
这些是压缩逻辑层数，不是 8 个时钟周期。

### 3.2 最终加法与规格化

归约结果的绝对值被无损左对齐到 37 位，同时调整指数，然后与 FP32 C 相加。
最终加法器使用软件多字整数完成精确对齐和有符号加减，再进行一次 RNE。
代码用 5 个 `uint64_t` 暂存最终加法的精确值，单位为 `2^-149`。

**这只是避免宿主机浮点误差的软件实现方法，不表示硬件含 320 位累加器。**
32 项点积在此之前已经经过了 37 位量化；最终加法不会找回丢掉的位。
FP16 有限乘积、37 位量化结果以及任意有限 FP32 C 均能在此暂存范围内精确表示。

### 3.3 特殊值、非规格化数及舍入

| 情况 | 约定 |
|---|---|
| FP16 正规数 | 支持全部有限编码 |
| FP16 subnormal | 默认渐进下溢；可配置输入 FTZ |
| FP32 C subnormal | 支持；不随 FP16 FTZ 配置改变 |
| 舍入 | 最终 RNE（最近，恰好中间时取偶） |
| 精确零 | 统一返回 +0，不保留全负零等情形的 -0 |
| NaN | 输出 canonical quiet NaN `0x7fc00000`，不保留 payload |
| sNaN | canonical NaN，设置 `invalid` |
| Inf * 0 | canonical NaN，设置 `invalid` |
| 正负 Inf 相加 | canonical NaN，设置 `invalid` |
| 单一符号 Inf | 输出对应符号 Inf |

NaN/Inf 由单独的特殊值通路处理，不尝试送入整数压缩树。
FTZ 在乘法前生效，因此 subnormal * Inf 在 FTZ 模式下会成为 0 * Inf。
这些规则用于让模型完整、确定，原文没有逐项定义这些行为。
原论文 BF16 实现不支持 subnormal；本工程默认支持 FP16 subnormal 是明确的扩展。

`alignment_truncated` 表示对阶时有信息丢失；
`final_round_inexact` 只表示最终加法到 FP32 的舍入不精确。
两者不能混为一谈，它们也不是完整的 IEEE 异常状态寄存器。

## 4. C++ 调用

```cpp
#include "dp32.hpp"
dp32::HalfVector a{}, b{};
a.fill(0x3c00); // FP16 1.0 的编码
b.fill(0x4000); // FP16 2.0 的编码
dp32::Trace trace;
auto r = dp32::dot(a, b, dp32::float_to_bits(3.0f), {}, &trace);
float d = dp32::bits_to_float(r.bits); // 67.0
```

`a[i] = 1` 表示 FP16 最小正 subnormal，不表示 FP16 的 1.0。
外部输入建议保留原始编码，避免通过 float/double 的隐式类型转换。
`trace` 可为空；非空时可观察各项乘积、对阶位移、截断标志、补码、
压缩树每层、归约结果和规格化结果。

只需纯点积时调用 `dp32::dot(a,b)`。本工程只实现单个 DP32；
不实现整个 32x32 MPU、A 双缓冲、PP 存储器和总线仲裁。

## 5. 流水延迟接口

`Pipeline9` 是可选的功能级时间适配器：接收时算出结果，再延迟输出。
它不是逐阶段执行内部算法的 SystemC/RTL 时序模型。

- 第 n 次 tick 接收输入，第 n+9 次 tick 输出该结果。
- 启动间隔 II=1；支持空泡和 tag，不支持输出反压。
- 空输入使用 `pipe.tick()`；复位使用 `pipe.reset()`。
- NaN/Inf 与普通数据采用同样的接口延迟。
- 所有请求相互独立；需跨请求累加时，由外部提供相应的 FP32 C。

论文对应的 9 周期数值工作划分是：

| 周期 | 操作 |
|---|---|
| 1 | 暂存、译码 |
| 2-3 | 乘法、指数、最大指数与对阶位移 |
| 4 | 对阶、转补码、前半压缩归约 |
| 5 | 后半归约、CPA、LZA 和符号转换 |
| 6 | 规格化 |
| 7-9 | 与 C 相加及最终舍入 |

## 6. Testbench 与独立参考

测试分为三个来源，避免把顺序 `std::fma` 误当成论文模型的位精确 golden：

1. **手写定向结果**：全 1、全负 1、RNE 奇偶 tie、subnormal、
   后加 C、抵消暴露截断等。
2. **FP16 全编码检查**：对 65536 个输入编码执行单有效乘积 `a*1`，
   分别检查默认和 FTZ 模式；使用独立 `ldexp` 转换作参考。
3. **Python Fraction golden**：用任意精度有理数计算每项绝对值量化和总和，
   不复制 C++ 37 位补码/CSA 的实现；同时生成完全精确点积的 FP32 RNE 结果。

包内包括 10054 个参考向量，随机 seed 为 20261016，覆盖：
宽指数分布、窄指数分布、正负抵消、稀疏、随机特殊编码、FP32 C、
FP16 输入 FTZ 开关等。每个向量检查输出 bits 和所有结果标志。
另外检查每层 CSA 的模和不变、符号恢复、位宽界限、规格化一致性，
以及流水接口的 valid、tag、延迟、空泡和复位。

修改或扩大测试向量时运行：

```bash
python3 tools/generate_vectors.py --count 5000 --seed 20261016
bash run_tests.sh
```

向量包含两种结果：

- `model`：按本工程公开的 37 位截断规则计算；C++ 必须逐位一致。
- `exact`：FP16 输入精确乘积求和加 C，最后只舍入一次；用于观察数值损失。

`model != exact` 的数量会被报告，不作为测试失败。
例如两个很大的相反乘积完全抵消，另有一个很小的乘积：小项可能在统一对阶时
已经被丢弃，因此该模型会得到 0，精确点积却非零。
**本工程不承诺与精确点积或顺序 FMA 的误差始终小于 16 ULP。**
整体量化过程决定了这种误差不存在普遍的小 ULP 上界。

## 7. 适用范围与交付验证

可用于点积数值行为研究、C-model 接入、与不同归约/截断规则比较。
若要匹配具体芯片，仍需拿到它的位布局、移位截断、sticky bit、
特殊值、舍入和累加语义，然后相应修改模型与独立参考。

交付时用 GCC 严格警告编译并执行完整 testbench，同时执行 ASan/UBSan。
运行环境限制了 LeakSanitizer 对 `/proc/task` 的访问，因此检查时关闭了
LeakSanitizer；ASan 的内存访问检查和 UBSan 仍然启用，不宣称通过泄漏检查。
CMake 文件和 Windows 命令作为可选入口提供；若未在对应环境运行，
不视为已经验证。实际编译选项和结果见 `TEST_REPORT.txt`。
