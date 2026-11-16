# 固定 SRAM 容量下的性能扫描

在同一个配置点上，单槽和双槽获得完全相同的 SRAM 容量、独立读写端口和计算资源。可以固定 tile 比较预取，也可以给定多个候选 tile，由每种策略在该点的容量约束内独立选择延迟最低的组合。扫描容量时，每一个容量都是一场独立比较。入口为 [perf_sweep.py](scripts/perf_sweep.py)，时序和数据流见[模型正文](pingpong_sram_tradeoff.md)。

## 1. 参数列表

一个 JSON 文件描述一个算子。表内列出的硬件和工作负载字段都可以是 `fixed` 中的单值或 `sweep` 中的离散值列表。每个适用字段须出现一次。未知量的候选范围由配置者给出；同一次扫描中未指定取值或范围的必要参数会报错。纯 GEMM 无需 Vector 参数，独立 Vector 算子无需 Tensor 参数。可以在 `fixed` 中保留芯片上其他引擎的规格，扫描未被本算子使用的引擎会报错。

| 硬件字段（精确 JSON 名） | 单位与含义 | 影响的事件 |
|---|---|---|
| `capacity_kib` | KiB，每个独立执行分区实际可用的共享 SRAM；1 KiB = 1024 B | 扣除其他任务与常驻空间后给定。先判峰值占用，再选 tile；可用小数表示小于 1 KiB 的容量。 |
| `ddr_read_gbs`、`ddr_read_ns` | DDR 持续读 GB/s、每次读路径启动 ns | DMA 装入输入、权重或 KV。 |
| `ddr_write_gbs`、`ddr_write_ns` | DDR 持续写 GB/s、每次写路径启动 ns | DMA 回写结果。 |
| `dma_gbs`、`dma_ns` | 一个 DMA 的持续 GB/s、每次非空逻辑传输启动 ns | 两个传输方向都经过此引擎；本文调度将输入、输出 DMA 串开。 |
| `sram_read_gbs`、`sram_read_ns` | SRAM 持续读 GB/s、每次逻辑读服务启动 ns | 计算引擎读操作数或旧部分和，DMA 读出最终结果。 |
| `sram_write_gbs`、`sram_write_ns` | SRAM 持续写 GB/s、每次逻辑写服务启动 ns | DMA 填充下一槽；Tensor/Vector 写中间值、部分和或最终结果。并发写共享此带宽。 |
| `tensor_tflops`、`tensor_ns` | 当前精度下的有效 TFLOP/s、每个矩阵乘板的启动 ns | GEMM、QK、PV。约定一次乘加为 2 FLOP。 |
| `vec_add_gops`、`vec_add_ns` | 十亿元素操作/s、每个非空逻辑原语启动 ns | 加、减共用该规格。 |
| `vec_mul_gops`、`vec_mul_ns` | 同上 | 乘法、缩放。 |
| `vec_reduce_gops`、`vec_reduce_ns` | 十亿次二元操作/s、每次归约或行最大值比较启动 ns | max/sum 按二元操作计数，宽度为 c 的一行需 c−1 次；SwiGLU 无此项。 |
| `vec_exp_gops`、`vec_exp_ns` | 十亿元素操作/s、每次原语启动 ns | 指数。 |
| `vec_inv_gops`、`vec_inv_ns` | 十亿元素操作/s、每次原语启动 ns | 倒数。 |

带宽采用十进制 GB/s，数值等于 B/ns；1 TFLOP/s = 1000 FLOP/ns；1 G 元素操作/s = 1 操作/ns。延迟可为零，容量与持续速率须为有限正数。启动延迟以一次模型事件为单位。用两种以上块长的实测时间拟合截距和斜率时，应采用相同的事件边界；若实测截距已覆盖整条 DMA 路径，应只填一次，以免与端口启动重复计时。

| `kernel` | 维度和输入/输出 | 数据类型字段 | `tiles` 候选轴、缓冲对象 |
|---|---|---|---|
| `gemm` | `M,K,N`；X[M,K]、W[K,N] → Y[M,N] | `x_dtype,w_dtype,y_dtype,partial_dtype` | `m,n,k`；分别枚举行块、列块、内积板。单槽、仅 X 双槽、仅 W 双槽、两者双槽。 |
| `flashattention` | `q_len,kv_len,d,dv`；Q[q_len,d]、K[kv_len,d]、V[kv_len,dv] → O[q_len,dv] | `q_dtype,k_dtype,v_dtype,score_dtype,prob_dtype,partial_dtype,out_dtype` | `r,c`；Q 行块和 KV 长度块。每次顺序处理一个 Q tile，给 K/V 对分配一槽或两槽。 |
| `online_softmax` | `rows,length`；输入分数和输出概率同形状 [rows,length] | `x_dtype,out_dtype` | `v`；每行每板的元素数。两遍读取输入，每行串行处理，输入单槽或双槽，输出另分一槽。 |
| `swiglu` | `rows,length`；gate、up、输出同形状 [rows,length] | `gate_dtype,up_dtype,out_dtype` | `v`；每个输入的槽数分别可取 1、2。`X_only` 对应 gate，`W_only` 对应 up。 |

dtype 接受 `fp8/int8/uint8`（1 B）、`bf16/fp16/int16`（2 B）、`fp32/int32`（4 B）、`fp64/int64`（8 B），或直接给正整数 B/元素。例如 `2` 表示每元素两个字节。维度与 dtype 决定各张量逻辑字节数，例如 BF16 X[2048,3584] 为 14,680,064 B，实际 DDR 读量还取决于被多少列块重复使用。类型用于计算容量和流量。若类型变化也改变算力，应分别给出匹配的算力配置；脚本不模拟低精度数值误差、隐式类型转换或亚字节打包。FlashAttention 的 `partial_dtype` 决定局部 PV 与历史分子在 SRAM 的存储宽度，行标量按寄存器驻留计算。独立 Vector 算子的寄存器中间值同理。

| 配置控制项 | 填法与含义 |
|---|---|
| `kernel` | 从上表选四种数据流之一。仅凭输入/输出形状不能确定重读次数、操作数或中间状态。 |
| `fixed` | 一组已知参数，名称到单值的对象。 |
| `sweep` | 名称到非空离散列表的对象，所有轴取笛卡尔积。可省略或写成空对象以评估一个点。已知参数与扫描参数不能重名。 |
| `tiles` | 每一轴给出正整数列表；各轴做笛卡尔积，超过整张量相应维度的候选截断并去重。每轴只给一个值即固定 tile。最优只针对给定候选集；实际硬件的对齐、阵列粒度应体现在候选值中。 |
| `plot_at` | 可选，从各扫描轴的候选中各选一个作参考。绘制一维曲线时只变化横轴；二维热图只变化两个带宽轴。省略的参考值取该轴列表第一个。 |

统一入口使用可行窗口内理想预取、额外槽位切换开销为零。原 GEMM 对照 CLI 的经验修正 `--overlap` 与 `--switch-ns` 分别对应 1 和 0。独立参数表描述单算子、单执行分区；batch、头间 KV 复用、跨算子并发、bank 冲突与缓存命中须有相应数据流才能计入。

## 2. 固定容量如何进入计算

对给定配置点和策略，先算每个候选 tile 的存活字节，超出容量即淘汰。以 GEMM 为例，令四种存储宽度分别为 bX、bW、bY、bP，输出与部分和复用空间，X、W 槽数为 sX、sW。峰值占用为

$$S=mn\max(b_P,b_Y)+k(s_X b_X m+s_W b_W n).$$

DDR 读 X、W 分别为 $b_X MK\lceil N/n\rceil$ 和 $b_W KN\lceil M/m\rceil$，写 Y 为 $b_Y MN$。相邻 K 板之间部分和落 SRAM，每方向额外产生 $b_P MN(\lceil K/k\rceil-1)$ B。第 1–4 节的 FP32 算例是 $b_P=4$ 的特例。因此同一容量下，改用双槽后选到更小的 m 或 n 会增加片外重读；若只缩短 k，DDR 字节不变，但部分和落地次数会上升。

对 FlashAttention 的完整候选 tile，Q/O 共用缓冲，score/P 共用 scratch，分子与局部 PV 各预留一份。K/V 成对开 s 份，峰值为

$$S=\max(rd\,b_Q,rd_v\,b_O)+s\,c(d\,b_K+d_v\,b_V)
+rc\max(b_S,b_{prob})+2rd_v\,b_P.$$

这里 $b_P$ 表示分子存储宽度，$b_{prob}$ 为概率宽度。每个 Q 块从 DDR 读一次对应的 Q，遍历全部 KV，再写一次 O。固定总形状时，仅减小 KV 板宽 c 会增加 KV 启动及历史状态更新次数；减小 r 还可能增加对整个 KV 的重读。实现支持单头、稠密、无掩码前向和尾板，沿 Q 块串行；因果跳块、GQA 共用 KV 和整网推理不在该算例中。

每次 DMA 的非空装入耗时为三个启动延迟之和，加上字节量除以 DDR 读、DMA、SRAM 写的最小带宽。写回沿 SRAM 读、DMA、DDR 写取对应的三项。单个计算阶段耗时取引擎算术、SRAM 读服务、SRAM 写服务三者的最大值。Tensor 时间为启动加 FLOP 数除以算力；Vector 依赖原语的时间依次相加，每项为启动加元素操作数除以该原语吞吐。GEMM 和两类 Vector 的单阶段预取按正文第 2 节的共享写时窗计算，首板装入、尾板执行、结果写回均计时。

FlashAttention 的一次 KV 预取可跨多个阶段。对非零启动延迟，示例采用一个具体的端口服务假设。令 $t_i,w_i$ 为当前计算阶段时间与其 SRAM 写字节，$a_i$ 为 DMA 装入路径剩余启动时间（初始为三个启动延迟之和），$e_i$ 为计算阶段的 SRAM 写启动延迟（无写则为 0）。引擎在前 $e_i$ 时间准备写端口，之后以恒定速率 $u_i=w_i/(t_i-e_i)$ 写出，无写时取零。第 i 阶段能预取的字节为

$$p_i=\min\!\left(z_{rem},\ [t_i-\max(a_i,e_i)]_+\min(B_{Dr},B_A,[B_W-u_i]_+)\right),\qquad [x]_+=\max(x,0).$$

逐阶段扣除 $p_i$，并更新 $a_{i+1}=[a_i-t_i]_+$。到下一板的消费屏障，剩余等待为 $a_{end}+z_{rem}/\min(B_{Dr},B_A,B_W)$。这样一条 DMA 只付一次启动；引擎写满本阶段 SRAM 写端口时，预取等待空闲窗口。零启动时该式退化为正文第 5.4 节公式。非零延迟结果取决于该阶段内端口服务假设，实际仲裁若允许在引擎启动期插入其他写，应按端口时序校准。

## 3. 运行与读取曲线

在本目录的 WSL shell 下运行，使用 Python 3.10+；示例出图使用 Matplotlib 3.10.5。传入 `--no-plot` 可只用 Python 标准库生成配置快照和结果表。每次实验指定独立的输出目录。

~~~bash
python3 -B scripts/perf_sweep.py configs/flashattention_sram.json --out artifacts/sweeps/flashattention_sram
python3 -B scripts/perf_sweep.py configs/gemm_capacity.json --out artifacts/sweeps/gemm_capacity
python3 -B -m unittest discover -s scripts -p 'test_*.py'
~~~

[FlashAttention 配置](configs/flashattention_sram.json)固定了形状、dtype、Tensor/Vector、DDR、DMA、112 KiB SRAM 及 32×64 tile。四轴的扫描部分如下，共 5×5×3×3 = 225 个点，每点两条策略记录。

~~~json
{
  "sweep": {
    "sram_read_gbs": [1000, 2000, 4000, 6000, 8000],
    "sram_write_gbs": [1000, 2000, 4000, 6000, 8000],
    "sram_read_ns": [0, 2, 8],
    "sram_write_ns": [0, 2, 8]
  }
}
~~~

要把容量也列为外层变量，可以从 `fixed` 删除 `capacity_kib`，在 `sweep` 加入 `"capacity_kib": [80, 96, 112]`。若想同时比较重分块，再将 `tiles` 改为 `{"r":[16,32],"c":[32,64]}`。前一项决定每个设计点的资源，后一项决定各策略能尝试的映射；若要测量某个特定 tile 的性能，给各轴一个候选即可。

每个目录输出 `config.json`（输入快照）、`results.csv`（所有组合）和 `curves.png`。CSV 中 `point` 与各扫描列共同标识条件，`policy` 标识策略，`tile_*` 是该策略的最快可行分块，`bytes_required` 是其峰值占用。四个 `*_bytes` 分别累计 DDR 读、DDR 写、SRAM 读、SRAM 写。比较指标为

$$\mathrm{gain\_ms}=T_{single}^*-T_s^*,\qquad \mathrm{speedup}=T_{single}^*/T_s^*.$$

其中 `ms` 是整个输入形状在该策略下的延迟。负收益和小于 1 的加速比保留；无可行候选的行标 `infeasible`，时间、tile 和流量留空，曲线在此处断开。图中连线只连接采样值，不能据此断定相邻容量之间的精确拐点。离散可行性变更可用容量式求出，再补扫该门槛两侧。

## 4. 两组结果如何解释

| FlashAttention，112 KiB，其他参数固定在参考点 | 双槽耗时 | 相对参考点 |
|---|---:|---:|
| 读、写各 4000 GB/s，两个端口启动均为 0 ns | 214.032 ns | 单槽为 239.568 ns；双槽省 25.536 ns |
| 只把读带宽调到 8000 GB/s | 209.168 ns | −4.864 ns |
| 只把写带宽调到 8000 GB/s | 214.032 ns | 0 ns |
| 只把写带宽降到 2000 GB/s | 225.296 ns | +11.264 ns |
| 只把读启动延迟调到 8 ns | 254.032 ns | +40.000 ns |
| 只把写启动延迟调到 8 ns | 280.912 ns | +66.880 ns |

参考点的 QK、PV 等阶段受 SRAM 读供数约束。当前写端口足以同时承接计算写和当时可以发出的 KV 预取，进一步提高写带宽没有缩短时间。降低写带宽或增加写启动延迟会挤占下一板预取窗口。读启动从 0 增至 8 ns 时，双槽相对单槽的加速比从 1.119 升至 1.164，但双槽本身也变慢了 40 ns，所以选硬件时应同时看绝对耗时和相对收益。各单轴曲线固定其余三轴，图底的热图固定两个启动延迟，展示读写带宽的联合影响。[FlashAttention 曲线与热图](artifacts/sweeps/flashattention_sram/curves.png) · [完整 450 行 CSV](artifacts/sweeps/flashattention_sram/results.csv)

[GEMM 配置](configs/gemm_capacity.json)取 5 个容量和 2 种部分和类型，共 10 个点、40 条策略记录。固定 FP32 部分和、k=64，320 KiB 时单槽可容纳 256×256 输出 tile，双槽只能选 128×256，两者分别为 14.202 和 13.140 ms。384 KiB 时双槽也能保留 256×256，DDR 少重读一部分权重，双槽降到 8.887 ms。320 KiB 下若部分和按 BF16 存储，双槽也能保留 256×256，耗时为 8.868 ms。后者假定相同的 Tensor 算力，仅评估部分和存储宽度对容量和字节的影响；累加精度和模型质量需另行验证。[GEMM 曲线](artifacts/sweeps/gemm_capacity/curves.png) · [完整 40 行 CSV](artifacts/sweeps/gemm_capacity/results.csv)

验证包括既有逐板 GEMM 回放及 FlashAttention 递推对照、可手算的两遍 softmax（单槽 54 ns、双槽 48 ns）、SwiGLU 的四种缓冲方式、不同部分和宽度、跨阶段启动、尾板、不可行容量与重选 tile。以上时间均为指定调度的分析模型值。
